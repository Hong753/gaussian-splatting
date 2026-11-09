#!/usr/bin/env python3
"""Render actual-image comparisons for saved train/eval views and multiple runs."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.view_comparisons import (atomic_json, image_id, model_info,
                                      shared_iteration, write_gallery)


def resolve_models(models, model_root, runs):
    if models:
        paths = [Path(p).expanduser().resolve() for p in models]
    elif model_root and runs:
        root = Path(model_root).expanduser().resolve()
        paths = []
        for name in runs:
            matches = sorted({p.parent.resolve() for p in root.rglob("cfg_args") if p.parent.name == name})
            if len(matches) != 1:
                raise ValueError(f"Expected one run named {name!r} under {root}, found {matches}")
            paths.extend(matches)
    else:
        raise ValueError("Specify --models RUN [RUN ...] or --model-root ROOT --runs NAME [NAME ...]")
    if len(paths) != len(set(paths)):
        raise ValueError("The same model was supplied more than once")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["train", "eval", "both", "all"])
    parser.add_argument("-m", "--model", "--models", nargs="+", dest="models")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--runs", nargs="+")
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--split", choices=["train", "eval", "both", "all"])
    split.add_argument("--train", action="store_const", const="train", dest="split")
    split.add_argument("--eval", action="store_const", const="eval", dest="split")
    parser.add_argument("--iteration", type=int, default=-1, help="-1: latest common saved iteration")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-views", type=int, default=0, help="0: all views; otherwise evenly spaced")
    parser.add_argument("--antialiasing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_views < 0 or args.iteration == 0 or args.iteration < -1:
        parser.error("max-views must be nonnegative; iteration must be positive or -1")
    if args.mode and args.split and args.mode != args.split:
        parser.error("Positional mode and --split disagree")
    mode = args.split or args.mode or "both"
    try:
        paths = resolve_models(args.models, args.model_root, args.runs)
        specs = [model_info(p) for p in paths]
        iteration = shared_iteration(specs, args.iteration)
        if args.output:
            output = args.output.expanduser().resolve()
        else:
            parent = paths[0] if len(paths) == 1 else Path(os.path.commonpath(paths))
            tag = "" if len(paths) == 1 else image_id("|".join(map(str, paths))) + "_"
            output = parent / "view_comparisons" / f"{tag}iteration_{iteration}"
        labels = [p.name for p in paths]
        if len(labels) != len(set(labels)):
            labels = [str(p) for p in paths]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        completed = []
        for split_name in (["train", "eval"] if mode == "both" else [mode]):
            sets = [s["splits"][split_name] for s in specs]
            names = sorted(set.intersection(*sets))
            if not names:
                if mode == "both" and split_name == "eval" and all(not s for s in sets):
                    print("No held-out eval views in these runs; train report only.")
                    continue
                raise ValueError(f"No shared {split_name} views in the supplied models")
            total = len(names)
            if args.max_views and len(names) > args.max_views:
                names = [names[round(i * (len(names)-1) / max(1, args.max_views-1))]
                         for i in range(args.max_views)]
            folder = output / split_name
            print(f"{split_name}: {len(names)}/{total} shared views at iteration {iteration}; {folder}", flush=True)
            if args.dry_run:
                continue
            atomic_json(folder / "selection.json", {"names": names, "split": split_name})
            model_rows = []
            for index, spec in enumerate(specs):
                destination = folder / f"model_{index}"
                python = spec.get("metadata", {}).get("python", sys.executable)
                cmd = [python, str(ROOT / "scripts/render_views_worker.py"),
                       "--model", str(spec["path"]), "--iteration", str(iteration),
                       "--selection", str(folder / "selection.json"), "--output", str(destination)]
                if args.antialiasing is not None:
                    cmd.append("--antialiasing" if args.antialiasing else "--no-antialiasing")
                print(shlex.join(cmd), flush=True)
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
                model_rows.append(json.loads((destination / "frames.json").read_text()))
            # Verify the actual targets agree before showing a single shared GT column.
            for name in names:
                if len({r[name]["target_sha256"] for r in model_rows}) != 1:
                    raise ValueError(f"Ground truth differs across models for {name}; use separate reports")
            rows = []
            for name in names:
                entries = []
                for index, records in enumerate(model_rows):
                    record = records[name]
                    entries.append({k: f"model_{index}/{record[k]}" for k in ("image", "error", "comparison")})
                    entries[-1]["psnr_db"] = record["psnr_db"]
                rows.append(dict(name=name, gt="model_0/" + model_rows[0][name]["gt"], renders=entries))
            models = [dict(label=label, path=str(s["path"]), type=s["type"], iteration=iteration)
                      for label, s in zip(labels, specs)]
            for model, records in zip(models, model_rows):
                mean = sum(float(records[n]["psnr_db"]) for n in names) / len(names)
                model["mean_psnr_db"] = round(mean, 5) if math.isfinite(mean) else "inf"
            write_gallery(folder, f"{split_name.title()} views · iteration {iteration}", rows, models,
                          dict(split=split_name, total_shared_views=total, rendered_views=len(names),
                               excluded_nonshared_views=[len(s)-total for s in sets],
                               pose_protocol="each model's saved source calibration",
                               exposure_protocol="base appearance, no per-image exposure",
                               metric="mean per-channel PSNR, matching scripts/eval_all_frames.py"))
            completed.append(folder / "index.html")
            print(f"Open {folder / 'index.html'}", flush=True)
        if completed:
            output.mkdir(parents=True, exist_ok=True)
            links = "".join(f'<li><a href="{p.relative_to(output).as_posix()}">{p.parent.name}</a></li>' for p in completed)
            (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Rendered views vs actual images</h1><ul>' + links + '</ul>')
            print(f"Comparison report: {output / 'index.html'}", flush=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Rendering failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

