#!/usr/bin/env python3
"""Print saved PSNR summaries and model paths; no CUDA and no invented metrics."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_all_frames import read_config, choose_iteration


def discover(root):
    skip = {"point_cloud", "proxygs_data", "failed_attempts", "view_comparisons", "comparisons", "coverage_cache"}
    for folder, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d not in skip]
        if "cfg_args" in files:
            yield Path(folder)
            directories[:] = []


def number(value):
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def metrics(path, iteration):
    files = list((path / "psnr_all_frames").glob("iteration_*/summary.json"))
    summaries = [json.loads(p.read_text()) for p in files]
    eligible = [s for s in summaries if iteration == -1 or s["iteration"] == iteration]
    return max(eligible, key=lambda s: s["iteration"]) if eligible else None


def print_table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        print("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--runs", nargs="+", help="Exact run folder names; omit to list all")
    parser.add_argument("--iteration", type=int, default=30000, help="-1: latest available metric per run, disclosed in table")
    parser.add_argument("--list", action="store_true", help="Print run paths and latest checkpoints only")
    args = parser.parse_args(argv)
    paths = sorted(discover(args.root.expanduser().resolve()))
    if args.runs:
        paths = [p for p in paths if p.name in args.runs]
        missing = set(args.runs) - {p.name for p in paths}
        if missing:
            parser.error(f"Run names not found: {sorted(missing)}")
    if not paths:
        parser.error("No cfg_args files found under the supplied root")
    rows, view_rows = [], []
    for path in paths:
        cfg = read_config(path / "cfg_args")
        proxy = (path / "proxygs_run.json").is_file()
        try:
            latest, _ = choose_iteration(path, -1, "proxygs" if proxy else cfg.get("model_type", "3dgs"))
        except FileNotFoundError:
            latest = "none"
        if args.list:
            rows.append([path.name, latest, str(path)])
            continue
        result = metrics(path, args.iteration)
        count = None
        if (path / "keyframes.json").is_file():
            count = len(json.loads((path / "keyframes.json").read_text())["selected_images"])
        if proxy:
            metadata = json.loads((path / "proxygs_run.json").read_text())
            prepared = json.loads(Path(metadata["preparation"]).read_text())
            count = sum(r["split"] == "train" for r in prepared["frames"])
        score = result.get("mean_psnr_db", {}) if result else {}
        counts = result.get("counts", {}) if result else {}
        aa = result.get("antialiasing") if result else cfg.get("antialiasing")
        rows.append([path.name, result["iteration"] if result else args.iteration,
            "N" if proxy else ("?" if aa is None else "Y" if aa else "N"),
            "Y" if cfg.get("train_test_exp", False) else "N", counts.get("selected", count or "—"),
            number(score.get("all")), number(score.get("selected")), number(score.get("unselected")),
            counts.get("all", "—"), latest])
        # Show train/eval gallery scores separately; their 'all' need not mean all registered frames.
        for file in sorted((path / "view_comparisons").glob("iteration_*/*/summary.json")):
            data = json.loads(file.read_text())
            for model in data.get("models", []):
                if Path(model["path"]).resolve() != path.resolve():
                    continue
                if args.iteration != -1 and model["iteration"] != args.iteration:
                    continue
                view_rows.append([path.name, model["iteration"], data["split"],
                    data["rendered_views"], data["total_shared_views"], number(model["mean_psnr_db"])])
    if args.list:
        print_table(["Run", "Latest checkpoint", "Model path"], rows)
    else:
        print("All registered-frame PSNR (base appearance; valid = unselected frames):")
        print_table(["Run", "Metric steps", "AA", "EC", "Keyframes", "PSNR all", "PSNR train", "PSNR valid", "Evaluated frames", "Latest checkpoint"], rows)
        if view_rows:
            print("\nSaved image-report scores (separate protocol; not necessarily all registered frames):")
            print_table(["Run", "Steps", "Split", "Rendered views", "Available shared views", "PSNR"], view_rows)
        print("\n— = no saved metric for this iteration; ? = AA absent from legacy config. No training status is inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
