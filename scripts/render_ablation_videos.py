#!/usr/bin/env python3
"""Shared random-camera videos or GT/captured-camera videos for saved ablations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts import render_model_comparison as core
from scripts.render import resolve_models
from scripts.view_comparisons import atomic_json, model_info, read_colmap_records, shared_iteration
from scripts.eval_all_frames import choose_iteration
from utils.colmap_paths import colmap_model_dir
from scripts.occlusion_options import add_arguments, apply_overrides


def camera(record):
    import numpy as np
    return SimpleNamespace(image_name=record.get("name", record.get("image_name")),
        R=np.asarray(record["R"]), T=np.asarray(record["T"]), width=record["width"],
        height=record["height"], FovX=record["fov_x"], FovY=record["fov_y"])


def camera_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def registered_source(spec, source):
    """Accept the original scene or a refinement sharing its unchanged LiDAR PLY."""
    saved = Path(spec.get("metadata", {}).get("source", spec["cfg"]["source_path"])).resolve()
    if saved != source:
        cloud = colmap_model_dir(source) / "points3D.ply"
        other = colmap_model_dir(saved) / "points3D.ply"
        if not cloud.is_file() or not other.is_file() or not os.path.samefile(cloud, other):
            raise ValueError(f"{spec['path']}: different source without a shared LiDAR PLY; cannot establish a common world frame")
    return saved


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, choices=["random", "actual"])
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--models", "--model", nargs="+")
    p.add_argument("--model-root", type=Path)
    p.add_argument("--runs", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--aa", nargs="+", choices=["on", "off", "auto"],
                   help="One setting per model; auto uses saved AA, or on for legacy native runs")
    p.add_argument("--iteration", type=int, default=-1, help="Latest common checkpoint by default")
    p.add_argument("--iterations", type=int, nargs="+", help="Explicit per-model iterations; labels disclose mismatches")
    p.add_argument("--actual-poses", choices=["source", "model"], default="source",
                   help="source: identical poses; model: each run's calibration for matching GT images")
    p.add_argument("--output", "--output-dir", required=True, type=Path)
    p.add_argument("--gpu", default="6")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--duration", type=float, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-window", type=int, default=600)
    p.add_argument("--control-step", type=int, default=24)
    p.add_argument("--deviation", type=float, default=3.0)
    p.add_argument("--max-distance-fraction", type=float, default=.10)
    p.add_argument("--view-degrees", type=float, default=12.)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--in-process", action="store_true", help="Render native 3DGS comparisons without replacing the Python process")
    add_arguments(p, per_model=True)
    return p


def build_plan(args):
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if not all(math.isfinite(x) and x > 0 for x in (args.fps, args.duration)) or not 0 <= args.crf <= 51:
        raise ValueError("fps/duration must be positive; CRF must be 0..51")
    if output.exists() and any(output.iterdir()) and not args.dry_run:
        raise ValueError(f"Use an empty output directory: {output}")
    paths = resolve_models(args.models, args.model_root, args.runs, allow_duplicates=True)
    if len(paths) < 2:
        raise ValueError("Supply at least two models for a comparison")
    specs = [model_info(path) for path in paths]
    if args.in_process and any(spec['type'] != '3dgs' for spec in specs):
        raise ValueError('--in-process videos require native 3DGS models')
    modes = args.occlusion or ["saved"] * len(specs)
    if len(modes) != len(specs):
        raise ValueError("Provide one --occlusion setting per model")
    specs = [apply_overrides(spec, args, mode) for spec, mode in zip(specs, modes)]
    labels = args.labels or [path.name for path in paths]
    aa = args.aa or ["auto"] * len(specs)
    if len(labels) != len(specs) or len(set(labels)) != len(labels) or len(aa) != len(specs):
        raise ValueError("Provide one unique label and one AA setting per model")
    if args.iterations:
        if len(args.iterations) != len(specs) or any(i < 1 for i in args.iterations):
            raise ValueError("--iterations requires one positive saved iteration per model")
        iterations = args.iterations
    else:
        if args.iteration != -1 and args.iteration < 1:
            raise ValueError("--iteration must be positive or -1")
        iterations = [shared_iteration(specs, args.iteration)] * len(specs)
    if len({bool(s["cfg"].get("white_background", False)) for s in specs}) != 1:
        raise ValueError("Model background colors differ")
    records = read_colmap_records(source)
    cameras = core.make_trajectory([camera(r) for r in records.values()], args.start, args.stride, args.max_frames)
    path_details = {}
    if args.video == "random":
        from scripts.render_novel_comparison import choose_source_segment, synthesize_random_path
        segment, selection = choose_source_segment(cameras, args.source_window, seed=args.seed)
        cameras, synthesis = synthesize_random_path(segment, cameras, max(2, round(args.duration * args.fps)),
            args.control_step, args.deviation, args.max_distance_fraction, args.seed, args.view_degrees)
        path_details = dict(source_segment=selection, synthesis=synthesis, seed=args.seed)
    width, height = core.video_dimensions(cameras, args.width)
    shared = core.trajectory_record(cameras)
    models = []
    for index, (spec, label, setting, iteration) in enumerate(zip(specs, labels, aa, iterations)):
        saved = registered_source(spec, source)
        kind = spec["type"]
        if kind not in ("3dgs", "scaffold", "textured", "proxygs"):
            raise ValueError(f"Unsupported model: {kind}")
        _, ply = choose_iteration(spec["path"], iteration, kind)
        metadata = spec.get("metadata", {})
        if kind in ("scaffold", "proxygs"):
            needed = ["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"]
            if kind == "proxygs" and spec["cfg"].get("use_feat_bank", False):
                needed.append("feature_bank_mlp.pt")
            missing = [name for name in needed if not (ply.parent / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete checkpoint {ply.parent}: {missing}")
        if not Path(metadata.get("python", sys.executable)).is_file():
            raise FileNotFoundError(f"Model interpreter is unavailable: {metadata['python']}")
        if kind == "proxygs":
            if not Path(metadata["mesh"]).is_file():
                raise FileNotFoundError(f"Proxy mesh missing: {metadata['mesh']}")
            if spec["cfg"].get("appearance_dim", 0):
                raise ValueError("Proxy video rendering requires appearance_dim=0")
        antialiasing = (spec["cfg"].get("antialiasing", kind not in ("textured", "proxygs"))
                        if setting == "auto" else setting == "on")
        if antialiasing and kind in ("proxygs", "textured"):
            raise ValueError(f"{kind} does not support Graphdeco AA; use off")
        selected_cameras = shared
        if args.video == "actual":
            own = records if saved == source else read_colmap_records(saved, spec["cfg"].get("images", "images"))
            for info in cameras:
                name = info.image_name
                if name not in own or Path(own[name]["image_path"]).resolve() != Path(records[name]["image_path"]).resolve():
                    raise ValueError(f"GT images differ for {label}: {name}")
            if args.actual_poses == "model":
                selected_cameras = core.trajectory_record([camera(own[c.image_name]) for c in cameras])
        models.append(dict(path=str(spec["path"]), cfg=spec["cfg"], metadata=metadata,
            type=kind, iteration=iteration, ply=str(ply), label=label, antialiasing=bool(antialiasing),
            python=metadata.get("python", sys.executable), train_count=len(spec["splits"]["train"]),
            cameras=selected_cameras, trajectory_sha256=camera_hash(selected_cameras),
            video=f"model_{index}.mp4"))
    if args.video == "actual":
        core.validate_originals(source / "images", cameras)
    return dict(version=1, mode=args.video, source=str(source), output=str(output),
        width=width, height=height, label_height=40, fps=args.fps, crf=args.crf, gpu=args.gpu,
        frame_count=len(cameras), cameras=shared, trajectory_sha256=camera_hash(shared),
        pose_protocol=("shared source trajectory" if args.video == "random" or args.actual_poses == "source"
                       else "matching filenames, each model's saved camera calibration"),
        color_protocol="base appearance; no per-image exposure correction",
        models=models, completed_models=[], complete=False, **path_details)


def labeled_frames(frames, label, width, height, band=40):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for frame in frames:
        image = Image.new("RGB", (width, height + band), "#111827")
        image.paste(Image.fromarray(frame), (0, band))
        ImageDraw.Draw(image).text((10, 8), label, fill="white", font=font)
        yield np.asarray(image)


def stack_videos(paths, output, crf):
    temp = output.with_name(output.stem + ".partial.mp4")
    inputs = [part for path in paths for part in ("-i", str(path))]
    graph = "".join(f"[{i}:v]" for i in range(len(paths))) + f"hstack=inputs={len(paths)}[v]"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
        "-filter_complex", graph, "-map", "[v]", "-an", "-c:v", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)]
    try:
        subprocess.run(command, check=True)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def execute_model(plan_path, index, chain=True):
    plan = json.loads(Path(plan_path).read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = plan["gpu"]
    model = plan["models"][index]
    spec = {**model, "path": Path(model["path"]), "ply": Path(model["ply"]),
            "splits": {"train": range(model["train_count"])}}
    frames = core.rendered_frames(spec, [camera(c) for c in model["cameras"]],
                                  plan["width"], plan["height"], model["antialiasing"])
    label = f"{model['label']} | {model['iteration']} steps | AA {'on' if model['antialiasing'] else 'off'}"
    if model["type"] == "3dgs":
        label += " | culling " + ("on" if model["cfg"].get("occlusion_mode") == "lidar" else "off")
    output = Path(plan["output"])
    core.write_video(output / model["video"], labeled_frames(frames, label, plan["width"], plan["height"]),
                     plan["width"], plan["height"] + plan["label_height"], plan["fps"], plan["crf"])
    plan["completed_models"].append(index)
    atomic_json(plan_path, plan)
    if index + 1 < len(plan["models"]) and chain:
        replace_with_model(plan_path, plan, index + 1)
    elif index + 1 == len(plan["models"]):
        videos = ([output / "original.mp4"] if plan["mode"] == "actual" else [])
        videos += [output / m["video"] for m in plan["models"]]
        stack_videos(videos, output / "side_by_side.mp4", plan["crf"])
        plan["complete"] = True
        atomic_json(plan_path, plan)
        print(f"Video complete: {output / 'side_by_side.mp4'}", flush=True)


def replace_with_model(path, plan, index):
    python = plan["models"][index]["python"]
    print(f"Rendering {plan['models'][index]['label']} in its environment", flush=True)
    os.execv(python, [python, "-u", str(Path(__file__).resolve()), "--_plan", str(path), str(index)])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_plan":
        execute_model(Path(argv[1]), int(argv[2]))
        return 0
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps({k: v for k, v in plan.items() if k not in ("cameras", "models")}, indent=2))
            for model in plan["models"]:
                print(f"{model['label']}: {model['path']} @ {model['iteration']} (AA={model['antialiasing']})")
            return 0
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg with libx264 is required")
        output = Path(plan["output"])
        output.mkdir(parents=True, exist_ok=True)
        plan_path = output / "trajectory.json"
        atomic_json(plan_path, plan)
        if plan["mode"] == "actual":
            frames = core.original_frames(Path(plan["source"]) / "images", [camera(c) for c in plan["cameras"]],
                                          plan["width"], plan["height"])
            core.write_video(output / "original.mp4", labeled_frames(frames, "GT image", plan["width"], plan["height"]),
                             plan["width"], plan["height"] + plan["label_height"], plan["fps"], plan["crf"])
        if args.in_process:
            for index in range(len(plan['models'])):
                execute_model(plan_path, index, chain=False)
        else:
            replace_with_model(plan_path, plan, 0)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f"Video preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
