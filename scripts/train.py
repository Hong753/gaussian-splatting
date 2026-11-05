#!/usr/bin/env python3
"""Train one or more COLMAP/NeRF scenes with the repository's train.py.

Edit the configuration below, or override it on the command line. Additional
native train.py options can be passed after `--`.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


# Editable defaults, in the same style as the PGSR scripts.
DATA_ROOT = Path("/data/scene")
SCENES = []  # e.g. ["scan65", "scan69"]
OUTPUT_ROOT = Path("/data/3dgs_outputs")
RUN_NAME = "3dgs"
GPU_ID = "0"
VIEWER_PORT = 6009
ITERATIONS = 30_000
RESOLUTION = -1  # -1: Graphdeco automatic resolution; 1: full resolution.
IMAGES = "images"
DATA_DEVICE = "cuda"
EVAL = False
ANTIALIASING = True
EXPOSURE_COMPENSATION = True
OPTIMIZER_TYPE = "default"  # sparse_adam requires the accelerated rasterizer.
DEPTHS = ""  # Optional depth-prior directory; empty means RGB-only.
SELECT_KEYFRAMES = False
KEYFRAME_GROUP_SIZE = 3
EXTRA_ARGS = []  # e.g. ["--save_iterations", "7000", "15000", "30000"]

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.colmap_paths import colmap_model_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train scenes sequentially. Edit defaults in this file or pass CLI options.",
        epilog="Pass additional native train.py options after --.",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Parent directory containing scene folders")
    parser.add_argument("--scenes", nargs="+", default=SCENES,
                        help="Scene paths relative to --data-root; use . for --data-root itself")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Outputs go to ROOT/SCENE/NAME")
    parser.add_argument("--name", default=RUN_NAME, help="Experiment name")
    parser.add_argument("--gpu", default=GPU_ID, help="GPU ID passed through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--viewer-port", type=int, default=VIEWER_PORT,
                        help="Live SIBR network viewer port (default: 6009)")
    parser.add_argument("--disable-viewer", action="store_true", help="Do not open the live viewer socket")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="-1 for automatic, 1/2/4/8 for downscale, or another positive target width")
    parser.add_argument("--images", default=IMAGES, help="Image folder relative to each scene")
    parser.add_argument("--data-device", choices=["cuda", "cpu"], default=DATA_DEVICE)
    parser.add_argument("--eval", action="store_true", default=EVAL, help="Reserve test views instead of training on all views")
    parser.add_argument("--antialiasing", action=argparse.BooleanOptionalAction, default=ANTIALIASING,
                        help="Use the Graphdeco antialiasing filter (default: enabled)")
    parser.add_argument("--exposure-compensation", action=argparse.BooleanOptionalAction,
                        default=EXPOSURE_COMPENSATION,
                        help="Optimize per-image exposure (default: enabled; incompatible with --eval)")
    parser.add_argument("--optimizer-type", choices=["default", "sparse_adam"], default=OPTIMIZER_TYPE,
                        help="sparse_adam requires the accelerated rasterizer")
    parser.add_argument("--depths", default=DEPTHS, help="Optional depth-prior directory for native depth regularization")
    parser.add_argument("--select-keyframes", action="store_true", default=SELECT_KEYFRAMES,
                        help="Select the sharpest registered training image per consecutive group")
    parser.add_argument("--keyframe-group-size", type=int, default=KEYFRAME_GROUP_SIZE)
    parser.add_argument("--selection-only", action="store_true", help="Select keyframes without launching training")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting training")
    parser.add_argument("--allow-existing", action="store_true", help="Allow training in a non-empty output directory")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Native train.py options after --")
    args = parser.parse_args(argv)

    if not args.scenes:
        parser.error("Set SCENES in scripts/train.py or pass --scenes SCENE [SCENE ...]")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not 1 <= args.viewer_port <= 65535:
        parser.error("--viewer-port must be between 1 and 65535")
    if args.resolution == 0 or args.resolution < -1:
        parser.error("--resolution must be -1 or a positive integer")
    if args.keyframe_group_size < 1:
        parser.error("--keyframe-group-size must be positive")
    if args.selection_only and not args.select_keyframes:
        parser.error("--selection-only requires --select-keyframes")
    if args.eval and args.exposure_compensation:
        parser.error("--eval requires --no-exposure-compensation: the native exposure flag changes the test split")
    if args.name in ("", ".", "..") or Path(args.name).name != args.name:
        parser.error("--name must be a single directory name")
    scene_paths = [Path(scene) for scene in args.scenes]
    if any(not scene or path.is_absolute() or ".." in path.parts
           for scene, path in zip(args.scenes, scene_paths)):
        parser.error("Scene paths must be relative to --data-root and must not contain '..'")
    if len(scene_paths) != len(set(scene_paths)):
        parser.error("Scene paths must be unique")

    extra = args.extra_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    # Keep source and output controlled by the launcher, including in multi-scene runs.
    if any(x in ("-s", "--source_path", "-m", "--model_path") or
           x.startswith(("--source_path=", "--model_path=")) for x in [*EXTRA_ARGS, *extra]):
        parser.error("Set scene and output paths using launcher options, not extra train.py arguments")
    if args.select_keyframes and any(x in ("--eval", "--train_test_exp", "-i", "--images", "--keyframe_manifest") or
            x.startswith(("--images=", "--keyframe_manifest=")) for x in [*EXTRA_ARGS, *extra]):
        parser.error("With keyframe selection, use launcher --eval/--images and do not pass --train_test_exp or --keyframe_manifest")

    root = args.data_root.expanduser().resolve()
    if Path(".") in scene_paths and not root.name:
        parser.error("Cannot derive an output scene name from the filesystem root")
    out_root = args.output_root.expanduser()
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root = out_root.resolve()
    jobs = []
    output_scenes = [Path(root.name) if path == Path(".") else path for path in scene_paths]
    if len(output_scenes) != len(set(output_scenes)):
        parser.error("Scene paths resolve to the same output directory")
    for scene, scene_path, output_scene in zip(args.scenes, scene_paths, output_scenes):
        source = root / scene_path
        output = out_root / output_scene / args.name
        if not source.is_dir():
            parser.error(f"Scene folder does not exist: {source}")
        try:
            colmap_model_dir(source)
            has_colmap = True
        except FileNotFoundError as exc:
            has_colmap = False
            if args.select_keyframes or not (source / "transforms_train.json").is_file():
                parser.error(str(exc))
        if args.select_keyframes and not has_colmap:
            parser.error("Keyframe selection currently supports COLMAP scenes only")
        if output.exists() and not output.is_dir():
            parser.error(f"Output path is not a directory: {output}")
        if output.is_dir() and any(output.iterdir()) and not args.allow_existing:
            parser.error(f"Output directory is not empty: {output} (choose another --name or use --allow-existing)")
        cmd = [sys.executable, str(REPO_ROOT / "train.py"), "-s", str(source), "-m", str(output),
               "--iterations", str(args.iterations), "-r", str(args.resolution),
               "-i", args.images, "--data_device", args.data_device]
        if args.disable_viewer:
            cmd.append("--disable_viewer")
        else:
            cmd.extend(["--port", str(args.viewer_port)])
        if args.eval:
            cmd.append("--eval")
        if args.antialiasing:
            cmd.append("--antialiasing")
        if args.exposure_compensation:
            # Graphdeco couples exposure optimization to --train_test_exp.
            # With --eval off, this enables exposure without changing the split.
            cmd.append("--train_test_exp")
        if args.optimizer_type != "default":
            cmd.extend(["--optimizer_type", args.optimizer_type])
        if args.depths:
            cmd.extend(["--depths", args.depths])
        if args.select_keyframes:
            cmd.extend(["--keyframe_manifest", str(output / "keyframes.json")])
        cmd.extend(EXTRA_ARGS)
        cmd.extend(extra)
        jobs.append((scene, source, output, cmd))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for scene, source, output, cmd in jobs:
        if args.select_keyframes:
            if args.dry_run:
                print(f"\n[{scene}] Would select the sharpest frame per {args.keyframe_group_size} "
                      f"registered training frames into {output / 'keyframes.json'}", flush=True)
            else:
                from keyframes import select_keyframes
                try:
                    select_keyframes(source, args.images, output, args.keyframe_group_size, args.eval)
                except (OSError, ValueError, ImportError) as exc:
                    print(f"Keyframe selection failed for {scene}: {exc}", file=sys.stderr)
                    return 1
        if args.selection_only:
            continue
        print(f"\n[{scene}] CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Training failed for {scene} (exit {exc.returncode}); remaining scenes were not started.", file=sys.stderr)
            return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
