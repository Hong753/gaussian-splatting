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
DATA_ROOT = Path("/workspace/colmap_scenes/DTU/dtu")
SCENES = []  # e.g. ["scan65", "scan69"]
OUTPUT_ROOT = Path("output")
RUN_NAME = "3dgs"
GPU_ID = "0"
ITERATIONS = 30_000
RESOLUTION = -1  # -1: Graphdeco automatic resolution; 1: full resolution.
IMAGES = "images"
DATA_DEVICE = "cuda"
EVAL = False
EXTRA_ARGS = []  # e.g. ["--save_iterations", "7000", "15000", "30000"]

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train scenes sequentially. Edit defaults in this file or pass CLI options.",
        epilog="Pass extra native train.py options after --; e.g. -- --antialiasing",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Parent directory containing scene folders")
    parser.add_argument("--scenes", nargs="+", default=SCENES, help="Scene folder names under --data-root")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Outputs go to ROOT/SCENE/NAME")
    parser.add_argument("--name", default=RUN_NAME, help="Experiment name")
    parser.add_argument("--gpu", default=GPU_ID, help="GPU ID passed through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="-1 for automatic, 1/2/4/8 for downscale, or another positive target width")
    parser.add_argument("--images", default=IMAGES, help="Image folder relative to each scene")
    parser.add_argument("--data-device", choices=["cuda", "cpu"], default=DATA_DEVICE)
    parser.add_argument("--eval", action="store_true", default=EVAL, help="Reserve test views instead of training on all views")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting training")
    parser.add_argument("--allow-existing", action="store_true", help="Allow training in a non-empty output directory")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Native train.py options after --")
    args = parser.parse_args(argv)

    if not args.scenes:
        parser.error("Set SCENES in scripts/train.py or pass --scenes SCENE [SCENE ...]")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.resolution == 0 or args.resolution < -1:
        parser.error("--resolution must be -1 or a positive integer")
    if args.name in ("", ".", "..") or Path(args.name).name != args.name:
        parser.error("--name must be a single directory name")
    if any(scene in ("", ".", "..") or Path(scene).name != scene for scene in args.scenes):
        parser.error("Each scene must be a single directory name under --data-root")
    if len(args.scenes) != len(set(args.scenes)):
        parser.error("Scene names must be unique")

    extra = args.extra_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    # Keep source and output controlled by the launcher, including in multi-scene runs.
    if any(x in ("-s", "--source_path", "-m", "--model_path") or
           x.startswith(("--source_path=", "--model_path=")) for x in [*EXTRA_ARGS, *extra]):
        parser.error("Set scene and output paths using launcher options, not extra train.py arguments")

    root = args.data_root.expanduser().resolve()
    out_root = args.output_root.expanduser()
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root = out_root.resolve()
    jobs = []
    for scene in args.scenes:
        source = root / scene
        output = out_root / scene / args.name
        if not source.is_dir():
            parser.error(f"Scene folder does not exist: {source}")
        if not ((source / "sparse" / "0").is_dir() or (source / "transforms_train.json").is_file()):
            parser.error(f"Expected COLMAP sparse/0 or transforms_train.json in {source}")
        if output.exists() and not output.is_dir():
            parser.error(f"Output path is not a directory: {output}")
        if output.is_dir() and any(output.iterdir()) and not args.allow_existing:
            parser.error(f"Output directory is not empty: {output} (choose another --name or use --allow-existing)")
        cmd = [sys.executable, str(REPO_ROOT / "train.py"), "-s", str(source), "-m", str(output),
               "--iterations", str(args.iterations), "-r", str(args.resolution),
               "-i", args.images, "--data_device", args.data_device]
        if args.eval:
            cmd.append("--eval")
        cmd.extend(EXTRA_ARGS)
        cmd.extend(extra)
        jobs.append((scene, cmd))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for scene, cmd in jobs:
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
