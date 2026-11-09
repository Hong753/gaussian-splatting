#!/usr/bin/env python3
"""Train one or more COLMAP/NeRF scenes with the repository's train.py.

Edit the configuration below, or override it on the command line. Additional
native train.py options can be passed after `--`.
"""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


# Editable defaults, in the same style as the PGSR scripts.
DATA_ROOT = Path("/data/scene")
SCENES = []  # e.g. ["scan65", "scan69"]
OUTPUT_ROOT = Path("/data/3dgs_outputs")
RUN_NAME = "3dgs"
MODEL_TYPE = "3dgs"  # or "scaffold"
SCAFFOLD_VOXEL_SIZE = 0.001
SCAFFOLD_N_OFFSETS = 10
SCAFFOLD_FEAT_DIM = 32
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
EVAL_ALL_FRAMES = True
RENDER_COMPARISONS = True
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
    parser.add_argument("--model-type", choices=["3dgs", "scaffold", "textured"], default=MODEL_TYPE)
    parser.add_argument("--textured-pretrain-iterations", type=int, default=30000)
    parser.add_argument("--textured-max-gaussians", type=int, default=10000)
    parser.add_argument("--textured-resolution", type=int, default=50)
    parser.add_argument("--scaffold-voxel-size", type=float, default=SCAFFOLD_VOXEL_SIZE)
    parser.add_argument("--scaffold-n-offsets", type=int, default=SCAFFOLD_N_OFFSETS)
    parser.add_argument("--scaffold-feat-dim", type=int, default=SCAFFOLD_FEAT_DIM)
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
    parser.add_argument("--absgs", action="store_true", help="Use AbsGS per-pixel absolute gradients for splitting (3DGS only)")
    parser.add_argument("--absgs-split-threshold", type=float, default=0.0004,
                        help="Absolute-gradient split threshold for AbsGS")
    parser.add_argument("--select-keyframes", action="store_true", default=SELECT_KEYFRAMES,
                        help="Select the sharpest registered training image per consecutive group")
    parser.add_argument("--keyframe-manifest", type=Path,
                        help="Reuse an existing keyframe manifest and copy it into the new run")
    parser.add_argument("--keyframe-group-size", type=int, default=KEYFRAME_GROUP_SIZE)
    parser.add_argument("--selection-only", action="store_true", help="Select keyframes without launching training")
    parser.add_argument("--eval-all-frames", action=argparse.BooleanOptionalAction, default=EVAL_ALL_FRAMES,
                        help="Evaluate every COLMAP frame after training (default: enabled)")
    parser.add_argument("--render-comparisons", action=argparse.BooleanOptionalAction, default=RENDER_COMPARISONS,
                        help="Save actual-image/render train and eval galleries after training (default: enabled)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate an existing saved model without starting training")
    parser.add_argument("--eval-iteration", type=int, default=-1,
                        help="Saved iteration for --evaluate-only (-1: latest)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting training")
    parser.add_argument("--allow-existing", action="store_true", help="Allow training in a non-empty output directory")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Native train.py options after --")
    args = parser.parse_args(argv)

    if not args.scenes:
        parser.error("Set SCENES in scripts/train.py or pass --scenes SCENE [SCENE ...]")
    if args.absgs and (args.model_type != "3dgs" or args.optimizer_type != "default"):
        parser.error("--absgs requires --model-type 3dgs and --optimizer-type default")
    if args.absgs_split_threshold <= 0:
        parser.error("--absgs-split-threshold must be positive")
    if args.model_type == "scaffold" and not args.evaluate_only:
        if args.eval or args.depths or args.optimizer_type != "default":
            parser.error("Scaffold-GS does not yet support --eval, --depths or sparse_adam")
        if args.scaffold_voxel_size <= 0 or args.scaffold_n_offsets < 1 or args.scaffold_feat_dim < 1:
            parser.error("Scaffold voxel size, offsets and feature dimension must be positive")
    if args.model_type == "textured" and not args.evaluate_only:
        if args.eval or args.depths or args.exposure_compensation or args.antialiasing or args.optimizer_type != "default":
            parser.error("Textured backend requires --no-antialiasing --no-exposure-compensation and does not support --eval, --depths or sparse_adam")
        if min(args.textured_pretrain_iterations, args.textured_max_gaussians, args.textured_resolution) < 1:
            parser.error("Textured pretraining iterations, Gaussian cap and texture resolution must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not 1 <= args.viewer_port <= 65535:
        parser.error("--viewer-port must be between 1 and 65535")
    if args.resolution == 0 or args.resolution < -1:
        parser.error("--resolution must be -1 or a positive integer")
    if args.keyframe_group_size < 1:
        parser.error("--keyframe-group-size must be positive")
    if args.keyframe_manifest and (args.select_keyframes or args.evaluate_only):
        parser.error("--keyframe-manifest cannot be combined with --select-keyframes or --evaluate-only")
    if args.keyframe_manifest and not args.keyframe_manifest.expanduser().is_file():
        parser.error(f"Keyframe manifest does not exist: {args.keyframe_manifest}")
    if args.selection_only and not args.select_keyframes:
        parser.error("--selection-only requires --select-keyframes")
    if args.evaluate_only and args.selection_only:
        parser.error("--evaluate-only cannot be combined with --selection-only")
    if args.evaluate_only and not args.eval_all_frames:
        parser.error("--evaluate-only requires --eval-all-frames")
    if args.eval_iteration != -1 and args.eval_iteration < 1:
        parser.error("--eval-iteration must be -1 or positive")
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
    if (args.select_keyframes or args.keyframe_manifest) and any(x in ("--eval", "--train_test_exp", "-i", "--images", "--keyframe_manifest") or
            x.startswith(("--images=", "--keyframe_manifest=")) for x in [*EXTRA_ARGS, *extra]):
        parser.error("With keyframes, use launcher --eval/--images and do not pass --train_test_exp or --keyframe_manifest")

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
        if args.keyframe_manifest and not has_colmap:
            parser.error("Keyframe manifests currently support COLMAP scenes only")
        if args.evaluate_only and not has_colmap:
            parser.error("All-frame PSNR currently supports COLMAP scenes only")
        if output.exists() and not output.is_dir():
            parser.error(f"Output path is not a directory: {output}")
        if output.is_dir() and any(output.iterdir()) and not args.allow_existing and not args.evaluate_only:
            parser.error(f"Output directory is not empty: {output} (choose another --name or use --allow-existing)")
        if args.evaluate_only and not output.is_dir():
            parser.error(f"Output directory does not exist: {output}")
        trainer_name = {"3dgs": "train.py", "scaffold": "models/train_scaffold.py",
                        "textured": "models/train_textured.py"}[args.model_type]
        trainer = REPO_ROOT / trainer_name
        cmd = [sys.executable, str(trainer), "-s", str(source), "-m", str(output),
               "--iterations", str(args.iterations), "-r", str(args.resolution),
               "-i", args.images, "--data_device", args.data_device]
        if args.model_type == "scaffold":
            cmd.extend(["--model_type", "scaffold", "--scaffold_voxel_size", str(args.scaffold_voxel_size),
                        "--scaffold_n_offsets", str(args.scaffold_n_offsets), "--scaffold_feat_dim", str(args.scaffold_feat_dim)])
        if args.model_type == "textured":
            cmd.extend(["--model_type", "textured",
                        "--textured_pretrain_iterations", str(args.textured_pretrain_iterations),
                        "--textured_max_gaussians", str(args.textured_max_gaussians),
                        "--textured_resolution", str(args.textured_resolution)])
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
        if args.absgs:
            cmd.extend(["--absgs", "--absgs_split_threshold", str(args.absgs_split_threshold)])
        if (args.select_keyframes or args.keyframe_manifest) and not args.evaluate_only:
            cmd.extend(["--keyframe_manifest", str(output / "keyframes.json")])
        cmd.extend(EXTRA_ARGS)
        cmd.extend(extra)
        jobs.append((scene, source, output, cmd, has_colmap))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for scene, source, output, cmd, has_colmap in jobs:
        if args.keyframe_manifest:
            manifest_source = args.keyframe_manifest.expanduser().resolve()
            manifest_destination = output / "keyframes.json"
            if args.dry_run:
                print(f"\n[{scene}] Would copy keyframe manifest {manifest_source} to {manifest_destination}", flush=True)
            else:
                output.mkdir(parents=True, exist_ok=True)
                if manifest_source != manifest_destination:
                    shutil.copyfile(manifest_source, manifest_destination)
        if args.select_keyframes and not args.evaluate_only:
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
        if not args.evaluate_only:
            if args.model_type == "scaffold":
                print(f"[{scene}] Scaffold-GS live viewer is not supported.", flush=True)
            if args.model_type == "textured":
                print(f"[{scene}] Textured Gaussians runs 2DGS pretraining followed by texture refinement; live viewer is not supported.", flush=True)
            print(f"\n[{scene}] CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {shlex.join(cmd)}", flush=True)
            if not args.dry_run:
                try:
                    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"Training failed for {scene} (exit {exc.returncode}); remaining scenes were not started.", file=sys.stderr)
                    return exc.returncode or 1
        if args.eval_all_frames and not has_colmap:
            print(f"[{scene}] Skipping all-frame PSNR: only COLMAP scenes are supported.", flush=True)
        if args.render_comparisons and has_colmap:
            iteration = args.eval_iteration if args.evaluate_only else args.iterations
            render_cmd = [sys.executable, str(REPO_ROOT / "scripts/render.py"), "--split", "both",
                          "--model", str(output), "--iteration", str(iteration), "--gpu", args.gpu,
                          "--antialiasing" if args.antialiasing else "--no-antialiasing"]
            print(f"\n[{scene}] {shlex.join(render_cmd)}", flush=True)
            if not args.dry_run:
                try:
                    subprocess.run(render_cmd, cwd=REPO_ROOT, env=env, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"View comparison rendering failed for {scene} (exit {exc.returncode}).", file=sys.stderr)
                    return exc.returncode or 1
        if args.eval_all_frames and has_colmap:
            iteration = args.eval_iteration if args.evaluate_only else args.iterations
            eval_cmd = [sys.executable, str(REPO_ROOT / "scripts/eval_all_frames.py"),
                        "--source", str(source), "--model", str(output), "--iteration", str(iteration),
                        "--antialiasing" if args.antialiasing else "--no-antialiasing"]
            print(f"\n[{scene}] CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {shlex.join(eval_cmd)}", flush=True)
            if not args.dry_run:
                try:
                    subprocess.run(eval_cmd, cwd=REPO_ROOT, env=env, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"All-frame evaluation failed for {scene} (exit {exc.returncode}).", file=sys.stderr)
                    return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





