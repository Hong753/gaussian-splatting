#!/usr/bin/env python3
"""Prepare Proxy-GS, then replace this process with its official trainer."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.proxygs_support import (REVISION, PATCH_VERSION, install, apply_upstream_patches,
                                     prepare_dataset, cache_depths)
from scripts.view_comparisons import atomic_json


def archive_failed_attempt(model):
    """Preserve failed-run metadata while retaining expensive preparation caches."""
    files = [model / name for name in ("cfg_args", "proxygs_run.json", "outputs.log",
                                      "launcher.log", "cameras.json")]
    files.extend(model.glob("events.out.tfevents.*"))
    files = [path for path in files if path.is_file()]
    if files:
        destination = model / "failed_attempts" / str(time.time_ns())
        destination.mkdir(parents=True)
        for path in files:
            path.rename(destination / path.name)
        print(f"Previous attempt metadata saved to {destination}", flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "patch", "run", "prepare"])
    parser.add_argument("--repo", type=Path, default=Path.home() / "Proxy-GS")
    parser.add_argument("--environment", type=Path, default=Path.home() / ".venvs/proxygs")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mesh", type=Path, help="Optional registered triangle mesh; otherwise build from LiDAR")
    parser.add_argument("--voxel-size", type=float, default=.05,
                        help="Proxy mesh spacing only, in map units; does not set anchor spacing")
    parser.add_argument("--base-layer", type=int, help="Anchor octree base layer (upstream default: 10); smaller values give coarser voxels")
    parser.add_argument("--levels", type=int, help="Anchor octree level count (upstream default: -1, adaptive)")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--restart", action="store_true",
                        help="Restart a failed run from step 0, retaining proxy/depth caches; refuses saved checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo, environment = args.repo.expanduser().resolve(), args.environment.expanduser().resolve()
    installation = environment / "proxygs_install.json"
    try:
        if args.action == "setup":
            if args.dry_run:
                print(f"Install {REVISION} into {repo}; isolated Python environment {environment}")
            else:
                print(f"Proxy-GS ready: {install(repo, environment)}")
            return 0
        installed = {}
        if not args.dry_run:
            if not installation.is_file():
                raise FileNotFoundError("Run scripts/run_proxygs.py setup first, from the CUDA-enabled 3dgs environment")
            installed = json.loads(installation.read_text())
            if installed["revision"] != REVISION or Path(installed["upstream"]) != repo:
                raise ValueError("Requested repository does not match the pinned environment installation")
        if args.action == "patch":
            if args.dry_run:
                print(f"Apply Python memory/runtime fixes to {repo}; retain installed dependencies")
            else:
                hashes = apply_upstream_patches(repo, installed.get("source_patch_sha256"))
                installed.update(patch_version=PATCH_VERSION, source_patch_sha256=hashes,
                                 training_patch_sha256=hashes["train.py"])
                atomic_json(installation, installed)
                print(f"Applied Proxy-GS patches in {repo}; no package rebuild required")
            return 0
        if not all((args.source, args.keyframes, args.model)):
            parser.error("run/prepare requires --source, --keyframes, and --model")
        if args.iterations < 1 or args.voxel_size <= 0 or args.resolution == 0 or args.resolution < -1:
            parser.error("Invalid iteration count, voxel size, or resolution")
        if args.base_layer is not None and args.base_layer < -1:
            parser.error("--base-layer must be -1 (adaptive) or nonnegative")
        if args.levels is not None and args.levels != -1 and args.levels < 1:
            parser.error("--levels must be -1 (adaptive) or positive")
        if not args.dry_run and installed.get("patch_version") != PATCH_VERSION:
            raise ValueError("Update the installed source first: python scripts/run_proxygs.py patch")
        model = args.model.expanduser().resolve()
        python = str(environment / "bin/python")
        prep = model / "proxygs_data/preparation.json"
        stage = prep.parent
        mesh_path, depth_dir = stage / "proxy.ply", stage / "depth"
        if args.action == "run" and not args.dry_run:
            saved = list((model / "point_cloud").glob("iteration_*/point_cloud.ply")) + list(model.glob("chkpnt*.pth"))
            if saved:
                raise ValueError("This run has saved checkpoints; choose a new --model directory")
            if (model / "cfg_args").exists() and not args.restart:
                raise ValueError("This run already has training outputs; use --restart for a failed run without saved checkpoints")
        checkpoints = sorted({min(7000, args.iterations), args.iterations})
        train_cmd = [python, "-u", str(repo / "train.py"), "-s", str(stage), "-m", str(model),
                     "--ply_path", str(stage / "input.ply"), "--ply_mesh", str(mesh_path),
                     "--depth_npy_dir", str(depth_dir), "--iterations", str(args.iterations),
                     "--resolution", str(args.resolution), "--appearance_dim", "0",
                     "--data_device", "cpu", "--test_iterations", *map(str, checkpoints),
                     "--save_iterations", *map(str, checkpoints)]
        for option, value in (("--base_layer", args.base_layer), ("--levels", args.levels)):
            if value is not None:
                train_cmd.extend((option, str(value)))
        manifest = json.loads(args.keyframes.read_text())
        if manifest.get("eval", False):
            train_cmd.append("--eval")
        report_args = ["--model", str(model), "--split", "both", "--iteration", str(args.iterations),
                       "--gpu", args.gpu, "--in-process"]
        if args.dry_run:
            print(f"Prepare selected cameras, LiDAR proxy and mesh-depth cache under {stage}")
            if args.action == "run":
                print(shlex.join(train_cmd))
                print(shlex.join([python, "-u", str(ROOT / "scripts/render.py"), *report_args]))
            return 0
        # exec replaces this process, retaining its PID; there is no Python supervisor.
        if Path(sys.prefix).resolve() != environment:
            os.execv(python, [python, "-u", str(Path(__file__).resolve()), *argv])
            return 0  # Only reached by mocked exec in CPU tests.
        print(f"Preparing cameras and proxy/depth cache in process {os.getpid()}", flush=True)
        prep = prepare_dataset(args.source, args.keyframes, model, args.mesh, args.voxel_size)
        cache_depths(prep, repo, args.resolution)
        if args.action == "prepare":
            print(f"Proxy previews and coverage: {depth_dir}")
            return 0
        if args.restart:
            archive_failed_attempt(model)
        atomic_json(model / "proxygs_run.json", dict(model_type="proxygs", upstream=str(repo),
            revision=REVISION, patch_version=PATCH_VERSION, python=python,
            preparation=str(prep), depth_dir=str(depth_dir), mesh=str(mesh_path),
            source=str(args.source.resolve()), iterations=args.iterations,
            train_command=train_cmd, antialiasing=False,
            notes="Official model; candidate-only densification allocations. No Graphdeco AA."))
        with (model / "launcher.log").open("a") as log:
            log.write(shlex.join(train_cmd) + "\n")
        os.environ.update(PROXYGS_EXTERNAL_REPORT="1", WANDB_MODE="disabled",
                          PROXYGS_REPORT_SCRIPT=str(ROOT / "scripts/render.py"),
                          PROXYGS_REPORT_ARGS=json.dumps(report_args))
        print("Starting training directly; any training traceback follows unchanged.", flush=True)
        print(shlex.join(train_cmd), flush=True)
        os.chdir(repo)
        os.execv(python, train_cmd)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Proxy-GS configuration failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
