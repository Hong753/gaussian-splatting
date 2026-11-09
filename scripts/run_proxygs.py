#!/usr/bin/env python3
"""Set up and run pinned official Proxy-GS on this project's selected LiDAR scene."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.proxygs_support import REVISION, install, prepare_dataset, cache_depths
from scripts.view_comparisons import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "run", "prepare"])
    parser.add_argument("--repo", type=Path, default=Path.home() / "Proxy-GS")
    parser.add_argument("--environment", type=Path, default=Path.home() / ".venvs/proxygs")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mesh", type=Path, help="Optional registered triangle mesh; otherwise build from LiDAR")
    parser.add_argument("--voxel-size", type=float, default=.05, help="Proxy construction spacing in input map units (metres)")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo, environment = args.repo.expanduser().resolve(), args.environment.expanduser().resolve()
    try:
        if args.action == "setup":
            if args.dry_run:
                print(f"Install {REVISION} into {repo}; isolated Python environment {environment}")
            else:
                python = install(repo, environment)
                print(f"Proxy-GS ready: {python}")
            return 0
        if not all((args.source, args.keyframes, args.model)):
            parser.error("run/prepare requires --source, --keyframes, and --model")
        if args.iterations < 1 or args.voxel_size <= 0 or args.resolution == 0 or args.resolution < -1:
            parser.error("Invalid iteration count, voxel size, or resolution")
        model = args.model.expanduser().resolve()
        installation = environment / "proxygs_install.json"
        if not args.dry_run and not installation.is_file():
            raise FileNotFoundError("Run scripts/run_proxygs.py setup first, from the CUDA-enabled 3dgs environment")
        if not args.dry_run:
            installed = json.loads(installation.read_text())
            if installed["revision"] != REVISION or Path(installed["upstream"]) != repo:
                raise ValueError("Requested repository does not match the pinned environment installation")
        python = str(environment / "bin/python")
        prep = model / "proxygs_data/preparation.json"
        stage = prep.parent
        mesh_path, depth_dir = stage / "proxy.ply", stage / "depth"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PROXYGS_EXTERNAL_REPORT="1", WANDB_MODE="disabled")
        prepare_cmd = [python, str(Path(__file__).resolve()), "prepare", "--repo", str(repo),
                       "--environment", str(environment), "--source", str(args.source.resolve()),
                       "--keyframes", str(args.keyframes.resolve()), "--model", str(model),
                       "--resolution", str(args.resolution), "--voxel-size", str(args.voxel_size),
                       "--gpu", args.gpu]
        if args.mesh:
            prepare_cmd += ["--mesh", str(args.mesh.resolve())]
        if args.action == "prepare":
            if args.dry_run:
                print(f"Prepare selected cameras, LiDAR proxy and mesh-depth cache under {stage}")
                return 0
            if Path(sys.prefix).resolve() != environment:
                subprocess.run(prepare_cmd, cwd=ROOT, env=env, check=True)
                return 0
            prep = prepare_dataset(args.source, args.keyframes, model, args.mesh, args.voxel_size)
            cache_depths(prep, repo, args.resolution)
            print(f"Proxy previews and coverage: {depth_dir}")
            return 0
        if not args.dry_run and ((model / "cfg_args").exists() or (model / "point_cloud").exists()):
            raise ValueError("This run already has training outputs; choose a new --model directory")
        checkpoints = sorted({min(7000, args.iterations), args.iterations})
        train_cmd = [python, str(repo / "train.py"), "-s", str(stage), "-m", str(model),
                     "--ply_path", str(stage / "input.ply"), "--ply_mesh", str(mesh_path),
                     "--depth_npy_dir", str(depth_dir), "--iterations", str(args.iterations),
                     "--resolution", str(args.resolution), "--appearance_dim", "0",
                     "--data_device", "cpu", "--test_iterations", *map(str, checkpoints),
                     "--save_iterations", *map(str, checkpoints)]
        # The adapter writes the exact held-out split in transforms_test.json.
        manifest = json.loads(args.keyframes.read_text())
        if manifest.get("eval", False):
            train_cmd.append("--eval")
        report_cmd = [sys.executable, str(ROOT / "scripts/render.py"), "--model", str(model),
                      "--split", "both", "--iteration", str(args.iterations), "--gpu", args.gpu]
        for command in (prepare_cmd, train_cmd, report_cmd):
            print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        subprocess.run(prepare_cmd, cwd=ROOT, env=env, check=True)
        atomic_json(model / "proxygs_run.json", dict(model_type="proxygs", upstream=str(repo),
            revision=REVISION, python=python, preparation=str(prep), depth_dir=str(depth_dir),
            mesh=str(mesh_path), source=str(args.source.resolve()), iterations=args.iterations,
            train_command=train_cmd, antialiasing=False,
            notes="Official anchor/MLP model and optimization defaults; requested iteration limit. No Graphdeco AA."))
        with (model / "launcher.log").open("a") as log:
            log.write(shlex.join(train_cmd) + "\n"); log.flush()
            # Inherit stdout/stderr so progress remains visible; the official trainer also writes outputs.log.
            subprocess.run(train_cmd, cwd=repo, env=env, check=True)
        subprocess.run(report_cmd, cwd=ROOT, env=env, check=True)
        print(f"Proxy-GS training and train/eval image comparisons complete: {model}")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Proxy-GS failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
