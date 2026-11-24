"""DA3 → AmbiSuR warm-up → joint Gaussian/mesh training → Orange4F trajectories."""

import argparse
from contextlib import contextmanager
import fcntl
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time

from scripts.joint_ambisur.report import dashboard, publish, write_json

FILES = Path(__file__).resolve().parent / "joint_ambisur"
ROOT = FILES.parents[1]
SOURCE = Path("/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full")
OUTPUT = Path("/data/3dgs_outputs/orange4f/3dgs_finetuning/ambisur_joint_mesh_w1600")
AMBI = "88d64054f53e0eba9ce49198282edf4a67fc8ca8"
MILO = "0c38aa65f8b768513964f13246ef1ae54f001c70"
DA3 = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
NVD = "253ac4fcea7de5f396371124af597e6cc957bfae"
OVERLAYS = {
    "runtime.py": "joint_runtime.py",
    "surface.py": "joint_surface.py",
    "report.py": "joint_report.py",
    "gpu_check.py": "joint_gpu_check.py",
}


def pinned(cache, name, sha):
    path = cache / f"{name.split('/')[-1]}-{sha[:12]}"
    if not path.is_dir():
        cache.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache) as tmp:
            checkout = Path(tmp) / "repo"
            subprocess.run(["git", "init", "-q", checkout], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    checkout,
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    f"https://github.com/{name}.git",
                    sha,
                ],
                check=True,
            )
            subprocess.run(["git", "-C", checkout, "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
            try:
                checkout.rename(path)
            except FileExistsError:
                pass
    actual = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    if actual != sha:
        raise ValueError(f"Wrong upstream revision in {path}: {actual}")
    return path


def build_runtime(run, ambi, milo):
    sha = hashlib.sha256((AMBI + MILO).encode())
    for name in ["ambisur.patch", *OVERLAYS]:
        sha.update((FILES / name).read_bytes())
    target = run / f"source_{sha.hexdigest()[:16]}"
    if not target.exists():
        with TemporaryDirectory(dir=run) as tmp:
            stage = Path(tmp) / "source"
            stage.mkdir()
            archive = Path(tmp) / "upstream.tar"
            subprocess.run(
                [
                    "git",
                    "-C",
                    ambi,
                    "archive",
                    "-o",
                    archive,
                    AMBI,
                    "train.py",
                    "scene",
                    "arguments",
                    "utils",
                    "gaussian_renderer",
                    "LICENSE.md",
                ],
                check=True,
            )
            subprocess.run(["tar", "-xf", archive, "-C", stage], check=True)
            subprocess.run(
                ["git", "apply", FILES / "ambisur.patch"],
                cwd=stage,
                check=True,
                env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(stage.parent)),
            )
            (stage / "joint_tetmesh.py").write_bytes(
                subprocess.check_output(["git", "-C", milo, "show", f"{MILO}:milo/utils/tetmesh.py"])
            )
            for name, dest in OVERLAYS.items():
                shutil.copyfile(FILES / name, stage / dest)
            stage.rename(target)
    return target


def dependency_setup(ambi):
    def available(code):
        return (
            subprocess.run(
                [sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).returncode
            == 0
        )

    def install(*spec, isolated=False):
        from importlib.metadata import version

        with TemporaryDirectory() as tmp:
            constraints = Path(tmp) / "constraints.txt"
            constraints.write_text("".join(f"{pkg}=={version(pkg)}\n" for pkg in ("torch", "torchvision")))
            build_flags = [] if isolated else ["--no-build-isolation"]
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *build_flags, "-c", constraints, *map(str, spec)],
                check=True,
            )

    if not available("import torch; from simple_knn._C import distCUDA2; assert torch.cuda.is_available()"):
        raise SystemExit(
            "Activate the working 3dgs environment and select an available CUDA GPU (torch + simple_knn required)."
        )
    if not available("import depth_anything_3.api"):
        install(f"git+https://github.com/ByteDance-Seed/depth-anything-3.git@{DA3}", isolated=True)
    if not available("from diff_plane_rasterization_ambisur import GaussianRasterizer"):
        install(ambi / "submodules/diff-plane-rasterization-ambisur")
    if not available("import nvdiffrast.torch"):
        install(f"git+https://github.com/NVlabs/nvdiffrast.git@{NVD}")
    for package, module in (
        ("plyfile", "plyfile"),
        ("imageio", "imageio"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
        ("opencv-python-headless", "cv2"),
    ):
        if importlib.util.find_spec(module) is None:
            install(package)


def invoke(run, cwd, label, cmd):
    line = f"cd {shlex.quote(str(cwd))} && JOINT_RUN={shlex.quote(str(run))} {shlex.join(map(str, cmd))}"
    print(line, flush=True)
    with (run / "commands.log").open("a") as log:
        log.write(line + "\n")
    env = dict(os.environ, JOINT_RUN=str(run))
    start = time.monotonic()
    with (
        (run / f"{label}.log").open("ab") as log,
        subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc,
    ):
        try:
            while chunk := proc.stdout.read1(65536):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    with (run / "timings.jsonl").open("a") as log:
        log.write(json.dumps(dict(stage=label, seconds=time.monotonic() - start, returncode=code)) + "\n")
    if code:
        raise subprocess.CalledProcessError(code, cmd)


@contextmanager
def lock(path):
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another process is writing this run ({path}).") from None
        yield


@contextmanager
def serve(run, port):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    if not port:
        yield
        return
    handler = partial(Handler, directory=str(run))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    publish(run / "preview_url.txt", url + "\n")
    print(f"Live previews on the training host: {url} (also open {run / 'index.html'} directly)", flush=True)
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--run", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/orange4f_joint")
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Install missing DA3, AmbiSuR and nvdiffrast dependencies",
    )
    parser.add_argument("--stage", choices=("all", "prepare", "check", "da3-worker"), default="all")
    parser.add_argument(
        "--evaluate",
        type=int,
        metavar="STEP",
        help="Render an already published joint checkpoint in a separate process",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue from the most recent published joint checkpoint"
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--da3-model", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--da3-resolution", type=int, default=504)
    parser.add_argument("--seed-points", type=int, default=500000)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--mesh-from", type=int, default=15000)
    parser.add_argument("--mesh-every", type=int, default=4)
    parser.add_argument("--retet-every", type=int, default=1000)
    parser.add_argument("--mesh-gaussians", type=int, default=32000)
    parser.add_argument(
        "--extra-sites",
        type=int,
        default=8192,
        help="Additional Gaussian sites for missing supported details",
    )
    parser.add_argument("--init-views", type=int, default=256)
    parser.add_argument("--preview-every", type=int, default=500)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[7000, 15000, 20000, 30000])
    parser.add_argument("--novel-frames", type=int, default=1200)
    parser.add_argument(
        "--port", type=int, default=8112, help="Loopback live preview server (0 for files only)"
    )
    a = parser.parse_args(argv)
    run = a.run.expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    cfg_file = run / "config.json"
    if a.stage == "da3-worker":
        from scripts.joint_ambisur.data import preprocess

        cfg = json.loads(cfg_file.read_text())
        preprocess(
            run,
            json.loads((run / "cameras.json").read_text())["frames"],
            cfg["da3_model"],
            cfg["da3_resolution"],
            cfg["seed_points"],
        )
        return
    if a.evaluate is not None:
        if a.resume or a.stage != "all":
            parser.error("--evaluate takes an existing training run")
        cfg = json.loads(cfg_file.read_text())
        source = run / cfg["runtime"]
        checkpoint = run / "checkpoints" / f"step_{a.evaluate:06d}.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint has not been published: {checkpoint}")
        invoke(
            run, source, f"eval_{a.evaluate:06d}", [sys.executable, "-u", "joint_runtime.py", str(a.evaluate)]
        )
        return
    if (
        not 1 <= a.mesh_from < a.iterations
        or min(
            a.mesh_every,
            a.preview_every,
            a.retet_every,
            a.stride - 1,
            a.seed_points - 3,
            a.mesh_gaussians - 3,
            a.init_views - 1,
            a.width - 1,
            a.novel_frames - 1,
            a.da3_resolution,
        )
        <= 0
        or a.retet_every % a.mesh_every
        or a.extra_sites < 0
    ):
        parser.error(
            "Require 1 <= mesh-from < iterations, positive sampling settings, and retet-every divisible by mesh-every"
        )
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required for the final trajectories")
    run_args = [sys.executable, "-u", "-m", "scripts.orange4f_joint_mesh"]
    run.mkdir(parents=True, exist_ok=True)
    with lock(run / "training.lock"):
        if cfg_file.is_file() and a.resume:
            cfg = json.loads(cfg_file.read_text())
            source = run / cfg["runtime"]
        else:
            ambi = pinned(a.cache, "Fictionarry/AmbiSuR", AMBI)
            milo = pinned(a.cache, "Anttwo/MILo", MILO)
            if a.install_missing:
                dependency_setup(ambi)
            source = build_runtime(run, ambi, milo)
            if a.stage == "check":
                invoke(run, source, "gpu_check", [sys.executable, "-u", "joint_gpu_check.py"])
                return
            cfg = {
                key: getattr(a, key)
                for key in (
                    "width",
                    "stride",
                    "da3_model",
                    "da3_resolution",
                    "seed_points",
                    "iterations",
                    "mesh_from",
                    "mesh_every",
                    "retet_every",
                    "mesh_gaussians",
                    "extra_sites",
                    "init_views",
                    "preview_every",
                    "novel_frames",
                )
            }
            cfg.update(
                runtime=source.name,
                ambi_revision=AMBI,
                milo_revision=MILO,
                source=str(a.source.resolve()),
                keyframes=str(a.keyframes.resolve()) if a.keyframes else None,
                checkpoints=sorted(
                    {i for i in a.checkpoints if 0 < i <= a.iterations} | {a.mesh_from, a.iterations}
                ),
                runner_revision=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
            )
            if cfg_file.exists() and json.loads(cfg_file.read_text()) != cfg:
                raise SystemExit(
                    f"Different experiment already configured in {run}. Use --resume or a fresh --run."
                )
            write_json(cfg_file, cfg)
        saved = sorted((run / "checkpoints").glob("step_*.pt"))
        if saved and not a.resume and a.stage == "all":
            raise SystemExit(f"Checkpoint exists at {saved[-1]}. Pass --resume to continue.")
        if a.resume and not saved:
            raise SystemExit(
                f"No published checkpoint to resume in {run}. Rerun without --resume to reuse the DA3 cache."
            )
        if a.resume and int(saved[-1].stem.split("_")[1]) == cfg["iterations"]:
            invoke(
                run,
                source,
                "final_render",
                [sys.executable, "-u", "joint_runtime.py", str(cfg["iterations"])],
            )
            return
        with serve(run, a.port):
            dashboard(run, "Checking CUDA renderers and preparing fitting-view DA3 priors.")
            invoke(run, source, "gpu_check", [sys.executable, "-u", "joint_gpu_check.py"])
            if not a.resume:
                from scripts.joint_ambisur.data import inputs

                inputs(a.source.resolve(), run, cfg["width"], cfg["stride"], a.keyframes, cfg["novel_frames"])
                invoke(run, ROOT, "da3", [*run_args, "--stage", "da3-worker", "--run", str(run)])
            if a.stage == "prepare":
                dashboard(run, "DA3 preprocessing complete.")
                return
            cmd = [
                sys.executable,
                "-u",
                "train.py",
                "-s",
                cfg["source"],
                "-m",
                str(run / "model"),
                "-r",
                "1",
                "--data_device",
                "cpu",
                "--iterations",
                str(cfg["iterations"]),
                "--densify_until_iter",
                str(min(15000, cfg["mesh_from"])),
                "--multi_view_sample_num",
                "8192",
            ]
            if a.resume:
                cmd.extend(["--start_checkpoint", str(saved[-1])])
            invoke(run, source, "train", cmd)


if __name__ == "__main__":
    main()
