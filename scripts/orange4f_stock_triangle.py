"""Train and render upstream Triangle Splatting on the full Orange4F capture."""

import argparse
import fcntl
import hashlib
from importlib.util import find_spec
import json
import math
from pathlib import Path
import resource
import shlex
import shutil
from statistics import mean
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from types import SimpleNamespace
from zipfile import is_zipfile

import numpy as np
from PIL import Image

from scripts.keyframes import natural_key, select_keyframes
from scripts.render_model_comparison import trajectory_record, write_video
from scripts.render_novel_comparison import choose_source_segment, synthesize_random_path
from scripts.view_comparisons import atomic_json, save_panel, write_gallery
from utils.colmap_paths import colmap_model_dir
from utils import read_write_model as colmap


SOURCE = Path("/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full")
OUTPUT = Path("/data/3dgs_outputs/orange4f/3dgs_finetuning")
UPSTREAM_SHA = "685ff270a8273afc3a2b094ae3148dd0b98e31e9"
VIDEO_FPS = 30
NOVEL_SECONDS = 40
NOVEL_SEED = 42
DENSIFICATION_PATCH = Path(__file__).resolve().parent / "indoor_triangles/stock_triangle_densification.patch"


def choose_keyframes(source, run, names, explicit, stride):
    manifest = explicit.resolve() if explicit else None
    if manifest is None and stride == 3:
        previous = (Path("/data/3dgs_outputs/orange_4F/1_full/sharp_keyframes_stride3_aa/keyframes.json"),
                    OUTPUT / "stride3_noaa_v1/shared/keyframes.json")
        for path in previous:
            if path.is_file():
                saved = json.loads(path.read_text())
                if (Path(saved.get("source_path", "")).resolve() == source and
                        saved.get("group_size") == 3 and saved.get("eval") is False):
                    manifest = path
                    break
    if manifest is None:
        manifest = select_keyframes(source, "images", run, stride, False)
    saved = json.loads(manifest.read_text())
    chosen = saved["selected_images"]
    if len(chosen) != len(set(chosen)) or len(chosen) < 2 or not set(chosen) < set(names):
        raise ValueError(f"Expected distinct registered fitting keyframes and a nonempty test set in {manifest}")
    return set(chosen), str(manifest)


def seed_path(sparse, explicit):
    candidates = [explicit] if explicit else [sparse / f"points3D.{ext}" for ext in ("bin", "txt", "ply")]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            if path.suffix == ".bin":
                header = stream.read(8)
                if len(header) == 8 and struct.unpack("<Q", header)[0] >= 4:
                    return path.resolve()
            if path.suffix == ".txt":
                if sum(bool(line.strip()) and not line.startswith(b"#") for line in stream) >= 4:
                    return path.resolve()
            if path.suffix == ".ply":
                return path.resolve()
    raise FileNotFoundError(f"No populated COLMAP point cloud in {sparse}. Supply --point-cloud with an XYZ/RGB PLY or COLMAP points3D.bin/.txt in the same coordinate frame.")


def stage_images(source, target, frames, cameras, width):
    target.mkdir(parents=True, exist_ok=True)
    for index, (row, frame) in enumerate(frames, 1):
        cam = cameras[frame.camera_id]
        if cam.model == "SIMPLE_PINHOLE":
            fx, cx, cy = cam.params
            fy = fx
        elif cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
        else:
            raise ValueError(f"Undistorted PINHOLE images required, found {cam.model} for {frame.name}")
        original = source / "images" / frame.name
        with Image.open(original) as image:
            if image.size != (cam.width, cam.height):
                raise ValueError(f"Calibration size differs from RGB for {original}: {cam.width}x{cam.height} versus {image.size}")
            w = min(width, cam.width)
            h = max(1, round(cam.height * w / cam.width))
            sx, sy = w / cam.width, h / cam.height
            dst = target / row["staged_name"]
            if not dst.is_file():
                if image.size == (w, h) and image.mode == "RGB":
                    dst = dst.with_suffix(original.suffix or ".png")
                    if not dst.is_file():
                        dst.symlink_to(original.resolve())
                    row["staged_name"] = dst.name
                else:
                    tmp = dst.with_suffix(".tmp")
                    image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS).save(tmp, "PNG", compress_level=1)
                    tmp.replace(dst)
        qnorm = np.linalg.norm(frame.qvec)
        if not np.isfinite(qnorm) or qnorm == 0 or not np.isfinite(frame.tvec).all():
            raise ValueError(f"Invalid pose for {frame.name}")
        row.update(width=w, height=h, original_size=[cam.width, cam.height],
                   intrinsics=[float(fx*sx), float(fy*sy), float(cx*sx), float(cy*sy)])
        if index % 250 == 0 or index == len(frames):
            print(f"Staged {index:,}/{len(frames):,} RGB images", flush=True)
        yield row, frame.qvec / qnorm, frame.tvec


def write_scene(path, rows, images, seed, *, fit=False):
    model = path / "sparse/0"
    model.mkdir(parents=True, exist_ok=True)
    image_dir = path / "images"
    if not image_dir.exists():
        image_dir.symlink_to(images, target_is_directory=True)
    with (model / "cameras.txt").open("w") as cameras, (model / "images.txt").open("w") as poses:
        for index, (row, qvec, tvec) in enumerate(rows, 1):
            intrinsics = " ".join(map(str, row["intrinsics"]))
            cameras.write(f"{index} PINHOLE {row['width']} {row['height']} {intrinsics}\n")
            pose = " ".join(map(str, (*qvec, *tvec)))
            poses.write(f"{index} {pose} {index} {row['staged_name']}\n\n")
    cloud = model / ("points3D" + seed.suffix)
    if not cloud.is_symlink() and not cloud.exists():
        if fit and seed.suffix == ".ply":
            from scripts.prepare_orange4f import canonical_cloud

            canonical_cloud(path, cloud, explicit=seed)
        else:
            cloud.symlink_to(seed)


def prepare_training_source(upstream, run):
    """Apply the coverage fix to a reproducible copy of the pinned training code."""
    patch_hash = hashlib.sha256(DENSIFICATION_PATCH.read_bytes()).hexdigest()
    source_dir = run / f"triangle_source_{UPSTREAM_SHA[:12]}_{patch_hash[:12]}"
    if not source_dir.is_dir():
        with TemporaryDirectory(prefix=".triangle_source_", dir=run) as temporary:
            root = Path(temporary)
            archive = root / "source.tar"
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "archive", "-o", archive, UPSTREAM_SHA], cwd=upstream, check=True)
            subprocess.run(["tar", "-xf", archive, "-C", source], check=True)
            subprocess.run(["git", "apply", "--unidiff-zero", DENSIFICATION_PATCH], cwd=source, check=True)
            source.rename(source_dir)
    atomic_json(run / "training_source.json", dict(path=str(source_dir), revision=UPSTREAM_SHA,
                                                  patch=str(DENSIFICATION_PATCH), patch_sha256=patch_hash))
    return source_dir


def invoke(upstream, run, label, options):
    command = [sys.executable, "-u", *map(str, options)]
    line = f"cd {shlex.quote(str(upstream))} && {shlex.join(command)}"
    print(line, flush=True)
    with (run / "commands.log").open("a") as commands:
        commands.write(line + "\n")
    start = perf_counter()
    with (run / f"{label}.log").open("w") as log:
        with subprocess.Popen(command, cwd=upstream, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
    return perf_counter() - start


def report_shard(directory, rows):
    names = {f"{i:05d}.png" for i in range(len(rows))}
    if any({p.name for p in (directory / folder).glob("*.png")} != names for folder in ("gt", "renders")):
        raise ValueError(f"Expected {len(rows)} matched RGB renders in {directory}")
    for index, row in enumerate(rows):
        name = f"{index:05d}.png"
        gt, rendered = directory / "gt" / name, directory / "renders" / name
        run = directory.parents[2]
        pair = run / "comparisons" / Path(row["staged_name"]).with_suffix(".jpg")
        with Image.open(gt) as actual, Image.open(rendered) as prediction:
            if actual.size != prediction.size or actual.size != (row["width"], row["height"]):
                raise ValueError(f"Unexpected render size in {directory} for {row['name']}")
            actual, prediction = actual.convert("RGB"), prediction.convert("RGB")
            error = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(prediction, dtype=np.float32))
            mse = float(np.mean((error / 255.0) ** 2))
            score = -10 * math.log10(max(mse, 1e-12))
            current_size = None
            if pair.is_file():
                with Image.open(pair) as comparison:
                    current_size = comparison.size
            if current_size != (2 * actual.width, actual.height + 36):
                temporary = pair.with_suffix(".tmp.jpg")
                save_panel(actual, prediction, temporary, "Triangle Splatting", score, max_width=None)
                temporary.replace(pair)
        error_path = directory / "error" / name
        error_path.parent.mkdir(exist_ok=True)
        Image.fromarray(np.clip(4 * error, 0, 255).astype(np.uint8)).save(error_path)
        yield dict(name=row["name"], split=row["split"], gt="../" + str(gt.relative_to(run)),
                   comparison="../" + str(pair.relative_to(run)),
                   renders=[dict(image="../" + str(rendered.relative_to(run)),
                                 error="../" + str(error_path.relative_to(run)), psnr_db=score)])


def render_batch(upstream, run, model, iteration, label, batch, images, seed, common):
    split, shard = label
    destination = run / "renders" / split / shard
    if not destination.exists():
        scene = run / "render_inputs" / split / shard
        write_scene(scene, batch, images, seed)
        stock_output = model / "train" / f"ours_{iteration}"
        invoke(upstream, run, f"render_{split}_{shard}",
               ["render.py", "-s", scene, *common, "--iteration", iteration, "--skip_test"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        stock_output.rename(destination)
    return destination


def encode_video(output, paths):
    if output.is_file():
        video_mtime = output.stat().st_mtime_ns
        if all(path.stat().st_mtime_ns <= video_mtime for path in paths):
            return
    sizes = set()
    for path in paths:
        with Image.open(path) as image:
            sizes.add(image.size)
    width = 2 * math.ceil(max(w for w, _ in sizes) / 2)
    height = 2 * math.ceil(max(h for _, h in sizes) / 2)

    def frames():
        for path in paths:
            with Image.open(path) as image:
                canvas = Image.new("RGB", (width, height))
                canvas.paste(image, ((width - image.width)//2, (height - image.height)//2))
                yield np.asarray(canvas)

    write_video(output, frames(), width, height, VIDEO_FPS, 18)


def novel_trajectory(run, staged):
    cameras = []
    for row, qvec, tvec in staged:
        w, h = row["width"], row["height"]
        fx, fy = row["intrinsics"][:2]
        cameras.append(SimpleNamespace(image_name=row["name"], R=colmap.qvec2rotmat(qvec).T, T=tvec,
                                       width=w, height=h, FovX=2*math.atan(w/(2*fx)), FovY=2*math.atan(h/(2*fy))))
    segment, selection = choose_source_segment(cameras, window=600, seed=NOVEL_SEED)
    path, details = synthesize_random_path(segment, cameras, VIDEO_FPS*NOVEL_SECONDS,
                                           deviation=2., seed=NOVEL_SEED)
    atomic_json(run / "random_trajectory.json", dict(fps=VIDEO_FPS, selection=selection,
                                                     synthesis=details, cameras=trajectory_record(path)))
    # The stock COLMAP renderer needs RGB files even for a novel-view pose.
    rgb = run / "random_rgb_placeholders"
    rgb.mkdir(exist_ok=True)
    blank = rgb / "blank.png"
    Image.new("RGB", (path[0].width, path[0].height)).save(blank)
    entries = []
    for view in path:
        filename = f"{view.image_name}.png"
        target = rgb / filename
        if not target.is_symlink():
            target.symlink_to(blank)
        w, h = view.width, view.height
        row = dict(staged_name=filename, width=w, height=h,
                   intrinsics=[w/(2*math.tan(view.FovX/2)), h/(2*math.tan(view.FovY/2)), w/2, h/2])
        entries.append((row, colmap.rotmat2qvec(view.R.T), view.T))
    return entries, rgb


def read_colmap_scene(sparse):
    if (sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file():
        return colmap.read_cameras_binary(sparse / "cameras.bin"), colmap.read_images_binary(sparse / "images.bin")
    return colmap.read_cameras_text(sparse / "cameras.txt"), colmap.read_images_text(sparse / "images.txt")


def evaluate_frames(upstream, run, model, iteration, staged, images, render_seed, common):
    (run / "comparisons").mkdir(exist_ok=True)
    scores = {}
    for split in ("train", "test"):
        entries = [entry for entry in staged if entry[0]["split"] == split]
        for offset in range(0, len(entries), 256):
            shard = f"{offset//256:03d}"
            batch = entries[offset:offset+256]
            destination = render_batch(upstream, run, model, iteration, (split, shard), batch,
                                       images, render_seed, common)
            scores.update((row["name"], row) for row in report_shard(destination, [row for row, _, _ in batch]))
    all_rows = [scores[row["name"]] for row, _, _ in staged]
    totals = {}
    for split in ("train", "test", "all"):
        scored = all_rows if split == "all" else [row for row in all_rows if row["split"] == split]
        totals[split] = dict(count=len(scored), mean_psnr_db=mean(row["renders"][0]["psnr_db"] for row in scored))
        gallery = run / split
        gallery.mkdir(exist_ok=True)
        write_gallery(gallery, f"Orange4F stock Triangle Splatting · {split}", scored,
                      [dict(label="Stock Triangle Splatting")], totals[split])
    return totals, all_rows


def evaluation_snapshot(training_model, evaluation, iteration):
    relative = Path("point_cloud") / f"iteration_{iteration}"
    checkpoint_files = [relative / name for name in ("point_cloud_state_dict.pt", "hyperparameters.pt")]
    model = evaluation / "model"
    if model.is_dir():
        return model
    files = [Path("cfg_args"), *checkpoint_files]
    print(f"Waiting for the complete {iteration:,}-step checkpoint in {training_model / relative}", flush=True)
    while not (training_model / "cfg_args").is_file() or not all(is_zipfile(training_model / path) for path in checkpoint_files):
        sleep(5)

    def identity():
        stats = [(training_model / path).stat() for path in files]
        return [(stat.st_ino, stat.st_size, stat.st_mtime_ns) for stat in stats]

    before = identity()
    with TemporaryDirectory(prefix=".model_", dir=evaluation) as directory:
        temporary = Path(directory) / "model"
        (temporary / relative).mkdir(parents=True)
        for path in files:
            shutil.copy2(training_model / path, temporary / path)
        if identity() != before or not all(is_zipfile(temporary / path) for path in checkpoint_files):
            raise RuntimeError(f"Checkpoint at {training_model / relative} changed during the snapshot. Rerun evaluation.")
        temporary.rename(model)
    return model


def evaluate_checkpoint(upstream, training_run, evaluation, iteration):
    manifest = json.loads((training_run / "split_manifest.json").read_text())
    images = training_run / "images"
    seed = training_run / "fit_input" / "sparse/0/points3D.ply"
    _, extrinsics = read_colmap_scene(Path(manifest["settings"]["colmap_model"]))
    by_name = {frame.name: frame for frame in extrinsics.values()}
    staged = []
    for row in manifest["frames"]:
        frame = by_name[row["name"]]
        if not (images / row["staged_name"]).is_file():
            raise FileNotFoundError(images / row["staged_name"])
        staged.append((row, frame.qvec / np.linalg.norm(frame.qvec), frame.tvec))

    training_model = training_run / "model"
    model = evaluation_snapshot(training_model, evaluation, iteration)
    common = ["-m", model, "-r", "1", "--data_device", "cpu"]
    print(f"Rendering checkpoint {iteration:,} on {len(staged):,} views. Output: {evaluation}", flush=True)
    totals, _ = evaluate_frames(upstream, evaluation, model, iteration, staged, images, seed, common)
    checkpoint = training_model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    atomic_json(evaluation / "results.json", dict(iteration=iteration, checkpoint=str(checkpoint), splits=totals))
    print(json.dumps(totals, indent=2), flush=True)


def run_scene(args, upstream, source, run):
    sparse = colmap_model_dir(source)
    cameras, extrinsics = read_colmap_scene(sparse)
    names = sorted((frame.name for frame in extrinsics.values()), key=natural_key)
    if (not names or len(names) != len(set(names)) or
            any(Path(name).is_absolute() or ".." in Path(name).parts for name in names)):
        raise ValueError("Expected unique, relative COLMAP image names")
    if args.expected_frames and len(names) != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames:,} registered views, found {len(names):,} in {sparse}")
    missing = [name for name in names if not (source / "images" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing):,} registered RGB images, including {missing[:5]}")
    fit, keyframes = choose_keyframes(source, run, names, args.keyframes, args.keyframe_stride)
    seed = seed_path(sparse, args.point_cloud)
    print(f"{len(names):,} registered RGB views: {len(fit):,} fitting, {len(names)-len(fit):,} held-out", flush=True)
    print(f"Fixed calibration: {sparse}\nInitialization: {seed}\nRun: {run}", flush=True)
    settings = dict(source=str(source), colmap_model=str(sparse), init_points=str(seed), keyframes=keyframes,
                    upstream=str(upstream), upstream_revision=UPSTREAM_SHA, width=args.width,
                    iterations=args.iterations, train_count=len(fit), heldout_count=len(names)-len(fit))
    manifest_file = run / "split_manifest.json"
    by_name = {frame.name: frame for frame in extrinsics.values()}
    frames = [(dict(name=name, staged_name=f"{index:08d}.png", split="train" if name in fit else "test"),
               by_name[name]) for index, name in enumerate(names)]
    if manifest_file.is_file():
        existing = json.loads(manifest_file.read_text())
        if existing["settings"] != settings or [(r["name"], r["split"]) for r in existing["frames"]] != [
                (r["name"], r["split"]) for r, _ in frames]:
            raise ValueError(f"Inputs differ from {manifest_file}; specify a fresh --run directory")
    images = run / "images"
    staged = list(stage_images(source, images, frames, cameras, args.width))
    fitting = [entry for entry in staged if entry[0]["split"] == "train"]
    fit_scene = run / "fit_input"
    write_scene(fit_scene, fitting, images, seed, fit=True)
    atomic_json(manifest_file, dict(settings=settings, frames=[row for row, _, _ in staged]))
    if args.prepare_only:
        return

    # Upstream opens all fitting images with PIL before loading the tensors.
    # Raise the soft limit for the trainer and retain RGB tensors in host memory.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = len(fitting) + 128
    if hard != resource.RLIM_INFINITY and hard < required:
        raise OSError(f"The hard open-file limit is {hard}; stock loading requires at least {required}")
    if soft != resource.RLIM_INFINITY and soft < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    model = run / "model"
    iteration = args.iterations
    checkpoint = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    common = ["-m", model, "-r", "1", "--data_device", "cpu"]
    train_seconds_file = run / "training_seconds.json"
    if not checkpoint.is_file() or not checkpoint.with_name("hyperparameters.pt").is_file():
        training_source = prepare_training_source(upstream, run)
        train_seconds = invoke(training_source, run, "train",
                               ["train.py", "-s", fit_scene, *common, "--iterations", iteration])
        if not checkpoint.is_file() or not checkpoint.with_name("hyperparameters.pt").is_file():
            raise FileNotFoundError(f"Incomplete checkpoint at {checkpoint.parent}")
        atomic_json(train_seconds_file, train_seconds)
    else:
        print(f"Reusing saved model {checkpoint.parent}", flush=True)

    mesh = run / "direct.off"
    if not mesh.is_file():
        temporary = run / "direct.tmp.off"
        invoke(upstream, run, "mesh_export", ["create_off.py", "--checkpoint_path", checkpoint, "--output_name", temporary])
        temporary.replace(mesh)

    render_seed = fit_scene / "sparse/0/points3D.ply"
    totals, all_rows = evaluate_frames(upstream, run, model, iteration, staged, images, render_seed, common)
    train_seconds = json.loads(train_seconds_file.read_text()) if train_seconds_file.is_file() else None
    results = dict(splits=totals, checkpoint=str(checkpoint), direct_mesh=str(mesh), training_seconds=train_seconds)
    atomic_json(run / "results.json", results)
    print(json.dumps(totals, indent=2), flush=True)

    original = run / "original_trajectory.mp4"
    print(f"Encoding {len(all_rows):,} captured views in natural filename order", flush=True)
    encode_video(original, [run / "all" / row["comparison"] for row in all_rows])
    random = run / "random_trajectory.mp4"
    if not random.is_file():
        novel, rgb = novel_trajectory(run, staged)
        novel_renders = []
        for offset in range(0, len(novel), 256):
            batch = novel[offset:offset+256]
            destination = render_batch(upstream, run, model, iteration, ("random", f"{offset//256:03d}"),
                                       batch, rgb, render_seed, common)
            novel_renders.extend(destination / "renders" / f"{i:05d}.png" for i in range(len(batch)))
        encode_video(random, novel_renders)
    results["videos"] = dict(original_trajectory=str(original), random_trajectory=str(random), fps=VIDEO_FPS)
    atomic_json(run / "results.json", results)
    print(f"Videos: {original}, {random}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path.home() / "triangle-splatting-upstream")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--run", type=Path)
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--keyframes", type=Path, help="Existing keyframes.json containing selected_images")
    split.add_argument("--keyframe-stride", type=int, default=3, help="Select the sharpest RGB in each group (default: 3)")
    parser.add_argument("--point-cloud", type=Path, help="Explicit initialization cloud in the supplied COLMAP coordinates")
    parser.add_argument("--width", type=int, default=1600, help="Maximum image width for training and rendering (default: 1600)")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--expected-frames", type=int, default=5009, help="Fail before training on an incomplete capture; 0 disables the check")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--evaluate-iteration", type=int, help="Render and score the given checkpoint in an existing --run")
    args = parser.parse_args()
    if args.width < 1 or args.iterations < 1 or args.expected_frames < 0 or args.keyframe_stride < 2:
        parser.error("width and iterations must be positive, expected-frames nonnegative, and keyframe-stride at least 2")
    if args.evaluate_iteration is not None and args.evaluate_iteration < 1:
        parser.error("--evaluate-iteration must be positive")
    upstream, source = args.upstream.expanduser().resolve(), args.source.expanduser().resolve()
    revision = subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", upstream, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != UPSTREAM_SHA or changes:
        parser.error(f"Use a clean upstream checkout at {UPSTREAM_SHA}: {upstream}")
    if not args.prepare_only:
        if find_spec("diff_triangle_rasterization") is None:
            rasterizer = "submodules/diff-triangle-rasterization"
            parser.error(
                f"Missing diff_triangle_rasterization in {sys.executable}. Install it in the active environment:\n"
                f"  cd {shlex.quote(str(upstream))}\n"
                f"  git submodule update --init --recursive -- {rasterizer}\n"
                f"  python -m pip install --no-build-isolation --no-deps ./{rasterizer}"
            )
        if args.evaluate_iteration is None and shutil.which("ffmpeg") is None:
            parser.error("ffmpeg with libx264 is required to encode the videos")
        preflight = subprocess.run([sys.executable, "-c", "import torch, lpips, plyfile; "
                                    "from simple_knn._C import distCUDA2; from triangle_renderer import render; "
                                    "assert torch.cuda.is_available(), 'CUDA GPU is unavailable'"], cwd=upstream)
        if preflight.returncode:
            raise SystemExit(preflight.returncode)
    cohort = "keyframes" if args.keyframes else f"stride{args.keyframe_stride}"
    run = (args.run or OUTPUT / f"triangle_splatting_stock_full_{cohort}_w{args.width}_{args.iterations}").resolve()
    if run.is_relative_to(source) or run.is_relative_to(upstream):
        parser.error("Run directory must be outside the input dataset and upstream checkout")
    if args.evaluate_iteration is not None:
        if not (run / "split_manifest.json").is_file():
            parser.error(f"Missing staged training run at {run}. Set --run to the active training run directory.")
        evaluation = run / f"evaluation_{args.evaluate_iteration}"
        evaluation.mkdir(exist_ok=True)
        with (evaluation / "run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            evaluate_checkpoint(upstream, run, evaluation, args.evaluate_iteration)
        return
    run.mkdir(parents=True, exist_ok=True)
    with (run / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_scene(args, upstream, source, run)


if __name__ == "__main__":
    main()
