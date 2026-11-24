"""Train the native vanilla 3DGS baseline on the completed Orange4F Spirula model."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image
from scripts.keyframes import natural_key, select_keyframes
from scripts.orange4f_stock_triangle import OUTPUT, REPO, encode_video, invoke
from scripts.view_comparisons import atomic_json, model_info, read_colmap_records, write_all_frame_metrics
from utils.checkpoint_snapshots import digest


SOURCE = OUTPUT / "spirula_sfm_rgb_w1600"
RUN = OUTPUT / "spirula_3dgs_stride3_w1600"
PRIOR_SPLITS = (
    Path("/data/3dgs_outputs/orange_4F/1_full/sharp_keyframes_stride3_aa/keyframes.json"),
    OUTPUT / "stride3_noaa_v1/shared/keyframes.json",
)


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another process is working in {directory}")
        yield


def link(source, target):
    source = source.resolve(strict=True)
    if target.is_symlink() and target.resolve() == source:
        return
    target.symlink_to(source, target_is_directory=source.is_dir())


def prepare(args):
    source, run = args.source, args.run
    stage, model = run / "scene", run / "model"
    sparse = source / "sparse/0"
    staged_sparse = stage / "sparse/0"
    staged_sparse.mkdir(parents=True, exist_ok=True)
    link(source / "images", stage / "images")
    hashes = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        link(sparse / name, staged_sparse / name)
        hashes[name] = digest(sparse / name)
    with (sparse / "points3D.bin").open("rb") as stream:
        points, = struct.unpack("<Q", stream.read(8))
    if points < 4:
        raise ValueError(f"Spirula exported only {points} seed points")
    records = read_colmap_records(stage)
    names = sorted(records, key=natural_key)
    if len(names) != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} registered images, got {len(names)} from {sparse}")
    for row in records.values():
        with Image.open(row["image_path"]) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"RGB and Spirula calibration dimensions disagree for {row['name']}")
        if not all(math.isfinite(row[k]) for k in ("fx", "fy", "cx", "cy")) or min(row["fx"], row["fy"]) <= 0:
            raise ValueError(f"Invalid pinhole calibration for {row['name']}")

    fingerprint = hashlib.sha256(json.dumps(sorted(names), ensure_ascii=False).encode()).hexdigest()
    saved_path = model / "keyframes.json"
    protocol_path = run / "protocol.json"
    old = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    if old and (old["sfm_sha256"] != hashes or old["source"] != str(source) or old["iterations"] != args.iterations):
        raise ValueError("SfM source or training schedule differs from this run; use a fresh --run")
    if not old and (model / "cfg_args").exists():
        raise ValueError(f"Existing model in {model} has no Spirula run protocol; use a fresh --run")
    if args.keyframes:
        selection = args.keyframes.resolve(strict=True)
    elif old:
        selection = saved_path
    else:
        selection = next((p for p in PRIOR_SPLITS if p.is_file() and
                          json.loads(p.read_text()).get("registered_names_sha256") == fingerprint), None)
    if selection is None:
        selection = select_keyframes(stage, "images", model, 3, False)
    manifest = json.loads(selection.read_text())
    chosen = manifest["selected_images"]
    if (manifest.get("registered_names_sha256") != fingerprint or manifest.get("group_size") != 3
            or manifest.get("eval") is not False or len(chosen) != math.ceil(len(names)/3)
            or any(n not in names[i*3:i*3+3] for i, n in enumerate(chosen))):
        raise ValueError(f"Keyframes do not match this capture's sharp stride-3 protocol: {selection}")
    if old and chosen != old["train"]:
        raise ValueError("Training split differs from the existing run")
    selected = set(chosen)
    report = dict(source=str(source), images=str((stage/"images").resolve()), sfm_sha256=hashes,
                  seed_points=points, pose_provenance="Spirula RGB SfM on all registered views; poses fixed",
                  seed_provenance=str(sparse / "points3D.bin"), iterations=args.iterations, width=1600,
                  antialiasing=False, principal_point="Spirula pinhole", keyframe_selection=
                  old["keyframe_selection"] if old else str(selection), train=chosen,
                  test=[n for n in names if n not in selected],
                  max_principal_offset_px_at_1600=max(1600*math.hypot(r["cx"]-r["width"]/2,
                      r["cy"]-r["height"]/2)/r["width"] for r in records.values()))
    if not old:
        report["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        atomic_json(saved_path, {**manifest, "source_path": str(stage)})
        atomic_json(protocol_path, report)
    print(f"Spirula: {len(names):,} cameras; {points:,} XYZ/RGB seeds; {len(chosen):,} train, "
          f"{len(names)-len(chosen):,} test. Fit and score at width 1600, AA off.\n"
          f"Run: {run}\nPreviews: {model/'debug_previews/index.html'}", flush=True)


def evaluate(args, step, videos):
    run, model = args.run, args.run / "model"
    output = run / f"evaluation/iteration_{step}"
    with lock(output):
        if not (output / "index.html").is_file():
            invoke(REPO, output, "render", ["scripts/render.py", "--split", "both", "--eval-set", "unselected",
                   "--model", model, "--iteration", step, "--gpu", args.gpu, "--output", output,
                   "--no-antialiasing", "--in-process"])
        records, panels, renders = {}, {}, {}
        for split in ("train", "eval"):
            folder = output / split / "model_0"
            rows = json.loads((folder / "frames.json").read_text())
            if records.keys() & rows.keys():
                raise ValueError("Overlapping training and held-out render outputs")
            records.update(rows)
            panels.update({n: folder/r["comparison"] for n, r in rows.items()})
            renders.update({n: folder/r["image"] for n, r in rows.items()})
        summary = write_all_frame_metrics(model_info(model), records, step, False)
        if summary is None:
            raise ValueError("Incomplete all-frame evaluation")
        groups = {"train": "selected", "test": "unselected", "all": "all"}
        scores = {label: summary["mean_psnr_db"][key] for label, key in groups.items()}
        atomic_json(output / "psnr.json", dict(iteration=step, psnr_db=scores, metric=summary["metric"],
                    counts={label: summary["counts"][key] for label, key in groups.items()}))
        print(f"PSNR (dB) {scores}\nGallery: {output/'index.html'}", flush=True)
        if videos:
            video = output / "videos"
            video.mkdir(exist_ok=True)
            names = sorted(records, key=natural_key)
            encode_video(video/"recorded.mp4", [renders[n] for n in names])
            encode_video(video/"gt_render.mp4", [panels[n] for n in names])
            if not (video / "random/render.mp4").is_file():
                with TemporaryDirectory(dir=video) as tmp:
                    destination = Path(tmp) / "random"
                    invoke(REPO, output, "novel", ["scripts/render_ablation_videos.py", "--video", "random",
                           "--model", model, "--source", run/"scene", "--iteration", step, "--gpu", args.gpu,
                           "--width", 1600, "--aa", "off", "--seed", 42, "--output", destination, "--in-process"])
                    plan = json.loads((destination/"trajectory.json").read_text())
                    atomic_json(destination/"trajectory.json", {**plan, "output": str(video/"random")})
                    destination.rename(video/"random")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--keyframes", type=Path)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--expected-frames", type=int, default=5009)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--evaluate-iteration", type=int)
    args = p.parse_args()
    args.source, args.run = args.source.expanduser().resolve(), args.run.expanduser().resolve()
    if not args.gpu or "," in args.gpu or args.iterations < 1 or args.expected_frames < 4:
        p.error("Select a single GPU, a positive training length and at least four cameras")
    if args.run == args.source or args.source in args.run.parents or args.run in args.source.parents:
        p.error("Use separate, non-overlapping source and run directories")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.evaluate_iteration is not None:
        evaluate(args, args.evaluate_iteration, False)
        return
    with lock(args.run):
        prepare(args)
        if args.prepare_only:
            return
        model = args.run / "model"
        done = args.run / "trained.json"
        if not done.is_file():
            steps = sorted({min(7000, args.iterations), args.iterations})
            checkpoints = [step for step in steps if (model/f"chkpnt{step}.pth").is_file()]
            resume = ["--start_checkpoint", model/f"chkpnt{max(checkpoints)}.pth"] if checkpoints else []
            invoke(REPO, args.run, "train", ["train.py", "-s", args.run/"scene", "-m", model,
                   "--keyframe_manifest", model/"keyframes.json", "-r", 1600, "--data_device", "cpu",
                   "--colmap_principal_point", "--disable_viewer", "--iterations", args.iterations,
                   "--checkpoint_iterations", *steps, "--save_iterations", *steps, *resume])
            atomic_json(done, dict(iteration=args.iterations))
        evaluate(args, args.iterations, True)


if __name__ == "__main__":
    main()
