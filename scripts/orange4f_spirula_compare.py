#!/usr/bin/env python3
"""Stage the orange_4F Spirula comparison and score the same held-out cameras."""

import argparse
import csv
from html import escape
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def read_json(path):
    return json.loads(Path(path).read_text())


def read_calibration(repo, source):
    from utils.colmap_paths import colmap_model_dir

    model_dir = colmap_model_dir(source)
    spec = importlib.util.spec_from_file_location(
        "comparison_colmap", repo / "scene" / "colmap_loader.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    if (model_dir / "images.bin").is_file() and (model_dir / "cameras.bin").is_file():
        images = loader.read_extrinsics_binary(str(model_dir / "images.bin"))
        cameras = loader.read_intrinsics_binary(str(model_dir / "cameras.bin"))
    else:
        images = loader.read_extrinsics_text(str(model_dir / "images.txt"))
        cameras = loader.read_intrinsics_text(str(model_dir / "cameras.txt"))
    return images, cameras


def check_training_cameras(snapshot_path, expected, records):
    import numpy as np

    snapshots = {cam["img_name"]: cam for cam in read_json(snapshot_path)}
    if not expected <= snapshots.keys():
        raise ValueError("The baseline camera snapshot is missing training frames.")
    for name, cam in snapshots.items():
        record = records[name]
        rotation = np.asarray(record["R"])
        center = -rotation @ np.asarray(record["T"])
        if (
            not np.allclose(rotation, cam["rotation"], rtol=0, atol=1e-6)
            or not np.allclose(center, cam["position"], rtol=0, atol=1e-5)
            or not np.allclose(
                [record[k] for k in ("width", "height", "fx", "fy")],
                [cam[k] for k in ("width", "height", "fx", "fy")],
                rtol=0, atol=1e-5,
            )
        ):
            raise ValueError(f"The current COLMAP source differs from the baseline for {name}.")


def prepare(args):
    from plyfile import PlyData
    from scripts.view_comparisons import model_info

    baseline = args.baseline.resolve(strict=True)
    run = args.run.resolve()
    if run.exists():
        raise FileExistsError(f"Use a new run directory: {run}")
    model = model_info(baseline)
    cfg, records = model["cfg"], model["records"]
    if (
        model["type"] != "3dgs"
        or cfg.get("resolution") != 1
        or not cfg.get("keyframe_manifest")
        or cfg.get("loss_mask_dir")
        or cfg.get("white_background")
        or cfg.get("train_test_exp")
    ):
        raise ValueError("Select the full-scene native-resolution stride-3 3DGS baseline.")
    if cfg.get("antialiasing") is False:
        raise ValueError("Select the AA baseline. This checkpoint was configured without AA.")
    source = Path(cfg["source_path"]).resolve(strict=True)
    seed = (baseline / "input.ply").resolve(strict=True)
    checkpoint = (baseline / "point_cloud" / f"iteration_{args.step}" / "point_cloud.ply")
    seed_vertices = PlyData.read(str(seed), mmap="r")["vertex"]
    if not {"x", "y", "z", "red", "green", "blue"} <= set(seed_vertices.data.dtype.names):
        raise ValueError("The baseline input.ply must contain XYZ and RGB seed points.")
    seed_count = seed_vertices.count
    baseline_count = PlyData.read(str(checkpoint), mmap="r")["vertex"].count
    if min(seed_count, baseline_count) <= 0:
        raise ValueError("The seed cloud and baseline checkpoint must both be nonempty.")

    train = model["splits"]["train"]
    held_out = model["splits"]["all"] - train
    if not train or not held_out:
        raise ValueError("The baseline must have both training and unselected frames.")
    check_training_cameras(baseline / "cameras.json", train, records)
    images, cameras = read_calibration(args.repo, source)
    by_name = {im.name: im for im in images.values()}

    run.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".orange4f-spirula-", dir=run.parent) as tmp:
        stage = Path(tmp) / "run"
        image_dir = stage / "dataset" / "images"
        sparse = stage / "dataset" / "sparse" / "0"
        image_dir.mkdir(parents=True)
        sparse.mkdir(parents=True)

        # Match the fork's centered projection using its focal lengths and image dimensions.
        max_offset = 0.0
        with (sparse / "cameras.txt").open("w") as file:
            for cid, cam in sorted(cameras.items()):
                if cam.model == "PINHOLE":
                    fx, fy, cx, cy = cam.params
                elif cam.model == "SIMPLE_PINHOLE":
                    fx, cx, cy = cam.params
                    fy = fx
                else:
                    raise ValueError(f"Expected an undistorted camera, got {cam.model}.")
                max_offset = max(max_offset, abs(cx - cam.width / 2), abs(cy - cam.height / 2))
                file.write(
                    f"{cid} PINHOLE {cam.width} {cam.height} "
                    f"{fx:.17g} {fy:.17g} {cam.width / 2:.17g} {cam.height / 2:.17g}\n"
                )

        (stage / "dataset" / "seed.ply").symlink_to(seed)
        aliases = {}
        with (sparse / "images.txt").open("w") as file:
            # A single index preserves the original capture order for sequential matching.
            for i, name in enumerate(sorted(records)):
                picture = Path(records[name]["image_path"]).resolve(strict=True)
                split = "train" if name in train else "eval"
                alias = (Path(name).parent / f"{i:06d}_{split}{picture.suffix}").as_posix()
                aliases[name] = alias
                (image_dir / alias).parent.mkdir(parents=True, exist_ok=True)
                (image_dir / alias).symlink_to(picture)
                im = by_name[name]
                pose = " ".join(f"{v:.17g}" for v in [*im.qvec, *im.tvec])
                file.write(f"{im.id} {pose} {im.camera_id} {alias}\n\n")
        manifest = {
            "baseline": str(baseline),
            "source": str(source),
            "iteration": args.step,
            "seed_count": seed_count,
            "baseline_splat_count": baseline_count,
            "cap_max": max(seed_count, baseline_count),
            "max_input_principal_offset_px": max_offset,
            "train": sorted(train),
            "eval": sorted(held_out),
            "aliases": aliases,
        }
        (stage / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(run)
    print(f"Train {len(train):,} | held out {len(held_out):,} | seeds {seed_count:,}")
    print(f"Spirula cap {manifest['cap_max']:,} | max centered principal-point change {max_offset:.3f} px")
    print(f"Prepared {run}")


def registration(repo, dataset, manifest):
    images, _ = read_calibration(repo, dataset)
    originals = {alias: name for name, alias in manifest["aliases"].items()}
    unknown = {im.name for im in images.values()} - originals.keys()
    if unknown:
        raise ValueError(f"Unexpected SfM image names: {sorted(unknown)[:5]}")
    registered = {originals[im.name] for im in images.values()}
    report = {}
    for split in ("train", "eval"):
        report[split] = sorted((name for name in manifest[split] if name in registered), key=manifest["aliases"].get)
        report[f"missing_{split}"] = [name for name in manifest[split] if name not in registered]
        print(f"{dataset.name}: registered {len(report[split]):,}/{len(manifest[split]):,} {split} views")
        if not report[split]:
            raise ValueError(f"SfM registered no {split} views in {dataset}.")
    (dataset / "registration.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def lidar_anchors(run, manifest, cloud):
    import numpy as np
    from scripts.view_comparisons import read_colmap_records

    records = read_colmap_records(run / "dataset")
    anchors = []
    for name in manifest["train"]:
        alias = manifest["aliases"][name]
        camera = records[alias]
        rotation = np.asarray(camera["R"])
        center = -rotation @ np.asarray(camera["T"])
        anchors.append({"name": alias, "c2w": np.column_stack((rotation, center)).tolist(), "scan": str(cloud)})
    return {"frame": "scan", "anchors": anchors}


def fresh_sfm(args):
    run = args.run.resolve(strict=True)
    spirula = args.spirula.resolve(strict=True)
    manifest = read_json(run / "split.json")
    if "aliases" not in manifest:
        raise ValueError("Run prepare in a new output directory to stage photos in capture order.")
    cloud = args.lidar_cloud.resolve(strict=True) if args.lidar_cloud else None
    dataset = run / "sfm"
    dataset.mkdir(exist_ok=True)
    images = dataset / "images"
    if not images.is_symlink():
        images.symlink_to(run / "dataset" / "images", target_is_directory=True)
    if images.resolve(strict=True) != run / "dataset" / "images":
        raise ValueError(f"Unexpected image directory: {images}")
    if not (dataset / "registration.json").is_file():
        subprocess.run([
            str(spirula), "sfm", "auto", str(images), "-o", str(dataset),
            "--device", args.device, "--quality", args.quality,
            "--data-type", "video", "--camera-model", "pinhole", "--features", "sift",
        ], check=True)
    registration(args.repo, dataset, manifest)
    if cloud is None:
        return

    aligned = run / "sfm_lidar"
    if not aligned.exists():
        with TemporaryDirectory(prefix=".sfm-lidar-", dir=run) as tmp:
            copy = Path(tmp) / "dataset"
            shutil.copytree(dataset / "sparse", copy / "sparse")
            (copy / "images").symlink_to(images.resolve(strict=True), target_is_directory=True)
            copy.rename(aligned)
    anchor_file = aligned / "anchors.json"
    anchors = lidar_anchors(run, manifest, cloud)
    if anchor_file.exists():
        if read_json(anchor_file) != anchors:
            raise ValueError(f"The LiDAR workspace already uses different anchors or a different scan: {aligned}")
    else:
        anchor_file.write_text(json.dumps(anchors, indent=2) + "\n")
    subprocess.run([
        str(spirula), "lidar", str(aligned), "--cloud", str(cloud),
        "--anchors", str(anchor_file), "--mode", "auto",
    ], check=True)
    registration(args.repo, aligned, manifest)
    if not all((aligned / folder).is_dir() for folder in ("depths", "normals")):
        raise RuntimeError(f"LiDAR supervision maps are missing under {aligned}.")


def psnr(pred, target):
    import numpy as np

    delta = pred.astype(np.float32) - target
    mse = np.mean(delta * delta, axis=(0, 1), dtype=np.float64)
    with np.errstate(divide="ignore"):
        return float(np.mean(10 * np.log10(255.0**2 / mse)))


def rgb(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def save_preview(run, index, name, target, rendered, labels, scores):
    import numpy as np
    from PIL import Image

    folder = run / "floor_crops"
    folder.mkdir(exist_ok=True)
    stem = f"{index:05d}"
    Image.fromarray(target).save(folder / f"{stem}_photo.png")
    columns = [f'<figure><img src="floor_crops/{stem}_photo.png"><figcaption>Photo</figcaption></figure>']
    for label, value in zip(labels, scores):
        image = rendered[label]
        Image.fromarray(image).save(folder / f"{stem}_{label}.png")
        error = np.abs(image.astype(np.int16) - target).mean(axis=2)
        Image.fromarray(np.minimum(255, error * 4).astype(np.uint8)).save(folder / f"{stem}_{label}_error.png")
        columns.append(
            f'<figure><img src="floor_crops/{stem}_{label}.png">'
            f'<figcaption>{label} · {value:.2f} dB</figcaption>'
            f'<img src="floor_crops/{stem}_{label}_error.png" alt="RGB error"></figure>'
        )
    return f"<section><h2>{escape(name)}</h2><div>{''.join(columns)}</div></section>"


def score(args):
    import numpy as np
    import torch
    from tqdm import tqdm

    from gaussian_renderer import render
    from scene.dataset_readers import readColmapCameras
    from scene.gaussian_model import GaussianModel
    from scripts.occlusion_options import configure_pipeline, inference_settings
    from scripts.eval_all_frames import read_config
    from utils.camera_utils import loadCam

    run = args.run.resolve(strict=True)
    manifest = read_json(run / "split.json")
    baseline = Path(manifest["baseline"])
    step = manifest["iteration"]
    dataset = run / "dataset"
    sfm_data = args.model_data.resolve(strict=True) if args.model_data else dataset
    same_poses = sfm_data == dataset
    names = manifest["eval"] if same_poses else registration(args.repo, sfm_data, manifest)["eval"]
    aliases = manifest.get("aliases")
    if same_poses and aliases:
        names = sorted(names, key=aliases.get)
    output = args.output_dir.resolve(strict=True) if args.output_dir else run
    runs = {label: output / label for label in ("warp_off", "warp_basic")}
    for label, folder in runs.items():
        if read_json(folder / "metrics.json")["num_eval_images"] != len(names):
            raise ValueError(f"{label} scored a different number of held-out views.")
        if len(list(folder.glob("eval-render-*.png"))) != len(names):
            raise ValueError(f"{label} is missing the saved held-out renders.")

    cfg = read_config(baseline / "cfg_args")
    extrinsics, intrinsics = read_calibration(args.repo, dataset)
    view_aliases = {aliases[name]: name for name in names} if aliases else {
        cam.name: name for cam, name in zip(
            sorted((im for im in extrinsics.values() if im.name.startswith("eval_")), key=lambda im: im.name),
            names, strict=True,
        )
    }
    selected = {i: cam for i, cam in extrinsics.items() if cam.name in view_aliases}
    staged_cameras = readColmapCameras(selected, intrinsics, None, str(dataset / "images"), "", [])
    cameras = {view_aliases[cam.image_name]: cam for cam in staged_cameras}
    camera_args = SimpleNamespace(resolution=1, data_device="cpu", train_test_exp=False, loss_mask_dir="")
    pipeline = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=True
    )
    configure_pipeline(pipeline, inference_settings(cfg))
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
        separate_sh = True
    except ImportError:
        separate_sh = False
    splats = GaussianModel(cfg.get("sh_degree", 3))
    splats.load_ply(str(baseline / "point_cloud" / f"iteration_{step}" / "point_cloud.ply"))
    background = torch.zeros(3, device="cuda")

    crops = {}
    if args.floor_protocol:
        floor = read_json(args.floor_protocol)
        crops = {
            name: floor["views"][name]["crop_xywh"]
            for name in floor["held_out"] if name in cameras
        }
        for name in sorted(set(floor["held_out"]) - crops.keys()):
            print(f"Skipping crop in {name}: outside the full-scene held-out set.")
    crops.update({name: tuple(map(int, xywh)) for name, *xywh in args.crop})
    if extra := crops.keys() - cameras.keys():
        raise ValueError(f"Crop files must be held out: {sorted(extra)}")
    default_previews = set(np.linspace(0, len(names) - 1, num=min(4, len(names)), dtype=int)) if not crops else set()

    labels = ["3dgs_aa", *runs]
    scores = {label: [] for label in labels}
    previews = []
    report_csv = output / "raw_per_frame.csv"
    partial_csv = report_csv.with_suffix(".tmp")
    with torch.inference_mode(), partial_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", *labels, *(f"crop_{label}" for label in labels)])
        for i, name in enumerate(tqdm(names, desc="Scoring held-out views")):
            camera = loadCam(camera_args, i, cameras[name], 1.0, False, False)
            target = camera.original_image[:3].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
            for label, folder in runs.items():
                actual = rgb(folder / f"eval-gt-{i:05d}.png")
                if actual.shape != target.shape or np.mean(np.abs(actual.astype(np.int16) - target)) > 2:
                    raise ValueError(f"Ground truth mismatch for {label}, frame {i}: {name}")

            prediction = render(
                camera, splats, pipeline, background,
                separate_sh=separate_sh, use_trained_exp=False,
            )["render"]
            rendered = {
                "3dgs_aa": prediction.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy(),
                **{label: rgb(folder / f"eval-render-{i:05d}.png") for label, folder in runs.items()},
            }
            if any(image.shape != target.shape for image in rendered.values()):
                raise ValueError(f"Render shape mismatch for {name}")
            frame_scores = [psnr(rendered[label], target) for label in labels]
            for label, value in zip(labels, frame_scores):
                scores[label].append(value)

            crop_scores = [""] * len(labels)
            if name in crops:
                x, y, w, h = crops[name]
                if min(x, y) < 0 or min(w, h) <= 0 or x + w > target.shape[1] or y + h > target.shape[0]:
                    raise ValueError(f"Out-of-bounds crop for {name}: {crops[name]}")
                ys, xs = slice(y, y + h), slice(x, x + w)
                crop_scores = [psnr(rendered[label][ys, xs], target[ys, xs]) for label in labels]
                previews.append(save_preview(
                    output, i, name, target[ys, xs], {label: image[ys, xs] for label, image in rendered.items()},
                    labels, crop_scores,
                ))
            elif i in default_previews:
                previews.append(save_preview(output, i, name, target, rendered, labels, frame_scores))
            writer.writerow([name, *frame_scores, *crop_scores])
    partial_csv.replace(report_csv)
    pose_note = "All runs use the baseline camera poses." if same_poses else (
        "The 3DGS baseline uses its original camera poses. Both Spirula runs use the same fresh SfM poses."
    )
    summary = " · ".join(f"{label} {np.mean(scores[label]):.3f} dB" for label in labels)
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>orange_4F comparison</title>'
        '<style>body{font:16px system-ui;margin:2rem;background:#111;color:#eee}'
        'section div{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}'
        'figure{margin:0}img{width:100%;height:auto}figcaption{margin:.4rem 0}a{color:#9cf}</style>'
        f'<h1>orange_4F comparison</h1><p>{pose_note} Scored {len(names):,} shared held-out views.</p>'
        f'<p>{summary}</p><p><a href="raw_per_frame.csv">Per-frame PSNR</a></p>'
        f"<p>Error maps show mean absolute RGB error at four times its original intensity.</p>{''.join(previews)}</html>"
    )
    (output / "comparison.html").write_text(page, encoding="utf-8")
    for label in labels:
        print(f"{label:12s} {np.mean(scores[label]):.3f} dB")
    print(f"Per-frame raw PSNR: {report_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to the 3DGS fork")
    sub = parser.add_subparsers(dest="action", required=True)
    stage = sub.add_parser("prepare", help="Stage the exact baseline data for Spirula")
    stage.add_argument("baseline", type=Path)
    stage.add_argument("run", type=Path)
    stage.add_argument("--step", type=int, default=30000)
    solve = sub.add_parser("sfm", help="Run Spirula SfM and optionally register to a LiDAR scan")
    solve.add_argument("run", type=Path)
    solve.add_argument("--spirula", required=True, type=Path)
    solve.add_argument("--device", default="auto")
    solve.add_argument("--quality", choices=("low", "medium", "high", "extreme"), default="high")
    solve.add_argument("--lidar-cloud", type=Path, help="Registered scan in the baseline camera world frame")
    measure = sub.add_parser("score", help="Score the baseline and Spirula runs with one metric")
    measure.add_argument("run", type=Path)
    measure.add_argument("--model-data", type=Path, help="Dataset used for both Spirula runs")
    measure.add_argument("--output-dir", type=Path, help="Parent directory of the two Spirula results")
    measure.add_argument("--floor-protocol", type=Path, help="Existing multiview floor protocol.json")
    measure.add_argument(
        "--crop", nargs=5, action="append", default=[],
        metavar=("IMAGE", "X", "Y", "WIDTH", "HEIGHT"),
    )
    args = parser.parse_args()
    args.repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(args.repo))
    {"prepare": prepare, "sfm": fresh_sfm, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
