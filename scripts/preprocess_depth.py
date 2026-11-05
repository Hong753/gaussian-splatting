#!/usr/bin/env python3
"""Predict DA-v2 inverse depth and align it to a LiDAR-derived point cloud."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.colmap_paths import colmap_model_dir
from utils.depth_alignment import fit_alignment

DA2_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
WEIGHTS_URL = ("https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/"
               "8e4a3952be8793aef563d85bacf827c12e830dc1/depth_anything_v2_vitl.pth")


def setup_da2(cache):
    source = cache / "Depth-Anything-V2"
    weights = cache / "depth_anything_v2_vitl.pth"
    cache.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").is_dir():
        if source.exists():
            raise ValueError(f"DA-v2 cache path exists but is not a git checkout: {source}")
        subprocess.run(["git", "clone", "https://github.com/DepthAnything/Depth-Anything-V2.git", str(source)], check=True)
    current = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if current != DA2_COMMIT:
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
            raise ValueError(f"DA-v2 checkout has local changes: {source}")
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", DA2_COMMIT], check=True)
    if not weights.is_file():
        temp = weights.with_suffix(".download")
        print(f"Downloading DA-v2 Large weights to {weights}", flush=True)
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, str(temp))
            os.replace(temp, weights)
        finally:
            temp.unlink(missing_ok=True)
    sys.path.insert(0, str(source))
    return weights


def load_colmap(model_dir):
    from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                     read_extrinsics_text, read_intrinsics_text)
    if (model_dir / "images.bin").is_file() and (model_dir / "cameras.bin").is_file():
        return read_extrinsics_binary(str(model_dir / "images.bin")), read_intrinsics_binary(str(model_dir / "cameras.bin"))
    return read_extrinsics_text(str(model_dir / "images.txt")), read_intrinsics_text(str(model_dir / "cameras.txt"))


def load_points(path):
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"].data
    return np.stack([vertex[key] for key in ("x", "y", "z")], axis=1).astype(np.float32)


def sample_projected_points(points, camera, intrinsic, image_shape, depth_png, cell_size, max_samples):
    """Z-buffer projected points in coarse cells; this is approximate visibility."""
    import torch
    from scene.colmap_loader import qvec2rotmat
    rotation = torch.tensor(qvec2rotmat(camera.qvec), dtype=torch.float32, device=points.device)
    translation = torch.tensor(camera.tvec, dtype=torch.float32, device=points.device)
    xyz = points @ rotation.T + translation
    z = xyz[:, 2]
    if intrinsic.model == "PINHOLE":
        fx, fy, cx, cy = intrinsic.params[:4]
    elif intrinsic.model == "SIMPLE_PINHOLE":
        fx, cx, cy = intrinsic.params[:3]
        fy = fx
    else:
        raise ValueError(f"Camera {intrinsic.id} uses {intrinsic.model}; undistorted pinhole images are required")
    u = fx * xyz[:, 0] / z + cx
    v = fy * xyz[:, 1] / z + cy
    mask = ((z > 1e-4) & torch.isfinite(z) & torch.isfinite(u) & torch.isfinite(v)
            & (u >= 0) & (v >= 0) & (u < intrinsic.width) & (v < intrinsic.height))
    u, v, z = u[mask], v[mask], z[mask]
    if not len(z):
        return np.array([]), np.array([])
    # Map COLMAP pixel coordinates to the exact saved depth image dimensions.
    height, width = image_shape
    u = u * (width / intrinsic.width)
    v = v * (height / intrinsic.height)
    grid_width = (width + cell_size - 1) // cell_size
    grid_height = (height + cell_size - 1) // cell_size
    cell = (v.long() // cell_size) * grid_width + u.long() // cell_size
    closest = torch.full((grid_width * grid_height,), float("inf"), device=points.device)
    closest.scatter_reduce_(0, cell, z, reduce="amin", include_self=True)
    visible = z <= closest[cell] + 1e-6
    u, v, z = u[visible], v[visible], z[visible]
    # Evenly limit samples in deterministic projected point order.
    if len(z) > max_samples:
        idx = torch.linspace(0, len(z) - 1, max_samples, device=points.device).long()
        u, v, z = u[idx], v[idx], z[idx]
    ui = u.long().cpu().numpy()
    vi = v.long().cpu().numpy()
    reference = 1.0 / z.cpu().numpy()
    predicted = depth_png[vi, ui].astype(np.float64) / 65536.0
    return predicted, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, help="Registered LiDAR PLY (default: COLMAP points3D.ply)")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/3dgs_depth")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-points", type=int, default=2000000)
    parser.add_argument("--cell-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--min-reliable-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if min(args.input_size, args.max_points, args.cell_size, args.max_samples) <= 0:
        parser.error("Size and point limits must be positive")
    if not 0 < args.min_reliable_fraction <= 1:
        parser.error("--min-reliable-fraction must be in (0, 1]")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = args.source.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    model_dir = Path(colmap_model_dir(source))
    cloud_path = args.point_cloud.expanduser().resolve() if args.point_cloud else model_dir / "points3D.ply"
    if not cloud_path.is_file():
        parser.error(f"Point cloud not found: {cloud_path}; pass --point-cloud with a registered LiDAR PLY")
    selected = json.loads(args.keyframes.read_text())["selected_images"]
    if not selected or len(selected) != len(set(selected)):
        parser.error("Keyframe list is empty or has duplicates")
    extrinsics, intrinsics = load_colmap(model_dir)
    images = {image.name: image for image in extrinsics.values()}
    missing = set(selected) - set(images)
    if missing:
        parser.error(f"{len(missing)} keyframes absent from COLMAP: {sorted(missing)[:5]}")
    preparation = {"source": str(source), "point_cloud": str(cloud_path),
                   "point_cloud_size": cloud_path.stat().st_size,
                   "point_cloud_mtime_ns": cloud_path.stat().st_mtime_ns,
                   "keyframes_sha256": hashlib.sha256(args.keyframes.read_bytes()).hexdigest(),
                   "depth_model_commit": DA2_COMMIT, "weights_url": WEIGHTS_URL,
                   "input_size": args.input_size, "max_points": args.max_points,
                   "cell_size": args.cell_size, "max_samples": args.max_samples}
    marker = output / "preparation.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != preparation:
            parser.error(f"Output has files from a different preparation: {output}")
        if (output / "depth_params.json").exists():
            parser.error(f"Depth preparation has already finished: {output}")
        print(f"Resuming interrupted preparation in {output}", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(preparation, indent=2, allow_nan=False) + "\n")
    weights = setup_da2(args.cache_dir.expanduser().resolve())
    import cv2
    import torch
    from tqdm import tqdm
    from depth_anything_v2.dpt import DepthAnythingV2
    if not torch.cuda.is_available():
        parser.error(f"GPU {args.gpu} is not available")
    cloud = load_points(cloud_path)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    cloud_count = len(cloud)
    if cloud_count < 50:
        parser.error("Point cloud has fewer than 50 finite points")
    if len(cloud) > args.max_points:
        indices = np.random.default_rng(42).choice(len(cloud), args.max_points, replace=False)
        cloud = cloud[indices]
    print(f"Alignment point cloud: {cloud_path} ({len(cloud):,} of {cloud_count:,} finite points)", flush=True)
    points = torch.from_numpy(cloud).to("cuda")
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    model = model.to("cuda").eval()
    params, records = {}, []
    with torch.inference_mode():
        for name in tqdm(selected, desc="Predicting and aligning depth", unit="image"):
            image_path = source / "images" / name
            rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Cannot read {image_path}")
            camera = images[name]
            intrinsic = intrinsics[camera.camera_id]
            if abs(rgb.shape[1] / rgb.shape[0] - intrinsic.width / intrinsic.height) > 1e-3:
                raise ValueError(f"Image aspect ratio does not match COLMAP for {name}")
            stem = str(Path(name).with_suffix(""))
            path = output / (stem + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
            if depth is None or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
                raw = np.asarray(model.infer_image(rgb, args.input_size), dtype=np.float32)
                if raw.shape != rgb.shape[:2] or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid depth prediction for {name}")
                low, high = float(raw.min()), float(raw.max())
                if high - low <= 1e-8:
                    depth = np.zeros(raw.shape, dtype=np.uint16)
                else:
                    depth = np.rint(np.clip((raw - low) / (high - low), 0, 1) * 65535).astype(np.uint16)
                temp = path.with_name(path.stem + ".partial.png")
                if not cv2.imwrite(str(temp), depth):
                    raise OSError(f"Failed to save {temp}")
                os.replace(temp, path)
            x, y = sample_projected_points(points, camera, intrinsic, depth.shape,
                                           depth, args.cell_size, args.max_samples)
            fit = fit_alignment(x, y)
            params[stem] = {"scale": fit["scale"], "offset": fit["offset"], "reliable": fit["reliable"]}
            records.append({"image": name, **fit})
    fraction = sum(record["reliable"] for record in records) / len(records)
    summary = {"source": str(source), "point_cloud": str(cloud_path), "points_total": cloud_count,
               "points_used": len(cloud), "keyframes": str(args.keyframes.resolve()),
               "keyframes_sha256": hashlib.sha256(args.keyframes.read_bytes()).hexdigest(),
               "depth_model": "Depth Anything V2 Large (relative)", "depth_model_commit": DA2_COMMIT,
               "weights": str(weights), "input_size": args.input_size, "cell_size_pixels": args.cell_size,
               "visibility": "nearest projected point per coarse image cell; approximate, not true occlusion",
               "frame_count": len(records), "reliable_count": sum(r["reliable"] for r in records),
               "reliable_fraction": fraction, "frames": records}
    (output / "depth_params.json").write_text(json.dumps(params, indent=2, allow_nan=False) + "\n")
    (output / "alignment_report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Reliable depth alignments: {summary['reliable_count']}/{len(records)}; report: {output / 'alignment_report.json'}")
    if fraction < args.min_reliable_fraction:
        raise RuntimeError(f"Only {fraction:.1%} frames passed alignment checks; inspect the report before training")


if __name__ == "__main__":
    main()
