"""Prepare fixed-camera DA3 depth and planar patches from Orange4F train views."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.orange4f_stock_triangle import read_colmap_scene
from scripts.prepare_da3_occlusion import save_npz, windows
from scripts.view_comparisons import atomic_json
from utils import read_write_model as colmap
from utils.da3_occlusion import cross_view_support, reliable_pixels


MODEL = "depth-anything/DA3NESTED-GIANT-LARGE"


def camera(row, frame):
    fx, fy, _, _ = row["intrinsics"]
    width, height = row["width"], row["height"]
    k = np.array([[fx, 0, (width - 1) / 2], [0, fy, (height - 1) / 2], [0, 0, 1]], np.float32)
    e = np.eye(4, dtype=np.float32)
    e[:3, :3] = colmap.qvec2rotmat(frame.qvec / np.linalg.norm(frame.qvec))
    e[:3, 3] = frame.tvec
    return k, e


def track_points(sparse, train_frames):
    if not any((f.point3D_ids >= 0).any() for f in train_frames):
        return {}
    for extension, reader in (("bin", colmap.read_points3D_binary), ("txt", colmap.read_points3D_text)):
        path = sparse / f"points3D.{extension}"
        if path.is_file():
            ids = {f.id for f in train_frames}
            return {pid: p.xyz for pid, p in reader(path).items()
                    if len(ids.intersection(p.image_ids)) >= 2}
    return {}


def track_ratios(frame, cam, cached_k, extrinsic, depth, valid, points):
    indices = [i for i, pid in enumerate(frame.point3D_ids) if pid in points]
    if not indices:
        return np.empty(0), np.empty(0, dtype=bool)
    xyz = np.stack([points[frame.point3D_ids[i]] for i in indices])
    xyz = xyz @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = xyz[:, 2]
    xy = xyz[:, :2] / np.maximum(z[:, None], 1e-6)
    if cam.model == "SIMPLE_PINHOLE":
        fx, cx, cy = cam.params
        fy = fx
    else:
        fx, fy, cx, cy = cam.params
    reprojection = xy * [fx, fy] + [cx, cy]
    good = (z > 0) & (np.linalg.norm(reprojection - frame.xys[indices], axis=1) <= 3)
    uv = np.rint(xy * cached_k.diagonal()[:2] + cached_k[:2, 2]).astype(int)
    u, v = uv.T
    height, width = depth.shape
    good &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx = np.flatnonzero(good)
    idx = idx[valid[v[idx], u[idx]]]
    ratios = np.log(z[idx] / depth[v[idx], u[idx]])
    return ratios, frame.point3D_ids[np.asarray(indices)[idx]] % 4 == 0


def calibrate_scales(track_matches):
    ratios = np.concatenate([values for values, _ in track_matches])
    holdout = np.concatenate([mask for _, mask in track_matches])
    if (~holdout).sum() < 32 or holdout.sum() < 10:
        return 1., {"scale_source": "pose_alignment", "anchors": len(ratios), "accepted": True}
    scale = float(np.median(ratios[~holdout]))
    error = np.abs(np.expm1(scale - ratios[holdout]))
    median = float(np.median(error))
    inlier_fraction = float(np.mean(error <= .10))
    return float(np.exp(scale)), {"scale_source": "sfm", "anchors": len(ratios),
                                  "heldout_median_relative_error": median,
                                  "heldout_inlier_fraction": inlier_fraction,
                                  "accepted": median <= .05 and inlier_fraction >= .70}


def fit_planar_tiles(depth, trusted, k, tile_size=48):
    """Fit camera-space n·X=b to trusted pixels in locally planar regions."""
    height, width = depth.shape
    ids = np.full((height, width), -1, np.int16)
    planes = []
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            tile = np.s_[top:top+tile_size, left:left+tile_size]
            valid = trusted[tile]
            yy, xx = np.where(valid)
            if len(xx) < max(24, .24 * valid.size) or min(np.ptp(xx), np.ptp(yy)) < 8:
                continue
            rays = np.stack(((xx + left - k[0, 2]) / k[0, 0],
                             (yy + top - k[1, 2]) / k[1, 1], np.ones(len(xx))), axis=1)
            z = depth[tile][valid]
            coeff, _, rank, _ = np.linalg.lstsq(rays, 1 / z, rcond=None)
            inliers = np.abs((rays @ coeff) * z - 1) <= .02
            if rank != 3 or inliers.mean() < .90:
                continue
            coeff, _, rank, _ = np.linalg.lstsq(rays[inliers], 1 / z[inliers], rcond=None)
            scale = np.linalg.norm(coeff)
            if rank == 3 and scale > 0:
                ids[tile][yy[inliers], xx[inliers]] = len(planes)
                planes.append([*(coeff / scale), 1. / scale])
    return ids, np.asarray(planes, np.float32).reshape(-1, 4)


def prepare(run, output, model_name, process_res, device):
    from depth_anything_3.api import DepthAnything3

    split = (run / "split_manifest.json").read_bytes()
    manifest = json.loads(split)
    rows = {r["name"]: r for r in manifest["frames"] if r["split"] == "train"}
    sparse = Path(manifest["settings"]["colmap_model"])
    cameras, all_frames = read_colmap_scene(sparse)
    frames = {f.name: f for f in all_frames.values() if f.name in rows}
    points = track_points(sparse, list(frames.values()))
    settings = dict(split_sha256=hashlib.sha256(split).hexdigest(), model=model_name,
                    process_res=process_res, core=8, context=3, train_views=len(rows), version=1)
    config = output / "config.json"
    if config.is_file() and json.loads(config.read_text()) != settings:
        raise ValueError(f"Different prior settings already stored at {config}")
    atomic_json(config, settings)
    if (output / "manifest.json").is_file():
        print(f"Reusing priors in {output}", flush=True)
        return

    model = DepthAnything3.from_pretrained(model_name).to(device).eval()
    reports = {}
    for batch, core in windows(list(rows), core=settings["core"], context=settings["context"]):
        paths = {n: output / "frames" / (Path(rows[n]["staged_name"]).stem + ".npz") for n in core}
        if all(p.is_file() for p in paths.values()):
            for name, path in paths.items():
                with np.load(path, allow_pickle=False) as data:
                    reports[name] = json.loads(str(data["report"]))
            continue
        k, e = zip(*(camera(rows[n], frames[n]) for n in batch))
        images = []
        for name in batch:
            with Image.open(run / "images" / rows[name]["staged_name"]) as rgb:
                images.append(np.asarray(rgb.convert("RGB")))
        if len({im.shape for im in images}) != 1:
            raise ValueError("DA3 windows need a common image size to retain full camera coverage")
        result = model.inference(images, intrinsics=np.stack(k), extrinsics=np.stack(e),
                                 align_to_input_ext_scale=True, process_res=process_res,
                                 process_res_method="upper_bound_resize")
        depth = result.depth.astype(np.float32)
        predicted_k = result.intrinsics
        valid = [reliable_pixels(z, confidence, percentile=45.) for z, confidence in zip(depth, result.conf)]
        ratios = [track_ratios(frames[n], cameras[frames[n].camera_id], pk, pose, z, v, points)
                  for n, pk, pose, z, v in zip(batch, predicted_k, e, depth, valid)]
        scale, calibration = calibrate_scales(ratios)
        depth = np.where(np.isfinite(depth) & (depth > 0), scale * depth, 0.)
        views = list(zip(depth, valid, predicted_k, e))
        for i, name in enumerate(batch):
            if name not in paths:
                continue
            z, mask, pk, pose = views[i]
            neighbors = views[max(0, i-4):i] + views[i+1:i+5]
            trusted = mask & (cross_view_support(z, mask, pk, pose, neighbors, tolerance=.05) >= 2)
            if not calibration["accepted"]:
                trusted.fill(False)
            plane_id, planes = fit_planar_tiles(z, trusted, pk)
            report = dict(**calibration, scale_correction=scale, trusted_fraction=float(trusted.mean()),
                          planar_fraction=float((plane_id >= 0).mean()), planes=len(planes))
            save_npz(paths[name], depth=z, valid=trusted, intrinsic=pk.astype(np.float32),
                     plane_id=plane_id, planes=planes, report=json.dumps(report))
            reports[name] = report
        print(f"Prepared {len(reports):,}/{len(rows):,} train views", flush=True)
    if not any(report["trusted_fraction"] > 0 for report in reports.values()):
        raise ValueError("No DA3 depths survived validation")
    atomic_json(output / "manifest.json", dict(**settings, frames=reports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Staged baseline run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.run.resolve(), args.out.resolve(), args.model, args.process_res, args.device)
