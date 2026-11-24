"""Transfer repeatable training-photo detail onto fixed-pose Orange4F renders."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

from scripts.orange4f_stock_triangle import OUTPUT, frame_output_names, read_colmap_scene
from utils import read_write_model as colmap


DEFAULT_RUN = OUTPUT / "triangle_splatting_stock_full_stride3_w1600_30000_da3_planes"


@dataclass
class Camera:
    width: int
    height: int
    k: np.ndarray
    rotation: np.ndarray  # World to camera
    center: np.ndarray


def camera(row, frame):
    rotation = colmap.qvec2rotmat(frame.qvec / np.linalg.norm(frame.qvec))
    w, h = row["width"], row["height"]
    fx, fy = row["intrinsics"][:2]
    # Match the stock renderer and the DA3 cache's integer pixel-center convention.
    k = np.array([[fx, 0, (w - 1) / 2], [0, fy, (h - 1) / 2], [0, 0, 1]])
    return Camera(w, h, k, rotation, -rotation.T @ frame.tvec)


def lowpass(image):
    return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)


def sample(array, y, x):
    return map_coordinates(array, [y, x], order=1, mode="nearest", prefilter=False)


def warp_source(src, dst, rgb, depth, valid, depth_k):
    """Forward z-buffer from one source, then fetch subpixel source texture at each hit."""
    h, w = dst.height, dst.width
    if rgb.shape != (src.height, src.width, 3) or depth.shape != valid.shape:
        raise ValueError("Source RGB or depth dimensions disagree with calibration")
    ys, xs = np.indices(rgb.shape[:2], dtype=np.float32)
    rx = (xs - src.k[0, 2]) / src.k[0, 0]
    ry = (ys - src.k[1, 2]) / src.k[1, 1]
    du = rx * depth_k[0, 0] + depth_k[0, 2]
    dv = ry * depth_k[1, 1] + depth_k[1, 2]
    inside = (du >= 0) & (du <= depth.shape[1]-1) & (dv >= 0) & (dv <= depth.shape[0]-1)
    u = np.rint(du).astype(np.intp).clip(0, depth.shape[1]-1)
    v = np.rint(dv).astype(np.intp).clip(0, depth.shape[0]-1)
    z = depth[v, u]
    ok = inside & np.isfinite(z) & (z > 0)
    trusted = ok & valid[v, u]
    xyz = np.stack([rx[ok], ry[ok], np.ones(ok.sum(), np.float32)], axis=1) * z[ok, None]
    rotation = (dst.rotation @ src.rotation.T).astype(np.float32)
    offset = (dst.rotation @ (src.center - dst.center)).astype(np.float32)
    xyz = xyz @ rotation.T + offset
    xyz = xyz[xyz[:, 2] > 0]
    px = np.rint(xyz[:, 0] / xyz[:, 2] * dst.k[0, 0] + dst.k[0, 2])
    py = np.rint(xyz[:, 1] / xyz[:, 2] * dst.k[1, 1] + dst.k[1, 2])
    hit = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    front = np.full((h, w), np.inf, np.float32)
    # Even an untrusted foreground depth can veto a farther color sample.
    np.minimum.at(front, (py[hit].astype(np.intp), px[hit].astype(np.intp)), xyz[hit, 2])
    ty, tx = np.where(np.isfinite(front))
    rays = np.stack([(tx-dst.k[0, 2])/dst.k[0, 0], (ty-dst.k[1, 2])/dst.k[1, 1],
                     np.ones(len(tx))], axis=1).astype(np.float32)
    back = (rays * front[ty, tx, None] - offset) @ rotation
    sz = back[:, 2]
    sx = back[:, 0] / np.maximum(sz, 1e-8) * src.k[0, 0] + src.k[0, 2]
    sy = back[:, 1] / np.maximum(sz, 1e-8) * src.k[1, 1] + src.k[1, 2]
    reliable = ((sz > 0) & (sx >= 0) & (sx < src.width-1) & (sy >= 0) & (sy < src.height-1))
    # Require valid source coverage and compatible depth across all four bilinear taps.
    x0 = np.floor(np.clip(sx, 0, src.width-2)).astype(np.intp)
    y0 = np.floor(np.clip(sy, 0, src.height-2)).astype(np.intp)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        reliable &= trusted[y0+dy, x0+dx] & (np.abs(z[y0+dy, x0+dx]-sz) <= .02 * sz)
    ty, tx, sy, sx = ty[reliable], tx[reliable], sy[reliable], sx[reliable]
    color = np.zeros((h, w, 3), np.uint8)
    detail = np.zeros((h, w, 3), np.int16)
    smooth = lowpass(rgb)
    for channel in range(3):
        values = sample(rgb[..., channel].astype(np.float32), sy, sx)
        color[ty, tx, channel] = np.rint(values)
        detail[ty, tx, channel] = np.rint(values - sample(smooth[..., channel], sy, sx))
    visible = np.zeros((h, w), bool)
    visible[ty, tx] = True
    return front, visible, color, detail


def transfer(base, warps):
    if len(warps) < 2:
        return base.copy(), np.zeros(base.shape[:2], bool)
    nearest = np.minimum.reduce([warp[0] for warp in warps])
    smooth = lowpass(base)
    candidate = base.astype(np.float32)
    accepted = np.zeros(nearest.shape, bool)
    # Inputs are ordered by camera proximity. Choose one texture source and require
    # a second photo to agree, so disagreeing fine detail is never averaged together.
    for i, (z, valid, rgb, detail) in enumerate(warps):
        eligible = valid & ~accepted & (z <= nearest * 1.02)
        eligible &= np.abs(rgb.astype(np.float32) - detail - smooth).mean(axis=2) <= 25
        for z2, valid2, rgb2, detail2 in warps[i+1:]:
            pair = valid & valid2 & (z <= nearest * 1.02) & (z2 <= nearest * 1.02)
            if not np.any(eligible & pair & ~accepted):
                continue
            color_error = np.abs(rgb.astype(np.int16) - rgb2).mean(axis=2, dtype=np.float32)
            detail_error = np.abs(detail - detail2).mean(axis=2, dtype=np.float32)
            support = uniform_filter(pair.astype(np.float32), size=5)
            local_error = uniform_filter(np.where(pair, detail_error, 0.), size=5)
            consistent = eligible & ~accepted & pair & (color_error <= 12) & (support >= .65)
            consistent &= (local_error <= 10 * support) & (detail_error <= 15)
            candidate[consistent] = smooth[consistent] + detail[consistent]
            accepted |= consistent
    weight = (accepted * gaussian_filter(accepted.astype(np.float32), 1))[..., None]
    result = np.rint(base + weight * (candidate-base)).clip(0, 255).astype(np.uint8)
    return result, accepted


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def score(actual, prediction):
    mse = np.square((actual.astype(np.float32) - prediction) / 255).mean()
    return float(-10 * np.log10(max(float(mse), 1e-12)))


def detail_rmse(image, target_detail):
    difference = (image - lowpass(image)) - target_detail
    return float(np.sqrt(np.mean(difference**2)))


def inspect_sources(directory, stem, actual, base, sources, warps):
    # The per-source mask tests its own depth/visibility only. Inter-source
    # agreement and similarity to the baseline would hide misregistered edges.
    luminance = actual @ np.array([.299, .587, .114], np.float32)
    gy, gx = np.gradient(luminance)
    edges = np.hypot(gx, gy) >= 16  # RGB on a 0–255 scale, at staged resolution
    reports = []
    for index, (name, (_, visible, color, _)) in enumerate(zip(sources, warps, strict=True), 1):
        panel = f"{stem}_source_{index}_gt_baseline_warp.jpg"
        shown = np.where(visible[..., None], color, np.uint8(128))
        Image.fromarray(np.concatenate([actual, base, shown], axis=1)).save(directory / panel, quality=95)
        mask_file = f"{stem}_source_{index}_valid.png"
        Image.fromarray(visible).save(directory / mask_file)
        report = dict(source=name, comparison=panel, validity_mask=mask_file,
                      valid_fraction=float(visible.mean()), target_edge_pixels=int(edges.sum()))
        for label, mask in (("valid", visible), ("valid_edge", edges & visible)):
            report[label + "_pixels"] = int(mask.sum())
            for method, rgb in (("baseline", base), ("warp", color)):
                report[label + "_" + method + "_psnr"] = score(actual[mask], rgb[mask]) if mask.any() else None
        reports.append(report)
    return reports


def run(args):
    run_dir = args.run.resolve()
    manifest = json.loads((run_dir / "split_manifest.json").read_text())
    rows = {r["name"]: r for r in manifest["frames"]}
    splits = manifest["settings"]
    prior = splits.get("geometry_priors")
    if not prior:
        raise ValueError("Use a DA3-guided run with its matching training-view depth cache")
    cache = Path(prior["cache"])
    if hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest() != prior["cache_sha256"]:
        raise ValueError("DA3 cache differs from the cache used for this run")
    names = frame_output_names(manifest["frames"])
    if args.all_held_out:
        iteration = splits["iterations"]
        inputs = {n: run_dir / "frames" / names[n] for n, r in rows.items() if r["split"] == "test"}
        if not all(p.is_file() for p in inputs.values()):
            raise FileNotFoundError("Full held-out renders are incomplete. Use the default live-preview comparison.")
        cohort = "all_held_out"
    else:
        preview = run_dir / "model/debug_previews"
        latest = json.loads((preview / "latest.json").read_text())
        iteration = latest["iteration"]
        inputs = {r["name"]: preview / r["render"] for r in latest["frames"] if r["split"] == "held-out"}
        cohort = "preview"
    if not inputs or any(rows[name]["split"] != "test" for name in inputs):
        raise ValueError("Expected held-out target cameras from the matching split")
    _, poses = read_colmap_scene(Path(splits["colmap_model"]))
    cameras = {f.name: camera(rows[f.name], f) for f in poses.values() if f.name in rows}
    train = [name for name, row in rows.items() if row["split"] == "train"]
    inspect = getattr(args, "inspect_warps", False)
    method = "photo_warps" if inspect else "photo_detail"
    output = run_dir / f"{method}_{cohort}_{iteration}"
    if (output / "results.json").is_file():
        print(f"Existing comparison: {output / 'results.json'}")
        return
    with TemporaryDirectory(dir=run_dir) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        results = []
        for name, baseline_path in inputs.items():
            target = cameras[name]
            qualified = [s for s in train if target.rotation[2] @ cameras[s].rotation[2] > .85]
            sources = sorted(qualified, key=lambda s: np.linalg.norm(cameras[s].center-target.center))[:4]
            warps = []
            for source in sources:
                staged = rows[source]["staged_name"]
                rgb = read_rgb(run_dir / "images" / staged)
                with np.load(cache / "frames" / (Path(staged).stem + ".npz"), allow_pickle=False) as data:
                    warps.append(warp_source(cameras[source], target, rgb, data["depth"], data["valid"],
                                             data["intrinsic"]))
            base = read_rgb(baseline_path)
            if base.shape != (target.height, target.width, 3):
                raise ValueError(f"Baseline RGB size differs from target camera for {name}")
            if not inspect:
                photo, supported = transfer(base, warps)
            # The target photograph first enters here, after reprojection and synthesis.
            actual = read_rgb(run_dir / "images" / rows[name]["staged_name"])
            if actual.shape != base.shape:
                raise ValueError(f"Reference RGB size differs from prediction for {name}")
            stem = Path(names[name]).stem
            if inspect:
                measurements = inspect_sources(stage, stem, actual, base, sources, warps)
                results.append(dict(frame=name, warps=measurements))
                for measurement in measurements:
                    print(name, json.dumps(measurement), flush=True)
                continue
            panel = stem + "_gt_baseline_photo.jpg"
            Image.fromarray(np.concatenate([actual, base, photo], axis=1)).save(stage / panel, quality=95)
            Image.fromarray(photo).save(stage / (stem + "_render.png"))
            Image.fromarray(supported).save(stage / (stem + "_support.png"))
            target_detail = actual - lowpass(actual)
            row = dict(frame=name, sources=sources, baseline_psnr=score(actual, base),
                       baseline_detail_rmse=detail_rmse(base, target_detail),
                       photo_detail_rmse=detail_rmse(photo, target_detail),
                       photo_psnr=score(actual, photo), coverage=float(supported.mean()), comparison=panel)
            results.append(row)
            print(json.dumps(row), flush=True)
        report = dict(iteration=iteration, cohort=cohort, frames=results)
        if not inspect:
            report.update(baseline_psnr=float(np.mean([r["baseline_psnr"] for r in results])),
                          photo_psnr=float(np.mean([r["photo_psnr"] for r in results])))
        (stage / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)
    if not inspect:
        print(f"Mean per-view PSNR {report['baseline_psnr']:.3f} -> {report['photo_psnr']:.3f} dB")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect-warps", action="store_true",
                      help="Show each training-photo warp on the latest held-out previews before detail transfer")
    mode.add_argument("--all-held-out", action="store_true",
                        help="Compare all held-out frames after the run finishes rendering")
    run(parser.parse_args())
