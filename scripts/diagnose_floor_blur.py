#!/usr/bin/env python3
"""Measure cross-view alignment in a floor ROI before fitting or rendering 3DGS."""
import argparse
from dataclasses import dataclass
import html
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.keyframe_pipeline.common import atomic_json, image_path, project, read_model


@dataclass
class FloorPatch:
    source: Path
    image_dir: str
    roi: list[int]
    rgb: np.ndarray
    gray: np.ndarray
    features: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    inliers: np.ndarray
    search: int

    def projected_inliers(self, reference, camera):
        uv, depth = project(self.inliers, reference.R, reference.C, camera.params)
        return uv[depth > .05] - (np.array(self.roi[:2]) + .5)


def load_rgb(path, camera):
    with Image.open(path) as raw:
        if raw.size != (camera.width, camera.height):
            raise ValueError(f"Cannot read {path} at its COLMAP resolution")
        return cv2.cvtColor(np.asarray(raw.convert("RGB")), cv2.COLOR_RGB2BGR)


def floor_points(path, reference, camera, roi, depth):
    vertices = PlyData.read(path, mmap="r")["vertex"].data
    if "scale_0" in vertices.dtype.names or "f_dc_0" in vertices.dtype.names:
        raise ValueError("Use the raw LiDAR PLY for plane fitting")
    x, y, w, h = roi
    result = []
    for start in range(0, len(vertices), 262144):
        chunk = vertices[start:start + 262144]
        points = np.column_stack([chunk[axis] for axis in "xyz"])
        uv, z = project(points, reference.R, reference.C, camera.params)
        uv -= .5
        use = (z >= depth[0]) & (z <= depth[1])
        use &= (uv[:, 0] >= x) & (uv[:, 0] < x + w)
        use &= (uv[:, 1] >= y) & (uv[:, 1] < y + h)
        result.append(points[use])
    return np.concatenate(result).astype(np.float64) if result else np.empty((0, 3))


def fit_plane(points, tolerance, up=None):
    if len(points) < 30:
        raise ValueError("Fewer than 30 LiDAR returns project into the floor ROI")
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(4096, len(points)), replace=False)]
    best = None
    count = 0
    for _ in range(512):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        if up is not None and abs(normal @ up) < math.cos(math.radians(20)):
            continue
        keep = np.abs((sample - a) @ normal) <= tolerance
        if keep.sum() > count:
            count, best = int(keep.sum()), (a, normal)
    if best is None or count < 30:
        raise ValueError("No supported floor plane found. Check ROI, --depth and --up.")
    center, normal = best
    for _ in range(3):
        inliers = points[np.abs((points - center) @ normal) <= tolerance]
        if len(inliers) < 30:
            raise ValueError("Insufficient LiDAR support after refitting the floor plane")
        center = inliers.mean(0)
        centered = inliers - center
        _, axes = np.linalg.eigh(centered.T @ centered)
        normal = axes[:, 0]
    if up is not None and abs(normal @ up) < math.cos(math.radians(20)):
        raise ValueError("Refitted plane disagrees with --up")
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    inliers = points[np.abs((points - center) @ normal) <= tolerance]
    if len(inliers) < 30:
        raise ValueError("Insufficient LiDAR support for the final floor plane")
    return center, normal, inliers


def plane_grid(reference, camera, roi, center, normal):
    x, y, w, h = roi
    yy, xx = np.mgrid[y:y + h, x:x + w]
    k = camera.params
    rays = np.stack(((xx + .5 - k[2]) / k[0], (yy + .5 - k[3]) / k[1],
                     np.ones_like(xx)), axis=-1) @ reference.R
    denominator = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = ((center - reference.C) @ normal) / denominator
    if not np.isfinite(depth).all() or depth.min() <= .05:
        raise ValueError("The plane is behind the camera or crosses the horizon in the ROI")
    return reference.C + rays * depth[..., None]


def target_grid(world, view, camera):
    uv, z = project(world.reshape(-1, 3), view.R, view.C, camera.params)
    xy = uv.reshape(*world.shape[:2], 2) - .5
    limit = np.array([camera.width - 1, camera.height - 1])
    valid = ((xy >= 0) & (xy <= limit)).all(axis=-1) & (z.reshape(xy.shape[:2]) > .05)
    return xy, valid


def local_template_matches(reference, warped, valid, features, search, radius=7):
    """Find subpixel displacements in reference ROI pixels using local ZNCC."""
    matches = []
    border = radius + search
    for x, y in features:
        ty, tx = slice(y - radius, y + radius + 1), slice(x - radius, x + radius + 1)
        sy, sx = slice(y - border, y + border + 1), slice(x - border, x + border + 1)
        if not valid[sy, sx].all() or reference[ty, tx].std() < 3:
            continue
        scores = cv2.matchTemplate(warped[sy, sx], reference[ty, tx], cv2.TM_CCOEFF_NORMED)
        iy, ix = np.unravel_index(scores.argmax(), scores.shape)
        peak = float(scores[iy, ix])
        if peak < .7 or min(ix, iy) == 0 or max(ix, iy) == 2 * search:
            continue
        alternative = scores.copy()
        alternative[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = -1
        if peak - alternative.max() < .035:
            continue
        neighbors = ((scores[iy, ix - 1], scores[iy, ix + 1]),
                     (scores[iy - 1, ix], scores[iy + 1, ix]))
        subpixel = [float(.5 * (low - high) / min(low - 2 * peak + high, -1e-5))
                    for low, high in neighbors]
        matches.append([int(x), int(y), ix - search + subpixel[0],
                        iy - search + subpixel[1], peak])
    return np.array(matches).reshape(-1, 5)


def roi_features(gray, search):
    border = search + 7
    mask = np.zeros(gray.shape, np.uint8)
    mask[border:-border, border:-border] = 255
    corners = cv2.goodFeaturesToTrack(gray, 240, .01, 10, mask=mask)
    if corners is None or len(corners) < 5:
        raise ValueError("Choose a sharp ROI containing multiple distinctly textured points")
    return np.rint(corners.reshape(-1, 2)).astype(int).clip(
        [border, border], [gray.shape[1] - border - 1, gray.shape[0] - border - 1])


def salience(reference, other, mask):
    """Reference-gradient response and agreement at the same salient pixels."""
    if not mask.any():
        return dict(edge_strength=None, gradient_agreement=None)

    def gradient(image):
        pixels = image.astype(np.float32)
        return np.stack([cv2.Sobel(pixels, cv2.CV_32F, x, y, ksize=3)
                         for x, y in ((1, 0), (0, 1))], axis=-1)
    base, test = gradient(reference), gradient(other)
    strength = np.linalg.norm(base, axis=-1)
    strong = mask & (strength >= max(5, np.percentile(strength[mask], 80)))
    a, b = base[strong].ravel(), test[strong].ravel()
    power_a, power_b = np.linalg.norm(a), np.linalg.norm(b)
    if not power_a:
        return dict(edge_strength=None, gradient_agreement=None)
    return dict(edge_strength=float(power_b / power_a),
                gradient_agreement=float(np.clip((a @ b) / (power_a * power_b), -1, 1)) if power_b else 0.)


def save_image(output, filename, pixels):
    if not cv2.imwrite(str(output / filename), pixels):
        raise OSError(f"Could not write {output / filename}")
    return filename


def measure_view(world, gray, features, source, image_dir, view, camera, search, output, index):
    xy, valid = target_grid(world, view, camera)
    row = dict(name=view.name, coverage=float(valid.mean()), status="insufficient ROI coverage")
    if valid.mean() < .9:
        return row, None
    # Singular values describe source pixels per native reference pixel.
    cy, cx = np.array(valid.shape) // 2
    jacobian = np.column_stack([(xy[cy, cx + 1] - xy[cy, cx - 1]) / 2,
                                (xy[cy + 1, cx] - xy[cy - 1, cx]) / 2])
    row["sampling_source_px_per_reference_px"] = np.linalg.svd(jacobian, compute_uv=False).tolist()
    target = load_rgb(image_path(source, image_dir, view.name), camera)
    warp = cv2.remap(target, xy.astype(np.float32), None, cv2.INTER_LINEAR)
    gray_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    matches = local_template_matches(gray, gray_warp, valid, features, search)
    overlay = cv2.merge((gray_warp, gray, gray_warp))
    overlay[~valid] = 0
    for px, py, dx, dy, _ in matches:
        cv2.circle(overlay, (int(px), int(py)), 2, (0, 255, 255), -1)
        cv2.arrowedLine(overlay, (int(px), int(py)),
                        (round(px + 3 * dx), round(py + 3 * dy)), (0, 165, 255), 1)
    row.update(matches=len(matches), warp=save_image(output, f"view_{index:03d}.png", warp),
               overlay=save_image(output, f"overlay_{index:03d}.png", overlay),
               feature_displacements=matches.tolist(), status="weak or ambiguous matches")
    if len(matches) < max(5, math.ceil(.15 * len(features))):
        return row, None

    shift = np.median(matches[:, 2:4], axis=0)
    distance = np.linalg.norm(matches[:, 2:4] - shift, axis=1)
    shift_norm = np.linalg.norm(matches[:, 2:4], axis=1)
    # Require two-dimensional support before treating a translation as a patch correction.
    occupied = np.unique((2 * matches[:, :2] / np.array(gray.shape[::-1])).astype(int), axis=0)
    row.update(status="measured", matched_quadrants=len(occupied), shift_xy_px=shift.tolist(),
               median_displacement_px=float(np.median(shift_norm)), p90_displacement_px=float(np.percentile(shift_norm, 90)),
               p90_after_translation_px=float(np.percentile(distance, 90)),
               median_zncc=float(np.median(matches[:, 4])))
    if len(occupied) < 3 or row["p90_after_translation_px"] > 2:
        row["status"] = "spatially uneven matches"
        return row, None

    # Shift samples back into reference coordinates, for a paired phantom-blur check.
    h, w = gray.shape
    transform = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32)
    aligned = cv2.warpAffine(gray_warp, transform, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    aligned_valid = cv2.warpAffine(valid.astype(np.float32), transform, (w, h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) > .999
    row["aligned"] = save_image(output, f"aligned_{index:03d}.png", aligned)
    common = valid & aligned_valid
    interior = cv2.erode(common.astype(np.uint8), np.ones((5, 5), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    row["aligned_gradients"] = salience(gray, aligned, interior)
    return row, (gray_warp, aligned, common)


def composite(rows, gray, output, key):
    if len(rows) < 2:
        return dict(views=len(rows))
    mask = cv2.erode(np.logical_and.reduce([v[2] for v in rows]).astype(np.uint8),
                     np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if mask.mean() < .65:
        return dict(views=len(rows), common_coverage=float(mask.mean()))
    before, after = [], []
    # The same gain/offset in both averages isolates the effect of translation.
    target = gray[mask].astype(np.float32)
    for raw, aligned, _ in rows:
        samples = aligned[mask].astype(np.float32)
        gain = target.std() / max(samples.std(), 1.)
        offset = target.mean() - gain * samples.mean()
        before.append(raw * gain + offset)
        after.append(aligned * gain + offset)
    result = dict(views=len(rows), common_coverage=float(mask.mean()))
    for label, stack in (("uncorrected", before), ("corrected", after)):
        mean = np.mean(stack, axis=0).clip(0, 255).astype(np.uint8)
        result[label] = salience(gray, mean, mask)
        mean[~mask] = 0
        result[label]["image"] = save_image(output, f"{key}_{label}.png", mean)
    return result


def choose_views(images, cameras, reference, world, groups, limit):
    """Rank for native-pixel support while retaining views across the trajectory."""
    chosen = []
    grid = world[::max(1, world.shape[0] // 12), ::max(1, world.shape[1] // 12)]
    for group, names in groups.items():
        ranked = []
        for i, view in enumerate(images):
            if view.name not in names or view.name == reference.name:
                continue
            xy, valid = target_grid(grid, view, cameras[view.camera_id])
            if valid.mean() >= .9:
                dx = np.diff(xy, axis=1) / max(1, world.shape[1] // 12)
                dy = np.diff(xy, axis=0) / max(1, world.shape[0] // 12)
                score = np.median(np.minimum(np.linalg.norm(dx[:-1], axis=-1),
                                             np.linalg.norm(dy[:, :-1], axis=-1)))
                ranked.append((i, float(score), view))
        bins = np.array_split(np.arange(len(ranked)), min(limit, len(ranked))) if ranked else []
        chosen.extend((group, ranked[max(indices, key=lambda j: ranked[j][1])][2]) for indices in bins)
    return chosen


def principal_point_offset(camera):
    return (camera.params[2:] - np.array([camera.width, camera.height]) / 2).tolist()


def render_models(args, reference, reference_camera, patch, output):
    from scripts.view_comparisons import model_info, shared_iteration

    models = [model_info(path) for path in args.models]
    if any(spec["type"] not in {"3dgs", "scaffold"} for spec in models):
        raise ValueError("--models supports native 3DGS and Scaffold-GS checkpoints")
    iteration = shared_iteration(models, args.iteration)
    expected = dict(R=reference.R.T, T=-reference.R @ reference.C,
                    width=reference_camera.width, height=reference_camera.height)
    expected.update(zip(("fx", "fy", "cx", "cy"), reference_camera.params))
    for spec in models:
        record = spec["records"].get(reference.name)
        if record is None or any(not np.allclose(record[k], value, rtol=0, atol=1e-7)
                                 for k, value in expected.items()):
            raise ValueError(f"{spec['path']} lacks the identical reference camera")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from scripts.render_views_worker import render_views
    import torch

    x, y, w, h = patch.roi
    gray = patch.gray
    interior = np.zeros(gray.shape, bool)
    interior[3:-3, 3:-3] = True
    results = []
    for index, spec in enumerate(models):
        spec["cfg"] = {**spec["cfg"], "resolution": 1, "loss_mask_dir": ""}
        destination = output / f"render_{index}"
        render_views(spec, [reference.name], iteration, destination, args.antialiasing)
        frame = json.loads((destination / "frames.json").read_text())[reference.name]
        gt = load_rgb(destination / frame["gt"], reference_camera)
        if not np.array_equal(gt[y:y + h, x:x + w], patch.rgb):
            raise ValueError(f"{spec['path']} renders against a different reference RGB")
        rgb = load_rgb(destination / frame["image"], reference_camera)[y:y + h, x:x + w]
        pred = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        train = spec["splits"]["train"]
        error = np.mean((pred.astype(float) - gray) ** 2)
        results.append(dict(label=spec["path"].name, model=str(spec["path"]), iteration=iteration,
                            reference_in_train=reference.name in train, train_images=len(train),
                            renderer_principal_point="image center",
                            colmap_principal_point_offset_px=principal_point_offset(reference_camera),
                            antialiasing=spec["cfg"].get("antialiasing", True) if args.antialiasing is None else args.antialiasing,
                            image=save_image(output, f"render_{index}_crop.png", rgb),
                            luma_psnr_db=float(10 * np.log10(255**2 / error)) if error else "inf",
                            **salience(gray, pred, interior)))
        torch.cuda.empty_cache()
    return results


def write_report(output, report):
    def card(label, images):
        panels = "".join(f'<div><span>{html.escape(key)}</span><img loading="lazy" src="{value}"></div>'
                         for key, value in images.items())
        return f"<section><h2>{html.escape(label)}</h2><div class='row'>{panels}</div></section>"

    sections = [card(report["reference"], {"Reference crop": "reference.png", "LiDAR plane support": "plane.png"})]
    if "pose_comparison" in report:
        count = len(report["pose_comparison"])
        sections.append(f'<p><a href="alternate/index.html">Alternate COLMAP poses</a> · '
                        f'{count} paired measurements in <a href="report.json">report.json</a></p>')
    for group, result in report["composites"].items():
        if "corrected" in result:
            sections.append(card(f"{group} · {result['views'] - 1} source views plus reference", {
                f"{key} · gradient agreement {result[key]['gradient_agreement']:.2f}":
                result[key]["image"] for key in ("uncorrected", "corrected")
                if result[key]["gradient_agreement"] is not None}))
    for row in report["renders"]:
        membership = "training camera" if row["reference_in_train"] else "held-out camera"
        agreement = row["gradient_agreement"]
        label = "Render crop" if agreement is None else f"Render crop · gradient agreement {agreement:.2f}"
        sections.append(card(f"{row['label']} · {row['iteration']:,} steps · {membership}", {label: row["image"]}))
    for row in sorted(report["views"], key=lambda r: (r["group"], r["name"])):
        detail = f"{row['group']} · {row['name']} · {row['status']}"
        if "median_displacement_px" in row:
            detail += f" · median {row['median_displacement_px']:.2f} px · P90 {row['p90_displacement_px']:.2f} px"
        images = {label: row[key] for label, key in
                  (("Plane warp", "warp"), ("Green reference and magenta warp", "overlay"),
                   ("Translation corrected", "aligned")) if key in row}
        sections.append(card(detail, images))
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Floor alignment diagnostic</title>'
        '<style>body{background:#17202a;color:#eee;font:16px system-ui;margin:24px}'
        'a{color:#9ed4ff}img{display:block;max-width:100%;height:auto}.row{display:flex;gap:16px;flex-wrap:wrap}</style>'
        '<h1>Floor alignment diagnostic</h1><p>Displacements are in native reference pixels. '
        'Arrows show local displacements at 3× scale. Green dots indicate plane inliers. '
        'Inspect the floor crop and warp for occlusion, reflections, repeated patterns, and insufficient detail. '
        'Gradient agreement compares edge direction and location at strong reference gradients. '
        'An improvement after correction shows how much alignment can affect this source-view average. '
        'Disagreement can come from camera poses, calibration, or the LiDAR plane. '
        'Source warps use the recorded COLMAP principal points. Check their offsets in the metrics '
        'when interpreting renders from the centered camera used by native 3DGS and Scaffold.</p>'
        '<p><a href="report.json">Metrics and settings</a></p>' + "".join(sections))


def audit_views(output, patch, images, cameras, reference, chosen, groups, full_names, regional_train):
    world = plane_grid(reference, cameras[reference.camera_id], patch.roi, patch.center, patch.normal)
    save_image(output, "reference.png", patch.rgb)
    support = patch.rgb.copy()
    local = patch.projected_inliers(reference, cameras[reference.camera_id])
    w, h = patch.roi[2:]
    local = local[((local >= 0) & (local < [w, h])).all(axis=1)]
    for px, py in local[::max(1, len(local) // 3000)]:
        cv2.circle(support, (round(px), round(py)), 1, (0, 255, 0), -1)
    save_image(output, "plane.png", support)
    gray = patch.gray
    views, pairs = [], {label: [(gray, gray, np.ones(gray.shape, bool))] for label in groups}
    poses = {im.name: i for i, im in enumerate(images)}
    for index, (group, view) in enumerate(chosen):
        row, pair = measure_view(world, gray, patch.features, patch.source, patch.image_dir, view,
                                 cameras[view.camera_id], patch.search, output, index)
        row.update(group=group, colmap_principal_point_offset_px=principal_point_offset(cameras[view.camera_id]),
                   frame_index_delta=poses[view.name] - poses[reference.name],
                   reference_baseline=float(np.linalg.norm(view.C - reference.C)),
                   full_training_frame=view.name in full_names if full_names else None,
                   regional_training_frame=view.name in regional_train if regional_train is not None else None)
        views.append(row)
        if pair is not None:
            pairs[group].append(pair)
        print(f"{group} | {view.name} | {row['status']}", flush=True)
    return dict(reference=reference.name, features=len(patch.features), search_radius_px=patch.search,
                reference_principal_point_offset_px=principal_point_offset(cameras[reference.camera_id]),
                group_candidates={label: len(names - {reference.name}) for label, names in groups.items()},
                reference_in_full_train=reference.name in full_names if full_names else None,
                reference_in_regional_train=reference.name in regional_train if regional_train is not None else None,
                views=views, composites={label: composite(values, gray, output, f"group_{index}")
                                         for index, (label, values) in enumerate(pairs.items())}, renders=[])


def alternate_poses(path, source, reference, chosen, cameras):
    _, alt_cameras, alt_images, stamps = read_model(source, model_dir=path)
    lookup = {image.name: image for image in alt_images}
    for image in [reference, *(view for _, view in chosen)]:
        if image.name not in lookup:
            raise ValueError(f"Alternate poses lack {image.name}; choose shared views with --group")
        a, b = cameras[image.camera_id], alt_cameras[lookup[image.name].camera_id]
        if (a.width, a.height) != (b.width, b.height) or not np.allclose(a.params, b.params, atol=1e-7, rtol=0):
            raise ValueError(f"Alternate intrinsics differ for {image.name}; use the same calibration")
    return alt_images, alt_cameras, lookup[reference.name], [(group, lookup[view.name]) for group, view in chosen], stamps


def compare_poses(original, alternate):
    original_rows = {(row["group"], row["name"]): row for row in original}
    comparison = []
    for row in alternate:
        before = original_rows[row["group"], row["name"]]
        a, b = [{(x, y): math.hypot(dx, dy) for x, y, dx, dy, _ in r.get("feature_displacements", [])}
                for r in (before, row)]
        common = sorted(a.keys() & b.keys())
        if len(common) >= 5:
            comparison.append(dict(group=row["group"], name=row["name"], shared_features=len(common),
                                   original_median_px=float(np.median([a[k] for k in common])),
                                   alternate_median_px=float(np.median([b[k] for k in common]))))
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=Path, help="Regional export containing selection.json and raw crop input.ply")
    parser.add_argument("--source", type=Path, help="Undistorted original COLMAP dataset; defaults to the region source")
    parser.add_argument("--point-cloud", type=Path, help="Raw LiDAR PLY; defaults to regional input.ply or source/input.ply")
    parser.add_argument("--keyframes", type=Path, help="Full-run keyframes.json; defaults to the region's original manifest")
    parser.add_argument("--image-dir", help="Source image directory; defaults to the manifest setting or images")
    parser.add_argument("--reference", required=True, help="Exact COLMAP image name of a sharp floor view")
    parser.add_argument("--roi", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"),
                        help="Floor-only rectangle in native reference pixels, including a distinct seam or other texture")
    parser.add_argument("--group", nargs="+", action="append", metavar="LABEL_OR_IMAGE",
                        help="Explicit test group, e.g. --group nearby a.jpg b.jpg --group other_pass c.jpg d.jpg")
    parser.add_argument("--max-views", type=int, default=20, help="Maximum automatically chosen images per group")
    parser.add_argument("--search", type=int, default=16, help="Maximum displacement to search in reference pixels")
    parser.add_argument("--plane-tolerance", type=float, default=.025, help="LiDAR plane inlier tolerance in scene units")
    parser.add_argument("--depth", nargs=2, type=float, default=(.05, float("inf")), help="Candidate range in reference camera depth")
    parser.add_argument("--up", nargs=3, type=float, help="Optional world-up direction for the floor normal")
    parser.add_argument("--alternate-model", type=Path,
                        help="Optional COLMAP sparse model with refined poses, identical intrinsics and the same LiDAR world frame")
    parser.add_argument("--models", nargs="+", type=Path, help="Optional native 3DGS or Scaffold models to render at the exact reference camera")
    parser.add_argument("--iteration", type=int, default=-1, help="Optional shared checkpoint, or latest common checkpoint")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--antialiasing", action=argparse.BooleanOptionalAction, default=None,
                        help="Force the same renderer AA setting; default uses each training setting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.source is None and args.region is None:
        parser.error("Supply --source or --region")
    if (args.max_views < 1 or args.search < 2 or not math.isfinite(args.plane_tolerance)
            or args.plane_tolerance <= 0 or not .0 < args.depth[0] < args.depth[1]):
        parser.error("Invalid view limit, search radius, plane tolerance or depth range")
    up = None if args.up is None else np.asarray(args.up)
    if up is not None:
        if not np.isfinite(up).all() or np.linalg.norm(up) == 0:
            parser.error("--up must be a finite nonzero vector")
        up /= np.linalg.norm(up)
    region = json.loads((args.region / "selection.json").read_text()) if args.region else {}
    source = (args.source or Path(region["source"])).expanduser().resolve()
    cloud = args.point_cloud or (args.region or source) / "input.ply"
    manifest_path = args.keyframes or region.get("original_keyframes")
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else {}
    image_dir = args.image_dir or manifest.get("image_directory", "images")
    _, cameras, images, stamps = read_model(source)
    lookup = {view.name: view for view in images}
    if args.reference not in lookup:
        parser.error("Reference frame is absent from this COLMAP model")
    reference = lookup[args.reference]
    camera = cameras[reference.camera_id]
    x, y, w, h = args.roi
    if min(w, h) < max(64, 2 * (args.search + 8) + 12) or min(x, y) < 0 or x + w > camera.width or y + h > camera.height:
        parser.error("The ROI must be in bounds and large enough for the displacement search")
    rgb = load_rgb(image_path(source, image_dir, reference.name), camera)[y:y + h, x:x + w].copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    features = roi_features(gray, args.search)
    points = floor_points(cloud, reference, camera, args.roi, args.depth)
    center, normal, inliers = fit_plane(points, args.plane_tolerance, up)
    world = plane_grid(reference, camera, args.roi, center, normal)
    patch = FloorPatch(source, image_dir, args.roi, rgb, gray, features, center, normal, inliers, args.search)
    local = patch.projected_inliers(reference, camera)
    cell = np.floor(local / [w, h] * 4).astype(int).clip(0, 3)
    coverage = len(np.unique(cell, axis=0))
    if coverage < 8:
        raise ValueError(f"Plane inliers cover only {coverage}/16 ROI cells. Inspect the ROI or choose a tighter floor patch.")

    full_names = set(manifest.get("selected_images", []))
    regional_train = {row["name"] for row in region["selection"] if row["split"] == "train"} if region else None
    if args.group:
        labels = [group[0] for group in args.group]
        if len(set(labels)) != len(labels) or any(len(group) < 2 for group in args.group):
            parser.error("Each --group needs a unique label and at least one view")
        groups = {group[0]: set(group[1:]) for group in args.group}
        if any(len(group) - 1 != len(groups[group[0]]) for group in args.group):
            parser.error("Duplicate image in an explicit group")
        requested = set().union(*groups.values())
        if requested - lookup.keys() or args.reference in requested:
            parser.error("Explicit views must be registered and exclude the reference")
        chosen = [(label, view) for label, names in groups.items() for view in images if view.name in names]
    else:
        if region and full_names:
            groups = {"Regional training views": regional_train, "Additional full-scene training views": full_names - regional_train}
        else:
            groups = {"Source views": full_names or set(lookup)}
        chosen = choose_views(images, cameras, reference, world, groups, args.max_views)
    if not chosen:
        raise ValueError("No candidate views cover enough of the floor ROI")

    alternate = (alternate_poses(args.alternate_model, source, reference, chosen, cameras)
                 if args.alternate_model else None)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = dict(source=str(source), point_cloud=str(cloud.resolve()), roi=args.roi,
                  keyframes=str(manifest_path) if manifest_path else None,
                  plane=dict(center=center.tolist(), normal=normal.tolist(), candidates=len(points),
                             inliers=len(inliers), covered_roi_cells=coverage, tolerance=args.plane_tolerance,
                             p90_inlier_distance=float(np.percentile(np.abs((inliers - center) @ normal), 90))))
    report = {**common, "colmap_inputs": stamps,
              **audit_views(output, patch, images, cameras, reference, chosen, groups, full_names, regional_train)}
    if alternate:
        alt_images, alt_cameras, alt_reference, alt_chosen, alt_stamps = alternate
        alt_dir = output / "alternate"
        alt_dir.mkdir()
        alt_report = {**common, "colmap_inputs": alt_stamps,
                      **audit_views(alt_dir, patch, alt_images, alt_cameras, alt_reference, alt_chosen,
                                    groups, full_names, regional_train)}
        atomic_json(alt_dir / "report.json", alt_report)
        write_report(alt_dir, alt_report)
        report["pose_comparison"] = compare_poses(report["views"], alt_report["views"])
    atomic_json(output / "report.json", report)
    write_report(output, report)
    if args.models:
        report["renders"] = render_models(args, reference, camera, patch, output)
        atomic_json(output / "report.json", report)
        write_report(output, report)
    print(f"Open {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
