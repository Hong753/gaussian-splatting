"""Compare the real Orange4F floor in 09882.jpg with nearby stride-3 training photos."""
import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_floor_blur import (fit_plane, floor_points, load_rgb, local_template_matches,
                                         plane_grid, roi_features, save_image, target_grid)
from scripts.keyframe_pipeline.common import atomic_json, image_path, project, read_model

REFERENCE = "09882.jpg"
ROI_FRACTIONS = (.62, .82, .95, .95)  # Same foreground crop as pgsr_depth_eval.
SEARCH = 8
CELL = 8
TOLERANCE = .025
WINDOW = 48  # Registered images in natural filename order, on either side.
MAX_VIEWS = 6
REPORT_DIR = "floor_neighbors_09882_v2"


def candidates(images, cams, ref_index, selected, world):
    reference = images[ref_index]
    grid = world[::12, ::12]
    ranked, counts = [], Counter()
    for i in range(max(0, ref_index-WINDOW), min(len(images), ref_index+WINDOW+1)):
        view = images[i]
        if i == ref_index or view.name not in selected:
            continue
        xy, valid = target_grid(grid, view, cams[view.camera_id])
        baseline = float(np.linalg.norm(view.C - reference.C))
        if valid.mean() < .9 or baseline < .01:
            counts["little overlap or baseline under 1 cm"] += 1
            continue
        dx = np.diff(xy, axis=1)[:-1] / 12
        dy = np.diff(xy, axis=0)[:, :-1] / 12
        scales = np.linalg.svd(np.stack((dx, dy), axis=-1), compute_uv=False)
        low, high = float(np.percentile(scales[..., 1], 10)), float(np.percentile(scales[..., 0], 90))
        if low < .7 or high > 1.5:
            counts["different pixel scale"] += 1
            continue
        ranked.append((baseline, view, dict(name=view.name, baseline_m=baseline, frame_delta=i-ref_index,
                                            sampling_range=[low, high])))
    return sorted(ranked, key=lambda item: item[0]), counts


def local_cloud(path, world, centers):
    # The segment from any camera to any point in the patch lies inside this box.
    bounds = np.vstack([world.reshape(-1, 3), centers])
    low, high = bounds.min(axis=0)-.5, bounds.max(axis=0)+.5
    vertices = PlyData.read(path, mmap="r")["vertex"].data
    pieces = []
    for i in range(0, len(vertices), 262144):
        chunk = vertices[i:i+262144]
        xyz = np.column_stack([chunk[k] for k in "xyz"])
        pieces.append(xyz[((xyz >= low) & (xyz <= high)).all(axis=1)])
    return np.concatenate(pieces)


def visible_floor(points, center, normal, view, camera, xy, valid):
    # Use the closest surface evidence in each camera separately. Surface support can
    # extend one 8px cell. A foreground return vetoes the same neighborhood.
    uv, z = project(points, view.R, view.C, camera.params)
    ids = (z > .05) & (uv >= 0).all(axis=1) & (uv < [camera.width, camera.height]).all(axis=1)
    px, py = (uv[ids] / CELL).astype(int).T
    signed = ((points[ids] - center) @ normal) * np.sign((view.C - center) @ normal)
    shape = ((camera.height+CELL-1)//CELL, (camera.width+CELL-1)//CELL)
    support, front = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    near_plane = np.abs(signed) <= TOLERANCE
    support[py[near_plane], px[near_plane]] = 1
    nearer = signed > TOLERANCE
    front[py[nearer], px[nearer]] = 1
    kernel = np.ones((3, 3), np.uint8)
    cells = (cv2.dilate(support, kernel) > 0) & (cv2.dilate(front, kernel) == 0)
    coords = np.zeros(xy.shape, dtype=int)
    coords[valid] = ((xy[valid] + .5) / CELL).astype(int)
    return valid & cells[coords[..., 1], coords[..., 0]]


def texture_strength(gray, mask):
    # Mean absolute high-frequency response on the exact same visible pixels.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap[mask])))


def measure(gray, warped, visible, features):
    border = SEARCH + 7
    valid = cv2.erode(visible.astype(np.uint8), np.ones((2*border+1, 2*border+1), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    eligible = features[valid[features[:, 1], features[:, 0]] > 0]
    matches = local_template_matches(gray, warped, visible, eligible, SEARCH)
    # A forward match must also find the starting pixel when searched in reverse.
    mutual = []
    for x, y, dx, dy, score in matches:
        q = np.rint([x+dx, y+dy]).astype(int)
        if not (border <= q[0] < gray.shape[1]-border and border <= q[1] < gray.shape[0]-border):
            continue
        back = local_template_matches(warped, gray, visible, [q], SEARCH)
        if len(back) and np.linalg.norm(q + back[0, 2:4] - [x, y]) <= 1.:
            mutual.append([x, y, dx, dy, score])
    matches = np.asarray(mutual).reshape(-1, 5)
    quadrants = np.floor(2 * matches[:, :2] / gray.shape[::-1]).astype(int)
    row = dict(eligible_features=len(eligible), matches=len(matches),
               quadrants=len(np.unique(quadrants, axis=0)), status="weak or ambiguous")
    if len(matches) >= max(12, .25 * len(eligible)) and row["quadrants"] >= 3:
        shifts = matches[:, 2:4]
        shift = np.median(shifts, axis=0)
        norms = np.linalg.norm(shifts, axis=1)
        p90 = float(np.percentile(norms, 90))
        residual = float(np.percentile(np.linalg.norm(shifts-shift, axis=1), 90))
        row.update(shift_xy_px=shift.tolist(), median_px=float(np.median(norms)), p90_px=p90,
                   p90_after_shift_px=residual, status="mixed displacements")
        if p90 <= 1.:
            row["status"] = "aligned"
        elif np.linalg.norm(shift) >= 1.5 and residual <= 1.5:
            row["status"] = "coherent shift"
    return row


def verdict(rows):
    supported = [r["status"] for r in rows if "p90_px" in r]
    if len(supported) < 3:
        return "Inconclusive. Fewer than three nearby photos gave spatially distributed, repeatable matches."
    if all(s == "aligned" for s in supported):
        return "The measured views repeat the texture within one pixel (p90). Test how training weights these close views."
    if all(s == "coherent shift" for s in supported):
        return "The measured views show coherent shifts. Inspect the floor plane, calibration and local poses before retraining."
    return "The measured views disagree in different ways. Inspect the per-view shifts before making a scene-wide change."


def analyze(suite, output):
    meta = json.loads((suite/"suite.json").read_text())
    source, cloud = Path(meta["source"]), Path(meta["cloud"]["cached"])
    manifest = json.loads(Path(meta["selection"]).read_text())
    selected = set(manifest["selected_images"])
    if REFERENCE not in selected:
        raise ValueError(f"{REFERENCE} is absent from this suite's training split")
    folder = manifest.get("image_directory", "images")
    _, cameras, images, _ = read_model(source)
    ref_index = next((i for i, view in enumerate(images) if view.name == REFERENCE), None)
    if ref_index is None:
        raise ValueError(f"{REFERENCE} is absent from the COLMAP model in {source}")
    ref = images[ref_index]
    cam = cameras[ref.camera_id]
    x, y, right, bottom = [round(a*b) for a, b in zip(ROI_FRACTIONS, (cam.width, cam.height)*2)]
    roi = (x, y, right-x, bottom-y)
    rgb = load_rgb(image_path(source, folder, ref.name), cam)[y:bottom, x:right].copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    save_image(output, "reference.png", rgb)
    rows = []
    report = dict(reference=REFERENCE, roi_xywh=roi, source=str(source), cloud=str(cloud),
                  keyframes=meta["selection"], search_radius_px=SEARCH, frame_window=WINDOW,
                  visibility_cell_px=CELL, max_sharp_neighbors=MAX_VIEWS, views=rows)
    report = compare_lidar(report, cloud, source, folder, images, cameras, ref_index, selected, roi, rgb, gray, output)
    if sum("p90_px" in row for row in rows) < 3:
        from scripts.floor_photo_matches import photo_check
        print(f"{report['verdict']} Checking real photo matches independently of LiDAR.", flush=True)
        report["photo_check"] = photo_check(source, folder, images, cameras, ref_index, selected, roi, gray,
                                           WINDOW, MAX_VIEWS, output)
    return report


def fit_visible_patch(points, ref, cam, roi, rgb, output):
    # Give each small patch of image comparable weight; points behind the nearest
    # return in that cell cannot establish the visible floor's orientation.
    uv, depth = project(points, ref.R, ref.C, cam.params)
    xy = uv - .5 - roi[:2]
    cells = np.floor(xy/4).astype(int)
    keys = cells[:, 1] * ((roi[2]+3)//4) + cells[:, 0]
    order = np.lexsort((depth, keys))
    _, first = np.unique(keys[order], return_index=True)
    ids = order[first]
    stats = dict(projected_returns=len(points), front_cells=len(ids), tolerance_m=TOLERANCE)
    overlay = rgb.copy()
    pixels = np.rint(xy[ids]).astype(int)
    for px, py in pixels:
        cv2.circle(overlay, (int(px), int(py)), 1, (0, 155, 255), -1)
    stats["overlay"] = save_image(output, "plane_support.png", overlay)
    if not len(ids):
        return None, stats, "No LiDAR returns project into the reference crop."
    stats["front_depth_p10_p50_p90"] = np.percentile(depth[ids], [10,50,90]).tolist()
    if len(ids) < 30:
        return None, stats, "Fewer than 30 frontmost LiDAR cells cover the crop."
    front = points[ids]
    try:
        center, normal, _ = fit_plane(front, TOLERANCE)
        world = plane_grid(ref, cam, roi, center, normal)
    except ValueError:
        return None, stats, "The frontmost returns do not define a supported plane in front of the camera across this crop."
    # Check observed camera depths as well as perpendicular plane distance. A
    # nearly edge-on plane can have small normal errors even at wildly different depths.
    rays = (front - ref.C) / depth[ids, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_depth = ((center-ref.C) @ normal) / (rays @ normal)
    keep = ((np.abs((front-center) @ normal) <= TOLERANCE)
            & (np.abs(predicted_depth-depth[ids]) <= np.maximum(.06, .03*depth[ids])))
    quadrants = np.floor(xy[ids][keep]/roi[2:]*4).astype(int).clip(0, 3)
    cover = len(np.unique(quadrants, axis=0))
    stats.update(center=center.tolist(), normal=normal.tolist(), inliers=int(keep.sum()),
                 fraction=float(keep.mean()), covered_cells=cover)
    for px, py in pixels[keep]:
        cv2.circle(overlay, (int(px), int(py)), 1, (0, 255, 0), -1)
    save_image(output, stats["overlay"], overlay)
    if keep.sum() < 30 or keep.mean() < .6 or cover < 12:
        return None, stats, (f"The fitted plane agrees with {keep.sum()}/{len(ids)} frontmost cells "
                             f"and covers {cover}/16 crop regions.")
    return (center, normal, world), stats, None


def compare_lidar(report, cloud, source, folder, images, cameras, ref_index, selected, roi, rgb, gray, output):
    ref = images[ref_index]
    cam = cameras[ref.camera_id]
    rows = report["views"]
    print(f"Fitting the visible floor in {REFERENCE} from {cloud}", flush=True)
    points = floor_points(cloud, ref, cam, roi, (.05, np.inf))
    fit, stats, reason = fit_visible_patch(points, ref, cam, roi, rgb, output)
    report["plane"] = stats
    print(f"{stats['projected_returns']} projected returns, {stats['front_cells']} frontmost cells, "
          f"{stats.get('inliers', 0)} accepted plane samples.", flush=True)
    if fit is None:
        return dict(report, verdict=f"Inconclusive. {reason}")
    center, normal, world = fit
    try:
        features = roi_features(gray, SEARCH)
    except ValueError:
        return dict(report, verdict="Inconclusive. Too few distinct corners in the reference floor crop.")
    ranked, counts = candidates(images, cameras, ref_index, selected, world)
    report["eligible_by_geometry"] = len(ranked)
    print(f"{len(ranked)} nearby training views pass the overlap and pixel-scale checks.", flush=True)
    if ranked:
        points = local_cloud(cloud, world, [ref.C] + [v.C for _, v, _ in ranked])
        xy, valid = target_grid(world, ref, cam)
        ref_mask = visible_floor(points, center, normal, ref, cam, xy, valid)
    used = 0
    for _, view, row in ranked:
        camera = cameras[view.camera_id]
        xy, valid = target_grid(world, view, camera)
        mask = ref_mask & visible_floor(points, center, normal, view, camera, xy, valid)
        row["mutual_coverage"] = float(mask.mean())
        rows.append(row)
        if mask.mean() < .5:
            row["status"] = "insufficient LiDAR visibility support"
        else:
            photo = load_rgb(image_path(source, folder, view.name), camera)
            warp = cv2.remap(photo, xy.astype(np.float32), None, cv2.INTER_LINEAR)
            warped = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            inside = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ratio = texture_strength(warped, inside) / max(texture_strength(gray, inside), .001) if inside.any() else 0.
            row["relative_detail"] = ratio
            if ratio < .55:
                row["status"] = "soft warped texture"
            else:
                used += 1
                row.update(measure(gray, warped, mask, features))
            overlay = cv2.merge((warped, gray, warped))
            overlay[~mask] = 0
            row["overlay"] = save_image(output, f"overlay_{len(rows)}.png", overlay)
            row["warp"] = save_image(output, f"warp_{len(rows)}.png", warp)
        print(f"{view.name}: {row['status']} ({mask.mean():.0%} mutual support)", flush=True)
        if used == MAX_VIEWS:
            break
    return dict(report, geometry_rejections=dict(counts), verdict=verdict(rows))


def write_report(output, report):
    atomic_json(output/"report.json", report)
    sections = []
    for row in report["views"]:
        title = escape(f"{row['name']} · {row['status']}")
        low, high = row['sampling_range']
        detail = (f"{row['baseline_m']:.2f} m baseline · {low:.2f}–{high:.2f} source px per reference px"
                  f" · {row['mutual_coverage']:.0%} supported")
        if "matches" in row:
            detail += f" · {row['matches']}/{row['eligible_features']} mutual matches"
        if "p90_px" in row:
            dx, dy = row["shift_xy_px"]
            detail += f" · median {row['median_px']:.2f} px · p90 {row['p90_px']:.2f} px · shift ({dx:+.2f}, {dy:+.2f}) px"
        pictures = {"Reference": "reference.png", "Warped neighbor": row["warp"],
                    "Green reference / magenta neighbor": row["overlay"]} if "warp" in row else {}
        panels = "".join(f'<figure><figcaption>{label}</figcaption><a href="{file}"><img src="{file}"></a></figure>'
                         for label, file in pictures.items())
        sections.append(f"<section><h2>{title}</h2><p>{detail}</p><div>{panels}</div></section>")
    plane = report["plane"]
    depth = plane.get("front_depth_p10_p50_p90", [])
    depth_text = ", ".join(f"{z:.2f}" for z in depth)
    sections.insert(0, f"<h2>Reference LiDAR support</h2><p>{plane['projected_returns']} projected returns · "
                    f"{plane['front_cells']} occupied 4 px cells · {plane.get('inliers', 0)} accepted plane samples. "
                    + (f"Camera depth p10, median and p90 in scene units are {depth_text}. " if depth else "")
                    + 'Green marks plane support. Orange marks other frontmost returns.</p>'
                    + '<a href="plane_support.png"><img style="max-width:650px" src="plane_support.png"></a>')
    if "photo_check" in report:
        photo = report["photo_check"]
        sections.append(f"<h2>Independent RGB check</h2><p>{escape(photo['verdict'])}</p>"
                        f"<p>{photo['reference_features']} distinct reference features. The photo warp is fitted "
                        "from RGB matches and scored on separate matches. The original-pose epipolar error "
                        "measures only the component perpendicular to the corresponding line.</p>")
        for row in photo["views"]:
            text = f"{row['matches']} mutual matches · {row['status']}"
            if "photo_p90_px" in row:
                text += f" · {row['check_matches']} check matches · photo warp p90 {row['photo_p90_px']:.2f} px"
            if "epipolar_median_px" in row:
                text += f" · original-pose epipolar median {row['epipolar_median_px']:.2f} px"
            sections.append(f"<h3>{escape(row['name'])}</h3><p>{text}</p>")
            if "overlay" in row:
                sections.append(f'<a href="{row["overlay"]}">Photo warp overlay</a>')
    headline = report.get("photo_check", report)["verdict"]
    lidar_result = f'<p>LiDAR warp · {escape(report["verdict"])}</p>' if "photo_check" in report else ''
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Orange4F floor alignment</title><style>body{font:16px system-ui;margin:2vw;background:#18212b;color:#eef}'
        'a{color:#ade}div{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0;width:min(420px,95vw)}img{width:100%}</style>'
        f'<h1>Nearby floor photos · {REFERENCE}</h1><h2>{escape(headline)}</h2>{lidar_result}'
        '<p>Click a crop for native pixels. Black in the LiDAR warp overlay marks missing support or a possible foreground surface in either view. '
        'Occluders absent from LiDAR may still pass. Shifts are measured in native reference pixels.</p>'
        '<a href="reference.png">Reference crop</a> · <a href="report.json">Measurements</a>' + ''.join(sections))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("ORANGE4F_OUTPUT_ROOT", "/data/3dgs_outputs/orange4f/3dgs_finetuning"))
    parser.add_argument("--suite", type=Path, default=root/"stride3_noaa_v1")
    parser.add_argument("--output", type=Path, help="Use a fresh directory to repeat the test")
    args = parser.parse_args(argv)
    suite = args.suite.expanduser().resolve()
    output = (args.output or suite/REPORT_DIR).expanduser().resolve()
    if (output/"index.html").is_file() and (output/"report.json").is_file():
        print(f"Existing report: {output/'index.html'}")
        report = json.loads((output/"report.json").read_text())
        print(report["verdict"])
        if "photo_check" in report:
            print(report["photo_check"]["verdict"])
        return
    if output.exists():
        parser.error(f"Output already exists. Use --output with a new directory: {output}")
    if not (suite/"suite.json").is_file():
        parser.error(f"Orange4F suite missing at {suite}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        stage = Path(work)/"report"
        stage.mkdir()
        report = analyze(suite, stage)
        write_report(stage, report)
        stage.rename(output)
    if output == suite/REPORT_DIR:
        from scripts.refresh_orange4f_dashboard import refresh
        try:
            refresh(suite)
        except (OSError, ValueError) as exc:
            print(f"Could not add the dashboard link: {exc}. Use the report path below.")
    print(report["verdict"])
    if "photo_check" in report:
        print(report["photo_check"]["verdict"])
    print(f"Open {output/'index.html'}", flush=True)


if __name__ == "__main__":
    main()
