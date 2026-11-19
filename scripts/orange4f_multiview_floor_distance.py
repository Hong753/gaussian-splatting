"""Train two native 3DGS models on the same physical floor patch from multiple photos."""

import argparse
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_floor_blur import floor_points, load_rgb, plane_grid, target_grid
from scripts.keyframe_pipeline.common import image_path, project, read_model
from scripts.orange4f_detail import SUITE, experiment_lock, select_gpu, split_settings, training_command
from scripts.orange4f_floor_neighbors import (
    REFERENCE,
    ROI_FRACTIONS,
    fit_visible_patch,
    local_cloud,
    visible_floor,
)
from scripts.orange4f_one_view_weight import crop_result
from scripts.orange4f_splat_footprints import NEIGHBOR, bright_cores, validate_audit_camera
from scripts.view_comparisons import image_id, model_info
from utils.checkpoint_snapshots import atomic_json, atomic_write, digest, read_snapshot
from utils.region_loss import mask_path

REPORT = "multiview_floor_distance_v1"
ARMS = ("uniform", "distance_weighted")


def inverse_square_weights(distances):
    values = np.array(list(distances.values()), dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Camera-to-patch distances must be finite and positive")
    inverse_square = (values.min() / values) ** 2
    return dict(zip(distances, (inverse_square / inverse_square.mean()).tolist()))


def native_projection(world, ref, ref_cam, ref_rect, view, camera, center, normal):
    """Map native pixels of the projected crop back to the reference floor grid."""
    if view.name == ref.name:
        return ref_rect, None
    corners = world[[0, 0, -1, -1], [0, -1, -1, 0]]
    uv, depth = project(corners, view.R, view.C, camera.params)
    if (depth <= 0.05).any():
        raise ValueError("Floor patch crosses the camera plane")
    lo = np.maximum(np.floor((uv - 0.5).min(axis=0)).astype(int), 0)
    hi = np.minimum(np.ceil((uv - 0.5).max(axis=0)).astype(int) + 1, [camera.width, camera.height])
    if (hi - lo < 11).any():
        raise ValueError("Projected floor patch is too small for regional SSIM")
    rect = tuple(map(int, [*lo, *(hi - lo)]))
    grid = plane_grid(view, camera, rect, center, normal)
    back, valid = target_grid(grid, ref, ref_cam)
    back -= ref_rect[:2]
    back[~valid] = -1
    return rect, back.astype(np.float32)


def native_mask(common, mapping):
    if mapping is None:
        return common.copy()
    return cv2.remap(common.astype(np.uint8), mapping, None, cv2.INTER_NEAREST).astype(bool)


def mask_interior(mask, radius):
    size = 2 * radius + 1
    return cv2.erode(
        mask.astype(np.uint8),
        np.ones((size, size), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def find_floor_views(suite, output, allowed, count, search_window):
    """Use geometry to identify the same supported floor texels in native photographs."""
    meta = json.loads((suite / "suite.json").read_text())
    source, cloud = Path(meta["source"]), Path(meta["cloud"]["cached"])
    folder = json.loads(Path(meta["selection"]).read_text()).get("image_directory", "images")
    _, cameras, images, stamps = read_model(source)
    records = {v.name: v for v in images}
    ref, heldout = (records[n] for n in (REFERENCE, NEIGHBOR))
    ref_cam = cameras[ref.camera_id]
    x, y, right, bottom = [round(v * n) for v, n in zip(ROI_FRACTIONS, (ref_cam.width, ref_cam.height) * 2)]
    rect = (x, y, right - x, bottom - y)
    photo = load_rgb(image_path(source, folder, REFERENCE), ref_cam)
    raw_floor = floor_points(cloud, ref, ref_cam, rect, (0.05, np.inf))
    fit, plane, reason = fit_visible_patch(raw_floor, ref, ref_cam, rect, photo[y:bottom, x:right], output)
    if fit is None:
        raise ValueError(f"Cannot fit the audited floor patch. {reason}")
    center, normal, world = fit
    ref_index = next(i for i, v in enumerate(images) if v.name == REFERENCE)
    nearby = images[max(0, ref_index - search_window) : ref_index + search_window + 1]
    candidates = []
    for view in nearby:
        if view.name in (REFERENCE, NEIGHBOR) or view.name not in allowed:
            continue
        _, in_image = target_grid(world[::8, ::8], view, cameras[view.camera_id])
        if in_image.mean() >= 0.8:
            candidates.append(view)
    print(f"{len(candidates)} training candidates project at least 80% of the reference crop", flush=True)
    points = local_cloud(cloud, world, [v.C for v in [ref, heldout, *candidates]])

    def support(view):
        camera = cameras[view.camera_id]
        xy, valid = target_grid(world, view, camera)
        return visible_floor(points, center, normal, view, camera, xy, valid)

    common = support(ref) & support(heldout)
    reference_support = int(common.sum())
    minimum = 0.5 * reference_support
    mappings = {
        v.name: native_projection(world, ref, ref_cam, rect, v, cameras[v.camera_id], center, normal)
        for v in (ref, heldout)
    }
    if not all(mask_interior(native_mask(common, mapping), 5).any() for _, mapping in mappings.values()):
        raise ValueError("Reference and held-out photos share no floor region large enough for SSIM")

    patch_center = world[common].mean(axis=0)

    def distance(view):
        return float(np.linalg.norm(view.C - patch_center))

    distances = {view.name: distance(view) for view in [ref, heldout, *candidates]}
    train = [ref]
    while candidates and len(train) < count:
        # Favor depth diversity, then prefer the smaller image name on a tie.
        def separation(candidate):
            log_distance = np.log(distances[candidate.name])
            return min(abs(log_distance - np.log(distances[v.name])) for v in train)

        view = min(candidates, key=lambda v: (-separation(v), v.name))
        # Pose dataclasses carry NumPy arrays, so avoid equality on them.
        candidates = [candidate for candidate in candidates if candidate.name != view.name]
        if min(np.linalg.norm(view.C - v.C) for v in train) < 0.02:
            continue
        intersection = common & support(view)
        if intersection.sum() < minimum:
            continue
        camera = cameras[view.camera_id]
        try:
            bounds, mapping = native_projection(world, ref, ref_cam, rect, view, camera, center, normal)
        except ValueError:
            continue
        projected_masks = [mapping] + [entry[1] for entry in mappings.values()]
        if not all(mask_interior(native_mask(intersection, m), 5).any() for m in projected_masks):
            continue
        train.append(view)
        mappings[view.name] = bounds, mapping
        common = intersection
        print(f"Selected {view.name}, {distances[view.name]:.2f} m from floor", flush=True)
    if len(train) < 2:
        raise ValueError("Need at least two training views of this floor. Try a wider --search-window.")

    patch_center = world[common].mean(axis=0)
    data = {}
    for view in [*train, heldout]:
        name = view.name
        camera = cameras[view.camera_id]
        (left, top, width, height), mapping = mappings[name]
        path = image_path(source, folder, name).resolve()
        rgb = load_rgb(path, camera)[top : top + height, left : left + width, ::-1].copy()
        mask = native_mask(common, mapping)
        # Keep the contrast statistic clear of dimmed pixels in the saved evaluation renders.
        interior = mask_interior(mask, 16)
        cores, detail = bright_cores(rgb, interior) if interior.any() else (interior, {"median_diameter_px": None})
        data[name] = dict(
            view=view,
            cam=camera,
            path=path,
            rect=(left, top, width, height),
            mask=mask,
            rgb=rgb,
            cores=cores,
            detail=detail,
            distance=distance(view),
        )
    geometry = dict(
        plane=plane,
        camera_files=stamps,
        patch_center_m=patch_center.tolist(),
        common_reference_pixels=int(common.sum()),
        reference_heldout_pixels=reference_support,
    )
    return [view.name for view in train], data, geometry


def write_gallery(root, status, iteration, train_names, data, weights, final):
    cameras = []
    panels = []
    for name in [REFERENCE, NEIGHBOR, *[n for n in train_names if n != REFERENCE]]:
        patch = data[name]
        label = "Training" if name in weights else "Held out"
        distance = patch["distance"]
        weight = f"{weights[name]:.3f}×" if name in weights else "—"
        cameras.append(f"<tr><td>{name}</td><td>{label}</td><td>{distance:.2f} m</td><td>{weight}</td></tr>")
        files = [("Photo", f"{Path(name).stem}_gt.png")]
        for arm in final:
            result = final[arm][name]
            contrast = result["spot_contrast"]
            contrast_label = f" · spot contrast {contrast:.1%}" if contrast is not None else ""
            title = "Uniform" if arm == "uniform" else "Distance weighted"
            files.append((f"{title} · {float(result['psnr_db']):.2f} dB{contrast_label}", result["render"]))
        images = "".join(
            f'<figure><figcaption>{caption}</figcaption><a href="{file}"><img src="{file}" alt="{caption}"></a></figure>'
            for caption, file in files
        )
        panels.append(f"<h2>{name} · {label}</h2><div class='panels'>{images}</div>")
    links = (
        '<a href="uniform/debug_previews/index.html">Uniform previews</a> · '
        '<a href="distance_weighted/debug_previews/index.html">Distance-weighted previews</a> · '
        '<a href="protocol.json">Cameras and settings</a>'
    )
    if status == "Complete":
        links += (
            ' · <a href="comparison/index.html">All renders and errors</a> · <a href="results.json">Scores</a>'
        )
    secondary_refresh = (
        '<meta http-equiv="refresh" content="15">' if status.startswith(("Training", "Rendering")) else ""
    )
    page = (
        f'<!doctype html><meta charset="utf-8">{secondary_refresh}'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Multiview floor loss</title><style>body{margin:2vw;font:16px system-ui;background:#14202b;color:white}"
        ".panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;height:auto}a{color:#9de0ff}th,td{padding:5px 14px;text-align:left}"
        "@media(max-width:720px){.panels{grid-template-columns:1fr}}</style>"
        f"<h1>Multiview floor loss</h1><p>{escape(status)} · {iteration:,} steps per run</p>"
        "<p>Both runs use the same floor region and original cameras. Each training view has weight 1× "
        "in the uniform run. The distance-weighted run gives closer cameras greater weight.</p>"
        f"<p>{links}</p><table><tr><th>Photo</th><th>Split</th><th>Distance to floor</th><th>Distance weight</th></tr>"
        + "".join(cameras)
        + "</table>"
        + "".join(panels)
    )
    atomic_write(root / "index.html", lambda path: path.write_text(page))


def launch(suite, gpu=None, *, count=4, steps_per_view=2000, search_window=180, prepare_only=False):
    root = suite / REPORT
    base = model_info(suite / "3dgs")
    if REFERENCE not in base["splits"]["train"]:
        raise ValueError(f"{REFERENCE} is absent from the original stride-3 split")
    if base["cfg"].get("resolution") != 1:
        raise ValueError("The floor masks use the native-resolution stride-3 suite")
    extent, _, _ = split_settings(base)
    ratio = base["cfg"]["optimization_settings"]["percent_dense"]
    # The explicit manifest defines every trained view and the one held-out preview.
    native_base = dict(base, cfg=dict(base["cfg"], eval=False))
    with experiment_lock(root) as lock:
        root.mkdir(exist_ok=True)
        train_names, data, geometry = find_floor_views(suite, root, base["splits"]["train"], count, search_window)
        iteration = len(train_names) * steps_per_view
        weights = inverse_square_weights({name: data[name]["distance"] for name in train_names})
        masks = root / "masks"
        full_masks = {}
        for name, patch in data.items():
            validate_audit_camera(base["records"][name], name, patch)
            camera = patch["cam"]
            left, top, width, height = patch["rect"]
            full_mask = np.zeros((camera.height, camera.width), np.uint8)
            full_mask[top : top + height, left : left + width] = patch["mask"] * 255
            full_masks[name] = full_mask
        weights_file = root / "view_loss_weights.json"
        commands = {}
        for arm in ARMS:
            cmd = training_command(native_base, root / arm, ratio, steps=(iteration,))
            cmd += ["--training_scene_extent", str(extent), "--loss_mask_dir", str(masks)]
            cmd[cmd.index("--debug-preview-test") + 1] = "1"
            if arm == "distance_weighted":
                cmd += ["--view_loss_weights", str(weights_file)]
            commands[arm] = cmd
        plan = dict(
            commands=commands,
            train=train_names,
            held_out=[NEIGHBOR],
            iteration=iteration,
            steps_per_training_view=steps_per_view,
            original_cameras=True,
            geometry=geometry,
            training_weights=dict(uniform=dict.fromkeys(train_names, 1.0), distance_weighted=weights),
            init_sha256=digest(base["path"] / "input.ply"),
            views={
                name: dict(
                    distance_m=p["distance"],
                    crop_xywh=list(p["rect"]),
                    supported_pixels=int(p["mask"].sum()),
                    photo_sha256=digest(p["path"]),
                    mask_sha256=sha256(full_masks[name].tobytes()).hexdigest(),
                )
                for name, p in data.items()
            },
        )
        protocol = root / "protocol.json"
        models_exist = any((root / arm).exists() for arm in ARMS)
        if models_exist and protocol.exists() and json.loads(protocol.read_text()) != plan:
            raise ValueError(f"The existing run uses a different protocol at {protocol}")
        if models_exist and not protocol.exists():
            raise FileExistsError(f"Unrecognized run directories in {root}")
        atomic_json(protocol, plan)
        atomic_json(weights_file, weights)
        for name, patch in data.items():
            atomic_write(
                mask_path(masks, name), lambda path: Image.fromarray(full_masks[name]).save(path, format="PNG")
            )
            rgb = patch["rgb"].copy()
            rgb[~patch["mask"]] //= 5
            Image.fromarray(rgb).save(root / f"{Path(name).stem}_gt.png")
            print(
                f"{name}: {patch['distance']:.2f} m, "
                + (f"distance weight {weights[name]:.3f}×" if name in weights else "held out"),
                flush=True,
            )
        if max(weights.values()) / min(weights.values()) < 1.1:
            print(
                "The selected cameras are nearly equidistant. The loss multipliers differ by less than 10%.",
                flush=True,
            )
        print(f"Report: {root / 'index.html'}", flush=True)
        if prepare_only:
            if not (root / "results.json").exists():
                write_gallery(root, "Prepared", iteration, train_names, data, weights, {})
            return
        gpu = select_gpu(gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        manifest = dict(selected_images=train_names, preview_heldout_images=[NEIGHBOR])

        def run(status, command):
            print(status, flush=True)
            write_gallery(root, status, iteration, train_names, data, weights, {})
            try:
                subprocess.run(command, cwd=ROOT, env=env, check=True, pass_fds=(lock.fileno(),))
            except subprocess.CalledProcessError as exc:
                write_gallery(
                    root, f"Stopped with exit code {exc.returncode}", iteration, train_names, data, weights, {}
                )
                raise

        write_gallery(root, "Preparing training", iteration, train_names, data, weights, {})
        from scripts.refresh_orange4f_dashboard import refresh

        refresh(suite)
        for arm in ARMS:
            output = root / arm
            manifest_path = output / "keyframes.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"Unexpected training or preview split at {manifest_path}")
            if not manifest_path.exists():
                atomic_json(manifest_path, manifest)
            if not read_snapshot(output, iteration):
                run(f"Training {arm} on {len(train_names)} views", commands[arm])
            if not read_snapshot(output, iteration, verify=False):
                raise RuntimeError(f"Missing final snapshot for {arm}")

        run(
            "Rendering all training views and the held-out photo",
            [
                sys.executable,
                "-u",
                "scripts/render.py",
                "--models",
                *[str(root / a) for a in ARMS],
                "--iteration",
                str(iteration),
                "--split",
                "both",
                "--eval-set",
                "unselected",
                "--frames",
                *train_names,
                NEIGHBOR,
                "--gpu",
                gpu,
                "--output",
                str(root / "comparison"),
            ],
        )
        final = {}
        for index, arm in enumerate(ARMS):
            final[arm] = {}
            for name, patch in data.items():
                split = "train" if name in train_names else "eval"
                image = root / "comparison" / split / f"model_{index}" / "renders" / (image_id(name) + ".png")
                final[arm][name] = crop_result(image, patch, root / f"{arm}_{Path(name).stem}.png")
        atomic_json(
            root / "results.json", dict(iteration=iteration, train=train_names, held_out=[NEIGHBOR], metrics=final)
        )
        write_gallery(root, "Complete", iteration, train_names, data, weights, final)
        print(root / "index.html", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--gpu", help="Physical CUDA device. Defaults to an idle GPU.")
    parser.add_argument("--train-views", type=int, default=4, help="Target number including 09882.jpg")
    parser.add_argument(
        "--steps-per-view", type=int, default=2000, help="Total steps divided by the training view count"
    )
    parser.add_argument(
        "--search-window", type=int, default=180, help="Registered photos on either side of the reference"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Inspect the selected views and weights on CPU"
    )
    args = parser.parse_args(argv)
    if args.train_views < 2 or min(args.steps_per_view, args.search_window) < 1:
        parser.error("Need at least two training views and positive step and search budgets")
    launch(
        args.suite.expanduser().resolve(),
        args.gpu,
        count=args.train_views,
        steps_per_view=args.steps_per_view,
        search_window=args.search_window,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
