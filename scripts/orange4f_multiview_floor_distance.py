"""Train two native 3DGS models on the same physical floor patch from multiple photos."""

import argparse
from functools import lru_cache
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

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
from scripts.orange4f_floor_patch_search import alternate_patches
from scripts.orange4f_one_view_weight import crop_result
from scripts.orange4f_splat_footprints import bright_cores, validate_audit_camera
from scripts.view_comparisons import image_id, model_info
from utils.checkpoint_snapshots import atomic_json, atomic_write, digest, read_snapshot
from utils.region_loss import mask_path

REPORT = "multiview_floor_distance_v3"
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


def find_floor_views(suite, output, allowed, count, search_window, far_min_ratio):
    """Find near and far training and evaluation views of the same floor patch."""
    meta = json.loads((suite / "suite.json").read_text())
    source, cloud = Path(meta["source"]), Path(meta["cloud"]["cached"])
    folder = json.loads(Path(meta["selection"]).read_text()).get("image_directory", "images")
    _, cameras, images, stamps = read_model(source)
    records = {v.name: v for v in images}
    original = records[REFERENCE]
    camera = cameras[original.camera_id]
    x, y, right, bottom = [round(v * n) for v, n in zip(ROI_FRACTIONS, (camera.width, camera.height) * 2)]
    original_rect = (x, y, right - x, bottom - y)

    def fit_patch(view, rect):
        cam = cameras[view.camera_id]
        left, top, width, height = rect
        photo = load_rgb(image_path(source, folder, view.name), cam)
        points = floor_points(cloud, view, cam, rect, (0.05, np.inf))
        return fit_visible_patch(points, view, cam, rect, photo[top : top + height, left : left + width], output)

    original_fit, _, reason = fit_patch(original, original_rect)
    if original_fit is None:
        raise ValueError(f"Cannot establish the audited floor level. {reason}")
    origin, original_normal, original_world = original_fit
    patches, original_range, axes = alternate_patches(
        cloud,
        origin,
        original_normal,
        original,
        original_world,
        images,
        cameras,
        allowed,
        far_min_ratio,
    )
    old_corners = (original_world[[0, 0, -1, -1], [0, -1, -1, 0]] - origin) @ axes.T
    print(f"Found {len(patches)} alternative floor crops with near and distant framing", flush=True)
    for attempt, (_, ref, rect, near_options) in enumerate(patches, 1):
        print(f"Checking alternative floor crop {attempt}/{len(patches)} in {ref.name}, ROI {rect}", flush=True)
        fit, plane, reason = fit_patch(ref, rect)
        if fit is None:
            print(f"  {reason}", flush=True)
            continue
        center, normal, world = fit
        bounds = (world[[0, 0, -1, -1], [0, -1, -1, 0]] - origin) @ axes.T
        separate = (
            (bounds.max(axis=0) < old_corners.min(axis=0)) | (bounds.min(axis=0) > old_corners.max(axis=0))
        ).any()
        same_floor = abs(normal @ original_normal) >= 0.95 and abs((center - origin) @ original_normal) <= 0.15
        if not separate or not same_floor:
            print("  Projected crop overlaps the original patch or fits a different surface", flush=True)
            continue
        try:
            train, held_out, data, geometry = match_floor_views(
                source,
                cloud,
                folder,
                images,
                cameras,
                stamps,
                allowed,
                count,
                search_window,
                far_min_ratio,
                original_range,
                ref,
                rect,
                near_options,
                fit,
                plane,
            )
        except ValueError as exc:
            print(f"  {exc}", flush=True)
            continue
        geometry["original_patch_center_m"] = original_world.mean(axis=(0, 1)).tolist()
        print(f"Selected a different floor patch in {ref.name}, ROI {rect}", flush=True)
        return train, held_out, data, geometry
    raise ValueError(
        f"No alternative floor crop passed the near/far visibility checks after testing {len(patches)} candidates."
    )


def match_floor_views(
    source,
    cloud,
    folder,
    images,
    cameras,
    stamps,
    allowed,
    count,
    search_window,
    far_min_ratio,
    original_range,
    ref,
    rect,
    near_options,
    fit,
    plane,
):
    center, normal, world = fit
    ref_cam = cameras[ref.camera_id]
    records = {v.name: v for v in images}
    corners = world[[0, 0, -1, -1], [0, -1, -1, 0]]
    floor_min, floor_max = corners.min(axis=0), corners.max(axis=0)
    # Keep the distance anchor identical even when a candidate changes the common pixel mask.
    patch_center = world.mean(axis=(0, 1))

    def distance(view):
        return float(np.linalg.norm(view.C - patch_center))

    ref_distance = distance(ref)
    far_distance = far_min_ratio * max(original_range, ref_distance)
    ref_index = next(i for i, v in enumerate(images) if v.name == ref.name)
    prescreen = world[::8, ::8]
    near_candidates, far_candidates = [], []
    camera_side = np.sign((ref.C - center) @ normal)
    for index, view in enumerate(images):
        if view.name == ref.name or ((view.C - center) @ normal) * camera_side <= 0:
            continue
        is_far = distance(view) >= far_distance
        if not is_far and (
            view.name not in allowed or (search_window is not None and abs(index - ref_index) > search_window)
        ):
            continue
        _, in_image = target_grid(prescreen, view, cameras[view.camera_id])
        if in_image.mean() >= 0.5:
            (far_candidates if is_far else near_candidates).append(view)
    near_candidates.sort(key=lambda v: (distance(v), v.name))
    far_training = sorted(far_candidates, key=lambda v: (v.name not in allowed, -distance(v), v.name))
    far_evaluation = sorted(far_candidates, key=lambda v: (v.name in allowed, -distance(v), v.name))
    stride_candidates = sum(v.name in allowed for v in far_candidates)
    print(
        f"Full-scene search: {len(far_candidates)} distant candidates, including {stride_candidates} "
        f"stride-3 training candidates. Distant means at least {far_distance:.2f} m "
        f"(at least {far_min_ratio:g}× the new reference range).",
        flush=True,
    )
    if len(far_candidates) < 2:
        raise ValueError("Fewer than two distant cameras frame this crop.")
    points = local_cloud(cloud, world, [v.C for v in [ref, *near_options, *near_candidates, *far_candidates]])

    def support(view):
        # Exclude points outside the camera-to-patch box before projecting the LiDAR.
        low = np.minimum(floor_min, view.C) - 0.5
        high = np.maximum(floor_max, view.C) + 0.5
        local = points[((points >= low) & (points <= high)).all(axis=1)]
        camera = cameras[view.camera_id]
        xy, valid = target_grid(world, view, camera)
        return visible_floor(local, center, normal, view, camera, xy, valid)

    def projection(view):
        return native_projection(world, ref, ref_cam, rect, view, cameras[view.camera_id], center, normal)

    ref_mask = support(ref)
    ref_projection = projection(ref)
    if not mask_interior(ref_mask, 5).any():
        raise ValueError("The reference crop has no supported 11×11 floor window.")
    for near_eval in near_options:
        if distance(near_eval) >= far_distance:
            continue
        try:
            near_projection = projection(near_eval)
        except ValueError:
            continue
        base_mask = ref_mask & support(near_eval)
        if (
            base_mask.sum() >= 0.5 * ref_mask.sum()
            and mask_interior(base_mask, 5).any()
            and mask_interior(native_mask(base_mask, near_projection[1]), 5).any()
        ):
            break
    else:
        raise ValueError("No distinct near held-out camera shares enough visible floor with the reference.")
    reference_support = int(base_mask.sum())
    minimum = 0.5 * reference_support
    mappings = {ref.name: ref_projection, near_eval.name: near_projection}
    reference_projections = list(mappings.values())

    def usable(mask, projections):
        return mask.sum() >= minimum and all(
            mask_interior(native_mask(mask, mapping), 5).any() for _, mapping in projections
        )

    rejected = set()

    @lru_cache(maxsize=32)
    def observe(name):
        if name in rejected:
            return None
        view = records[name]
        try:
            projected = projection(view)
        except ValueError:
            rejected.add(name)
            return None
        mask = base_mask & support(view)
        if usable(mask, [projected, *reference_projections]):
            return mask, projected
        rejected.add(name)
        return None

    # Find a compatible distant train/eval pair before adding extra training views.
    train_far = eval_far = None
    for train_view in far_training:
        training = observe(train_view.name)
        if training is None:
            continue
        for eval_view in far_evaluation:
            if np.linalg.norm(eval_view.C - train_view.C) < 0.10:
                continue
            evaluation = observe(eval_view.name)
            if evaluation is None:
                continue
            common = training[0] & evaluation[0]
            if usable(common, [*mappings.values(), training[1], evaluation[1]]):
                train_far, eval_far = train_view, eval_view
                mappings[train_far.name], mappings[eval_far.name] = training[1], evaluation[1]
                break
        if train_far is not None:
            break
    if train_far is None:
        raise ValueError(
            f"No distant train/eval pair at {far_distance:.2f} m or farther shares enough visible floor."
        )

    train = [ref, train_far]
    heldout = [near_eval, eval_far]
    print(
        f"Distant train {train_far.name} ({distance(train_far):.2f} m). "
        f"Distant held out {eval_far.name} ({distance(eval_far):.2f} m).",
        flush=True,
    )
    for group, extra in ((near_candidates, (count - 1) // 2), (far_training, count // 2 - 1)):
        for view in group:
            if extra == 0:
                break
            if min(np.linalg.norm(view.C - v.C) for v in [*train, *heldout]) < 0.02:
                continue
            if np.linalg.norm(view.C - eval_far.C) < 0.10:
                continue
            observation = observe(view.name)
            if observation is None:
                continue
            mask, projected = observation
            intersection = common & mask
            if usable(intersection, [*mappings.values(), projected]):
                train.append(view)
                mappings[view.name] = projected
                common = intersection
                extra -= 1

    data = {}
    for view in [*train, *heldout]:
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
            distance_group="far" if distance(view) >= far_distance else "near",
        )
    geometry = dict(
        plane=plane,
        camera_files=stamps,
        patch_center_m=patch_center.tolist(),
        reference=ref.name,
        reference_roi_xywh=list(rect),
        original_floor_reference=REFERENCE,
        original_reference_range_m=original_range,
        far_min_ratio=far_min_ratio,
        far_min_distance_m=far_distance,
        common_reference_pixels=int(common.sum()),
        reference_heldout_pixels=reference_support,
    )
    return [view.name for view in train], [view.name for view in heldout], data, geometry


def write_gallery(root, status, iteration, train_names, data, weights, final, far_min_distance):
    cameras = []
    panels = []
    heldout_names = [name for name in data if name not in weights]
    for name in [*heldout_names, *train_names]:
        patch = data[name]
        split = "Training" if name in weights else "Held out"
        label = f"{split} · {patch['distance_group']}"
        escaped_name = escape(name)
        distance = patch["distance"]
        weight = f"{weights[name]:.3f}×" if name in weights else "—"
        cameras.append(
            f"<tr><td>{escaped_name}</td><td>{label}</td><td>{distance:.2f} m</td><td>{weight}</td></tr>"
        )
        files = [("Photo", f"{Path(name).stem}_gt.png")]
        for arm in final:
            result = final[arm][name]
            contrast = result["spot_contrast"]
            contrast_label = f" · spot contrast {contrast:.1%}" if contrast is not None else ""
            title = "Uniform" if arm == "uniform" else "Distance weighted"
            files.append((f"{title} · {float(result['psnr_db']):.2f} dB{contrast_label}", result["render"]))
        images = "".join(
            f'<figure><figcaption>{escape(caption)}</figcaption><a href="{escape(file)}">'
            f'<img src="{escape(file)}" alt="{escape(caption)}"></a></figure>'
            for caption, file in files
        )
        panels.append(f"<h2>{escaped_name} · {label}</h2><div class='panels'>{images}</div>")
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
        f"<p>Reference crop in {escape(train_names[0])}. <a href='plane_support.png'>LiDAR support</a></p>"
        "<p>Both runs use the same floor region and original cameras. Each training view has weight 1× "
        "in the uniform run. The distance-weighted run gives closer cameras greater weight. "
        f"Distant cameras are at least {far_min_distance:.2f} m from the patch. "
        "Both models are evaluated from a near and a distant held-out camera.</p>"
        f"<p>{links}</p><table><tr><th>Photo</th><th>Split</th><th>Distance to floor</th><th>Distance weight</th></tr>"
        + "".join(cameras)
        + "</table>"
        + "".join(panels)
    )
    atomic_write(root / "index.html", lambda path: path.write_text(page))


def launch(
    suite, gpu=None, *, count=4, steps_per_view=2000, search_window=None, far_min_ratio=2.0, prepare_only=False
):
    root = suite / REPORT
    base = model_info(suite / "3dgs")
    if REFERENCE not in base["splits"]["train"]:
        raise ValueError(f"{REFERENCE} is absent from the original stride-3 split")
    if base["cfg"].get("resolution") != 1:
        raise ValueError("The floor masks use the native-resolution stride-3 suite")
    extent, _, _ = split_settings(base)
    ratio = base["cfg"]["optimization_settings"]["percent_dense"]
    # The explicit manifest defines the training and held-out preview cameras.
    native_base = dict(base, cfg=dict(base["cfg"], eval=False))
    with experiment_lock(root) as lock:
        root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=root) as preparation:
            staging = Path(preparation)
            train_names, heldout_names, data, geometry = find_floor_views(
                suite,
                staging,
                base["splits"]["train"],
                count,
                search_window,
                far_min_ratio,
            )
            overlay = (staging / "plane_support.png").read_bytes()
        iteration = len(train_names) * steps_per_view
        far_min_distance = geometry["far_min_distance_m"]
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
            cmd[cmd.index("--debug-preview-test") + 1] = str(len(heldout_names))
            if arm == "distance_weighted":
                cmd += ["--view_loss_weights", str(weights_file)]
            commands[arm] = cmd
        plan = dict(
            commands=commands,
            train=train_names,
            held_out=heldout_names,
            iteration=iteration,
            steps_per_training_view=steps_per_view,
            original_cameras=True,
            geometry=geometry,
            training_weights=dict(uniform=dict.fromkeys(train_names, 1.0), distance_weighted=weights),
            init_sha256=digest(base["path"] / "input.ply"),
            views={
                name: dict(
                    distance_m=p["distance"],
                    distance_group=p["distance_group"],
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
        atomic_write(root / "plane_support.png", lambda path: path.write_bytes(overlay))
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
                + (f"distance weight {weights[name]:.3f}×" if name in weights else "held out")
                + f" ({patch['distance_group']})",
                flush=True,
            )
        print(f"Report: {root / 'index.html'}", flush=True)
        if prepare_only:
            if not (root / "results.json").exists():
                write_gallery(root, "Prepared", iteration, train_names, data, weights, {}, far_min_distance)
            return
        gpu = select_gpu(gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        manifest = dict(selected_images=train_names, preview_heldout_images=heldout_names)

        def run(status, command):
            print(status, flush=True)
            write_gallery(root, status, iteration, train_names, data, weights, {}, far_min_distance)
            try:
                subprocess.run(command, cwd=ROOT, env=env, check=True, pass_fds=(lock.fileno(),))
            except subprocess.CalledProcessError as exc:
                write_gallery(
                    root,
                    f"Stopped with exit code {exc.returncode}",
                    iteration,
                    train_names,
                    data,
                    weights,
                    {},
                    far_min_distance,
                )
                raise

        write_gallery(root, "Preparing training", iteration, train_names, data, weights, {}, far_min_distance)
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
            "Rendering the training cameras and both held-out photos",
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
                *heldout_names,
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
            root / "results.json",
            dict(iteration=iteration, train=train_names, held_out=heldout_names, metrics=final),
        )
        write_gallery(root, "Complete", iteration, train_names, data, weights, final, far_min_distance)
        print(root / "index.html", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--gpu", help="Physical CUDA device. Defaults to an idle GPU.")
    parser.add_argument("--train-views", type=int, default=4, help="Target number of training cameras")
    parser.add_argument(
        "--steps-per-view", type=int, default=2000, help="Total steps divided by the training view count"
    )
    parser.add_argument(
        "--search-window",
        type=int,
        help="Optional frame window for extra near training cameras. Distant views always use the full capture.",
    )
    parser.add_argument(
        "--far-min-ratio",
        type=float,
        default=2.0,
        help="Minimum distant camera range divided by the reference camera range",
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Inspect the selected views and weights on CPU"
    )
    args = parser.parse_args(argv)
    if (
        args.train_views < 2
        or args.steps_per_view < 1
        or (args.search_window is not None and args.search_window < 1)
    ):
        parser.error("Need at least two training views and positive step and search budgets")
    if not np.isfinite(args.far_min_ratio) or args.far_min_ratio <= 1:
        parser.error("--far-min-ratio must be finite and greater than 1")
    launch(
        args.suite.expanduser().resolve(),
        args.gpu,
        count=args.train_views,
        steps_per_view=args.steps_per_view,
        search_window=args.search_window,
        far_min_ratio=args.far_min_ratio,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
