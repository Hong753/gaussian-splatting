"""Fixed COLMAP cameras, fit-only DA3 inference and a reproducible RGB-D seed cloud."""

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.keyframes import natural_key
from scripts.orange4f_stock_triangle import choose_keyframes, read_colmap_scene
from scripts.prepare_da3_occlusion import save_npz, windows
from scripts.render_novel_comparison import choose_source_segment, synthesize_random_path
from scripts.view_comparisons import atomic_json
from utils.colmap_paths import colmap_model_dir
from utils.da3_occlusion import cross_view_support, reliable_pixels
from utils.read_write_model import qvec2rotmat


def inputs(source, run, width, stride, keys, novel_frames):
    sparse = colmap_model_dir(source)
    cams, views = read_colmap_scene(sparse)
    ordered = sorted(views.values(), key=lambda v: natural_key(v.name))
    names = [v.name for v in ordered]
    if len(names) < 4 or len(set(names)) != len(names):
        raise ValueError("Expected at least four distinct registered cameras")
    fit, selection = choose_keyframes(source, run, names, keys, stride)
    if len(fit) < 3:
        raise ValueError("Fit-only DA3 requires at least three fitting cameras")
    rows, cameras = [], []
    for i, view in enumerate(ordered):
        cam = cams[view.camera_id]
        if cam.model == "SIMPLE_PINHOLE":
            fx, cx, cy = cam.params
            fy = fx
        elif cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
        else:
            raise ValueError(f"Undistorted pinhole RGB required: {view.name} uses {cam.model}")
        image = source / "images" / view.name
        with Image.open(image) as im:
            if im.size != (cam.width, cam.height):
                raise ValueError(f"COLMAP/RGB size mismatch: {image}")
        w = min(width, cam.width)
        h = max(1, round(cam.height * w / cam.width))
        sx, sy = w / cam.width, h / cam.height
        # COLMAP uses (0.5,0.5) for the center of the first pixel. Tensor indices use (0,0).
        k = [fx * sx, fy * sy, cx * sx - 0.5, cy * sy - 0.5]
        pose = np.eye(4)
        pose[:3, :3] = qvec2rotmat(view.qvec / np.linalg.norm(view.qvec))
        pose[:3, 3] = view.tvec
        if not np.isfinite(pose).all() or not np.isfinite(k).all() or min(k[:2]) <= 0:
            raise ValueError(f"Invalid calibration: {view.name}")
        rows.append(
            dict(
                id=i,
                name=view.name,
                image=str(image),
                split="train" if view.name in fit else "test",
                width=w,
                height=h,
                K=k,
                w2c=pose.tolist(),
            )
        )
        cameras.append(
            SimpleNamespace(
                image_name=view.name,
                R=pose[:3, :3].T,
                T=pose[:3, 3],
                width=w,
                height=h,
                FovX=2 * math.atan(w / (2 * k[0])),
                FovY=2 * math.atan(h / (2 * k[1])),
            )
        )
    segment, selection_info = choose_source_segment(cameras, window=600, seed=42)
    path, details = synthesize_random_path(segment, cameras, novel_frames, seed=42)
    by_name = {r["name"]: r for r in rows}
    novel = []
    for i, cam in enumerate(path):
        # Use the interpolated focal lengths and principal point of the nearest time sample.
        ref = by_name[segment[round(i * (len(segment) - 1) / (len(path) - 1))].image_name]
        k = [
            cam.width / (2 * math.tan(cam.FovX / 2)),
            cam.height / (2 * math.tan(cam.FovY / 2)),
            (ref["K"][2] + 0.5) * cam.width / ref["width"] - 0.5,
            (ref["K"][3] + 0.5) * cam.height / ref["height"] - 0.5,
        ]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cam.R.T, cam.T
        novel.append(
            dict(id=i, name=cam.image_name, width=cam.width, height=cam.height, K=k, w2c=pose.tolist())
        )
    manifest = dict(
        calibration=str(sparse),
        keyframes=selection,
        frames=rows,
        random=novel,
        random_path=dict(**selection_info, **details),
        pose_source="supplied COLMAP world-to-camera",
    )
    path = run / "cameras.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"Camera selection or calibration changed in {run}. Use a fresh run directory.")
    atomic_json(path, manifest)
    print(
        f"{len(rows):,} registered cameras, {len(fit):,} fitting keyframes, {len(rows) - len(fit):,} held out",
        flush=True,
    )
    return rows


def rgb(row):
    with Image.open(row["image"]) as im:
        return np.array(im.convert("RGB").resize((row["width"], row["height"]), Image.Resampling.LANCZOS))


def matrix(row):
    fx, fy, cx, cy = row["K"]
    # DA3 scales principal points directly when resizing its inputs. Supply the
    # image-boundary convention here, then convert its output K to array indices.
    return np.array([[fx, 0, cx + 0.5], [0, fy, cy + 0.5], [0, 0, 1]], np.float32)


def prior_path(run, row):
    return run / "da3" / f"{row['id']:08d}.npz"


def preprocess(run, rows, model_id, resolution, seed_points):
    fit = [row for row in rows if row["split"] == "train"]
    if len({(r["width"], r["height"]) for r in fit}) != 1:
        raise ValueError("DA3 fitting windows require a common image size")
    meta = dict(
        version=1,
        model=model_id,
        resolution=resolution,
        seed_points=seed_points,
        fitting_sha256=hashlib.sha256(json.dumps(fit, sort_keys=True).encode()).hexdigest(),
    )
    path = run / "da3" / "config.json"
    if path.exists() and json.loads(path.read_text()) != meta:
        raise ValueError("DA3 cache belongs to different images, cameras or settings")
    atomic_json(path, meta)
    seed = run / "da3" / "seed.npz"
    if seed.is_file() and all(prior_path(run, r).is_file() for r in fit):
        print(f"Reusing {len(fit)} fitting priors and {seed}", flush=True)
        return
    model = None
    for batch, core in windows(fit, core=8, context=3):
        if all(prior_path(run, r).is_file() for r in core):
            continue
        if model is None:
            from depth_anything_3.api import DepthAnything3

            model = DepthAnything3.from_pretrained(model_id).to("cuda").eval()
        extr = np.asarray([r["w2c"] for r in batch], np.float32)
        pred = model.inference(
            [rgb(r) for r in batch],
            extrinsics=extr,
            intrinsics=np.stack([matrix(r) for r in batch]),
            align_to_input_ext_scale=True,
            process_res=resolution,
            process_res_method="upper_bound_resize",
        )
        zs = np.asarray(pred.depth, np.float32)
        masks = [reliable_pixels(z, conf, percentile=35) for z, conf in zip(zs, pred.conf, strict=True)]
        calibrated_k = np.array(pred.intrinsics, copy=True)
        calibrated_k[:, :2, 2] -= 0.5
        views = list(zip(zs, masks, calibrated_k, extr, strict=True))
        core_ids = {r["id"] for r in core}
        for i, r in enumerate(batch):
            if r["id"] in core_ids:
                z, mask, k, pose = views[i]
                trusted = mask & (cross_view_support(z, mask, k, pose, views[:i] + views[i + 1 :]) >= 2)
                save_npz(
                    prior_path(run, r),
                    depth=np.where(np.isfinite(z) & (z > 0), z, 0),
                    valid=trusted,
                    K=np.asarray(k, np.float32),
                )
        from scripts.joint_ambisur.report import dashboard

        message = f"DA3 preprocessing · fitting frame {core[-1]['name']}"
        dashboard(run, message)
        print(message, flush=True)
    del model
    rng = np.random.default_rng(42)
    xyz, colors = [], []
    per_view = max(32, math.ceil(2 * seed_points / len(fit)))
    for r in fit:
        with np.load(prior_path(run, r)) as p:
            y, x = np.where(p["valid"])
            ids = rng.choice(len(x), min(per_view, len(x)), replace=False)
            y, x = y[ids], x[ids]
            z, k = p["depth"][y, x], p["K"]
        ray = np.stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones(len(x))), axis=1)
        pose, (fx, fy, cx, cy) = np.asarray(r["w2c"]), r["K"]
        xyz.append((ray * z[:, None] - pose[:3, 3]) @ pose[:3, :3])
        u = np.rint(ray[:, 0] * fx + cx).astype(int).clip(0, r["width"] - 1)
        v = np.rint(ray[:, 1] * fy + cy).astype(int).clip(0, r["height"] - 1)
        colors.append(rgb(r)[v, u])
    xyz, colors = np.concatenate(xyz), np.concatenate(colors)
    if len(xyz) < 4:
        raise ValueError("DA3 produced fewer than four multiview-supported seed points")
    ids = rng.choice(len(xyz), min(seed_points, len(xyz)), replace=False)
    save_npz(seed, xyz=xyz[ids].astype(np.float32), rgb=colors[ids])
    print(f"Wrote {len(ids):,} DA3 seed points from {len(fit)} fitting views", flush=True)
