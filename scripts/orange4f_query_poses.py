"""Localize individual capture frames in a frozen keyframe map. All distances are in image pixels."""
from functools import lru_cache
import hashlib
import json
import sqlite3

import cv2
import numpy as np

from scripts.keyframes import natural_key
from scripts.view_comparisons import atomic_json, read_colmap_records
from utils import read_write_model as colmap
from utils.checkpoint_snapshots import atomic_write, digest


def track(ref, target, xy):
    """Reject tracks that fail to return to their reference measurement."""
    if not len(xy) or ref.shape != target.shape:
        return np.empty(0, int), np.empty((0, 2))
    opts = dict(winSize=(21, 21), maxLevel=4)
    uv, valid, residual = cv2.calcOpticalFlowPyrLK(ref, target, xy, None, **opts)
    back, back_valid, _ = cv2.calcOpticalFlowPyrLK(target, ref, uv, None, **opts)
    h, w = target.shape
    good = (valid.ravel().astype(bool) & back_valid.ravel().astype(bool)
            & (np.linalg.norm(xy - back, axis=1) < 1.0) & (residual.ravel() < 24)
            & (uv[:, 0] >= 2) & (uv[:, 0] < w-2) & (uv[:, 1] >= 2) & (uv[:, 1] < h-2))
    return np.flatnonzero(good), uv[good]


def pnp(xyz, uv, K, width, height):
    report = dict(matches=len(uv), inliers=0, status="too_few_matches")
    if len(uv) < 12:
        return report
    ok, r, t, idx = cv2.solvePnPRansac(xyz, uv, K, None, iterationsCount=500, reprojectionError=3.0,
                                       confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or idx is None or len(idx) < 12:
        return {**report, "status": "pnp_failed"}
    idx = idx.ravel()
    r, t = cv2.solvePnPRefineLM(xyz[idx], uv[idx], K, None, r, t)
    R = cv2.Rodrigues(r)[0]
    projected = cv2.projectPoints(xyz, r, t, K, None)[0].reshape(-1, 2)
    errors = np.linalg.norm(projected-uv, axis=1)
    keep = (errors <= 3) & ((xyz @ R[2] + t[2, 0]) > 0)
    count = int(keep.sum())
    report.update(status="weak_pose", inliers=count)
    if count < max(12, 0.25*len(uv)):
        return report
    fraction = cv2.contourArea(cv2.convexHull(uv[keep].astype(np.float32))) / (width*height)
    report.update(image_hull_fraction=fraction, median_error_px=float(np.median(errors[keep])))
    if fraction < 0.025 or report["median_error_px"] > 2:
        return report
    report.update(status="localized", qvec=colmap.rotmat2qvec(R).tolist(), tvec=t.ravel().tolist())
    return report


def localize(map_dir, stage, capture, names, train, run):
    """Keep mapped poses/landmarks fixed. Input capture supplies query intrinsics only."""
    cameras, mapped, points = colmap.read_model(map_dir, ".bin")
    refs = {v.name: v for v in mapped.values()}
    if not refs or not set(refs) <= set(train) or not points:
        raise ValueError("Expected a sparse model with tracks from selected keyframes only")
    records = read_colmap_records(capture)
    ordered = sorted(names, key=natural_key)
    if set(names) - records.keys():
        raise ValueError("Missing supplied intrinsics for capture images")
    order = {name: i for i, name in enumerate(ordered)}
    lookup = {i: pt.xyz for i, pt in points.items()}
    images = stage / "images"
    cv2.setNumThreads(4)

    @lru_cache(maxsize=5)
    def grayscale(name):
        path = images / name
        pixel = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if pixel is None or pixel.shape[::-1] != (records[name]["width"], records[name]["height"]):
            raise ValueError(f"RGB dimensions disagree with capture calibration: {path}")
        # Spirula mapper observations are in original image coordinates. Track at up to 1200px.
        scale = min(1.0, 1200 / pixel.shape[1])
        if scale < 1:
            pixel = cv2.resize(pixel, (1200, round(scale*pixel.shape[0])), interpolation=cv2.INTER_AREA)
        return pixel

    @lru_cache(maxsize=5)
    def reference(name):
        im = refs[name]
        cam = cameras[im.camera_id]
        if (cam.width, cam.height) != (records[name]["width"], records[name]["height"]):
            raise ValueError(f"Keyframe map and original RGB dimensions differ: {name}")
        gray = grayscale(name)
        valid = np.array([i for i, pid in enumerate(im.point3D_ids) if pid in lookup], dtype=int)
        ids = im.point3D_ids[valid]
        scale = np.array(gray.shape[::-1]) / (cam.width, cam.height)
        xy = (im.xys[valid] * scale - 0.5).astype(np.float32)
        return gray, ids, xy

    # PnP uses supplied image intrinsics. No source extrinsics enter this calculation.
    def solve(name):
        record = records[name]
        w, h = record["width"], record["height"]
        query = grayscale(name)
        matches = {}
        neighbors = sorted(refs, key=lambda n: (abs(order[n]-order[name]), order[n]))[:4]
        for neighbor in neighbors:
            img, ids, xy = reference(neighbor)
            index, uv = track(img, query, xy)
            for pid, point in zip(ids[index], uv):
                matches.setdefault(pid, point)
        uv = np.asarray(list(matches.values()), dtype=float).reshape(-1, 2)
        scale = np.array((w, h)) / query.shape[::-1]
        uv = (uv + 0.5) * scale - 0.5
        xyz = np.array([lookup[i] for i in matches], dtype=float).reshape(-1, 3)
        K = np.array([[record["fx"], 0, record["cx"]-0.5], [0, record["fy"], record["cy"]-0.5], [0, 0, 1.]])
        cv2.setRNGSeed(42)
        return {**pnp(xyz, uv, K, w, h), "references": neighbors}

    key = hashlib.sha256()
    for file in ("cameras.bin", "images.bin", "points3D.bin"):
        key.update(digest(map_dir / file).encode())
    intrinsics = {n: tuple(records[n][k] for k in ("width", "height", "fx", "fy", "cx", "cy")) for n in ordered}
    key.update(json.dumps([1, cv2.__version__, str(images.resolve()), intrinsics]).encode())
    db = sqlite3.connect(run / f"query_poses_{key.hexdigest()[:20]}.sqlite")
    failures = {}
    next_id = max(mapped) + 1
    ks = {}
    try:
        db.execute("CREATE TABLE IF NOT EXISTS poses (name TEXT PRIMARY KEY, result TEXT NOT NULL)")
        for index, name in enumerate(ordered):
            if name in refs:
                continue
            row = db.execute("SELECT result FROM poses WHERE name=?", (name,)).fetchone()
            if row:
                result = json.loads(row[0])
            else:
                result = solve(name)
                with db:
                    db.execute("INSERT INTO poses VALUES (?, ?)", (name, json.dumps(result)))
            if result["status"] != "localized":
                failures[name] = result["status"]
            else:
                w, h, *params = intrinsics[name]
                spec = intrinsics[name]
                if spec not in ks:
                    cid = max(cameras) + 1
                    ks[spec] = cid
                    cameras[cid] = colmap.Camera(cid, "PINHOLE", w, h, np.array(params))
                mapped[next_id] = colmap.Image(next_id, np.array(result["qvec"]), np.array(result["tvec"]),
                                               ks[spec], name, np.empty((0, 2)), np.empty(0, dtype=np.int64))
                next_id += 1
            if (index+1) % 100 == 0:
                print(f"Localized {len(mapped)-len(refs)} queries through capture frame {index+1}/{len(ordered)}; "
                      f"{len(failures)} failures", flush=True)
    finally:
        db.close()
    missing_train = sorted(set(train) & failures.keys())
    atomic_json(run / "localization.json", dict(mapped_keyframes=len(refs), localized=len(mapped)-len(refs),
                capture=len(ordered), missing=failures, missing_train=missing_train))
    if missing_train:
        raise ValueError(f"{len(missing_train)} selected keyframes lack poses. See {run/'localization.json'}; "
                         "cannot fit the matched training split.")
    sparse = stage / "sparse/0"
    atomic_write(sparse / "cameras.bin", lambda p: colmap.write_cameras_binary(cameras, p))
    atomic_write(sparse / "images.bin", lambda p: colmap.write_images_binary(mapped, p))
    return set(ordered)-failures.keys()
