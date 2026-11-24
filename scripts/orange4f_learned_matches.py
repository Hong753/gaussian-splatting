"""SuperPoint and indoor SuperGlue correspondences for Spirula v2026.11.12 sfm map."""

from functools import lru_cache
from pathlib import Path
import struct

import cv2
import numpy as np

from scripts.keyframes import natural_key
from scripts import region_superglue as sg
from utils.checkpoint_snapshots import atomic_write, atomic_json


def native_features(path, image, feat):
    # Mapping consumes keypoints, scale of measurement, score and color. The
    # true float descriptors are stored separately in SuperGlue's feature cache.
    width, height = map(int, feat["original_size"])
    xy = feat["keypoints_colmap"]
    pixels = cv2.imread(str(image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if pixels is None or pixels.shape[:2] != (height, width):
        raise ValueError(f"Invalid RGB dimensions: {image}")
    u, v = np.floor(xy).astype(int).T
    colors = pixels[np.clip(v, 0, height - 1), np.clip(u, 0, width - 1), ::-1]
    keys = np.zeros((len(xy), 4), dtype="<f4")
    keys[:, :2] = xy

    def write(tmp):
        with tmp.open("wb") as f:
            f.write(struct.pack("<4sIiiIII", b"VKFT", 6, width, height, len(keys), 0, 0))
            f.write(keys.tobytes())
            f.write(b"\x01" + colors.tobytes())
            f.write(struct.pack("<dIiiB", 0.0, 0, *feat["image_size"], 1))
            f.write(feat["scores"].astype("<f4").tobytes() + b"\x01")

    atomic_write(path, write)


def pairs(vectors):
    count = len(vectors)
    planned = set()
    for i in range(count):
        for step in (1, 2, 3, 4, 6, 8, 12, 16, 24):
            if i + step < count:
                planned.add((i, i + step))
    # Nonlocal proposals are chosen solely by RGB appearance, even when an
    # earlier SIFT model happened to connect the whole sequence.
    for start in range(0, count, 128):
        similarity = vectors[start : start + 128] @ vectors.T
        for i, row in enumerate(similarity, start):
            row[max(0, i - 24) : i + 25] = -np.inf
            for j in np.argsort(row)[-4:]:
                if np.isfinite(row[j]):
                    planned.add(tuple(sorted((i, int(j)))))
    return sorted(planned)


def verified(a, b, matches):
    if len(matches) < 15:
        return 1, matches[:0]
    # RANSAC in a shared 1600-long-edge coordinate system, including portrait inputs.
    xy = [
        feat["keypoints_colmap"][matches[:, i]] * (1600 / max(feat["original_size"]))
        for i, feat in enumerate((a, b))
    ]
    _, f = cv2.findFundamentalMat(*xy, cv2.USAC_MAGSAC, 2.0, 0.999, 5000)
    _, h = cv2.findHomography(*xy, cv2.USAC_MAGSAC, 2.0, maxIters=5000, confidence=0.999)

    def mask(m):
        return np.zeros(len(matches), dtype=bool) if m is None else m.ravel().astype(bool)

    fm, hm = mask(f), mask(h)
    if hm.sum() >= 15 and hm.sum() >= 0.8 * fm.sum():
        return 3, matches[hm]
    if fm.sum() >= 15:
        return 2, matches[fm]
    return 1, matches[:0]


def build(work, photos, records, repo, device):
    import torch

    names = sorted(records, key=natural_key)
    stems = [str(Path(n).with_suffix("")) for n in names]
    if len(set(stems)) != len(names):
        raise ValueError("Spirula requires distinct extension-free image names")
    config = dict(
        repository=str(repo),
        names=names,
        max_keypoints=2048,
        max_size=1600,
        nms_radius=4,
        keypoint_threshold=0.005,
    )
    if not (work / "images").exists():
        (work / "images").symlink_to(photos, target_is_directory=True)
    sg.extract(work, config, device)
    counts, groups, camera_ids, vectors = [], {}, [], []
    for name, stem in zip(names, stems):
        feat = sg.read_feature(work, name)
        rec = records[name]
        if tuple(feat["original_size"]) != (rec["width"], rec["height"]):
            raise ValueError(f"Calibration/RGB mismatch: {name}")
        fpath = work / "features" / (stem + ".bin")
        if not fpath.exists():
            native_features(fpath, photos / name, feat)
        counts.append(len(feat["scores"]))
        k = tuple(rec[n] for n in ("width", "height", "fx", "fy", "cx", "cy"))
        camera_ids.append(groups.setdefault(k, len(groups) + 1))
        desc = np.average(feat["descriptors"], axis=1, weights=feat["scores"])
        vectors.append(desc / max(np.linalg.norm(desc), 1e-8))
    planned = pairs(np.asarray(vectors))
    manifest = work / "matches.bin"
    if manifest.is_file() and (work / "pairing.json").is_file():
        return
    model = (
        sg.model_class(repo, "superglue")(
            {"weights": "indoor", "sinkhorn_iterations": 100, "match_threshold": 0.2}
        )
        .eval()
        .to(device)
    )
    feature = lru_cache(maxsize=25)(lambda i: sg.read_feature(work, names[i]))

    def write(tmp):
        kept = total = 0
        with tmp.open("wb+") as out:
            out.write(struct.pack("<4sII", b"VKMT", 5, len(names)))
            for name, count in zip(stems, counts):
                data = name.encode()
                out.write(struct.pack("<I", len(data)) + data + struct.pack("<I", count))
            offset = out.tell()
            out.write(struct.pack("<I", 0))
            for j, (a, b) in enumerate(planned, 1):
                cache = sg.cache_path(work, "verified", [names[a], names[b]])
                if not cache.exists():
                    fa, fb = feature(a), feature(b)
                    matches, _ = sg.match_one(model, fa, fb, device)
                    kind, inliers = verified(fa, fb, matches)
                    sg.save_npz(cache, kind=kind, inliers=inliers)
                with np.load(cache, allow_pickle=False) as z:
                    inliers, kind = z["inliers"], int(z["kind"])
                if len(inliers):
                    out.write(struct.pack("<IIiI", a, b, kind, len(inliers)))
                    out.write(inliers.astype("<u4", copy=False).tobytes())
                    kept += 1
                    total += len(inliers)
                if j % 100 == 0 or j == len(planned):
                    print(
                        f"SuperGlue pairs {j:,}/{len(planned):,}, {kept:,} geometrically verified", flush=True
                    )
            out.write(struct.pack("<I", len(groups)))
            for (w, h, fx, fy, cx, cy), cid in groups.items():
                out.write(
                    struct.pack("<IiiiBdI4d", cid, w, h, 1, 1, max(1.0, max(w, h) / 1600), 4, fx, fy, cx, cy)
                )
            out.write(struct.pack("<I", len(names)) + np.asarray(camera_ids, dtype="<u4").tobytes())
            out.write(bytes(len(groups) + 1))
            out.seek(offset)
            out.write(struct.pack("<I", kept))
        atomic_json(
            work / "pairing.json",
            dict(
                images=len(names),
                proposed=len(planned),
                verified=kept,
                inlier_observations=total,
                weights="indoor",
                max_keypoints=2048,
            ),
        )

    with torch.inference_mode():
        atomic_write(manifest, write)
