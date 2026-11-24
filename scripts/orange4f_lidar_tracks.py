"""Read the original Orange4F scan coordinates and their stored image measurements."""

from collections import Counter
import struct

import numpy as np
from plyfile import PlyData

from scripts.view_sparse_observations import (
    FEATURE,
    PlyIndex,
    PointIndex,
    mapped,
    read_frames,
)
from utils.colmap_paths import colmap_model_dir
from utils.read_write_model import read_images_text


def scan_inputs(original, explicit=None):
    model = colmap_model_dir(original)
    if explicit is not None:
        explicit = explicit.expanduser().absolute()
    if explicit is not None and explicit.suffix.lower() != ".ply":
        raise ValueError(f"Expected a registered PLY override, got {explicit}")
    points = None
    for ext in ("bin", "txt"):
        path = model / f"points3D.{ext}"
        if not path.is_file():
            continue
        with path.open("rb") as f:
            populated = (
                struct.unpack("<Q", f.read(8))[0]
                if ext == "bin"
                else any(line.strip() and not line.lstrip().startswith(b"#") for line in f)
            )
        if populated:
            points = path
            break
    cloud = explicit or next(
        (
            p
            for p in (model / "points3D.ply", points, original / "input.ply")
            if p is not None and p.is_file()
        ),
        None,
    )
    if cloud is None or not cloud.is_file():
        raise FileNotFoundError(
            f"Missing original Orange4F scan under {model} (points3D.bin/.txt/.ply) or at {cloud}"
        )
    return cloud, points


def scan_xyz(path):
    if path.suffix.lower() != ".ply":
        return PointIndex(path).xyz
    vertices = PlyData.read(path, mmap="r")["vertex"].data
    if not set("xyz") <= set(vertices.dtype.names) or "f_dc_0" in vertices.dtype.names:
        raise ValueError(f"Expected an XYZ scan in {path}")
    return vertices


def stored_pixels(model):
    if (model / "images.bin").is_file() and (model / "cameras.bin").is_file():
        with mapped(model / "images.bin") as buf:
            for frame in read_frames(buf).values():
                data = np.frombuffer(buf, FEATURE, frame.count, frame.offset)
                yield frame.id, frame.name, data["xy"].copy(), data["id"].copy()
                del data
    else:
        for frame in read_images_text(model / "images.txt").values():
            yield frame.id, frame.name, frame.xys, frame.point3D_ids


def linked_rows(points, iid, ids):
    # Merge forward and reverse references. Never use a COLMAP ID as a row number.
    rows = np.full(len(ids), -1, dtype=np.int64)
    valid = ids >= 0
    rows[valid] = points.rows(ids[valid].astype(np.uint64))
    back = points.for_image(iid)
    in_range = (back["feature"] >= 0) & (back["feature"] < len(ids))
    feat = back["feature"][in_range]
    rev_rows = back["row"][in_range].astype(np.int64)
    unique, first, counts = np.unique(feat, return_index=True, return_counts=True)
    row = rev_rows[first]
    conflict = (counts != 1) | ((ids[unique] >= 0) & (rows[unique] != row))
    rows[unique] = np.where(conflict, -1, row)
    return rows, int(conflict.sum() + (~in_range).sum())


def collect(original, cloud, points_path, reference):
    points = PointIndex(points_path) if points_path is not None else None
    # A separate scan PLY and COLMAP model may contain different points.
    # Reuse the sparse-observation viewer's exact coordinate join in that case.
    match_scan = cloud.suffix.lower() == ".ply" and points is not None
    if match_scan:
        vertices = scan_xyz(cloud)
        ply = PlyIndex(cloud, True)
        scan_row = np.empty(len(points.ids), dtype=np.int64)
        for start in range(0, len(scan_row), 262144):
            section = slice(start, start + 262144)
            scan_row[section] = ply.match(points.xyz[section])
        del ply
        count = len(vertices)
    else:
        count = len(points.xyz) if points is not None else len(scan_xyz(cloud))
    if not count:
        raise ValueError(f"Empty scan: {cloud}")
    stats = Counter(raw_reverse_links=len(points.reverse) if points is not None else 0)
    names = sorted(reference)
    image_indices = {name: i for i, name in enumerate(names)}
    batches = []
    if points is not None:
        for iid, name, xy, ids in stored_pixels(colmap_model_dir(original)):
            stats["raw_forward_links"] += int(np.count_nonzero(ids >= 0))
            rows, rejected = linked_rows(points, iid, ids)
            stats["conflicting_or_invalid_reverse_entries"] += rejected
            idx = np.flatnonzero(rows >= 0)
            stats["resolved_pixel_links"] += len(idx)
            if match_scan:
                idx = idx[scan_row[rows[idx]] >= 0]
            xyz = points.xyz[rows[idx]]
            size = np.array([reference[name]["width"], reference[name]["height"]])
            idx = idx[
                np.isfinite(xyz).all(1)
                & np.isfinite(xy[idx]).all(1)
                & ((xy[idx] >= 0) & (xy[idx] < size)).all(1)
            ]
            # A point associated to two different pixels in one image is ambiguous.
            _, first, multiplicity = np.unique(rows[idx], return_index=True, return_counts=True)
            idx = idx[first[multiplicity == 1]]
            if not len(idx):
                continue
            stats["usable_pixel_links"] += len(idx)
            stats["images_with_links"] += 1
            # Balance camera influence and cover the sensor. Keep one actual
            # measurement from each cell on a 16 by 12 grid.
            idx = np.random.default_rng(image_indices[name]).permutation(idx)
            u, v = (xy[idx] / size * [16, 12]).astype(int).T
            _, first = np.unique(u + 16 * v, return_index=True)
            idx = idx[first]
            rows = rows[idx]
            xyz = (
                points.xyz[rows]
                if not match_scan
                else np.column_stack([vertices[k][scan_row[rows]] for k in "xyz"])
            )
            batches.append(
                (
                    np.full(len(idx), image_indices[name], dtype=np.int32),
                    points.ids[rows],
                    xyz,
                    xy[idx],
                )
            )
    empty = (
        np.empty(0, np.int32),
        np.empty(0, np.uint64),
        np.empty((0, 3)),
        np.empty((0, 2)),
    )
    image, pid, xyz, xy = map(np.concatenate, zip(*batches)) if batches else empty
    stats.update(scan_points=count, sampled_pixel_links=len(pid))
    return dict(names=np.asarray(names), image=image, pid=pid, xyz=xyz, xy=xy), dict(stats)
