"""Propose other patches on the audited floor level using LiDAR and camera geometry."""

from collections import Counter

import numpy as np
from plyfile import PlyData


def floor_tiles(cloud, center, normal, axes, *, spacing=0.75):
    """Count returns near the floor in world-space tiles, reading the PLY once."""
    counts = Counter()
    coordinates = PlyData.read(cloud, mmap="r")["vertex"].data
    for start in range(0, len(coordinates), 262144):
        chunk = coordinates[start : start + 262144]
        points = np.column_stack([chunk[key] for key in ("x", "y", "z")]) - center
        on_level = points[np.abs(points @ normal) <= 0.10]
        cells = np.floor((on_level @ axes.T) / spacing).astype(np.int64)
        keys, numbers = np.unique(cells, axis=0, return_counts=True)
        counts.update({tuple(key): int(count) for key, count in zip(keys, numbers)})
    tiles = [(key, count) for key, count in counts.items() if count >= 30]
    xy = (np.array([key for key, _ in tiles], dtype=float).reshape(-1, 2) + 0.5) * spacing
    return xy, [count for _, count in tiles]


def alternate_patches(cloud, center, normal, original, original_world, images, cameras, allowed, far_ratio):
    """Rank nearby-reference crops whose floor tile is in frame from distant cameras."""
    normal = normal * np.sign((original.C - center) @ normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = np.cross(normal, axis)
    tangent /= np.linalg.norm(tangent)
    axes = np.stack([tangent, np.cross(normal, tangent)])
    tiles, counts = floor_tiles(cloud, center, normal, axes)
    original_bounds = (original_world[[0, 0, -1, -1], [0, -1, -1, 0]] - center) @ axes.T
    original_range = float(np.linalg.norm(original.C - original_world.mean(axis=(0, 1))))
    centers = np.array([view.C for view in images])
    rotations = np.array([view.R for view in images])
    intrinsics = np.array([cameras[view.camera_id].params for view in images])
    sizes = np.array([[cameras[view.camera_id].width, cameras[view.camera_id].height] for view in images])
    training = np.array([view.name in allowed for view in images])
    same_side = (centers - center) @ normal > 0.2
    proposed = []
    # A one-meter world square, separated from the original cropped floor even at its edges.
    offsets = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    for tile, point_count in zip(tiles, counts):
        corners = tile + offsets
        if not (
            (corners.max(axis=0) < original_bounds.min(axis=0) - 0.15)
            | (corners.min(axis=0) > original_bounds.max(axis=0) + 0.15)
        ).any():
            continue
        world = center + corners @ axes
        ranges = np.linalg.norm(centers - world.mean(axis=0), axis=1)
        xyz = np.einsum("nij,nkj->nki", rotations, world[None] - centers[:, None])
        depth = xyz[..., 2]
        uv = xyz[..., :2] / np.maximum(depth[..., None], 1e-6)
        uv = uv * intrinsics[:, None, :2] + intrinsics[:, None, 2:] - 0.5
        low, high = uv.min(axis=1), uv.max(axis=1)
        short_side = (high - low).min(axis=1)
        visible = same_side & (depth > 0.05).all(axis=1) & (low >= 0).all(axis=1)
        visible &= (high <= sizes - 1).all(axis=1)
        references = np.flatnonzero(visible & training & (short_side >= 40))
        # Prefer views with more pixels across the crop, then geographically separate alternatives.
        picked = []
        for index in sorted(references, key=lambda i: (-short_side[i], ranges[i], images[i].name)):
            if picked and min(np.linalg.norm(centers[index] - centers[i]) for i in picked) < 0.20:
                continue
            cutoff = far_ratio * max(original_range, ranges[index])
            far = np.flatnonzero(visible & (ranges >= cutoff) & (short_side >= 12))
            if len(far) < 2 or np.linalg.norm(np.ptp(centers[far], axis=0)) < 0.10:
                continue
            baselines = np.linalg.norm(centers - centers[index], axis=1)
            near = np.flatnonzero(visible & (ranges < cutoff) & (short_side >= 16) & (baselines >= 0.02))
            if not len(near):
                continue
            near = sorted(near, key=lambda i: (baselines[i], images[i].name))
            left, top = np.floor(low[index]).astype(int)
            right, bottom = np.ceil(high[index]).astype(int) + 1
            rect = (int(left), int(top), int(right - left), int(bottom - top))
            score = (len(picked), -min(len(far), 30), -float(short_side[index]), -point_count, images[index].name)
            proposed.append((score, images[index], rect, [images[i] for i in near]))
            picked.append(index)
            if len(picked) == 3:
                break
    return sorted(proposed, key=lambda item: item[0]), original_range, axes
