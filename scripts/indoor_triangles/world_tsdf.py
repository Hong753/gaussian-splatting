"""Block-sparse projective TSDF fusion and indexed surface-nets extraction."""
from itertools import product

import numpy as np


B = 8
OFFSETS = np.array(list(product(range(B), repeat=3)), np.int32)
CORNERS = list(product(range(2), repeat=3))
AXES = np.eye(3, dtype=int)


def integrate(volume, z, valid, k, rot, center, voxel, far):
    valid = valid & np.isfinite(z) & (z > .05) & (z < far)
    y, x = np.where(valid)
    if not len(x):
        return
    rays = np.column_stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones(len(x)))) @ rot
    distances = z[y, x, None]
    trunc = 3 * voxel
    blocks = np.concatenate([np.floor((rays*(distances+d)+center)/(B*voxel)).astype(np.int32)
                             for d in (-trunc, 0, trunc)])
    # Unique block coordinates as packed records avoid Python work per depth pixel.
    packed = np.ascontiguousarray(blocks).view(np.dtype((np.void, 12))).ravel()
    blocks = np.unique(packed).view(np.int32).reshape(-1, 3)
    for start in range(0, len(blocks), 64):
        keys = blocks[start:start+64]
        world = ((keys[:, None]*B+OFFSETS+.5)*voxel).reshape(-1, 3)
        camera = (world - center) @ rot.T
        depth = camera[:, 2]
        uv = camera[:, :2] / np.maximum(depth[:, None], .01) * k.diagonal()[:2] + k[:2, 2]
        u, v = np.rint(uv).astype(int).T
        inside = (depth > .05) & (u >= 0) & (u < z.shape[1]) & (v >= 0) & (v < z.shape[0])
        u, v = u.clip(0, z.shape[1]-1), v.clip(0, z.shape[0]-1)
        residual = z[v, u] - depth
        observed = (inside & valid[v, u] & (np.abs(residual) <= trunc)).reshape(len(keys), -1)
        sdf = np.where(observed, residual.reshape(observed.shape)/trunc, 0).astype(np.float32)
        for i in np.where(observed.any(axis=1))[0]:
            key = tuple(keys[i])
            if key not in volume:
                volume[key] = np.zeros((2, B**3), np.float32)
            volume[key][0] += sdf[i]
            volume[key][1] += observed[i]


def halo(blocks, key, default, dtype):
    """Fetch this block and the one-sample positive halo for cells crossing its boundary."""
    out = np.full((B+1,)*3, default, dtype=dtype)
    for d in CORNERS:
        neighbor = blocks.get(tuple(np.add(key, d)))
        if neighbor is None:
            continue
        src = tuple(slice(0, 1 if i else B) for i in d)
        dst = tuple(slice(B, B+1) if i else slice(0, B) for i in d)
        out[dst] = neighbor[src]
    return out


def sub(block, offset):
    return block[tuple(slice(d, d+B) for d in offset)]


def surface(volume, voxel, min_views=2):
    field = {}
    for key, (total, count) in volume.items():
        value = np.full(B**3, np.nan, np.float32)
        np.divide(total, count, out=value, where=count >= min_views)
        if np.isfinite(value).any():
            field[key] = value.reshape((B,)*3)
    vertices, ids, count_vertices = [], {}, 0
    for key in field:
        f = halo(field, key, np.nan, np.float32)
        corners = np.stack([sub(f, c) for c in CORNERS])
        active = np.isfinite(corners).all(axis=0) & (corners.min(0) < 0) & (corners.max(0) >= 0)
        if not active.any():
            continue
        cells = np.column_stack(np.where(active))
        tsdf = corners[:, active]
        positions = np.zeros_like(cells, dtype=np.float32)
        counts = np.zeros(len(cells), np.int32)
        for j, corner in enumerate(CORNERS):
            for axis in range(3):
                if corner[axis]:
                    continue
                end = tuple(np.add(corner, AXES[axis]))
                left, right = tsdf[j], tsdf[CORNERS.index(end)]
                crossing = (left < 0) != (right < 0)
                t = left[crossing]/(left[crossing]-right[crossing])
                positions[crossing] += corner + t[:, None]*AXES[axis]
                counts[crossing] += 1
        positions = (positions/counts[:, None]+cells+np.asarray(key)*B+.5)*voxel
        vertices.append(positions.astype(np.float32))
        local = np.full((B,)*3, -1, np.int32)
        local[active] = np.arange(count_vertices, count_vertices+len(cells))
        count_vertices += len(cells)
        ids[key] = local
    faces = []
    for key in ids:
        f = halo(field, key, np.nan, np.float32)
        block = halo(ids, key, -1, np.int32)
        for axis in range(3):
            u, v = AXES[(axis+1)%3], AXES[(axis+2)%3]
            a, b = sub(f, u+v), sub(f, u+v+AXES[axis])
            crossing = np.isfinite(a) & np.isfinite(b) & ((a < 0) != (b < 0))
            quads = np.stack([sub(block, off)[crossing] for off in ((0, 0, 0), u, u+v, v)], axis=1)
            reverse = a[crossing] >= 0
            quads[reverse] = quads[reverse, ::-1]
            quads = quads[(quads >= 0).all(axis=1)]
            if len(quads):
                faces.extend((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    if not faces:
        raise ValueError("No connected surface with sufficient overlapping valid depth observations")
    return np.concatenate(vertices), np.concatenate(faces)
