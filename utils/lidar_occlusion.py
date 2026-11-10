"""Build fixed occluder surfels directly from the Gaussian initialization cloud.

No global surface completion: local planar patches have bounded radii and must
have supporting neighbors around their centers. Geometry is never optimized.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

VERSION = 1
DEFAULT_BUILD = dict(voxel_size=0.05, max_radius=0.1, neighbors=24, confidence=0.6)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_build(voxel_size, max_radius, neighbors, confidence):
    if (not np.isfinite([voxel_size, max_radius, confidence]).all() or
            voxel_size <= 0 or max_radius <= 0 or neighbors < 6 or not 0 < confidence <= 1):
        raise ValueError('Require positive voxel size/radius, >=6 neighbors, and confidence in (0,1]')


def fit_surfels(points, voxel_size=.05, max_radius=.1, neighbors=24,
                confidence=.6, batch_size=32768, workers=-1):
    """PCA normals, bounded footprints, and angular support; no hole filling."""
    from scipy.spatial import cKDTree
    validate_build(voxel_size, max_radius, neighbors, confidence)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite N x 3 input points')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    # Retain actual measured points rather than averaging across thin surfaces.
    cells = np.floor(points.astype(np.float64) / voxel_size).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    samples = points[np.sort(indices)]
    if len(samples) < neighbors:
        raise ValueError(f'Only {len(samples)} voxel representatives; need {neighbors} for local surfaces')
    tree = cKDTree(samples)
    centers, normals, radii = [], [], []
    for start in range(0, len(samples), batch_size):
        query = samples[start:start + batch_size]
        distances, ids = tree.query(query, k=neighbors, workers=workers)
        offsets = samples[ids].astype(np.float64) - query[:, None]
        centered = offsets - offsets.mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', centered, centered) / neighbors
        values, vectors = np.linalg.eigh(covariance)
        values = values.clip(0)
        planarity = np.clip(1 - values[:, 0] / np.maximum(values[:, 1], 1e-20) / .15, 0, 1)
        spread = np.clip(values[:, 1] / np.maximum(values[:, 2], 1e-20) / .15, 0, 1)
        radius = np.minimum(max_radius, 1.2 * np.median(distances[:, 1:7], axis=1))
        uv = offsets @ vectors[:, :, 1:]
        radial = np.linalg.norm(uv, axis=2)
        supported = (radial > 1e-8) & (distances <= 2.5 * radius[:, None])
        angles = np.sort(np.where(supported, np.arctan2(uv[:, :, 1], uv[:, :, 0]), np.inf), axis=1)
        count = supported.sum(axis=1)
        safe_count = np.maximum(count, 1)
        last = angles[np.arange(len(query)), safe_count-1]
        # Avoid inf-inf arithmetic for unused neighbor slots.
        finite_angles = np.where(np.isfinite(angles), angles, 0.)
        gaps = np.diff(finite_angles, axis=1)
        gaps[np.arange(neighbors-1)[None] >= count[:, None]-1] = 0
        wrap = np.where(count > 0, finite_angles[:, 0] + 2*np.pi - np.where(np.isfinite(last), last, 0.), np.inf)
        max_gap = np.maximum(gaps.max(axis=1), wrap)
        good = ((planarity * spread >= confidence) & (values[:, 1] > 1e-16) &
                (distances[:, -1] <= 3*max_radius) & (radius > 1e-8) &
                (count >= 5) & (max_gap < .9*np.pi))
        centers.append(query[good])
        normals.append(vectors[good, :, 0].astype(np.float32))
        radii.append(radius[good].astype(np.float32))
    result = dict(centers=np.concatenate(centers), normals=np.concatenate(normals), radii=np.concatenate(radii))
    if not len(result['centers']):
        raise ValueError('No supported planar LiDAR patches survived; inspect cloud density and geometry settings')
    return result, dict(input_points=len(points), voxel_representatives=len(samples),
                        surfels=len(result['centers']), trusted_fraction=len(result['centers'])/len(samples))


def save_geometry(path, centers, normals, radii, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz')
    try:
        np.savez(temporary, centers=np.asarray(centers, np.float32), normals=np.asarray(normals, np.float32),
                 radii=np.asarray(radii, np.float32), metadata=json.dumps(dict(metadata or {}, version=VERSION)))
        load_geometry(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_geometry(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].astype(np.float32) for key in ('centers', 'normals', 'radii')}
        metadata = json.loads(str(data['metadata']))
    n = len(arrays['radii'])
    if (metadata.get('version') != VERSION or not n or arrays['radii'].shape != (n,) or
            any(arrays[k].shape != (n, 3) for k in ('centers', 'normals')) or
            any(not np.isfinite(a).all() for a in arrays.values()) or np.any(arrays['radii'] <= 0) or
            not np.allclose(np.linalg.norm(arrays['normals'], axis=1), 1, atol=1e-4)):
        raise ValueError(f'Invalid LiDAR occlusion geometry: {path}')
    return arrays, metadata


def prepare_geometry(point_cloud, output, **options):
    from plyfile import PlyData
    point_cloud, output = Path(point_cloud).resolve(), Path(output).resolve()
    config = dict(DEFAULT_BUILD, **options)
    validate_build(**config)
    identity = dict(source_sha256=file_digest(point_cloud), build=config)
    if output.exists():
        _, metadata = load_geometry(output)
        if any(metadata.get(k) != v for k, v in identity.items()):
            raise ValueError(f'Existing occlusion geometry has different inputs/settings: {output}')
        print(f'Reusing LiDAR occlusion geometry: {output}', flush=True)
        return metadata
    vertex = PlyData.read(str(point_cloud))['vertex']
    points = np.stack([vertex[k] for k in ('x', 'y', 'z')], axis=1).astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    print(f'Fitting fixed occluders from {finite.sum():,} initialization points...', flush=True)
    arrays, stats = fit_surfels(points[finite], **config)
    metadata = dict(identity, **stats, source_point_cloud=str(point_cloud), rejected_nonfinite=int((~finite).sum()))
    save_geometry(output, **arrays, metadata=metadata)
    print(f'LiDAR occluders: {stats["surfels"]:,} patches ({stats["trusted_fraction"]:.1%} of representatives): {output}', flush=True)
    return metadata


def surfel_triangles(centers, normals, radii, sides=8):
    """Temporary bounded octagons, built on the same device as the points."""
    import torch
    import torch.nn.functional as F
    axis = F.one_hot(normals.abs().argmin(dim=1), 3).to(normals.dtype)
    tangent = F.normalize(torch.cross(normals, axis, dim=1), dim=1)
    bitangent = torch.cross(normals, tangent, dim=1)
    angles = torch.arange(sides, device=centers.device, dtype=centers.dtype)*(2*np.pi/sides)
    ring = centers[:, None] + radii[:, None, None]*(tangent[:, None]*angles.cos()[None, :, None] + bitangent[:, None]*angles.sin()[None, :, None])
    vertices = torch.cat((centers[:, None], ring), dim=1).reshape(-1, 3)
    indices = torch.arange(sides, device=centers.device, dtype=torch.int32)
    local = torch.stack((torch.zeros_like(indices), indices+1, (indices+1)%sides+1), dim=1)
    faces = (local[None] + torch.arange(len(centers), device=centers.device, dtype=torch.int32)[:, None, None]*(sides+1)).reshape(-1, 3)
    return vertices.contiguous(), faces.contiguous()
