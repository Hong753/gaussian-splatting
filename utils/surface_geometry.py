"""LiDAR patch estimation and identity checks for surface-aware 3DGS."""
import hashlib
import numpy as np


def points_digest(points, batch=100_000):
    digest = hashlib.sha256()
    for start in range(0, len(points), batch):
        digest.update(np.ascontiguousarray(points[start:start + batch], dtype='<f4').tobytes())
    return digest.hexdigest()


def estimate_patches(points, neighbors=24, batch=50_000, workers=-1, max_support=None):
    """Yield normal, confidence and support arrays; confidence rejects lines/corners."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=np.float32)
    if len(points) < neighbors or neighbors < 6 or not np.isfinite(points).all():
        raise ValueError('Point cloud must be finite and have at least the requested neighbors (minimum 6)')
    tree = cKDTree(points)
    for start in range(0, len(points), batch):
        query = points[start:start + batch]
        distances, indices = tree.query(query, k=neighbors, workers=workers)
        neighborhood = points[indices].astype(np.float64)
        neighborhood -= neighborhood.mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', neighborhood, neighborhood) / neighbors
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 0)
        # A plane has two well-supported directions and little normal scatter.
        planar = 1.0 - values[:, 0] / np.maximum(values[:, 1], 1e-20) / 0.15
        spread = values[:, 1] / np.maximum(values[:, 2], 1e-20) / 0.15
        confidence = np.clip(planar, 0, 1) * np.clip(spread, 0, 1)
        support = distances[:, -1]
        confidence[(support <= 1e-8) | (values[:, 1] <= 1e-16)] = 0
        if max_support is not None:
            confidence[support > max_support] = 0
        yield start, vectors[:, :, 0].astype(np.float32), confidence.astype(np.float32), support.astype(np.float32)
