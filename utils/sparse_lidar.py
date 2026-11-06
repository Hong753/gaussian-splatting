"""Storage and loss for sparse, projected LiDAR depth supervision."""
import hashlib
import json
from pathlib import Path
import numpy as np


def frame_filename(name):
    return hashlib.sha256(name.encode('utf-8')).hexdigest() + '.npz'


def check_dataset(directory, source, cameras, resolution=None):
    root = Path(directory)
    report = json.loads((root / 'metadata.json').read_text())
    if not report.get('complete') or Path(report['source']).resolve() != Path(source).resolve():
        raise ValueError('LiDAR depth set is incomplete or belongs to a different source')
    if resolution is not None and report.get('resolution') != resolution:
        raise ValueError('LiDAR depth preprocessing resolution differs from training resolution')
    selected = set(report['images'])
    missing = [c.image_name for c in cameras if c.image_name not in selected or
               not (root / frame_filename(c.image_name)).is_file()]
    if missing:
        raise ValueError(f'Missing LiDAR depth for {len(missing)} training cameras, first: {missing[0]}')
    return report


def load_frame(directory, camera):
    path = Path(directory) / frame_filename(camera.image_name)
    with np.load(path, allow_pickle=False) as data:
        if str(data['image']) != camera.image_name:
            raise ValueError(f'Wrong image in sparse depth file {path}')
        if tuple(data['shape']) != (camera.image_height, camera.image_width):
            raise ValueError(f'Sparse depth resolution does not match training camera: {path}')
        u, v, z = data['u'].astype(np.int64), data['v'].astype(np.int64), data['z'].astype(np.float32)
    if u.ndim != 1 or v.shape != u.shape or z.shape != u.shape or not len(z):
        raise ValueError(f'Invalid or empty sparse depth: {path}')
    if (np.any(u < 0) or np.any(u >= camera.image_width) or np.any(v < 0) or
            np.any(v >= camera.image_height) or not np.isfinite(z).all() or np.any(z <= 0)):
        raise ValueError(f'Out-of-range sparse depth samples: {path}')
    return u, v, z


def lidar_loss(inverse_depth, u, v, z, alpha_mask=None):
    import torch
    import torch.nn.functional as F
    device = inverse_depth.device
    u = torch.as_tensor(u, device=device, dtype=torch.long)
    v = torch.as_tensor(v, device=device, dtype=torch.long)
    z = torch.as_tensor(z, device=device, dtype=inverse_depth.dtype)
    predicted = inverse_depth[0, v, u]
    if alpha_mask is not None:
        valid = alpha_mask.to(device)[0, v, u] > 0
        if not valid.any():
            return predicted.sum() * 0
        predicted, z = predicted[valid], z[valid]
    # Dimensionless, robust inverse-depth discrepancy; targets are camera z.
    return F.smooth_l1_loss(predicted * z, torch.ones_like(predicted), beta=0.1)
