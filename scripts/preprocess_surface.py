#!/usr/bin/env python3
"""Fit local LiDAR patches in PLY vertex order for optional 3DGS training."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.colmap_paths import colmap_model_dir
from utils.surface_geometry import estimate_patches, points_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--neighbors', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=50000)
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--max-support', type=float, default=None, help='Reject patches with larger radii, in scene units')
    args = parser.parse_args()
    if args.batch_size < 1 or args.neighbors < 6 or (args.max_support is not None and args.max_support <= 0):
        parser.error('batch-size must be positive and neighbors at least 6')
    from plyfile import PlyData
    ply = colmap_model_dir(args.source) / 'points3D.ply'
    vertices = PlyData.read(str(ply))['vertex']
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1).astype(np.float32)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    arrays = {name: np.lib.format.open_memmap(out / (name + '.npy'), mode='w+', dtype=np.float32, shape=shape)
              for name, shape in [('normals', (len(points), 3)), ('confidence', (len(points),)), ('support', (len(points),))]}
    for start, normals, confidence, support in estimate_patches(points, args.neighbors, args.batch_size, args.workers, args.max_support):
        end = start + len(normals)
        arrays['normals'][start:end] = normals
        arrays['confidence'][start:end] = confidence
        arrays['support'][start:end] = support
        print(f'Fitted {end:,}/{len(points):,} points', flush=True)
    for array in arrays.values():
        array.flush()
    report = {'version': 1, 'source_ply': str(ply.resolve()), 'points': len(points),
              'points_sha256_float32': points_digest(points), 'neighbors': args.neighbors,
              'max_support': args.max_support,
              'sampled_support_median_p95': np.percentile(arrays['support'][::1000], [50, 95]).tolist(),
              'trusted_fraction_at_0_6': float((arrays['confidence'][:] >= 0.6).mean())}
    (out / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
