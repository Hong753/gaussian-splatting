#!/usr/bin/env python3
"""Measure sampled Gaussian-center to nearest input LiDAR-point distances."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData


def xyz(path):
    vertices = PlyData.read(str(path))['vertex']
    return np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lidar', type=Path, required=True)
    parser.add_argument('--gaussians', type=Path, nargs='+', required=True)
    parser.add_argument('--samples', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    lidar = xyz(args.lidar)
    if not np.isfinite(lidar).all():
        raise ValueError('LiDAR point cloud contains non-finite points')
    print(f'Building LiDAR tree for {len(lidar):,} points', flush=True)
    tree = cKDTree(lidar)
    results = []
    for path in args.gaussians:
        centers = xyz(path)
        if not np.isfinite(centers).all() or len(centers) == 0:
            raise ValueError(f'Empty or non-finite Gaussian centers: {path}')
        rng = np.random.default_rng(args.seed)
        if len(centers) > args.samples:
            centers = centers[rng.choice(len(centers), args.samples, replace=False)]
        distances, _ = tree.query(centers, workers=args.workers)
        result = {'gaussians': str(path.resolve()), 'sample_count': len(centers),
                  'median': float(np.median(distances)), 'mean': float(np.mean(distances)),
                  'p95': float(np.percentile(distances, 95))}
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'lidar': str(args.lidar.resolve()), 'seed': args.seed,
        'distance_units': 'scene units', 'results': results}, indent=2) + '\n')

if __name__ == '__main__':
    main()
