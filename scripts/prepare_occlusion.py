#!/usr/bin/env python3
"""Prepare standalone fixed occlusion geometry from a registered point cloud."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-cloud', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--voxel-size', type=float, default=.05)
    parser.add_argument('--max-radius', type=float, default=.1)
    parser.add_argument('--neighbors', type=int, default=24)
    parser.add_argument('--confidence', type=float, default=.6)
    args = parser.parse_args(argv)
    from utils.lidar_occlusion import prepare_geometry
    prepare_geometry(args.point_cloud, args.output, voxel_size=args.voxel_size,
                     max_radius=args.max_radius, neighbors=args.neighbors, confidence=args.confidence)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
