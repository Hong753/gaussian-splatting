#!/usr/bin/env python3
"""Cache fixed LiDAR occlusion depths for all original COLMAP cameras."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--edge-pixels", type=int, default=1)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args(argv)
    if args.resolution == 0 or args.resolution < -1 or args.edge_pixels < 0:
        parser.error("Invalid image resolution or edge padding")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from tqdm import tqdm
    from scripts.view_comparisons import atomic_json, read_colmap_records
    from utils.occlusion_depth import LidarDepthCache, record_camera
    records = read_colmap_records(args.source, args.images)
    provider = LidarDepthCache(args.geometry, args.cache, args.edge_pixels)
    rows = []
    with torch.no_grad():
        for name in tqdm(sorted(records), desc="Caching LiDAR occlusion", unit="frame"):
            camera = record_camera(records[name], args.resolution)
            tiles = provider.tiles(camera, torch.device("cuda:0"))
            rows.append(dict(name=name, reliable_tile_fraction=float(torch.isfinite(tiles).float().mean())))
    report = dict(source=str(args.source.resolve()), geometry=str(args.geometry.resolve()), geometry_sha256=provider.geometry_sha256,
                  resolution=args.resolution, edge_pixels=args.edge_pixels, frames=rows, complete=True)
    # Cache filenames independently encode exact pose, intrinsics, size and edge padding.
    report_path = provider.directory / f"coverage_resolution_{args.resolution}_edge_{args.edge_pixels}.json"
    atomic_json(report_path, report)
    print(f"Cached {len(rows)} original views: {report_path}")
    print(f"Mean reliable tile coverage: {sum(r['reliable_tile_fraction'] for r in rows)/len(rows):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
