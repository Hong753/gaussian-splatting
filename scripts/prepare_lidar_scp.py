#!/usr/bin/env python3
"""Cache a measured-LiDAR local surface field for the fifth regional ablation."""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.lidar_scp import prepare_cache


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-cloud',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--voxel-size',type=float)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args(argv)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prepare_cache(args.point_cloud,args.output,args.voxel_size,args.workers)
    return 0


if __name__=='__main__': raise SystemExit(main())
