#!/usr/bin/env python3
"""3DGS + AA with fixed measured coordinates, adaptive activation and size pruning."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_fixed_lidar, run_soft_surface_long as periodic
from scripts.run_occlusion import invoke


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--keyframes', type=Path)
    p.add_argument('--point-cloud', type=Path)
    p.add_argument('--cache', type=Path, help='Reusable measured-coordinate kNN cache; default MODEL/discrete_cache')
    p.add_argument('--gpu', default='2')
    p.add_argument('--iterations', type=int, default=100000)
    p.add_argument('--checkpoint-every', type=int, default=10000)
    p.add_argument('--prune-sigma', type=float, default=.05, help='Delete when any sigma axis exceeds this scene-unit size')
    p.add_argument('--seed-sigma-ratio', type=float, default=.5, help='Initial/child sigma <= ratio * prune-sigma')
    p.add_argument('--initial-fraction', type=float, default=.5)
    p.add_argument('--init-voxel', type=float, default=.05, help='Preserve at least one initial center per occupied voxel')
    p.add_argument('--neighbors', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cache-workers', type=int, default=8)
    p.add_argument('--growth-radius', type=float, default=.1)
    p.add_argument('--max-new', type=int, default=50000)
    p.add_argument('--min-observations', type=int, default=1)
    p.add_argument('--densification-interval', type=int, default=100)
    p.add_argument('--densify-from', type=int, default=500)
    p.add_argument('--gradient-threshold', type=float, default=.0002)
    p.add_argument('--resolution', type=int, default=-1)
    p.add_argument('--expected-frames', type=int, default=5009)
    p.add_argument('--report-max-views', type=int, default=64)
    p.add_argument('--eval-memory-fraction', type=float, default=.2)
    p.add_argument('--poll-seconds', type=float, default=15.)
    p.add_argument('--dry-run', action='store_true')
    return p


def build_plan(args):
    # Reuse the strict source/PLY/manifest/fresh-directory validation and common
    # checkpoint protocol, then replace the fixed-population policy explicitly.
    common = SimpleNamespace(**vars(args), max_scale_ratio=2., max_scale_world=0.,
                             scale_control='clamp', scale_regularization_weight=.01)
    plan = run_fixed_lidar.build_plan(common)
    if (not all(math.isfinite(x) and x > 0 for x in (args.prune_sigma, args.init_voxel, args.growth_radius)) or
            not 0 < args.seed_sigma_ratio < 1 or not 0 < args.initial_fraction <= 1 or
            not 3 <= args.neighbors <= 128 or args.seed < 0 or args.cache_workers < 1 or
            args.max_new < 1 or args.min_observations < 1 or args.densification_interval < 1 or
            not 0 <= args.densify_from < args.iterations or
            not math.isfinite(args.gradient_threshold) or args.gradient_threshold <= 0):
        raise ValueError('Invalid discrete LiDAR initialization, growth or size-pruning settings')
    cache = args.cache or Path(plan['model'])/'discrete_cache'
    command = plan['stages'][0]['argv']
    command.remove('--fixed_lidar')
    for flag in ('--max_scale_ratio', '--max_scale_world', '--densify_until_iter'):
        index = command.index(flag)
        del command[index:index+2]
    command += ['--discrete_lidar', '--discrete_lidar_cache', str(cache.expanduser().resolve()),
        '--discrete_lidar_prune_sigma', str(args.prune_sigma),
        '--discrete_lidar_seed_sigma_ratio', str(args.seed_sigma_ratio),
        '--discrete_lidar_radius', str(args.growth_radius),
        '--discrete_lidar_max_new', str(args.max_new),
        '--discrete_lidar_min_observations', str(args.min_observations),
        '--densification_interval', str(args.densification_interval),
        '--densify_from_iter', str(args.densify_from),
        '--densify_until_iter', str(args.iterations), '--densify_grad_threshold', str(args.gradient_threshold)]
    del plan['max_scale_ratio'], plan['max_scale_world']
    plan['input_points'] = plan.pop('initial_points')
    plan.update(positions='fixed original LiDAR coordinates only', population='adaptive unique candidate activation + scale pruning',
                opacity_pruning=False, prune_sigma=args.prune_sigma, seed_sigma_ratio=args.seed_sigma_ratio,
                initial_fraction=args.initial_fraction, init_voxel=args.init_voxel,
                growth_through_iteration=args.iterations, cache=str(cache.expanduser().resolve()),
                scale_limit_semantics='Size pruning at population updates; no per-step scale clamp or penalty')
    return plan


def prepare(argv):
    args = parser().parse_args(argv)
    if args.baseline:
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
    return periodic.prepare_experiment(args, build_plan(args), 'discrete_lidar', [args.keyframes, args.point_cloud])


def execute_plan(args, plan):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Activate the CUDA-enabled 3dgs environment')
    model = Path(plan['model'])
    model.mkdir(parents=True, exist_ok=True)
    invoke(['scripts/smoke_discrete_lidar.py', '--gpu', args.gpu,
            '--output', str(model/'discrete_lidar_cuda_smoke.json')])
    from utils.discrete_lidar_cache import prepare as prepare_cache
    metadata = prepare_cache(args.point_cloud, plan['cache'], fraction=args.initial_fraction,
        voxel=args.init_voxel, neighbors=args.neighbors, seed=args.seed, workers=args.cache_workers)
    from utils.checkpoint_snapshots import atomic_json
    atomic_json(model/'discrete_lidar_initialization.json', metadata)
    if args.keyframes.resolve() != model/'keyframes.json':
        shutil.copyfile(args.keyframes, model/'keyframes.json')
    print(json.dumps({k: v for k, v in plan.items() if k != 'stages'}, indent=2), flush=True)
    invoke(plan['stages'][0]['argv'])
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'prepare':
        return prepare(argv[1:])
    return periodic.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
