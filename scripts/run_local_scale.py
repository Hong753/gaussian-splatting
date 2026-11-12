#!/usr/bin/env python3
"""Fixed LiDAR 3DGS + AA: local KNN hard ceilings and overlap-oriented soft targets."""
import os
import math
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_fixed_lidar, run_soft_surface_long as periodic
from scripts.run_occlusion import invoke
from utils.local_scale import validate


def parser():
    p = run_fixed_lidar.parser()
    p.description = __doc__
    p.set_defaults(gpu='3', max_scale_ratio=1., max_scale_world=.015, scale_control='hybrid')
    p.add_argument('--neighbors', type=int, default=10)
    p.add_argument('--support-sigma', type=float, default=3.)
    p.add_argument('--overlap-factor', type=float, default=2.)
    p.add_argument('--audit-samples', type=int, default=2048)
    p.add_argument('--audit-pair-budget', type=int, default=2000000)
    p.add_argument('--cache-workers', type=int, default=8)
    p.add_argument('--ceiling-multiplier',type=float,default=1.)
    p.add_argument('--target-multiplier',type=float,default=1.)
    p.add_argument('--reuse-cache',type=Path,help='Reuse original 1x KNN bounds; rebuild overlap audit at the new ceilings')
    p.add_argument('--training-schedule',type=json.loads,help='Explicit JSON cap/LR policy; tune_local_scale.py constructs it')
    return p


def build_plan(args):
    if args.scale_control != 'hybrid':
        raise ValueError('Local scale experiment requires hybrid scale control')
    validate(args.neighbors, args.support_sigma, args.overlap_factor, args.max_scale_ratio,
             args.max_scale_world, args.audit_samples, args.audit_pair_budget, args.cache_workers)
    cache = args.local_scale_cache or args.model/'local_scale_cache'
    common = SimpleNamespace(**vars(args))
    common.local_scale_cache = cache
    plan = run_fixed_lidar.build_plan(common)
    if plan['initial_points'] <= args.neighbors:
        raise ValueError('Local-scale initialization needs more points than the requested neighbor count')
    plan.update(local_scale_cache=str(cache.expanduser().resolve()), neighbors=args.neighbors,
        support_sigma=args.support_sigma, overlap_factor=args.overlap_factor,
        audit_samples=args.audit_samples, audit_pair_budget=args.audit_pair_budget,
        scale_rule='ceiling=min(ratio*sigma_init, dK/support_sigma, optional_world_cap); '
                   'target=min(ceiling, dK/(support_sigma*overlap_factor))')
    ceiling = getattr(args,'ceiling_multiplier',1.)
    target = getattr(args,'target_multiplier',1.)
    if not all(math.isfinite(v) and v > 0 for v in (ceiling,target)):
        raise ValueError('Local scale multipliers must be positive and finite')
    if ceiling != 1. or target != 1.:
        plan.update(ceiling_multiplier=ceiling,target_multiplier=target,
            effective_max_scale_world=args.max_scale_world*ceiling,
            scale_rule='ceiling=ceiling_multiplier*base_ceiling; target=min(ceiling,target_multiplier*base_target); initialization unchanged when relaxing')
        plan['stages'][0]['argv'] += ['--local_scale_ceiling_multiplier',str(ceiling),
                                      '--local_scale_target_multiplier',str(target)]
    if getattr(args,'reuse_cache',None) is not None:
        plan['reuse_cache'] = str(Path(args.reuse_cache).expanduser().resolve())
    if getattr(args,'training_schedule',None) is not None:
        from utils.local_scale_schedule import validate as validate_schedule
        policy = validate_schedule(args.training_schedule, ceiling, args.iterations)
        plan['training_schedule'] = policy
        plan['stages'][0]['argv'] += ['--local_scale_schedule',json.dumps(policy,sort_keys=True)]
    return plan


def prepare(argv):
    args = parser().parse_args(argv)
    if args.baseline:
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
    return periodic.prepare_experiment(args, build_plan(args), 'local_scale', [args.keyframes, args.point_cloud])


def execute_plan(args, plan):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Activate the CUDA-enabled 3dgs environment')
    model = Path(plan['model'])
    model.mkdir(parents=True, exist_ok=True)
    smoke = ['scripts/smoke_local_scale.py', '--gpu', args.gpu,
            '--ceiling-multiplier',str(getattr(args,'ceiling_multiplier',1.)),
            '--target-multiplier',str(getattr(args,'target_multiplier',1.)),
            '--output', str(model/'local_scale_cuda_smoke.json')]
    if getattr(args,'training_schedule',None) is not None:
        smoke += ['--training-schedule',json.dumps(args.training_schedule,sort_keys=True)]
    invoke(smoke)
    from utils.local_scale import prepare as prepare_cache
    data = prepare_cache(args.point_cloud, plan['local_scale_cache'], neighbors=args.neighbors,
        support_sigma=args.support_sigma, overlap_factor=args.overlap_factor,
        ratio=args.max_scale_ratio, world=args.max_scale_world,
        samples=args.audit_samples, pair_budget=args.audit_pair_budget, workers=args.cache_workers,
        ceiling_multiplier=getattr(args,'ceiling_multiplier',1.),target_multiplier=getattr(args,'target_multiplier',1.),
        reuse_cache=getattr(args,'reuse_cache',None))
    from utils.checkpoint_snapshots import atomic_json
    atomic_json(model/'local_scale_initialization.json', data)
    return run_fixed_lidar.execute_plan(args, plan, smoke=False)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'prepare':
        return prepare(argv[1:])
    return periodic.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
