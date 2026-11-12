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
    p.add_argument('--pose-policy',type=json.loads,help='Optional bounded train-camera refinement policy')
    p.add_argument('--lidar-offset-policy',type=json.loads,help='Bounded displacement from each original LiDAR row')
    p.add_argument('--pose-culling-policy',type=json.loads,help='Pose-aware primary LiDAR mesh cutoffs')
    p.add_argument('--scene-geometry',type=Path)
    p.add_argument('--lidar-activation-policy',type=json.loads)
    p.add_argument('--lidar-activation-cache',type=Path)
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
    if getattr(args,'pose_policy',None) is not None:
        from utils.pose_refinement import validate as validate_pose
        policy=validate_pose(args.pose_policy,args.iterations)
        plan['pose_refinement']=policy
        plan['stages'][0]['argv'] += ['--pose_refinement',json.dumps(policy,sort_keys=True)]
    if getattr(args,'lidar_offset_policy',None) is not None:
        from utils.lidar_offsets import validate as validate_offsets
        if getattr(args,'pose_policy',None) is None:
            raise ValueError('LiDAR offset experiment requires alternating pose refinement')
        policy=validate_offsets(args.lidar_offset_policy,args.iterations)
        plan['lidar_offsets']=policy
        plan['positions']='bounded displacement from immutable original LiDAR rows; no population changes'
        plan['stages'][0]['argv'] += ['--lidar_offsets',json.dumps(policy,sort_keys=True)]
    if getattr(args, 'pose_culling_policy', None) is not None:
        from utils.pose_culling import validate as validate_culling, geometry_inputs
        if not getattr(args, 'lidar_offset_policy', None) or not getattr(args, 'scene_geometry', None):
            raise ValueError('Pose culling requires bounded LiDAR offsets and the scene mesh')
        policy=validate_culling(args.pose_culling_policy)
        geometry_inputs(args.scene_geometry, args.point_cloud, args.source)
        command=plan['stages'][0]['argv']
        command[command.index('--occlusion_mode')+1]='mesh'
        command += ['--occlusion_backend','ray','--occlusion_geometry',str(args.scene_geometry.resolve()),
                    '--occlusion_margin','0','--occlusion_edge_pixels',str(policy['edge_pixels']),
                    '--pose_culling',json.dumps(policy,sort_keys=True)]
        plan.update(hard_culling=True,pose_culling=policy,scene_geometry=str(args.scene_geometry.resolve()),
            culling_protocol='primary measured mesh; corrected camera; per-pixel CUDA clipping; no-contribution Adam rows frozen',
            inference_protocol='culling off; original/refined poses; additional matched culling-on previews')
    if getattr(args,'lidar_activation_policy',None) is not None:
        from utils.lidar_activation import validate as validate_activation
        if not plan.get('pose_culling') or not getattr(args,'lidar_activation_cache',None):
            raise ValueError('Adaptive LiDAR activation requires pose-aware culling and a candidate cache')
        policy=validate_activation(args.lidar_activation_policy,args.iterations)
        plan.update(lidar_activation=policy,lidar_activation_cache=str(args.lidar_activation_cache.resolve()),
            positions='unique discrete original LiDAR anchors with bounded offsets; reserved rows activate adaptively; no pruning',
            population='fixed-size optimizer arrays; inactive rows frozen; only active Gaussians exported')
        plan['stages'][0]['argv'] += ['--lidar_activation',json.dumps(policy,sort_keys=True),
                                    '--lidar_activation_cache',plan['lidar_activation_cache']]
    return plan


def prepare(argv):
    args = parser().parse_args(argv)
    if args.baseline:
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
    inputs=[args.keyframes,args.point_cloud]
    if args.pose_culling_policy:
        from utils.pose_culling import geometry_inputs
        inputs += geometry_inputs(args.scene_geometry,args.point_cloud,args.source)
    return periodic.prepare_experiment(args, build_plan(args), 'local_scale', inputs)


def execute_plan(args, plan):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Activate the CUDA-enabled 3dgs environment')
    model = Path(plan['model'])
    model.mkdir(parents=True, exist_ok=True)
    if getattr(args,'pose_policy',None) is not None:
        invoke(['scripts/smoke_pose_refinement.py','--gpu',args.gpu,
                '--output',str(model/'pose_refinement_cuda_smoke.json')])
    smoke = ['scripts/smoke_local_scale.py', '--gpu', args.gpu,
            '--ceiling-multiplier',str(getattr(args,'ceiling_multiplier',1.)),
            '--target-multiplier',str(getattr(args,'target_multiplier',1.)),
            '--output', str(model/'local_scale_cuda_smoke.json')]
    if getattr(args,'training_schedule',None) is not None:
        smoke += ['--training-schedule',json.dumps(args.training_schedule,sort_keys=True)]
    invoke(smoke)
    if getattr(args,'lidar_offset_policy',None) is not None:
        invoke(['scripts/smoke_lidar_offsets.py','--gpu',args.gpu,
                '--ceiling-multiplier',str(getattr(args,'ceiling_multiplier',1.)),
                '--target-multiplier',str(getattr(args,'target_multiplier',1.)),
                '--output',str(model/'lidar_offsets_cuda_smoke.json')]+
                (['--pose-culling'] if getattr(args,'pose_culling_policy',None) is not None else [])+
                (['--lidar-activation'] if getattr(args,'lidar_activation_policy',None) is not None else []))
    if getattr(args,'pose_culling_policy',None) is not None:
        invoke(['scripts/smoke_ray_occlusion.py','--gpu',args.gpu,
                '--output',str(model/'ray_occlusion_cuda_smoke.json')])
        invoke(['scripts/smoke_pose_culling.py','--gpu',args.gpu,
                '--output',str(model/'pose_culling_cuda_smoke.json')])
    from utils.local_scale import prepare as prepare_cache
    data = prepare_cache(args.point_cloud, plan['local_scale_cache'], neighbors=args.neighbors,
        support_sigma=args.support_sigma, overlap_factor=args.overlap_factor,
        ratio=args.max_scale_ratio, world=args.max_scale_world,
        samples=args.audit_samples, pair_budget=args.audit_pair_budget, workers=args.cache_workers,
        ceiling_multiplier=getattr(args,'ceiling_multiplier',1.),target_multiplier=getattr(args,'target_multiplier',1.),
        reuse_cache=getattr(args,'reuse_cache',None),
        position_padding=(getattr(args,'lidar_offset_policy',None) or {}).get('max_world',0.))
    from utils.checkpoint_snapshots import atomic_json
    atomic_json(model/'local_scale_initialization.json', data)
    if getattr(args,'lidar_activation_policy',None) is not None:
        from utils.discrete_lidar_cache import prepare as prepare_candidates
        cfg=args.lidar_activation_policy
        data=prepare_candidates(args.point_cloud,plan['lidar_activation_cache'],workers=args.cache_workers,
            **{k:cfg[k] for k in ('fraction','voxel','neighbors','seed')})
        atomic_json(model/'lidar_activation_initialization.json',data)
    return run_fixed_lidar.execute_plan(args, plan, smoke=False)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'prepare':
        return prepare(argv[1:])
    return periodic.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
