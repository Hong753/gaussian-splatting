#!/usr/bin/env python3
"""Native 3DGS + AA from LiDAR, with reduced XYZ LR and native densification.

Scales are unconstrained. Native pruning and opacity resets are explicit options.
"""
import math
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_fixed_lidar as fixed
from scripts import run_soft_surface_long as periodic
from scripts.run_occlusion import invoke


def parser():
    p = fixed.parser()
    p.description = __doc__
    p.set_defaults(gpu='1', iterations=30000, scale_control='off')
    p.add_argument('--position-lr-divisor', type=float, default=50.)
    p.add_argument('--freeze-lidar-positions', action='store_true',
                   help='Freeze original XYZ and disable cloning/splitting while retaining optional pruning')
    p.add_argument('--pruning', action='store_true', help='Enable native opacity/size pruning during densification')
    p.add_argument('--opacity-resets', action='store_true',
                   help='Enable native 3000-step opacity resets during the active densification/pruning phase')
    p.add_argument('--percent-dense', type=float, default=None,
                   help='Clone/split sigma boundary as a fraction of scene extent (native default: 0.01)')
    for name in ('feature-lr', 'opacity-lr', 'rotation-lr', 'scaling-lr', 'densify-grad-threshold', 'prune-world-scale-ratio', 'prune-screen-size'):
        p.add_argument('--'+name, type=float, default=None)
    for name in ('densify-from-iter', 'densify-until-iter', 'densification-interval', 'prune-size-from-iter'):
        p.add_argument('--'+name, type=int, default=None)
    p.add_argument('--preserve-densification-radii', action='store_true',
                   help='Keep measured radius history through native growth until the pruning check')
    p.add_argument('--lidar-scp-cache',type=Path)
    p.add_argument('--lidar-scp-policy',default='',help='JSON overrides for the LiDAR adaptation of GSPrior')
    return p


def build_plan(args):
    # Missing on previously saved no-pruning plans: keep those reproducible.
    pruning = bool(getattr(args, 'pruning', False))
    percent_dense = getattr(args, 'percent_dense', None)
    if percent_dense is not None and (not math.isfinite(percent_dense) or not 0 < percent_dense <= 1):
        raise ValueError('percent-dense must be finite and in (0, 1]')
    if not math.isfinite(args.position_lr_divisor) or args.position_lr_divisor < 1:
        raise ValueError('Position LR divisor must be finite and >= 1')
    if args.scale_control != 'off' or args.distance_weighting or args.local_scale_cache:
        raise ValueError('This ablation requires unconstrained scales and no distance/local-scale policies')
    for name in ('feature_lr', 'opacity_lr', 'rotation_lr', 'scaling_lr', 'densify_grad_threshold', 'prune_world_scale_ratio', 'prune_screen_size'):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value < 0 or
                (name in ('feature_lr', 'opacity_lr', 'rotation_lr', 'scaling_lr', 'densify_grad_threshold') and value == 0)):
            raise ValueError(f'Invalid {name}: {value}')
    for name in ('densify_from_iter', 'densify_until_iter', 'densification_interval', 'prune_size_from_iter'):
        value = getattr(args, name, None)
        if value is not None and (value < 0 or
                (name in ('densify_until_iter', 'densification_interval') and value == 0)):
            raise ValueError(f'Invalid {name}: {value}')
    start = getattr(args, 'densify_from_iter', None)
    end = getattr(args, 'densify_until_iter', None)
    if (500 if start is None else start) >= (15000 if end is None else end):
        raise ValueError('Densification start must precede its end')
    if end is not None and end > args.iterations:
        raise ValueError('Explicit densification end must not exceed training iterations')
    plan = fixed.build_plan(args)  # Shared input/manifest validation, checkpoint and preview settings.
    command = plan['stages'][0]['argv']
    command.remove('--fixed_lidar')
    for name in ('--max_scale_ratio', '--max_scale_world'):
        i = command.index(name)
        del command[i:i+2]
    initial, final = .00016/args.position_lr_divisor, .0000016/args.position_lr_divisor
    for name, value in (('--position_lr_init', initial), ('--position_lr_final', final),
                        ('--densify_until_iter', 15000)):
        command[command.index(name)+1] = str(value)
    resets = bool(getattr(args, 'opacity_resets', False))
    command += ['--position_lr_max_steps', str(args.iterations),
        *(['--opacity_reset_interval', '3000'] if resets else ['--disable_opacity_resets']),
        *([] if pruning else ['--disable_pruning']),
        '--densify_from_iter', '500', '--densification_interval', '100',
        '--densify_grad_threshold', '0.0002']
    for name in ('max_scale_ratio', 'max_scale_world'):
        plan.pop(name)
    population = ('native cloning/splitting with opacity/size pruning' if pruning else
                  'native cloning/splitting; no opacity/size pruning')
    plan.update(positions='optimized from step 1', population=population,
        pruning=pruning, position_lr=dict(init=initial, final=final, divisor=args.position_lr_divisor,
            max_steps=args.iterations, scene_extent_scaling=True, warmup_steps=0),
        densification=dict(start=500, until=15000, interval=100, gradient_threshold=.0002))
    # Only explicit overrides alter the command/metadata, so old saved plans
    # reproduce exactly when training is launched after a code update.
    if percent_dense is not None:
        command += ['--percent_dense', str(percent_dense)]
        plan['densification']['percent_dense'] = percent_dense
    for field, key in (('densify_from_iter', 'start'), ('densify_until_iter', 'until'),
                       ('densification_interval', 'interval'), ('densify_grad_threshold', 'gradient_threshold')):
        value = getattr(args, field, None)
        if value is not None:
            command[command.index('--'+field)+1] = str(value)
            plan['densification'][key] = value
    for field in ('feature_lr', 'opacity_lr', 'rotation_lr', 'scaling_lr', 'prune_world_scale_ratio', 'prune_screen_size', 'prune_size_from_iter'):
        value = getattr(args, field, None)
        if value is not None:
            command += ['--'+field, str(value)]
            plan[field] = value
    if resets:
        plan['opacity_resets'] = True
        plan['opacity_reset_schedule'] = dict(interval=3000,
            active_until_exclusive=plan['densification']['until'],
            steps=list(range(3000,min(args.iterations+1,plan['densification']['until']),3000)))
    if getattr(args, 'preserve_densification_radii', False):
        command.append('--preserve_densification_radii')
        plan['preserve_densification_radii'] = True
    if getattr(args, 'freeze_lidar_positions', False):
        command.append('--freeze_lidar_positions')
        for field in ('--position_lr_init', '--position_lr_final'):
            command[command.index(field)+1] = '0'
        plan.update(positions='fixed throughout; surviving original LiDAR coordinates',
                    freeze_lidar_positions=True,
                    population='pruning only; no cloning/splitting' if pruning else 'constant',
                    position_lr=dict(init=0., final=0., max_steps=args.iterations, warmup_steps=0))
        plan['densification']['enabled'] = False
    if getattr(args,'lidar_scp_cache',None):
        from utils.lidar_scp import policy
        from utils.checkpoint_snapshots import digest
        if getattr(args,'freeze_lidar_positions',False):
            raise ValueError('LiDAR SCP pull requires learnable Gaussian centers')
        cache=Path(args.lidar_scp_cache).resolve()
        meta=json.loads((cache/'metadata.json').read_text())
        if meta['source_sha256']!=digest(args.point_cloud) or meta['field_sha256']!=digest(cache/'field.npz'):
            raise ValueError('LiDAR prior cache does not match this initialization')
        cfg=policy(getattr(args,'lidar_scp_policy',''))
        if cfg['start']>=args.iterations: raise ValueError('LiDAR SCP must start before training ends')
        command+=['--lidar_scp_cache',str(cache),'--lidar_scp_policy',json.dumps(cfg,sort_keys=True)]
        plan['lidar_scp']=dict(cache=str(cache),policy=cfg,metadata=meta)
    return plan


def prepare(argv):
    args = parser().parse_args(argv)
    if args.baseline:
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
    plan = build_plan(args)
    inputs=[args.keyframes,args.point_cloud,*fixed.mask_inputs(args)]
    if args.lidar_scp_cache: inputs += [args.lidar_scp_cache/name for name in ('metadata.json','field.npz')]
    return periodic.prepare_experiment(args, plan, 'native_lidar', inputs)


def execute_plan(args, plan):
    # Direct in-process CUDA check, then the shared native trainer invocation.
    smoke = ['scripts/smoke_native_lidar.py', '--gpu', args.gpu,
            '--percent-dense', str(plan['densification'].get('percent_dense', .01)),
            '--position-lr-divisor', str(args.position_lr_divisor),
            '--scaling-lr', str(plan.get('scaling_lr', .005)), '--iterations', str(args.iterations),
            '--output', str(Path(plan['model'])/'native_lidar_cuda_smoke.json')]
    if plan.get('preserve_densification_radii'):
        smoke.append('--preserve-densification-radii')
    if plan.get('freeze_lidar_positions'):
        smoke.append('--freeze-lidar-positions')
    if getattr(args,'loss_mask_dir',None):
        smoke.append('--region-loss')
    for field in ('feature_lr', 'opacity_lr', 'rotation_lr'):
        if field in plan: smoke += ['--'+field.replace('_','-'),str(plan[field])]
    if plan['opacity_resets']: smoke.append('--opacity-resets')
    if not plan['antialiasing']: smoke.append('--no-antialiasing')
    invoke(smoke)
    if plan.get('lidar_scp'):
        invoke(['scripts/smoke_lidar_scp.py','--gpu',args.gpu,
                '--output',str(Path(plan['model'])/'lidar_scp_cuda_smoke.json')])
    return fixed.execute_plan(args, plan, smoke=False)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return prepare(argv[1:]) if argv and argv[0] == 'prepare' else periodic.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
