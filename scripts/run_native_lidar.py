#!/usr/bin/env python3
"""Native 3DGS + AA from LiDAR, with reduced XYZ LR and native densification.

Scales are unconstrained. Opacity resets stay disabled; native pruning is optional.
"""
import math
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
    p.add_argument('--pruning', action='store_true', help='Enable native opacity/size pruning during densification')
    p.add_argument('--percent-dense', type=float, default=None,
                   help='Clone/split sigma boundary as a fraction of scene extent (native default: 0.01)')
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
    command += ['--position_lr_max_steps', str(args.iterations),
        '--disable_opacity_resets', *([] if pruning else ['--disable_pruning']),
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
    return plan


def prepare(argv):
    args = parser().parse_args(argv)
    if args.baseline:
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
    plan = build_plan(args)
    return periodic.prepare_experiment(args, plan, 'native_lidar', [args.keyframes, args.point_cloud])


def execute_plan(args, plan):
    # Direct in-process CUDA check, then the shared native trainer invocation.
    invoke(['scripts/smoke_native_lidar.py', '--gpu', args.gpu,
            '--percent-dense', str(plan['densification'].get('percent_dense', .01)),
            '--output', str(Path(plan['model'])/'native_lidar_cuda_smoke.json')])
    return fixed.execute_plan(args, plan, smoke=False)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return prepare(argv[1:]) if argv and argv[0] == 'prepare' else periodic.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
