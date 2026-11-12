#!/usr/bin/env python3
"""3DGS + AA: fixed LiDAR centers, hard/soft scale limits, constant population.

100k iterations by default; every 10k publish a checkpoint for the independent
all-frame evaluator. Uses only native 3DGS (no mesh, DA3, Proxy-GS or nvdiffrast).
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_soft_surface_long as periodic
from scripts.run_occlusion import invoke


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--keyframes', type=Path)
    p.add_argument('--point-cloud', type=Path, help='Original colored LiDAR PLY; defaults to BASELINE/input.ply')
    p.add_argument('--gpu', default='0')
    p.add_argument('--iterations', type=int, default=100000)
    p.add_argument('--checkpoint-every', type=int, default=10000)
    p.add_argument('--max-scale-ratio', type=float, default=2., help='Principal sigma reference / initial nearest-neighbor sigma')
    p.add_argument('--max-scale-world', type=float, default=0., help='Optional absolute reference sigma in scene units (0: off)')
    p.add_argument('--scale-control', choices=('clamp', 'regularize', 'hybrid'), default='clamp',
                   help='Hard cap, soft penalty, or local soft target with a hard ceiling')
    p.add_argument('--scale-regularization-weight', type=float, default=.01)
    p.add_argument('--local-scale-cache', type=Path, help='Precomputed local bounds, required by hybrid mode')
    p.add_argument('--resolution', type=int, default=-1)
    p.add_argument('--expected-frames', type=int, default=5009)
    p.add_argument('--report-max-views', type=int, default=64)
    p.add_argument('--eval-memory-fraction', type=float, default=.20)
    p.add_argument('--poll-seconds', type=float, default=15.)
    p.add_argument('--dry-run', action='store_true')
    return p


def build_plan(args):
    from utils.colmap_paths import colmap_model_dir
    from plyfile import PlyData
    source, model = args.source.expanduser().resolve(), args.model.expanduser().resolve()
    control = getattr(args, 'scale_control', 'clamp')
    weight = getattr(args, 'scale_regularization_weight', .01)
    colmap_model_dir(source)
    if (args.iterations < 1 or args.checkpoint_every < 1 or args.expected_frames < 1 or
            args.report_max_views < 1 or args.resolution == 0 or args.resolution < -1 or
            not math.isfinite(args.max_scale_ratio) or args.max_scale_ratio <= 0 or
            not math.isfinite(args.max_scale_world) or args.max_scale_world < 0 or
            control not in ('clamp', 'regularize', 'hybrid') or not math.isfinite(weight) or weight <= 0 or
            not 0 < args.eval_memory_fraction <= 1 or not 0 < args.poll_seconds <= 60):
        raise ValueError('Invalid scale bounds, iteration/frame counts, resolution or evaluator settings')
    for path in (args.keyframes, args.point_cloud):
        if path is None or not path.expanduser().is_file():
            raise FileNotFoundError(f'Need original LiDAR PLY and training manifest: {path}')
    if (any(model.glob('point_cloud/iteration_*/point_cloud.ply')) or (model/'cfg_args').exists() or
            (args.baseline and args.baseline.resolve() == model)):
        raise ValueError('Choose a fresh model directory; existing trained weights are never reused')
    # Binary LiDAR vertex arrays are memory mapped here, not copied into RAM.
    ply = PlyData.read(str(args.point_cloud.expanduser()), mmap='r')
    vertices = ply['vertex']
    fields = set(vertices.data.dtype.names)
    if not len(vertices.data) or not {'x','y','z','red','green','blue','nx','ny','nz'} <= fields:
        raise ValueError('Need original colored LiDAR PLY with x/y/z, RGB and normal fields (BASELINE/input.ply)')
    if 'f_dc_0' in fields or 'scale_0' in fields:
        raise ValueError('Pass original input.ply, not a learned Gaussian checkpoint')
    steps = sorted(set(range(args.checkpoint_every, args.iterations+1, args.checkpoint_every)) | {args.iterations})
    command = ['train.py', '-s', str(source), '-m', str(model),
        '--initial_point_cloud', str(args.point_cloud.expanduser().resolve()),
        '--keyframe_manifest', str(model/'keyframes.json'),
        '--iterations', str(args.iterations), '--resolution', str(args.resolution),
        '--data_device', 'cpu', '--disable_viewer', '--antialiasing',
        '--fixed_lidar', '--max_scale_ratio', str(args.max_scale_ratio),
        '--max_scale_world', str(args.max_scale_world),
        '--position_lr_init', '0', '--position_lr_final', '0', '--densify_until_iter', '0',
        '--occlusion_mode', 'off', '--occlusion_inference_mode', 'off',
        '--test_iterations', '-1', '--save_iterations', *map(str, steps),
        '--checkpoint_iterations', *map(str, steps), '--publish_checkpoint_snapshots']
    policy = {}
    if control in ('regularize', 'hybrid'):
        command += ['--scale_control', control, '--scale_regularization_weight', str(weight)]
        policy = dict(scale_control=control, scale_regularization_weight=weight,
                      scale_limit_semantics='Soft penalty target, not a strict training-time maximum')
    if control == 'hybrid':
        if not getattr(args, 'local_scale_cache', None):
            raise ValueError('Hybrid mode requires --local-scale-cache; run_local_scale.py can prepare it')
        command += ['--local_scale_cache', str(args.local_scale_cache.expanduser().resolve())]
        policy['scale_limit_semantics'] = 'Local soft target plus an independent hard ceiling after every Adam step'
    return dict(source=str(source), model=str(model), gpu=args.gpu, initial_points=len(vertices.data),
                max_scale_ratio=args.max_scale_ratio, max_scale_world=args.max_scale_world,
                positions='fixed throughout', population='constant', antialiasing=True,
                hard_culling=False, soft_surface=False, opacity_resets=False,
                stages=[dict(name='train', argv=command)], **policy)


def prepare(argv):
    args = parser().parse_args(argv)
    if args.baseline:
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
    plan = build_plan(args)
    return periodic.prepare_experiment(args, plan, 'fixed_lidar', [args.keyframes, args.point_cloud])


def execute_plan(args, plan, *, smoke=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Activate the CUDA-enabled 3dgs environment')
    model = Path(plan['model'])
    model.mkdir(parents=True, exist_ok=True)
    # Exercise the actual native CUDA rasterizer and optimizer path on a tiny
    # fixed cloud before loading the large scene. Failures stay direct tracebacks.
    if smoke and getattr(args, 'scale_control', 'clamp') == 'hybrid':
        invoke(['scripts/smoke_local_scale.py', '--gpu', args.gpu,
                '--output', str(model/'local_scale_cuda_smoke.json')])
    elif smoke:
        invoke(['scripts/smoke_fixed_lidar.py', '--gpu', args.gpu,
            '--scale-control', getattr(args, 'scale_control', 'clamp'),
            '--output', str(model/'fixed_lidar_cuda_smoke.json')])
    if args.keyframes.resolve() != model/'keyframes.json':
        shutil.copyfile(args.keyframes, model/'keyframes.json')
    print(json.dumps({k: v for k, v in plan.items() if k != 'stages'}, indent=2), flush=True)
    invoke(plan['stages'][0]['argv'])
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'prepare':
        return prepare(argv[1:])
    # train/watch reuse the atomic snapshot queue and evaluator; train dispatches
    # to this module from the saved experiment kind, independent of CLI defaults.
    return periodic.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
