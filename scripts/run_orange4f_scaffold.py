#!/usr/bin/env python3
"""Prepare a Scaffold-GS variant against the existing immutable orange4f suite."""
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
from utils.checkpoint_snapshots import digest


def build_plan(args):
    from scripts.run_fixed_lidar import checkpoint_steps
    fixed = args.variant == 'fixed'
    command = ['models/train_scaffold.py', '--model_type', 'scaffold',
        '-s', str(args.source), '-m', str(args.model), '--initial_point_cloud', str(args.point_cloud),
        '--keyframe_manifest', str(args.keyframes), '--resolution', '1', '--data_device', 'cpu',
        '--scaffold_voxel_size', str(args.voxel_size), '--iterations', str(args.iterations),
        '--position_lr_init', str(0. if fixed else args.anchor_lr_init),
        '--position_lr_final', str(0. if fixed else args.anchor_lr_final),
        '--position_lr_max_steps', str(args.iterations), '--publish_snapshots',
        '--save_iterations', *map(str, checkpoint_steps(args)), '--test_iterations', *map(str, checkpoint_steps(args)),
        '--debug_preview_every', '500']
    if fixed:
        command += ['--freeze_anchors', '--disable_anchor_growth']
    return dict(model=str(args.model), source=str(args.source), model_type='scaffold',
        antialiasing=False, freeze_anchors=fixed, anchor_growth=not fixed, pruning=True,
        voxel_size=args.voxel_size, learned_offsets=True, appearance_dim=0,
        hard_culling=False, stages=[dict(argv=command)])


def execute_plan(args, plan):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    import simple_knn._C as knn
    if not torch.cuda.is_available() or not callable(getattr(knn, 'distCUDA2', None)):
        raise RuntimeError('Use the CUDA-enabled 3dgs environment with the existing simple_knn extension')
    print(f'Reusing simple_knn: {knn.__file__}', flush=True)
    if list(Path(args.model).glob('point_cloud/iteration_*')):
        raise FileExistsError('Saved iterations already exist; use a new --run-version, or resume the trainer explicitly')
    invoke(['scripts/smoke_scaffold.py'])
    print(json.dumps(plan, indent=2), flush=True)
    invoke(plan['stages'][0]['argv'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--variant', choices=['learnable', 'fixed'], required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--voxel-size', type=float, default=.02,
                   help='Anchor spacing in input scene units; this is not a Gaussian scale cap')
    p.add_argument('--anchor-lr-init', type=float, default=.00016)
    p.add_argument('--anchor-lr-final', type=float, default=.0000016)
    p.add_argument('--run-version', default='v1')
    args = p.parse_args(argv)
    if (not args.run_version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_version)
            or not all(math.isfinite(x) and x > 0 for x in (args.voxel_size, args.anchor_lr_init, args.anchor_lr_final))):
        p.error('Need positive finite voxel/LR values and a simple run version')
    root = args.suite.expanduser().resolve()
    suite = json.loads((root/'suite.json').read_text())
    for name, expected in suite['input_fingerprints'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Shared input changed: {name}')
    args.source = Path(suite['source'])
    args.model = root/f'scaffold_{args.variant}_anchors_{args.run_version}'
    args.point_cloud = Path(suite['cloud']['cached'])
    args.keyframes = args.model/'keyframes.json'
    args.iterations, args.checkpoint_every = 30000, 10000
    args.expected_frames, args.report_max_views = suite['frames'], 32
    args.eval_memory_fraction, args.poll_seconds, args.dry_run = .25, 15., False
    plan = build_plan(args)
    if (periodic.directory(args.model)/'plan.json').exists():
        previous = periodic.load_plan(args.model)
        if previous['plan'] != plan or previous['gpu'] != args.gpu:
            raise ValueError('Prepared Scaffold settings changed; use another --run-version')
        for name, expected in previous['input_fingerprints'].items():
            if digest(Path(name)) != expected:
                raise ValueError(f'Prepared Scaffold input changed: {name}')
        print(f'Reusing Scaffold plan: {args.model}', flush=True)
        return 0
    args.model.mkdir(exist_ok=False)
    shutil.copyfile(suite['selection'], args.keyframes)
    # Avoid copying the full dense cloud once per model; Scene recognizes this link.
    (args.model/'input.ply').symlink_to(args.point_cloud)
    result = periodic.prepare_experiment(args, plan, 'scaffold',
        [args.keyframes, *map(Path, suite['input_fingerprints'])])
    # Keep the original four-run manifest immutable. Add independent dashboard links.
    from utils.checkpoint_snapshots import atomic_write
    index = root/'index.html'
    link = f'{args.model.name}/debug_previews/index.html'
    content = index.read_text()
    if link not in content:
        content += f'<p>GPU {args.gpu}: <a href="{link}">{args.model.name}</a></p>'
        atomic_write(index, lambda path: path.write_text(content))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
