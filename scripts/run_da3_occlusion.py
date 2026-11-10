#!/usr/bin/env python3
"""Train native 3DGS+AA with calibrated DA3 culling; evaluate without depth inputs."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_occlusion
from scripts.view_comparisons import atomic_json


def parser():
    p = run_occlusion.parser(depth_maps=True)
    p.description = __doc__
    p.add_argument('--depth-dir', type=Path, required=True, help='Complete prepare_da3_occlusion.py output')
    p.set_defaults(gpu='3', position_warmup_steps=7000, position_ramp_steps=2000, position_lr_scale=.02)
    return p


def build_plan(args):
    if args.baseline and not args.keyframes:
        args.keyframes = args.baseline/'keyframes.json'
    report_path = args.depth_dir.expanduser().resolve()/'metadata.json'
    if not report_path.is_file():
        raise RuntimeError(f'DA3 preprocessing has not completed: missing {report_path}. '
                           'Fix the preprocessing error and rerun scripts/prepare_da3_occlusion.py '
                           'before starting training (or use scripts/run_da3_ablation.sh).')
    report = json.loads(report_path.read_text())
    if (not report.get('complete') or Path(report['source']).resolve() != args.source.resolve() or
            report['resolution'] != args.resolution or report['edge_pixels'] != 1):
        raise ValueError('Aligned DA3 cache is incomplete or incompatible with this training run')
    if not args.keyframes:
        raise ValueError('Pass the exact keyframe manifest used to prepare the DA3 cache')
    selected = json.loads(args.keyframes.read_text())['selected_images']
    if set(selected) != set(report['images']):
        raise ValueError('DA3 cache and training keyframe manifest differ')
    plan = run_occlusion.build_plan(args)
    stages = []
    for stage in plan['stages']:
        name, command = stage['name'], stage['argv'][:]
        if name.startswith('trained_off') or name.startswith('baseline_on'):
            continue
        if name == 'train':
            for flag in ('--occlusion_voxel_size', '--occlusion_max_radius'):
                index = command.index(flag)
                del command[index:index+2]
            command[command.index('--occlusion_mode')+1] = 'off'
            command += ['--train_depth_occlusion_dir', str(args.depth_dir.resolve())]
        if name.endswith('_video'):
            command = [c.replace('LiDAR culling', 'DA3 culling during training') for c in command]
        stages.append(dict(name=name.replace('trained_on', 'trained_da3_render_off'), argv=command))
    plan.update(stages=stages, training_occlusion='LiDAR-calibrated DA3 depths',
                render_occlusion='off', depth_directory=str(args.depth_dir.resolve()))
    return plan


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Activate the native CUDA-enabled 3dgs conda environment')
    model = Path(plan['model'])
    model.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        run_occlusion.invoke(['scripts/smoke_da3_occlusion.py', '--gpu', args.gpu])
        if args.keyframes.resolve() != model/'keyframes.json':
            shutil.copyfile(args.keyframes, model/'keyframes.json')
    status = dict(plan, completed=[], active=None, complete=False)
    path = model/'da3_occlusion_workflow.json'
    if args.evaluate_only and path.exists():
        previous = json.loads(path.read_text())
        commands = {s['name']: s['argv'] for s in previous['stages']}
        status['completed'] = [s['name'] for s in plan['stages'] if s['name'] in previous['completed']
                               and commands.get(s['name']) == s['argv']]
    for stage in plan['stages']:
        if stage['name'] in status['completed']:
            continue
        status['active'] = stage['name']
        atomic_json(path, status)
        print(f"DA3 occlusion workflow: {stage['name']}", flush=True)
        run_occlusion.invoke(stage['argv'])
        status['completed'].append(stage['name'])
    status.update(active=None, complete=True)
    atomic_json(path, status)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
