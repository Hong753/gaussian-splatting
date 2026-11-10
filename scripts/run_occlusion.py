#!/usr/bin/env python3
"""Standalone 3DGS + AA + fixed LiDAR occlusion, in one Python process."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--keyframes', type=Path, help='Reuse this manifest; otherwise select sharp stride-3 keyframes')
    p.add_argument('--baseline', type=Path, help='Optional native 3DGS+AA checkpoint for four-way scores and both comparison videos')
    p.add_argument('--video-output', type=Path, help='Comparison video root; defaults to MODEL/comparisons')
    p.add_argument('--gpu', default='2')
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--resolution', type=int, default=-1)
    p.add_argument('--voxel-size', type=float, default=.05)
    p.add_argument('--max-radius', type=float, default=.1)
    p.add_argument('--margin', type=float, default=.05)
    p.add_argument('--report-max-views', type=int, default=64)
    p.add_argument('--evaluate-only', action='store_true', help='Retry post-training evaluation without retraining')
    p.add_argument('--dry-run', action='store_true')
    return p


def build_plan(args):
    from utils.colmap_paths import colmap_model_dir
    from utils.lidar_occlusion import validate_build
    import math
    source, model = args.source.expanduser().resolve(), args.model.expanduser().resolve()
    colmap_model_dir(source)
    validate_build(args.voxel_size, args.max_radius, 24, .6)
    if (args.iterations < 1 or args.report_max_views < 1 or args.resolution == 0 or args.resolution < -1 or
            not math.isfinite(args.margin) or args.margin < 0):
        raise ValueError('Invalid iterations, report limit, resolution or margin')
    if args.keyframes and not args.keyframes.expanduser().is_file():
        raise FileNotFoundError(args.keyframes)
    if not args.evaluate_only and any((model/'point_cloud').glob('iteration_*/point_cloud.ply')):
        raise ValueError('Model already has checkpoints; choose a fresh model or use --evaluate-only')
    if args.evaluate_only and not (model/f'point_cloud/iteration_{args.iterations}/point_cloud.ply').is_file():
        raise FileNotFoundError('Requested saved iteration is missing')
    steps = sorted(set(i for i in (7000, args.iterations) if i <= args.iterations))
    train = ['train.py', '-s', str(source), '-m', str(model), '--iterations', str(args.iterations),
             '--resolution', str(args.resolution), '--data_device', 'cpu', '--disable_viewer', '--antialiasing',
             '--keyframe_manifest', str(model/'keyframes.json'), '--occlusion_mode', 'lidar',
             '--occlusion_voxel_size', str(args.voxel_size), '--occlusion_max_radius', str(args.max_radius),
             '--occlusion_margin', str(args.margin), '--test_iterations', *map(str, steps),
             '--save_iterations', *map(str, steps)]
    stages = [] if args.evaluate_only else [dict(name='train', argv=train)]
    geometry, cache = str(model/'occlusion_geometry.npz'), str(model/'occlusion_depth_cache')
    def evaluation(path, mode, name):
        options = ['--model', str(path), '--iteration', str(args.iterations), '--antialiasing', '--occlusion', mode]
        if mode == 'on':
            options += ['--occlusion-geometry', geometry, '--occlusion-cache', cache, '--occlusion-margin', str(args.margin)]
        stages.append(dict(name=name+'_psnr', argv=['scripts/eval_all_frames.py', '--source', str(source), *options]))
        stages.append(dict(name=name+'_images', argv=['scripts/render.py', '--split', 'both', '--max-views',
            str(args.report_max_views), '--gpu', args.gpu, '--in-process', *options]))
    evaluation(model, 'saved', 'trained_on')
    evaluation(model, 'off', 'trained_off')
    if args.baseline:
        baseline = args.baseline.expanduser().resolve()
        if baseline == model or not (baseline/f'point_cloud/iteration_{args.iterations}/point_cloud.ply').is_file():
            raise ValueError('Baseline must be a different native checkpoint at the requested iteration')
        from scripts.eval_all_frames import read_config
        cfg = read_config(baseline/'cfg_args')
        if (cfg.get('model_type', '3dgs') != '3dgs' or (baseline/'proxygs_run.json').exists() or
                cfg.get('train_test_exp', False) or cfg.get('occlusion_mode', 'off') != 'off'):
            raise ValueError('Baseline must be native 3DGS without exposure compensation or culling')
        evaluation(baseline, 'off', 'baseline_off')
        evaluation(baseline, 'on', 'baseline_on')
        video_root = args.video_output.expanduser().resolve() if args.video_output else model/'comparisons'
        for video in ('random', 'actual'):
            command = ['scripts/render.py', '--video', video, '--source', str(source), '--models', str(baseline), str(model),
                '--labels', '3DGS + AA', '3DGS + AA + LiDAR culling', '--aa', 'on', 'on', '--occlusion', 'off', 'saved',
                '--iteration', str(args.iterations), '--gpu', args.gpu, '--in-process', '--width', '1280', '--fps', '30',
                '--output', str(video_root/video)]
            if video == 'random':
                command += ['--duration', '40', '--seed', '42']
            else:
                command += ['--actual-poses', 'source']
            stages.append(dict(name=video+'_video', argv=command))
    return dict(source=str(source), model=str(model), gpu=args.gpu, stages=stages)


def invoke(command):
    """Keep training tracebacks direct; no Python supervisor subprocess."""
    argv, stdout = sys.argv, sys.stdout
    try:
        sys.argv = [str(ROOT/command[0]), *command[1:]]
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        sys.argv, sys.stdout = argv, stdout
        gc.collect()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.baseline and not args.keyframes:
        args.keyframes = args.baseline/'keyframes.json'
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run this workflow in the CUDA-enabled native 3dgs environment')
    try:
        import nvdiffrast.torch
    except ImportError as exc:
        raise RuntimeError('Install requirements-occlusion.txt in the native 3dgs environment first') from exc
    if not args.evaluate_only:
        # Compile/check CUDA before spending time fitting a large cloud.
        invoke(['scripts/smoke_occlusion.py', '--gpu', args.gpu])
    model = Path(plan['model'])
    model.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        if args.keyframes:
            if args.keyframes.resolve() != model/'keyframes.json':
                shutil.copyfile(args.keyframes, model/'keyframes.json')
        else:
            from scripts.keyframes import select_keyframes
            select_keyframes(Path(plan['source']), 'images', model, 3, False)
    from scripts.view_comparisons import atomic_json
    status = dict(plan, completed=[], active=None, complete=False)
    previous_path = model/'occlusion_workflow.json'
    if args.evaluate_only and previous_path.exists():
        previous = json.loads(previous_path.read_text())
        previous_commands = {s['name']: s['argv'] for s in previous['stages']}
        status['completed'] = [s['name'] for s in plan['stages'] if s['name'] in previous['completed']
                               and previous_commands.get(s['name']) == s['argv']]
    for stage in plan['stages']:
        if stage['name'] in status['completed']:
            print(f"Already completed: {stage['name']}", flush=True)
            continue
        status['active'] = stage['name']
        atomic_json(model/'occlusion_workflow.json', status)
        print(f"\nOcclusion workflow: {stage['name']}", flush=True)
        invoke(stage['argv'])
        status['completed'].append(stage['name'])
    status.update(active=None, complete=True)
    atomic_json(model/'occlusion_workflow.json', status)
    print(f'Standalone occlusion workflow complete: {model}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
