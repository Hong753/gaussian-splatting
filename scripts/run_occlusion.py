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


def parser(*, depth_maps=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--keyframes', type=Path, help=('Exact training manifest used for depth preparation; defaults to baseline manifest'
                   if depth_maps else 'Reuse this manifest; otherwise select sharp stride-3 keyframes'))
    p.add_argument('--baseline', type=Path, help=('Optional native 3DGS+AA checkpoint for scores and both comparison videos'
                   if depth_maps else 'Optional native 3DGS+AA checkpoint for four-way scores and both comparison videos'))
    p.add_argument('--video-output', type=Path, help='Comparison video root; defaults to MODEL/comparisons')
    p.add_argument('--gpu', default='0')
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--resolution', type=int, default=-1)
    if depth_maps:
        p.set_defaults(voxel_size=.05, max_radius=.1)
    else:
        p.add_argument('--voxel-size', type=float, default=.05)
        p.add_argument('--max-radius', type=float, default=.1)
        p.add_argument('--scene-geometry', type=Path, help='Use a fixed whole-scene scene.json instead of LiDAR patches')
    p.add_argument('--margin', type=float, default=.05)
    p.add_argument('--occlusion-backend', choices=('whole', 'ray'), default='whole')
    p.add_argument('--inference-occlusion', choices=('saved', 'off'), default=None,
                   help='off trains with culling, but saves mesh-free inference and skips culling-on evaluations')
    p.add_argument('--prepare-visibility', action='store_true',
                   help='Finish/resume all-frame CUDA visibility preprocessing before mesh training')
    p.add_argument('--position-warmup-steps', type=int, default=0,
                   help='Freeze xyz and its Adam state for this many initial steps')
    p.add_argument('--position-ramp-steps', type=int, default=0,
                   help='Linearly ramp xyz LR after the fixed-position warmup')
    p.add_argument('--position-lr-scale', type=float, default=1.,
                   help='Multiply both baseline xyz LR endpoints; 0.02 is 50x smaller')
    p.add_argument('--position-lr-max-steps', type=int,
                   help='Exponential decay duration; default: iterations minus warmup and ramp')
    p.add_argument('--densification-phase-offset', type=int,
                   help='Shift densification and opacity resets; default: position warmup steps')
    p.add_argument('--report-max-views', type=int, default=64)
    p.add_argument('--view-tangent', action='store_true', help='Grow only inside trusted visible surface neighborhoods')
    # Keep this launcher importable without PyTorch, for CPU plan inspection.
    from arguments import OptimizationParams
    defaults = OptimizationParams(argparse.ArgumentParser(add_help=False))
    for name in ('radius_pixels', 'max_radius', 'depth_abs', 'depth_relative',
                 'min_observations', 'max_parents', 'max_points'):
        value = getattr(defaults, 'view_tangent_'+name)
        p.add_argument('--view-tangent-'+name.replace('_', '-'), type=type(value), default=value)
    p.add_argument('--evaluate-only', action='store_true', help='Retry post-training evaluation without retraining')
    p.add_argument('--dry-run', action='store_true')
    return p


def training_schedule(args):
    import math
    from arguments import OptimizationParams
    defaults = OptimizationParams(argparse.ArgumentParser(add_help=False))
    if args.evaluate_only:
        # Report the schedule that produced these weights, not new CLI defaults.
        from scripts.eval_all_frames import read_config
        cfg = read_config(args.model.expanduser().resolve()/'cfg_args')
        saved = cfg.get('optimization_settings', {})
        keys = ('position_warmup_steps', 'position_ramp_steps', 'position_lr_init', 'position_lr_final',
                'position_lr_max_steps', 'densification_phase_offset')
        return {key: cfg.get(key, saved.get(key, getattr(defaults, key))) for key in keys}
    warmup, ramp = args.position_warmup_steps, args.position_ramp_steps
    offset = warmup if args.densification_phase_offset is None else args.densification_phase_offset
    decay = args.iterations-warmup-ramp if args.position_lr_max_steps is None else args.position_lr_max_steps
    if (not math.isfinite(args.position_lr_scale) or args.position_lr_scale <= 0 or
            min(warmup, ramp, offset) < 0 or warmup+ramp >= args.iterations or decay < 1):
        raise ValueError('Require positive finite xyz LR scale, nonnegative warmup/ramp/offset, and a positive decay phase')
    if offset < warmup:
        raise ValueError('Densification phase offset must cover the entire fixed-position warmup')
    result = dict(position_warmup_steps=warmup, position_ramp_steps=ramp,
                  position_lr_init=defaults.position_lr_init*args.position_lr_scale,
                  position_lr_final=defaults.position_lr_final*args.position_lr_scale,
                  position_lr_max_steps=decay, densification_phase_offset=offset)
    return result


def schedule_label(schedule):
    additions = []
    if schedule['position_warmup_steps']:
        additions.append(f"xyz warmup {schedule['position_warmup_steps']}")
    if schedule['position_lr_init'] < 0.00016:
        additions.append('reduced xyz LR')
    return '3DGS + AA + LiDAR culling' + ((' + ' + ' + '.join(additions)) if additions else '')


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
    schedule = training_schedule(args)
    backend = args.occlusion_backend
    inference_mode = getattr(args, 'inference_occlusion', None)
    if args.evaluate_only:
        from scripts.eval_all_frames import read_config
        saved_cfg = read_config(model/'cfg_args')
        backend = saved_cfg.get('occlusion_backend', 'whole')
        if inference_mode is None:
            inference_mode = saved_cfg.get('occlusion_inference_mode', 'saved')
    inference_mode = inference_mode or 'saved'
    steps = sorted(set(i for i in (7000, args.iterations) if i <= args.iterations))
    train = ['train.py', '-s', str(source), '-m', str(model), '--iterations', str(args.iterations),
             '--resolution', str(args.resolution), '--data_device', 'cpu', '--disable_viewer', '--antialiasing',
             '--keyframe_manifest', str(model/'keyframes.json'), '--occlusion_mode', 'lidar',
             '--occlusion_voxel_size', str(args.voxel_size), '--occlusion_max_radius', str(args.max_radius),
             '--occlusion_margin', str(args.margin), '--occlusion_backend', backend,
             '--occlusion_inference_mode', inference_mode,
             '--test_iterations', *map(str, steps),
             '--save_iterations', *map(str, steps)]
    scene_geometry = getattr(args, 'scene_geometry', None)
    preparation_stages = []
    if args.prepare_visibility and (not scene_geometry or args.evaluate_only):
        raise ValueError('--prepare-visibility requires --scene-geometry and a new training run')
    if scene_geometry and (not args.evaluate_only or inference_mode != 'off'):
        from utils.scene_occlusion import load_scene
        report, _ = load_scene(scene_geometry, source=source)
        if report.get('diagnostic_only'):
            raise ValueError('Finish full-scene fusion before training; this is a diagnostic preview mesh')
        if not args.keyframes:
            raise ValueError('Scene mesh training requires its exact keyframe manifest')
        if set(json.loads(args.keyframes.read_text())['selected_images']) != set(report['training_images']):
            raise ValueError('Scene fusion and training keyframes differ')
        visibility_path = scene_geometry.resolve().parent/'visibility/metadata.json'
        visibility = json.loads(visibility_path.read_text()) if visibility_path.is_file() else {}
        if not visibility.get('complete') and not args.prepare_visibility:
            raise ValueError('Finish whole-scene per-view visibility preprocessing and inspect its coverage before training')
        from utils.lidar_occlusion import file_digest
        if visibility and (visibility['scene_sha256'] != file_digest(scene_geometry) or
                not set(report['training_images']).issubset(visibility['images'])):
            raise ValueError('Per-view visibility belongs to a different scene or omits training views')
        if args.prepare_visibility:
            if not report.get('point_cloud'):
                raise ValueError('Scene manifest is missing its initialization point-cloud path')
            load_scene(scene_geometry, point_cloud=report['point_cloud'], source=source)
            preparation_stages.append(dict(name='visibility', argv=[
                'scripts/prepare_scene_occlusion.py', 'views', '--scene', str(scene_geometry.resolve()),
                '--point-cloud', report['point_cloud'], '--resolution', str(args.resolution),
                '--margin', str(args.margin), '--split', 'all', '--device', 'cuda',
                '--visibility-backend', 'cuda', '--gpu', args.gpu]))
        train[train.index('--occlusion_mode')+1] = 'mesh'
        train += ['--occlusion_geometry', str(scene_geometry.resolve()), '--occlusion_cache',
                  str(scene_geometry.resolve().parent/'occlusion_depth_cache')]
    for key, value in schedule.items():
        train += ['--'+key, str(value)]
    tangent_options = {}
    if args.view_tangent:
        tangent_options = {key: value for key, value in vars(args).items() if key.startswith('view_tangent_')}
        if (any(not math.isfinite(float(v)) for v in tangent_options.values()) or
                args.view_tangent_radius_pixels < 2 or args.view_tangent_max_radius <= 0 or
                min(args.view_tangent_depth_abs, args.view_tangent_depth_relative) < 0 or
                args.view_tangent_depth_abs + args.view_tangent_depth_relative <= 0 or
                min(args.view_tangent_min_observations, args.view_tangent_max_parents, args.view_tangent_max_points) < 1):
            raise ValueError('Invalid view-tangent settings')
        train += ['--surface_mode', 'view_tangent']
        for key, value in tangent_options.items():
            train += ['--'+key, str(value)]
    stages = [] if args.evaluate_only else preparation_stages + [dict(name='train', argv=train)]
    geometry, cache = str(model/'occlusion_geometry.npz'), str(model/'occlusion_depth_cache')
    if scene_geometry:
        geometry, cache = str(scene_geometry.resolve()), str(scene_geometry.resolve().parent/'occlusion_depth_cache')
    def evaluation(path, mode, name):
        options = ['--model', str(path), '--iteration', str(args.iterations), '--antialiasing', '--occlusion', mode]
        if mode == 'on':
            options += ['--occlusion-geometry', geometry, '--occlusion-cache', cache, '--occlusion-margin', str(args.margin)]
            options += ['--occlusion-backend', backend]
        stages.append(dict(name=name+'_psnr', argv=['scripts/eval_all_frames.py', '--source', str(source), *options]))
        stages.append(dict(name=name+'_images', argv=['scripts/render.py', '--split', 'both', '--max-views',
            str(args.report_max_views), '--gpu', args.gpu, '--in-process', *options]))
    if inference_mode == 'off':
        evaluation(model, 'off', 'trained_off')
    else:
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
        if inference_mode != 'off':
            evaluation(baseline, 'on', 'baseline_on')
        video_root = args.video_output.expanduser().resolve() if args.video_output else model/'comparisons'
        for video in ('random', 'actual'):
            command = ['scripts/render.py', '--video', video, '--source', str(source), '--models', str(baseline), str(model),
                '--labels', '3DGS + AA', schedule_label(schedule).replace('LiDAR culling', 'fixed scene mesh culling' if scene_geometry else 'LiDAR culling') + (' + view-tangent growth' if args.view_tangent else '') + (' | culling during training only' if inference_mode == 'off' else ''),
                '--aa', 'on', 'on', '--occlusion', 'off', 'off' if inference_mode == 'off' else 'saved',
                '--iteration', str(args.iterations), '--gpu', args.gpu, '--in-process', '--width', '1280', '--fps', '30',
                '--output', str(video_root/video)]
            if video == 'random':
                command += ['--duration', '40', '--seed', '42']
            else:
                command += ['--actual-poses', 'source']
            stages.append(dict(name=video+'_video', argv=command))
    return dict(source=str(source), model=str(model), gpu=args.gpu, training_schedule=schedule,
                view_tangent_options=tangent_options, inference_occlusion=inference_mode, stages=stages)


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
    print('Position/densification schedule (LR endpoints before scene-extent scaling):\n' +
          json.dumps(plan['training_schedule'], indent=2), flush=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run this workflow in the CUDA-enabled native 3dgs environment')
    if not args.evaluate_only or plan['inference_occlusion'] != 'off':
        try:
            import nvdiffrast.torch
        except ImportError as exc:
            raise RuntimeError('Install requirements-occlusion.txt in the native 3dgs environment first') from exc
    if not args.evaluate_only:
        # Compile/check CUDA before spending time fitting a large cloud.
        if args.occlusion_backend == 'ray':
            invoke(['scripts/smoke_ray_occlusion.py', '--gpu', args.gpu])
        invoke(['scripts/smoke_occlusion.py', '--gpu', args.gpu, '--backend', args.occlusion_backend,
                *(['--mesh'] if args.scene_geometry else [])])
        if args.view_tangent:
            invoke(['scripts/smoke_view_tangent.py', '--gpu', args.gpu,
                    '--backend', args.occlusion_backend,
                    '--output', str(Path(plan['model'])/'view_tangent_cuda_smoke.json')])
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
