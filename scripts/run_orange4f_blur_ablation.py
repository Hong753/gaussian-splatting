#!/usr/bin/env python3
"""Matched, fresh Orange4F experiments derived from the completed 200k baseline."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

REFERENCE = Path('/data/3dgs_outputs/orange4f/3dgs_finetuning/original_lidar_full_200k')
VARIANTS = {
    'baseline': (False, False, False),
    'split3cm': (True, False, False),
    'noreset': (False, True, False),
    'split3cm_noreset': (True, True, False),
    'split3cm_noreset_scalelr5x': (True, True, True),
}
STEPS = [15000, 30000, 60000, 100000, 150000, 200000]
ROOT = Path(__file__).resolve().parents[1]


def read_reference(path):
    node = ast.parse((path/'cfg_args').read_text(), mode='eval').body
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
            or node.func.id != 'Namespace' or node.args or any(k.arg is None for k in node.keywords)):
        raise ValueError('Expected a literal Namespace in reference cfg_args')
    cfg = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    opt = cfg['optimization_settings']
    expected_model = dict(model_type='3dgs', eval=False, train_test_exp=False,
                          keyframe_manifest='', data_device='cpu', colmap_principal_point=True,
                          resolution=1600, antialiasing=False, occlusion_mode='off')
    expected_opt = dict(iterations=200000, position_warmup_steps=15000, position_ramp_steps=5000,
                        densification_phase_offset=15000, position_lr_init=3.2e-6,
                        position_lr_final=3.2e-8, position_lr_max_steps=185000,
                        percent_dense=.01, disable_opacity_resets=False, disable_pruning=False,
                        preserve_densification_radii=False, scaling_lr=.005,
                        densification_interval=500, opacity_reset_interval=15000,
                        densify_from_iter=2500, densify_until_iter=100000,
                        fixed_lidar=False, freeze_lidar_positions=False, absgs=False, surface_mode='off')
    for actual, expected in ((cfg, expected_model), (opt, expected_opt)):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f'Reference {key}: expected {value!r}, got {actual.get(key)!r}')
    with (path/'densification.jsonl').open() as f:
        extents = {float(json.loads(line)['scene_extent']) for line in f if line.strip()}
    if len(extents) != 1:
        raise ValueError(f'Expected one reference scene extent, got {extents}')
    extent = extents.pop()
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError('Invalid reference scene extent')
    return cfg, extent


def variant_options(opt, variant, extent):
    split, noreset, scale_lr = VARIANTS[variant]
    result = dict(opt)
    if split:
        result['percent_dense'] = .03/extent
    if noreset:
        result['disable_opacity_resets'] = True
    if scale_lr:
        result['scaling_lr'] = .001
    return result


def reproduce_arguments(parser, values):
    """Replay saved values; check the round trip through the native parser."""
    defaults = vars(parser.parse_args([]))
    if set(defaults) != set(values):
        raise ValueError(f'Native argument schema changed: missing={sorted(set(defaults)-set(values))}, '
                         f'extra={sorted(set(values)-set(defaults))}')
    argv = []
    for key, default in defaults.items():
        value = values[key]
        if value == default:
            continue
        if isinstance(default, bool):
            if value is not True:
                raise ValueError(f'Cannot replay false --{key} with native store_true parser')
            argv.append('--'+key)
        else:
            argv.extend(['--'+key, str(value)])
    if vars(parser.parse_args(argv)) != values:
        raise ValueError('Native argument replay does not match the requested values')
    return argv


def training_arguments(cfg, opt, initial_point_cloud, output):
    from arguments import ModelParams, OptimizationParams, PipelineParams
    parser = argparse.ArgumentParser(add_help=False)
    model = ModelParams(parser)
    OptimizationParams(parser)
    pipe = PipelineParams(parser)
    defaults = parser.parse_args([])
    model_keys = vars(model.extract(defaults))
    pipe_keys = vars(pipe.extract(defaults))
    values = {k: cfg[k] for k in model_keys}
    values.update({k:cfg.get(k, v) for k,v in pipe_keys.items()})
    # Legacy reference cfg_args predates the optional native scale controls.
    from utils.native_scale_control import DEFAULTS as native_scale_defaults
    values.update({k:v for k,v in native_scale_defaults.items() if k in vars(defaults)})
    values.update(opt)
    values.update(model_path=str(output), initial_point_cloud=str(initial_point_cloud))
    return reproduce_arguments(parser, values)


def validate_source(reference, cfg, extent):
    import numpy as np
    from scripts.view_comparisons import read_colmap_records
    from utils.densification_diagnostics import training_extent
    if not (reference/'input.ply').is_file():
        raise FileNotFoundError(reference/'input.ply')
    records = read_colmap_records(cfg['source_path'], cfg.get('images','images'))
    cameras = json.loads((reference/'cameras.json').read_text())
    saved = {c['img_name']:c for c in cameras}
    if len(cameras) != len(saved) or set(saved) != set(records) or len(records) != 5009:
        raise ValueError('Current COLMAP camera set does not match all 5009 saved reference views')
    for name, record in records.items():
        camera = saved[name]
        # Match camera_to_JSON exactly: it inverts the homogeneous
        # world-to-camera matrix, rather than assuming R is orthonormal.
        # COLMAP files exported from LiDAR may contain rounded quaternions.
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = np.asarray(record['R'], dtype=np.float64).T
        world_to_camera[:3, 3] = np.asarray(record['T'], dtype=np.float64)
        camera_to_world = np.linalg.inv(world_to_camera)
        expected = dict(rotation=camera_to_world[:3, :3],
                        position=camera_to_world[:3, 3],
                        width=record['width'], height=record['height'],
                        fx=record['width']/(2*math.tan(record['fov_x']/2)),
                        fy=record['height']/(2*math.tan(record['fov_y']/2)),
                        cx=record['width']*(record['cx']/record['width']),
                        cy=record['height']*(record['cy']/record['height']))
        differences = {}
        for key, value in expected.items():
            if key not in camera:
                differences[key] = {'error': 'missing from saved camera'}
                continue
            actual = np.asarray(camera[key], dtype=np.float64)
            value = np.asarray(value, dtype=np.float64)
            atol = 1e-5 if key in ('rotation','position') else 1e-4
            if actual.shape != value.shape or not np.allclose(actual, value, rtol=1e-6, atol=atol):
                differences[key] = dict(saved=actual.tolist(), current=value.tolist())
                if actual.shape == value.shape:
                    differences[key]['max_abs_difference'] = float(np.max(np.abs(actual-value)))
        if differences:
            raise ValueError(f'Saved and current camera differ: {name}\\n'
                             + json.dumps(differences, indent=2))
        if not Path(record['image_path']).is_file():
            raise FileNotFoundError(record['image_path'])
    current_extent = training_extent(records, set(records))
    if not math.isclose(current_extent,extent,rel_tol=1e-5,abs_tol=1e-5):
        raise ValueError(f'Camera extent changed: {current_extent} vs {extent}')
    print(f'Validated {len(records)} original training cameras; extent={current_extent:.8f}', flush=True)


def alternate_source(cfg, source, explicit_cloud, output):
    """Use the new dataset's own camera model and initialization, never old poses."""
    from scripts.view_comparisons import read_colmap_records
    from utils.colmap_paths import colmap_model_dir
    from utils.densification_diagnostics import training_extent
    source = source.expanduser().resolve()
    records = read_colmap_records(source, 'images')
    for record in records.values():
        if not Path(record['image_path']).is_file():
            raise FileNotFoundError(record['image_path'])
    extent = training_extent(records, set(records))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError('Invalid new dataset scene extent')
    sparse = colmap_model_dir(source)
    convert = None
    if explicit_cloud is not None:
        cloud = explicit_cloud.expanduser().resolve()
        if not cloud.is_file():
            raise FileNotFoundError(cloud)
    elif (sparse/'points3D.ply').is_file():
        cloud = sparse/'points3D.ply'
    else:
        convert = next((p for p in (sparse/'points3D.bin', sparse/'points3D.txt') if p.is_file()), None)
        if convert is None:
            raise FileNotFoundError(f'No points3D.ply, points3D.bin or points3D.txt in {sparse}')
        # Convert inside the new model directory; never write to the shared
        # source or make concurrent runs race to create the dataset PLY.
        cloud = output/'input.ply'
    new_cfg = dict(cfg, source_path=str(source), images='images', depths='',
                   keyframe_manifest='', loss_mask_dir='', eval=False,
                   train_test_exp=False, initial_point_cloud=str(cloud),
                   training_scene_extent=extent)
    print(f'New source: {source}; {len(records)} registered training views; '
          f'extent={extent:.8f}; initialization={convert or cloud}', flush=True)
    return new_cfg, extent, cloud, convert, len(records)


def convert_initial_cloud(source, output):
    from scene.colmap_loader import read_points3D_binary, read_points3D_text
    from scene.dataset_readers import storePly
    from utils.checkpoint_snapshots import atomic_write
    reader = read_points3D_binary if source.suffix == '.bin' else read_points3D_text
    xyz, rgb, _ = reader(str(source))
    atomic_write(output, lambda path: storePly(str(path), xyz, rgb))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--gpu', type=int, required=True, help='Physical GPU index')
    p.add_argument('--reference', type=Path, default=REFERENCE)
    p.add_argument('--source', type=Path, help='Alternate COLMAP dataset; defaults to the saved reference source')
    p.add_argument('--initial-point-cloud', type=Path, help='Optional PLY aligned with --source; default is the new source points3D')
    p.add_argument('--output', type=Path)
    p.add_argument('--render-max-views', type=int, default=0, help='0 renders all saved views after training')
    p.add_argument('--dry-run', action='store_true', help='Print and validate arguments without allocating CUDA or launching training')
    args = p.parse_args(argv)
    if args.gpu < 0 or args.render_max_views < 0:
        p.error('GPU and max views must be nonnegative')
    reference = args.reference.expanduser().resolve()
    if args.initial_point_cloud is not None and args.source is None:
        p.error('--initial-point-cloud requires --source')
    prefix = ('colmap_'+args.source.expanduser().resolve().name+'_full_200k_' if args.source is not None
              else 'original_lidar_full_200k_')
    output = (args.output or reference.with_name(prefix+args.variant)).expanduser().resolve()
    if output == reference or reference in output.parents or output in reference.parents or output.exists():
        p.error(f'Output must be a fresh separate directory: {output}')
    cfg, extent = read_reference(reference)
    reference_source = cfg['source_path']
    reference_extent = extent
    conversion = None
    count = 5009
    initial_cloud = reference/'input.ply'
    if args.source is not None:
        new_source = args.source.expanduser().resolve()
        if output == new_source or new_source in output.parents or output in new_source.parents:
            p.error('Output must be outside the source dataset')
        cfg, extent, initial_cloud, conversion, count = alternate_source(
            cfg, new_source, args.initial_point_cloud, output)
    opt = variant_options(cfg['optimization_settings'], args.variant, extent)
    changes = {k: v for k,v in opt.items() if v != cfg['optimization_settings'][k]}
    command = [sys.executable, '-u', 'train.py', *training_arguments(cfg,opt,initial_cloud,output),
               '--test_iterations', *map(str,STEPS), '--save_iterations', *map(str,STEPS),
               '--checkpoint_iterations', *map(str,STEPS), '--debug_preview_every', '500',
               '--debug_preview_train', '2', '--debug_preview_test', '2', '--debug_preview_seed', '0',
               '--disable_viewer']
    render = [sys.executable, '-u', 'scripts/render.py', '--model', str(output), '--split', 'both',
              '--iteration', '200000', '--gpu', str(args.gpu), '--max-views', str(args.render_max_views)]
    audit = [sys.executable, '-u', '-m', 'scripts.audit_splat_blur', '--model', str(output),
             '--iteration', '200000', '--views', '12', '--history']
    plan = dict(reference=str(reference), output=str(output), variant=args.variant, gpu=args.gpu,
                scene_extent=extent, split_sigma=opt['percent_dense']*extent, changed_optimization_settings=changes,
                reference_source=reference_source, reference_extent=reference_extent,
                source=cfg['source_path'], training_views=count,
                initial_point_cloud=str(initial_cloud), converted_from=str(conversion) if conversion else None,
                same_source_as_reference=args.source is None,
                commands=[command,render,audit],
                note='Fresh training from unoptimized input points. Alternate source uses its own calibration and points; old reference supplies hyperparameters only.')
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:
        return 0
    if args.source is None:
        validate_source(reference,cfg,extent)
    output.mkdir(parents=True,exist_ok=False)
    if conversion is not None:
        convert_initial_cloud(conversion, initial_cloud)
    (output/'blur_ablation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    for cmd in (command,render,audit):
        print('CUDA_VISIBLE_DEVICES='+str(args.gpu)+' '+shlex.join(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
