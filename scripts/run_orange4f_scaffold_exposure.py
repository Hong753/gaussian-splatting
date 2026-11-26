#!/usr/bin/env python3
"""Matched Scaffold-GS anchor-growth/offset experiments on corrected Orange4F RGB."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = Path('/data/colmap_scenes/orange4f')
EXPOSURE_ROOT = Path('/data/3dgs_outputs/orange4f/exposure_check_v1')
OUTPUT_ROOT = Path('/data/3dgs_outputs/orange4f/scaffold_exposure_v1')
VARIANTS = ('baseline', 'fixed_anchors', 'bounded_offsets', 'fixed_anchors_bounded_offsets')
STEPS = [5000, 6000, 10000, 15000, 20000, 30000, 60000, 100000, 120000]


def command_options(variant, offset_radius=.05):
    """A 2x2 test: growing versus initial anchor pool, free versus bounded offsets.

    Coordinates of surviving anchors are fixed in every variant, as in the
    zero-position-LR Scaffold baseline. Pruning stays enabled in all four, so
    the ordinary Adam and volume-loss paths are identical across the suite.
    """
    if variant not in VARIANTS:
        raise ValueError(f'Unknown Scaffold exposure variant: {variant}')
    if not math.isfinite(offset_radius) or offset_radius <= 0:
        raise ValueError('The experimental offset radius must be positive and finite')
    return dict(
        iterations=120000, freeze_anchors=True,
        disable_anchor_growth=variant in ('fixed_anchors', 'fixed_anchors_bounded_offsets'),
        disable_anchor_pruning=False,
        offset_max_radius=offset_radius if variant in ('bounded_offsets', 'fixed_anchors_bounded_offsets') else 0.,
        position_lr_init=0., position_lr_final=0., position_lr_max_steps=120000,
        offset_lr_init=.01, offset_lr_final=.0001, offset_lr_max_steps=120000,
        feature_lr=.0075, opacity_lr=.02, scaling_lr=.007, rotation_lr=.002,
        mlp_opacity_lr_init=.002, mlp_opacity_lr_final=.00002, mlp_opacity_lr_max_steps=120000,
        mlp_cov_lr_init=.004, mlp_cov_lr_final=.004, mlp_cov_lr_max_steps=120000,
        mlp_color_lr_init=.008, mlp_color_lr_final=.00005, mlp_color_lr_max_steps=120000,
        mlp_featurebank_lr_max_steps=120000, appearance_lr_max_steps=120000,
        start_stat=2000, update_from=6000, update_until=60000, update_interval=400,
        min_opacity=.005, success_threshold=.8, densify_grad_threshold=.0002,
        lambda_dssim=.2)


def flags(values):
    result = []
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                result.append('--' + key)
        else:
            result.extend(('--' + key, str(value)))
    return result


def build_plan(args):
    from scripts.run_orange4f_blur_ablation import alternate_source
    from scripts.view_comparisons import read_colmap_records
    from utils.exposure_training import validate_exposure_export

    source = SOURCE.expanduser().resolve()
    out = (args.output_root.expanduser() / args.variant).resolve()
    if out == source or source in out.parents:
        raise ValueError('Run output must be outside the source dataset')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {out}')
    images, exposure = validate_exposure_export(args.exposure_dir, source)
    export = Path(exposure['directory'])
    if out == export or export in out.parents:
        raise ValueError('Run output must be outside the exposure export')
    cfg, extent, cloud, conversion, count = alternate_source({}, source, args.initial_point_cloud, out)
    names = sorted(read_colmap_records(source))
    if len(names) != count or exposure['registered_images'] != count:
        raise ValueError('Registered camera set changed while preparing the run')
    opt = command_options(args.variant, getattr(args, 'offset_radius', .05))
    keyframes = out / 'keyframes.json'
    preflight = [sys.executable, '-u', 'scripts/smoke_scaffold.py',
                 '--offset-max-radius', str(opt['offset_max_radius'])]
    train = [sys.executable, '-u', 'models/train_scaffold.py', '--model_type', 'scaffold',
             '-s', cfg['source_path'], '--images', str(images), '-m', str(out),
             '--initial_point_cloud', str(cloud), '--training_scene_extent', str(extent),
             '--keyframe_manifest', str(keyframes), '-r', '1600', '--data_device', 'cpu',
             '--colmap_principal_point', '--scaffold_voxel_size', '.02',
             '--scaffold_n_offsets', '10', '--scaffold_feat_dim', '32',
             *flags(opt), '--publish_snapshots',
             '--save_iterations', *map(str, STEPS), '--test_iterations', *map(str, STEPS),
             '--debug_preview_every', '500', '--debug_preview_train', '2', '--debug_preview_test', '2',
             '--debug_preview_seed', '0', '--disable_viewer']
    render = [sys.executable, '-u', 'scripts/render.py', '--model', str(out), '--split', 'both',
              '--iteration', '120000', '--gpu', str(args.gpu),
              '--max-views', str(args.render_max_views)]
    return dict(
        schema=1, source=cfg['source_path'], images=str(images), model=str(out),
        variant=args.variant, gpu=args.gpu, model_type='scaffold',
        training_views=count, registered_names=names, scene_extent=extent,
        initial_point_cloud=str(cloud), converted_from=str(conversion) if conversion else None,
        keyframe_manifest=str(keyframes), exposure_preprocessing=exposure,
        learned_exposure=False, appearance_dim=0, antialiasing=False,
        scaffold_voxel_size=.02, scaffold_n_offsets=10, scaffold_feat_dim=32,
        options=opt, save_iterations=STEPS, commands=[preflight, train, render],
        first_anchor_update=6400, last_anchor_update=59600,
        note='All registered views train on the completed corrected export. Original source cameras and '
             'SfM coordinates initialize every run; source colors do not initialize Scaffold features. '
             'This is the repository-adapted Scaffold baseline with a 4x training schedule. '
             'Surviving anchor coordinates are fixed in all variants; fixed_anchors disables new anchor '
             'growth but still permits pruning. Child centers remain learned in every variant. '
             'A nonzero offset_max_radius bounds scaled child-center displacement after every optimizer '
             'step, not Gaussian size; distances are scene units and metric scale is not verified. '
             'No pose, appearance-embedding, or learned-exposure changes are introduced.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--gpu', type=int, required=True, help='Physical GPU index')
    parser.add_argument('--exposure-dir', type=Path, default=EXPOSURE_ROOT,
                        help='Completed normalize_image_exposure output containing corrected/')
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT,
                        help='New parent directory; each variant gets a separate model folder')
    parser.add_argument('--initial-point-cloud', type=Path,
                        help='Optional PLY aligned to the original source; default: source SfM cloud')
    parser.add_argument('--offset-radius', type=float, default=.05,
                        help='Radius in scene units for bounded variants only (default .05)')
    parser.add_argument('--render-max-views', type=int, default=0, help='0: render every registered view')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print commands without CUDA or writes')
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.render_max_views < 0:
        parser.error('GPU and render view count must be nonnegative')
    if not math.isfinite(args.offset_radius) or args.offset_radius <= 0:
        parser.error('offset-radius must be positive and finite')
    plan = build_plan(args)
    # The complete list is stored in keyframes.json rather than printing 3k names.
    saved_plan = {key: value for key, value in plan.items() if key != 'registered_names'}
    print(json.dumps(saved_plan, indent=2, allow_nan=False), flush=True)
    if args.dry_run:
        return 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))

    def run(command):
        print('CUDA_VISIBLE_DEVICES=' + str(args.gpu) + ' ' + shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    # A failed CUDA/extension preflight must not leave a reserved experiment folder.
    run(plan['commands'][0])
    out = Path(plan['model'])
    out.mkdir(parents=True, exist_ok=False)
    from utils.checkpoint_snapshots import atomic_json
    from utils.exposure_training import snapshot_exposure_metadata
    snapshot_exposure_metadata(plan['exposure_preprocessing'], out / 'exposure_preprocessing')
    atomic_json(out / 'keyframes.json', dict(schema=1, source=plan['source'],
                selection='all registered source images', selected_images=plan['registered_names']))
    atomic_json(out / 'scaffold_exposure_plan.json', saved_plan)
    if plan['converted_from']:
        from scripts.run_orange4f_blur_ablation import convert_initial_cloud
        convert_initial_cloud(Path(plan['converted_from']), Path(plan['initial_point_cloud']))
    run(plan['commands'][1])
    run(plan['commands'][2])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
