#!/usr/bin/env python3
"""Prepare a Scaffold-GS variant against the existing immutable orange4f suite."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_soft_surface_long as periodic
from scripts.run_occlusion import invoke
from utils.checkpoint_snapshots import digest


INDEXED_VARIANTS = ("indexed", "indexed-only", "indexed-opacity")


def build_plan(args):
    from scripts.run_fixed_lidar import checkpoint_steps
    fixed = args.variant != 'learnable'
    hard = args.variant in ('indexed-only', 'indexed-opacity')
    gated = args.variant in (*INDEXED_VARIANTS, 'sensed', 'mesh-visible')
    static = gated or args.variant == 'fixed-static'
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
    if static:
        command += ['--disable_anchor_pruning']
    if gated:
        command += ['--sensed_membership', str(Path(args.model)/'observations')]
    if hard:
        command += ['--indexed_forward_only']
    if args.variant == 'indexed-opacity':
        command += ['--indexed_opacity_weight', '0.05', '--indexed_opacity_target', '0.9']
    return dict(model=str(args.model), source=str(args.source), model_type='scaffold',
        antialiasing=False, freeze_anchors=fixed, anchor_growth=not fixed, pruning=not static,
        voxel_size=args.voxel_size, learned_offsets=True, appearance_dim=0,
        hard_culling=hard, stages=[dict(argv=command)])


def execute_plan(args, plan):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    import simple_knn._C as knn
    if not torch.cuda.is_available() or not callable(getattr(knn, 'distCUDA2', None)):
        raise RuntimeError('Use the CUDA-enabled 3dgs environment with the existing simple_knn extension')
    print(f'Reusing simple_knn: {knn.__file__}', flush=True)
    if list(Path(args.model).glob('point_cloud/iteration_*')):
        raise FileExistsError('Saved iterations already exist; use a new --run-version, or resume the trainer explicitly')
    smoke = ['scripts/smoke_scaffold.py']
    if args.variant == 'indexed-opacity':
        smoke.append('--indexed-opacity')
    invoke(smoke)
    print(json.dumps(plan, indent=2), flush=True)
    invoke(plan['stages'][0]['argv'])


def prepare_indexed(args, suite, staged, manifest, names):
    from scripts.prepare_orange4f import canonical_cloud
    from scripts.view_sparse_observations import Inspector
    from utils.scaffold_observations import prepare_membership

    viewer = Inspector(args.source, args.indexed_ply, manifest=manifest)
    cloud = staged/'input.ply'
    if Path(suite['cloud']['original']).resolve() == args.indexed_ply:
        cloud.symlink_to(Path(suite['cloud']['cached']).resolve())
    else:
        canonical_cloud(args.source, cloud, explicit=args.indexed_ply)
    prepare_membership(staged/'observations', viewer.manifest_path, cloud, args.indexed_ply,
                       names, args.voxel_size, indexed_viewer=viewer, prepare_hits=args.variant == 'indexed-opacity')

    inputs = [args.indexed_ply, args.point_cloud]
    binary = Path(viewer.summary['model'])/'points3D.bin'
    if binary.is_file():
        inputs.append(binary)
    if viewer.manifest_path:
        inputs.append(viewer.manifest_path)
        inputs.extend(viewer.manifest[n] for n in names if n in viewer.manifest)
    return inputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--variant', choices=['learnable', 'fixed', 'fixed-static', *INDEXED_VARIANTS, 'sensed','mesh-visible'], required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--observations', type=Path, default=os.environ.get('SCAFFOLD_OBSERVATIONS'),
                   help='Optional viewer row manifest for indexed, measured associations for sensed, or mesh visibility metadata')
    p.add_argument('--indexed-ply', type=Path, default=os.environ.get('SCAFFOLD_INDEXED_PLY'),
                   help="Use the PLY selected with the viewer's --ply override")
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
    manifest = None
    indexed = args.variant in INDEXED_VARIANTS
    if args.indexed_ply and not indexed:
        p.error('--indexed-ply applies to the indexed variant')
    if indexed:
        from scripts.view_sparse_observations import find_row_manifest, viewer_cloud_path
        from utils.colmap_paths import colmap_model_dir
        sparse = colmap_model_dir(args.source)
        args.indexed_ply = viewer_cloud_path(args.source, sparse, args.indexed_ply).resolve(strict=True)
        args.point_cloud = args.model/'input.ply'
        manifest, _ = find_row_manifest(args.source, sparse, args.observations)
        args.observations = str(manifest) if manifest else None
        args.indexed_tracks_present = (sparse/'points3D.bin').is_file()
    elif args.variant in ('sensed','mesh-visible'):
        from utils.scaffold_observations import discover
        if args.variant == 'mesh-visible' and not args.observations:
            p.error('For mesh-visible set SCAFFOLD_OBSERVATIONS to visibility/metadata.json from the first-hit mesh cache')
        manifest = discover(args.source, args.observations)
        args.observations = str(manifest)
    elif args.observations:
        p.error('--observations applies to indexed, sensed or mesh-visible')
    plan = build_plan(args)
    if (periodic.directory(args.model)/'plan.json').exists():
        previous = periodic.load_plan(args.model)
        if (previous['plan'] != plan or previous['gpu'] != args.gpu
                or (indexed and (str(args.indexed_ply) != previous['training_args'].get('indexed_ply')
                                 or args.indexed_tracks_present != previous['training_args'].get('indexed_tracks_present')
                                 or args.observations != previous['training_args'].get('observations')))
                or (manifest and str(manifest) != previous['training_args'].get('observations'))):
            raise ValueError('Prepared Scaffold settings changed; use another --run-version')
        for name, expected in previous['input_fingerprints'].items():
            if digest(Path(name)) != expected:
                raise ValueError(f'Prepared Scaffold input changed: {name}')
        print(f'Reusing Scaffold plan: {args.model}', flush=True)
        return 0
    if args.model.exists():
        raise FileExistsError(f'Existing model directory: {args.model}')
    # Publish the membership, exact initialization and runnable plan together.
    with tempfile.TemporaryDirectory(dir=root) as temp:
        staged = Path(temp)/'model'
        staged.mkdir()
        shutil.copyfile(suite['selection'], staged/'keyframes.json')
        names = json.loads((staged/'keyframes.json').read_text())['selected_images']
        inputs = [args.keyframes, *map(Path, suite['input_fingerprints'])]
        if indexed:
            inputs.extend(prepare_indexed(args, suite, staged, manifest, names))
        else:
            (staged/'input.ply').symlink_to(args.point_cloud)
            if manifest:
                from utils.scaffold_observations import prepare_membership
                prepare_membership(staged/'observations', manifest, args.point_cloud, suite['cloud']['original'],
                                   names, args.voxel_size, mesh_source=args.source if args.variant == 'mesh-visible' else None)
                inputs.append(manifest)
        if indexed or manifest:
            directory = args.model/'observations'
            inputs.extend([directory/'metadata.json', directory/'parents.npy'])
            if args.variant == 'indexed-opacity':
                inputs.append(directory/'hit_pixels.npy')
        result = periodic.prepare_experiment(args, plan, 'scaffold', inputs, staged_model=staged)
        staged.rename(args.model)
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
