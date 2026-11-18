#!/usr/bin/env python3
"""Run the released Scaffold-GS optimizer on our Orange4F split and preview protocol."""
import argparse
import ast
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REVISION = '59c833b56bbbf510f3f64d40f81721995caced66'
STEPS = (10000, 20000, 30000)


def prepare(args):
    from utils.checkpoint_snapshots import atomic_json, digest
    from scripts.view_comparisons import read_colmap_records
    suite = json.loads((args.suite/'suite.json').read_text())
    for path, sha in suite['input_fingerprints'].items():
        if digest(path) != sha:
            raise ValueError(f'Suite input changed: {path}')
    from utils.colmap_paths import colmap_model_dir
    from scripts.view_sparse_observations import viewer_cloud_path
    source = Path(suite['source'])
    cloud = viewer_cloud_path(source, colmap_model_dir(source), args.point_cloud)
    if not cloud.is_file() and args.point_cloud is None:
        cloud = Path(suite['cloud']['original'])
    cloud = cloud.resolve(strict=True)
    original = Path(suite['cloud']['original']).resolve()
    selection = Path(suite['selection'])
    names = json.loads(selection.read_text())['selected_images']
    records = read_colmap_records(suite['source'])
    if not names or len(set(names)) != len(names) or not set(names) < records.keys():
        raise ValueError('The comparison needs distinct training names and a disjoint held-out set')
    if not math.isfinite(args.voxel_size) or args.voxel_size <= 0:
        raise ValueError('Voxel size must be positive and finite')
    plan = dict(variant=args.variant, revision=REVISION, checkout=str(args.source_code.resolve()),
                dependencies=str(args.dependencies.resolve()), voxel_size=args.voxel_size,
                source=suite['source'], cloud=str(cloud), inputs={**suite['input_fingerprints'], str(cloud):digest(cloud)},
                iterations=STEPS[-1], train_views=len(names), heldout_views=len(records)-len(names))
    path = args.model/'upstream.json'
    if args.model.exists():
        if (not path.is_file() or json.loads(path.read_text()) != plan
                or digest(args.model/'keyframes.json') != digest(selection)):
            raise ValueError('Existing run differs. Use a fresh SCAFFOLD_RUN_VERSION.')
    else:
        with tempfile.TemporaryDirectory(dir=args.model.parent) as temp:
            staged = Path(temp)/'model'
            staged.mkdir()
            atomic_json(staged/'upstream.json', plan)
            shutil.copyfile(selection, staged/'keyframes.json')
            if cloud == original:
                (staged/'input.ply').symlink_to(Path(suite['cloud']['cached']).resolve())
            else:
                from scripts.prepare_orange4f import canonical_cloud
                canonical_cloud(source, staged/'input.ply', explicit=cloud)
            staged.rename(args.model)
    print(f"Full initialization cloud: {cloud}", flush=True)
    print(f"{args.variant}: {len(names)} train / {plan['heldout_views']} held out · {args.model}", flush=True)


def parameters(api, model, plan):
    parser = argparse.ArgumentParser()
    mp = api.arguments.ModelParams(parser)
    op = api.arguments.OptimizationParams(parser)
    pp = api.arguments.PipelineParams(parser)
    flags = ['-s', plan['source'], '-m', str(model), '-r', '1', '--data_device', 'cpu',
             '--appearance_dim', '0', '--voxel_size', str(plan['voxel_size'])]
    if plan['variant'] == 'fixed':
        flags += ['--update_until', '0']
    args = parser.parse_args(flags)
    return mp.extract(args), op.extract(args), pp.extract(args)


def runtime(plan):
    # Run in a fresh process so the two incompatible extensions never share an import name.
    checkout = Path(plan['checkout'])
    head = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'], text=True).strip()
    if head != REVISION or plan['revision'] != REVISION:
        raise ValueError(f'Expected upstream Scaffold-GS {REVISION}, found {head}')
    sys.path[:0] = [str(checkout), plan['dependencies']]
    import arguments
    import gaussian_renderer
    from gaussian_renderer import network_gui
    import scene
    import diff_gaussian_rasterization as raster
    from scene.dataset_readers import fetchPly, getNerfppNorm
    from utils.camera_utils import loadCam, camera_to_JSON
    from utils.general_utils import safe_state
    from utils.loss_utils import l1_loss, ssim
    if (not Path(gaussian_renderer.__file__).resolve().is_relative_to(checkout)
            or not Path(scene.__file__).resolve().is_relative_to(checkout)
            or not Path(raster.__file__).resolve().is_relative_to(Path(plan['dependencies']))
            or not hasattr(raster.GaussianRasterizer, 'visible_filter')):
        raise ImportError('Expected the upstream Scaffold model, renderer and CUDA visible_filter extension')
    return SimpleNamespace(arguments=arguments, renderer=gaussian_renderer, scene=scene, fetch=fetchPly,
                           norm=getNerfppNorm, load_cam=loadCam, seed=safe_state, l1=l1_loss, ssim=ssim, gui=network_gui, camera_json=camera_to_JSON)


class CameraView:
    """Keep cameras addressable by name without holding thousands of full RGB tensors."""
    def __init__(self, name, load):
        self.image_name, self._load = name, load

    def __getattr__(self, key):
        return getattr(self._load(self.image_name), key)


def cameras(api, dataset, records):
    import numpy as np
    from PIL import Image
    indices = {name: i for i, name in enumerate(sorted(records))}

    @lru_cache(maxsize=4)
    def load(name):
        row = records[name]
        with Image.open(row['image_path']) as image:
            info = SimpleNamespace(uid=indices[name], image_name=name, image=image,
                                   R=np.array(row['R']), T=np.array(row['T']),
                                   FovX=row['fov_x'], FovY=row['fov_y'])
            return api.load_cam(dataset, indices[name], info, 1.)
    return {n: CameraView(n, load) for n in records}


def training_function(source, bindings):
    # Importing upstream train.py also chooses a different GPU and loads VGG-LPIPS on
    # that GPU. Load its unmodified training function, binding our dataset and reporting.
    tree = ast.parse(source.read_text(), filename=str(source))
    func, = (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'training')
    module = ast.Module(body=[func], type_ignores=[])
    exec(compile(module, str(source), 'exec'), bindings)
    return bindings['training']


def train(api, model, plan, dataset, opt, pipe, records, selected, views):
    import numpy as np
    import torch
    from tqdm import tqdm
    from utils.checkpoint_snapshots import atomic_json, atomic_write
    from utils.training_previews import DEFAULTS, TrainingPreviewer, select_names

    if (model/'trained.json').exists():
        print(f'Training already completed in {model}.', flush=True)
        return
    if (model/'debug_previews/history.json').exists() or (model/'point_cloud').exists():
        raise FileExistsError('Partial upstream run exists. Use a fresh SCAFFOLD_RUN_VERSION for a new training run.')
    api.seed(False)
    cfg = argparse.Namespace(**vars(dataset))
    cfg.model_type, cfg.scaffold_upstream = 'scaffold', True
    cfg.scaffold_feat_dim, cfg.scaffold_n_offsets, cfg.scaffold_voxel_size = dataset.feat_dim, dataset.n_offsets, dataset.voxel_size
    cfg.keyframe_manifest, cfg.antialiasing = str(model/'keyframes.json'), False
    atomic_write(model/'cfg_args', lambda p: p.write_text(str(cfg)))
    preview_names = select_names(selected, set(records)-set(selected), DEFAULTS)
    frames = [(split, views[name]) for split, names in preview_names.items() for name in names]
    train_views = [views[name] for name in sorted(selected)]
    poses = []
    for name in sorted(selected):
        row = records[name]
        poses.append(SimpleNamespace(R=np.array(row['R']), T=np.array(row['T']), image_name=name,
                                     width=row['width'], height=row['height'], FovX=row['fov_x'], FovY=row['fov_y']))
    atomic_json(model/'cameras.json', [api.camera_json(i, cam) for i, cam in enumerate(poses)])
    extent = api.norm(poses)['radius']

    class SuiteScene:
        def __init__(self, dataset, gaussians, **unused):
            self.model_path, self.gaussians = str(model), gaussians
            gaussians.set_appearance(len(train_views))
            gaussians.create_from_pcd(api.fetch(model/'input.ply'), extent)
            self.preview = TrainingPreviewer(model/'debug_previews', frames, DEFAULTS, dict(
                source=plan['source'], eligible_train=len(train_views), eligible_heldout=len(records)-len(train_views),
                heldout_protocol='Stride-3 keyframe complement', state_timing='upstream reporting point, before optimizer step'))

        def getTrainCameras(self):
            return train_views

        def save(self, iteration):
            api.scene.Scene.save(self, iteration)

    def report(writer, name, step, ll1, loss, loss_fn, elapsed, testing, scene, render, render_args, *unused):
        if step == 1 or step % 500 == 0:
            print(f'[{step}/{opt.iterations}] anchors={len(scene.gaussians.get_anchor):,} loss={loss.item():.5f}', flush=True)
        if scene.preview.due(step, opt.iterations):
            scene.gaussians.eval()
            try:
                scene.preview.capture(step, scene.gaussians, draw, *render_args)
            finally:
                scene.gaussians.train()

    def draw(cam, gs, pipeline, bg, **unused):
        visible = api.renderer.prefilter_voxel(cam, gs, pipeline, bg)
        return api.renderer.render(cam, gs, pipeline, bg, visible_mask=visible)

    loop = training_function(Path(plan['checkout'])/'train.py', dict(
        GaussianModel=api.scene.GaussianModel, Scene=SuiteScene, torch=torch, tqdm=tqdm, randint=random.randint,
        network_gui=api.gui, prefilter_voxel=api.renderer.prefilter_voxel, render=api.renderer.render,
        l1_loss=api.l1, ssim=api.ssim, prepare_output_and_logger=lambda dataset: None, training_report=report))
    loop(dataset, opt, pipe, 'orange4f', [], STEPS, [], None, -1, logger=logging.getLogger('scaffold'))
    atomic_json(model/'trained.json', dict(iteration=opt.iterations, revision=REVISION))


def load_saved_model(api, dataset, folder):
    gs = api.scene.GaussianModel(feat_dim=dataset.feat_dim, n_offsets=dataset.n_offsets,
                                 voxel_size=dataset.voxel_size, appearance_dim=dataset.appearance_dim)
    gs.load_ply_sparse_gaussian(str(folder/'point_cloud.ply'))
    gs.load_mlp_checkpoints(str(folder))
    gs.eval()
    return gs


def saved_renderer(api, model, dataset, pipe, iteration):
    import torch
    gs = load_saved_model(api, dataset, model/f'point_cloud/iteration_{iteration}')
    bg = torch.zeros(3, device='cuda')
    def draw(camera):
        visible = api.renderer.prefilter_voxel(camera, gs, pipe, bg)
        return api.renderer.render(camera, gs, pipe, bg, visible_mask=visible)['render']
    return draw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--suite', type=Path, required=True)
    prep.add_argument('--variant', choices=('fixed','adaptive'), required=True)
    prep.add_argument('--source-code', type=Path, required=True)
    prep.add_argument('--dependencies', type=Path, required=True)
    prep.add_argument('--point-cloud', type=Path, default=os.environ.get('SCAFFOLD_INDEXED_PLY'),
                      help='Override the full viewer cloud, using the same PLY as the indexed runs')
    prep.add_argument('--voxel-size', type=float, required=True)
    for action in ('train','score','render'):
        child = commands.add_parser(action)
        child.add_argument('--model', type=Path, required=True)
        if action == 'render':
            child.add_argument('--selection', type=Path, required=True)
            child.add_argument('--iteration', type=int, required=True)
            child.add_argument('--output', type=Path, required=True)
    prep.add_argument('--model', type=Path, required=True)
    args = p.parse_args()
    model = args.model = args.model.resolve()
    if args.action == 'prepare':
        return prepare(args)
    plan = json.loads((model/'upstream.json').read_text())
    api = runtime(plan)
    dataset, opt, pipe = parameters(api, model, plan)
    from scripts.view_comparisons import read_colmap_records
    records = read_colmap_records(plan['source'])
    selected = json.loads((model/'keyframes.json').read_text())['selected_images']
    views = cameras(api, dataset, records)
    if args.action == 'train':
        return train(api, model, plan, dataset, opt, pipe, records, selected, views)

    import torch
    from scripts.view_comparisons import model_info, write_all_frame_metrics
    from utils.region_loss import score
    from tqdm import tqdm
    spec = model_info(model)
    with torch.no_grad():
        draw = saved_renderer(api, model, dataset, pipe, getattr(args, 'iteration', STEPS[-1]))
        if args.action == 'render':
            from scripts.render_views_worker import save_rendered_views
            names = json.loads(args.selection.read_text())['names']
            return save_rendered_views(spec, names, args.output, draw, lambda row, i: views[row['name']], None)
        rows = {}
        for name in tqdm(sorted(records), desc='Scoring all registered frames'):
            camera = views[name]
            pred = draw(camera).clamp(0, 1)
            target = camera.original_image[:3].to(pred.device)
            if pred.shape != target.shape or not torch.isfinite(pred).all():
                raise ValueError(f'Invalid render for {name}')
            rows[name] = dict(psnr_db=score(pred, target, None).item())
        summary = write_all_frame_metrics(spec, rows, STEPS[-1], False)
        print(summary['mean_psnr_db'], flush=True)


if __name__ == '__main__':
    main()
