"""Execute the pinned PGSR trainer or render its checkpoint with its own rasterizer."""
import argparse
import json
import os
from pathlib import Path
import sys
import csv


def evaluate(scene, pipe, render, iteration, cfg):
    import torch
    from tqdm import tqdm
    from utils.checkpoint_snapshots import atomic_json, atomic_write
    from utils.training_previews import TrainingPreviewer, select_names, DEFAULTS
    from utils.region_loss import score
    root = Path(scene.model_path)
    metrics = root/'psnr_all_frames'/f'iteration_{iteration}'
    pool = {'train': scene.getTrainCameras(), 'held-out': scene.getTestCameras()}
    split = json.loads((Path(cfg['source'])/'split.json').read_text())
    chosen = select_names(split['train'], split['test'], {**DEFAULTS, 'train': 8, 'test': 8})
    background = torch.zeros(3, device='cuda')
    frames, rows, means = [], [], {}
    with torch.no_grad():
        for group, cameras in pool.items():
            cameras = sorted(cameras, key=lambda c: c.image_name)
            expected = split['train' if group == 'train' else 'test']
            if [c.image_name for c in cameras] != sorted(expected):
                raise ValueError(f'Incomplete evaluation for {group}')
            values = []
            for camera in tqdm(cameras, desc=f'PGSR {group} {iteration}'):
                image = render(camera, scene.gaussians, pipe, background)['render'].clamp(0, 1)
                target = camera.get_image()[0].clamp(0, 1)
                value = score(image, target).item()
                values.append(value)
                rows.append((camera.image_name, group, value))
                if camera.image_name in chosen[group]:
                    camera.original_image = target.cpu()
                    frames.append((group, camera))
            means[group] = sum(values)/len(values) if values else None

    def write_rows(path):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('image_name', 'split', 'psnr_db'))
            writer.writerows(rows)

    gallery = root/'view_comparisons'/f'iteration_{iteration}'
    preview = TrainingPreviewer(gallery, frames, DEFAULTS, dict(source=cfg['original_source']))
    if not (gallery/f'iteration_{iteration:06d}'/'frames.json').exists():
        preview.capture(iteration, scene.gaussians, render, pipe, background)
    atomic_write(metrics/'per_frame.csv', write_rows)
    atomic_json(metrics/'summary.json', dict(iteration=iteration, model_type='pgsr', frames=len(rows),
        mean_psnr_db=means, metric='mean per-channel PSNR, then mean over images', gallery=str(gallery/'index.html')))
    print(f'{means} — {gallery / "index.html"}', flush=True)


def cuda_smoke(opt, pipe):
    import math
    import numpy as np
    import torch
    from gaussian_renderer import render
    from scene import GaussianModel
    from scene.cameras import Camera
    from utils.graphics_utils import BasicPointCloud
    from pgsr_lidar import PlaneGuidance, position_schedule

    grid = np.linspace(-.3, .3, 6, dtype=np.float32)
    x, y = np.meshgrid(grid, grid)
    xyz = np.stack((x, y, 2 + .3*x - .2*y), axis=-1).reshape(-1, 3)
    model = GaussianModel(3)
    model.create_from_pcd(BasicPointCloud(xyz, np.full_like(xyz, .5), np.zeros_like(xyz)), 1.)
    model.training_setup(opt)
    model.lidar = PlaneGuidance.fit(model, True)
    cam = Camera(0, np.eye(3), np.zeros(3), math.pi/2, math.pi/2, 48, 48, '', 'smoke', 0,
                 preload_img=False, data_device='cpu')
    bg = torch.zeros(3, device='cuda')
    initial = model.get_xyz.detach().clone()
    for phase in (0, 2000):
        position_schedule(model, phase)
        out = render(cam, model, pipe, bg, return_plane=phase > 0, return_depth_normal=phase > 0)
        loss = (out['render'] - .7).square().mean()
        if phase:
            loss += .001 * (out['plane_depth'].mean() + out['depth_normal'].square().mean())
        loss.backward()
        for group in model.optimizer.param_groups:
            grad = group['params'][0].grad
            if grad is not None and not torch.isfinite(grad).all():
                raise RuntimeError(f'PGSR CUDA smoke produced nonfinite {group["name"]} gradients')
        model.lidar.step(model)
        if phase == 0 and not torch.equal(initial, model.get_xyz):
            raise RuntimeError('PGSR CUDA smoke moved frozen centers')
        model.optimizer.zero_grad(set_to_none=True)
    print('PGSR CUDA RGB, plane backward, and fixed-center warmup smoke passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'eval'))
    parser.add_argument('checkout', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('iteration', type=int, default=0, nargs='?')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(args.protocol.read_text())
    os.environ['PGSR_LIDAR_CONFIG'] = str(args.protocol.resolve())
    checkout = args.checkout.resolve()
    sys.path[:0] = [str(checkout/'extension'), str(checkout/'src')]
    import utils
    utils.__path__ = [str(checkout/'src/utils'), str(root/'utils')]
    import torch
    import diff_plane_rasterization
    if not Path(diff_plane_rasterization.__file__).resolve().is_relative_to(checkout/'extension'):
        raise RuntimeError('Expected the isolated PGSR plane rasterizer')
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from pgsr_lidar import rgb_renderer, STEPS, WARMUP, RAMP
    p = argparse.ArgumentParser()
    mp, pp, op = ModelParams(p), PipelineParams(p), OptimizationParams(p)
    cli = p.parse_args(['-s', cfg['source'], '-m', cfg['model'], '-r', '1', '--data_device', 'cpu',
                        '--iterations', str(STEPS[-1]), '--position_lr_init', str(cfg['position_lr_endpoints'][0]),
                        '--position_lr_final', str(cfg['position_lr_endpoints'][1]),
                        '--position_lr_max_steps', str(STEPS[-1] - WARMUP - RAMP),
                        '--max_all_points', str(cfg['max_all_points'])])
    ds, pipe, opt = mp.extract(cli), pp.extract(cli), op.extract(cli)
    ds.model_type = 'pgsr'
    ds.keyframe_manifest = str(Path(cfg['model'])/'keyframes.json')
    if args.mode == 'train':
        # Explicitly load official train.py (the native fork also has a train.py).
        import train
        if Path(train.__file__).resolve() != checkout/'src/train.py':
            raise RuntimeError('Wrong PGSR training module')
        torch.set_num_threads(8)
        cuda_smoke(opt, pipe)
        train.safe_state(False)
        checkpoint = str(Path(cfg['model'])/f'chkpnt{args.iteration}.pth') if args.iteration else None
        train.training(ds, opt, pipe, [], [], [], checkpoint, -100)
    else:
        from scene import GaussianModel, Scene
        from gaussian_renderer import render
        model = GaussianModel(ds.sh_degree)
        scene = Scene(ds, model, load_iteration=args.iteration, shuffle=False)
        evaluate(scene, pipe, rgb_renderer(render), args.iteration, cfg)


if __name__ == '__main__':
    main()
