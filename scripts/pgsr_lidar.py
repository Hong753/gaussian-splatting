"""LiDAR plane guidance for the pinned PGSR trainer. All lengths use scene units."""
import json
import math
import os
from pathlib import Path
import time

import torch

WARMUP = 7000
RAMP = 2000
STEPS = (WARMUP, 30000, 50000)
NORMAL_STEP = .1


def config():
    return json.loads(Path(os.environ['PGSR_LIDAR_CONFIG']).read_text())


class PlaneGuidance:
    def __init__(self, table, ids, guided):
        # Original anchor XYZ, trusted unit normal (zero if ambiguous), support radius, log sigma cap.
        self.table, self.ids, self.guided = table, ids, guided

    @classmethod
    @torch.no_grad()
    def fit(cls, model, guided):
        from utils.surface_geometry import estimate_patches
        xyz = model.get_xyz.detach()
        points = xyz.cpu().numpy()
        table = xyz.new_empty((len(xyz), 8))
        table[:, :3] = xyz
        table[:, 7] = model._scaling[:, 0] + math.log(2.)
        trusted_count = 0
        for start, normals, confidence, support in estimate_patches(points, neighbors=24, batch=50000):
            rows = slice(start, start + len(normals))
            normal = torch.as_tensor(normals, device=xyz.device)
            keep = torch.as_tensor(confidence >= .6, device=xyz.device)
            normal = torch.where(normal[:, 2:] < 0, -normal, normal)
            table[rows, 3:6] = normal * keep[:, None]
            table[rows, 6] = torch.as_tensor(support, device=xyz.device)
            # Rotate local +z onto the sign-equivalent upward PCA normal.
            w = ((1 + normal[:, 2]) * .5).sqrt()
            quat = torch.stack((w, -normal[:, 1] / (2*w), normal[:, 0] / (2*w), torch.zeros_like(w)), 1)
            model._rotation[rows][keep] = quat[keep]
            model._scaling[rows, 2] += keep * math.log(.2)
            trusted_count += int(keep.sum())
            print(f'Fitted LiDAR planes {rows.stop:,}/{len(points):,} ({trusted_count:,} trusted)', flush=True)
        return cls(table, torch.arange(len(xyz), device=xyz.device), guided)

    def local_normals(self, xyz, ids):
        rows = self.table[ids]
        inside = (xyz - rows[:, :3]).square().sum(1) <= rows[:, 6].square()
        return rows[:, 3:6] * inside[:, None]

    @staticmethod
    def damp(delta, normal, fraction):
        return delta - (1 - fraction) * (delta * normal).sum(1, keepdim=True) * normal

    def children(self, centers, mask, offsets, copies):
        parents = centers[mask].repeat(copies, 1)
        ids = self.ids[mask].repeat(copies)
        if self.guided:
            offsets = self.damp(offsets, self.local_normals(parents, ids), 0.)
        self.ids = torch.cat((self.ids, ids))
        return parents + offsets

    @torch.no_grad()
    def step(self, model):
        xyz = model.get_xyz
        # Adam preconditions each world axis separately. Project its actual
        # displacement after step(), including momentum, on an oblique plane.
        before = xyz.detach().clone() if self.guided and xyz.grad is not None else None
        model.optimizer.step()
        moments = model.optimizer.state.get(model._scaling, {})
        for start in range(0, len(xyz), 131072):
            rows = slice(start, start + 131072)
            ids = self.ids[rows]
            if before is not None:
                normal = self.local_normals(before[rows], ids)
                delta = xyz[rows] - before[rows]
                xyz[rows].sub_((1 - NORMAL_STEP) * (delta * normal).sum(1, keepdim=True) * normal)
            scale = model._scaling[rows]
            cap = self.table[ids, 7:8]
            clipped = scale > cap
            scale.copy_(torch.minimum(scale, cap))
            for name in ('exp_avg', 'exp_avg_sq'):
                if name in moments:
                    moments[name][rows].masked_fill_(clipped, 0.)


def position_schedule(model, phase):
    model._xyz.requires_grad_(phase > 0)
    rate = model.update_learning_rate(max(0, phase - RAMP)) * min(1., max(0., phase / RAMP))
    for group in model.optimizer.param_groups:
        if group['name'] == 'xyz':
            group['lr'] = rate
    return rate


def setup(scene, opt, pipe, render, checkpoint):
    from utils.training_previews import DEFAULTS, select_names, TrainingPreviewer
    cfg = config()
    model = scene.gaussians
    guided = cfg['variant'] == 'guided'
    start = 0
    if checkpoint:
        state, start = torch.load(checkpoint, weights_only=False)
        table, ids = state[-2:]
        model.restore(state[:-2], opt)
        model.lidar = PlaneGuidance(table, ids, guided)
    else:
        model.lidar = PlaneGuidance.fit(model, guided)
    if len(model.lidar.ids) != len(model.get_xyz):
        raise ValueError('Point and plane-reference counts differ')
    train, test = scene.getTrainCameras(), scene.getTestCameras()
    expected = json.loads((Path(cfg['source'])/'split.json').read_text())
    for cameras, split in ((train, 'train'), (test, 'test')):
        actual = sorted(c.image_name for c in cameras)
        if actual != sorted(expected[split]):
            raise ValueError(f'Loaded PGSR {split} cameras differ from the selected split')
    selected = select_names(expected['train'], expected['test'], DEFAULTS)
    frames = []
    for cameras, split in ((train, 'train'), (test, 'held-out')):
        by_name = {c.image_name: c for c in cameras}
        for name in selected[split]:
            camera = by_name[name]
            camera.original_image = camera.get_image()[0].cpu()
            frames.append((split, camera))
    root = Path(scene.model_path)/'debug_previews'
    if root.exists() and any(p.is_dir() and int(p.name.split('_')[-1]) > start for p in root.glob('iteration_[0-9]*')):
        root.rename(root.with_name(f'debug_previews_before_restart_{time.time_ns()}'))
    scene._previewer = TrainingPreviewer(root, frames, DEFAULTS, dict(source=cfg['original_source'],
        eligible_train=len(train), eligible_heldout=len(test), heldout_protocol='Sharp stride-3 complement'))
    scene._preview_render = rgb_renderer(render)
    neighbors = sum(bool(c.nearest_id) for c in train)
    print(f'PGSR {cfg["variant"]}: {len(train)} train, {len(test)} held-out, '
          f'{neighbors}/{len(train)} training views have a multi-view neighbor. Previews: {root}', flush=True)
    return start


def rgb_renderer(render):
    def rgb(camera, model, pipe, bg, **kwargs):
        return render(camera, model, pipe, bg, return_plane=False, return_depth_normal=False)
    return rgb


def finish(scene, iteration, pipe, background):
    from utils.checkpoint_snapshots import publish_snapshot
    if scene._previewer.due(iteration, STEPS[-1]):
        scene._previewer.capture(iteration, scene.gaussians, scene._preview_render, pipe, background)
    if iteration in STEPS:
        publish_snapshot(scene, iteration)
