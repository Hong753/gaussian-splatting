"""Sparse camera-z supervision for the pinned Orange4F PGSR trainer."""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.checkpoint_snapshots import atomic_write, digest


def scan_records(manifest):
    """Each file holds first-return XYZ deskewed into the listed scanner frame."""
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    if data.get('format') != 'scanner_xyz_v1' or not data.get('scans'):
        raise ValueError(f'Expected scanner_xyz_v1 and a nonempty scans array in {manifest}')
    records = []
    for row in data['scans']:
        pose = np.asarray(row['scanner_to_world'], dtype=np.float64)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
            raise ValueError(f'Invalid rigid scanner_to_world for {row["xyz"]}')
        path = (manifest.parent / row['xyz']).resolve(strict=True)
        records.append((path, pose))
    return records


def depth_config(source, requested_manifest):
    manifest = requested_manifest or Path(source)/'lidar_scans.json'
    mode = 'scanner' if requested_manifest or manifest.is_file() else 'projected'
    cfg = dict(mode=mode, warmup=7000, ramp_end=9000, taper_start=30000, end=50000,
               peak_weight=.05, final_weight=.01, beta=.05, min_alpha=.5, min_incidence=.08,
               max_targets=8192, cell_px=4 if mode == 'projected' else 1,
               interval=4 if mode == 'scanner' else 1)
    if mode == 'scanner':
        records = scan_records(manifest)
        h = hashlib.sha256()
        for path, _ in records:
            points = np.load(path, mmap_mode='r', allow_pickle=False)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or points.dtype.kind != 'f':
                raise ValueError(f'Expected nonempty floating-point Nx3 scanner returns in {path}')
            h.update(bytes.fromhex(digest(path)))
        cfg.update(manifest=str(manifest.resolve()), manifest_sha256=digest(manifest),
                   scans_sha256=h.hexdigest(), scans=len(records), face_px=512)
    return cfg


def scheduled_weight(step, cfg):
    if step <= cfg['warmup']:
        return 0.
    ramp = min(1., (step-cfg['warmup']) / (cfg['ramp_end']-cfg['warmup']))
    taper = min(1., max(0., (step-cfg['taper_start']) / (cfg['end']-cfg['taper_start'])))
    return ramp * ((1-taper)*cfg['peak_weight'] + taper*cfg['final_weight'])


@torch.no_grad()
def front_samples(xyz, height, width, fx, fy, cell, conservative, limit, seed):
    """Use the plane rasterizer's ray center (W/2,H/2) and optical-axis z."""
    z = xyz[:, 2]
    u = torch.floor(fx*xyz[:, 0]/z + width*.5 + .5).long()
    v = torch.floor(fy*xyz[:, 1]/z + height*.5 + .5).long()
    valid = (z > .2) & (z < 100) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[valid], v[valid], z[valid]
    nx, ny = math.ceil(width/cell), math.ceil(height/cell)
    index = (v//cell)*nx + u//cell
    front = z.new_full((nx*ny,), float('inf'))
    front.scatter_reduce_(0, index, z, reduce='amin')
    # Resolve equal-depth ties deterministically, keeping an actual observed pixel.
    pixels = v*width + u
    nearest = torch.full((nx*ny,), width*height, device=z.device, dtype=torch.long)
    wins = z == front[index]
    nearest.scatter_reduce_(0, index[wins], pixels[wins], reduce='amin')
    keep = front.isfinite()
    if conservative:
        # Ignore background leaking through small gaps and isolated projections.
        grid = front.reshape(ny, nx)
        padded = F.pad(grid, (1, 1, 1, 1), value=float('inf'))
        support = torch.zeros_like(grid, dtype=torch.int32)
        closest = grid.clone()
        tolerance = .03 + .03*grid
        for dy in range(3):
            for dx in range(3):
                neighbor = padded[dy:dy+ny, dx:dx+nx]
                support += neighbor.isfinite() & ((neighbor-grid).abs() <= tolerance)
                closest = torch.minimum(closest, neighbor)
        keep &= ((support >= 4) & (grid <= closest+tolerance)).ravel()
    ids = torch.where(keep)[0]
    if len(ids) > limit:
        rng = torch.Generator(device=z.device).manual_seed(seed)
        ids = ids[torch.randperm(len(ids), device=z.device, generator=rng)[:limit]]
    return nearest[ids], front[ids]


def relative_loss(pkg, samples, cfg, cam):
    depths = pkg['plane_depth'].reshape(-1)
    pixels, target = (t.to(depths.device) for t in samples)
    pred = depths[pixels]
    alpha = pkg['rendered_alpha'].detach().reshape(-1)[pixels]
    with torch.no_grad():
        x = (pixels % cam.image_width - cam.Cx)/cam.Fx
        y = (pixels // cam.image_width - cam.Cy)/cam.Fy
        normal = pkg['rendered_normal'].detach().reshape(3, -1)[:, pixels]
        incidence = (normal[0]*x + normal[1]*y + normal[2]).abs() / (1+x*x+y*y).sqrt()
        valid = ((alpha >= cfg['min_alpha']) & (incidence >= cfg['min_incidence']*alpha)
                 & pred.isfinite() & (pred > 0))
    # Select before arithmetic so unsupported or singular plane depths contribute zero.
    error = (pred[valid]-target[valid]) / target[valid]
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='sum', beta=cfg['beta']) / max(1, len(error))
    return loss, len(error)


def cuda_depth_smoke(pipe, render, cfg):
    from scene import GaussianModel
    from scene.cameras import Camera
    from utils.graphics_utils import BasicPointCloud

    coords = np.linspace(-.4, .4, 9, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)
    xyz = np.stack((x, y, np.full_like(x, 2)), axis=-1).reshape(-1, 3)
    model = GaussianModel(0)
    model.create_from_pcd(BasicPointCloud(xyz, np.full_like(xyz, .5), np.zeros_like(xyz)), 1.)
    with torch.no_grad():
        model._scaling[:, 2] -= 3
        model._opacity.fill_(3)
    cam = Camera(0, np.eye(3), np.zeros(3), math.pi/2, math.pi/2, 64, 64, '', 'depth_smoke', 0,
                 preload_img=False, data_device='cpu')
    pkg = render(cam, model, pipe, torch.zeros(3, device='cuda'), return_plane=True, return_depth_normal=False)
    pixels, z = front_samples(model.get_xyz.detach(), 64, 64, 32., 32., 1, False, 128, 0)
    loss, n = relative_loss(pkg, (pixels, z + .1), cfg, cam)
    grad, = torch.autograd.grad(loss, model.get_xyz)
    if not n or not torch.isfinite(grad).all() or grad[:, 2].sum() >= 0:
        raise RuntimeError('PGSR CUDA depth smoke failed: expected a finite gradient moving the plane farther away')
    print(f'PGSR CUDA sparse-depth gradient passed on {n} observed pixels.', flush=True)


def face_basis(face):
    forward = np.eye(3)[face//2] * (1 if face%2 == 0 else -1)
    ref = np.eye(3)[(face//2+1)%3]
    right = np.cross(ref, forward)
    return np.column_stack((right, np.cross(forward, right), forward))


class DepthSupervisor:
    def __init__(self, cfg, scene, start=0):
        self.cfg = cfg
        self.model = scene.gaussians
        self.cache = OrderedDict()
        self.scans = scan_records(cfg['manifest']) if cfg['mode'] == 'scanner' else []
        self.history = Path(scene.model_path)/'lidar_depth_metrics.csv'
        if self.history.is_file():
            rows = self.history.read_text().splitlines()
            kept = [r for i, r in enumerate(rows) if i == 0 or int(r.split(',')[0]) <= start]
            if len(kept) < len(rows):
                atomic_write(self.history, lambda p: p.write_text('\n'.join(kept)+'\n'))
        self.totals = np.zeros(4)
        print(f'LiDAR depth: {cfg["mode"]}, weight {cfg["peak_weight"]} after ramp, '
              f'one depth update per {cfg["interval"]} RGB steps. Metrics: {self.history}', flush=True)

    def projected(self, cam):
        name = cam.image_name
        if name not in self.cache:
            transform = cam.world_view_transform
            # PlaneGuidance.table always retains the original registered cloud.
            xyz = self.model.lidar.table[:, :3] @ transform[:3, :3] + transform[3, :3]
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')
            samples = front_samples(xyz, cam.image_height, cam.image_width, cam.Fx, cam.Fy,
                                    self.cfg['cell_px'], True, self.cfg['max_targets'], seed)
            self.cache[name] = tuple(t.cpu() for t in samples)
        return self.cache[name]

    def scanner(self, step):
        from scene.cameras import Camera
        rng = np.random.default_rng(step)
        scan_id = int(rng.integers(len(self.scans)))
        path, pose = self.scans[scan_id]
        if scan_id not in self.cache:
            points = np.load(path, allow_pickle=False)
            points = points[np.isfinite(points).all(axis=1)]
            if not len(points):
                raise ValueError(f'No finite scanner returns in {path}')
            axis = np.abs(points).argmax(axis=1)
            face_ids = 2*axis + (points[np.arange(len(points)), axis] < 0)
            faces = []
            size = self.cfg['face_px']
            for face in np.unique(face_ids):
                basis = face_basis(int(face))
                xyz = torch.from_numpy((points[face_ids == face] @ basis).astype(np.float32))
                samples = front_samples(xyz, size, size, size/2, size/2, 1, False,
                                        self.cfg['max_targets'], scan_id*6+int(face))
                if len(samples[0]):
                    rotation = pose[:3, :3] @ basis
                    cam = Camera(0, rotation, -rotation.T @ pose[:3, 3], math.pi/2, math.pi/2,
                                 size, size, '', f'lidar_{scan_id}_{face}', 0, preload_img=False, data_device='cpu')
                    faces.append((cam, samples))
            if not faces:
                raise ValueError(f'No in-range scanner returns in {path}')
            if len(self.cache) >= 8:
                self.cache.popitem(last=False)
            self.cache[scan_id] = faces
        self.cache.move_to_end(scan_id)
        faces = self.cache[scan_id]
        return faces[int(rng.integers(len(faces)))]

    def backward(self, rgb_loss, step, camera, pkg, pipe, background, render):
        cfg = self.cfg
        weight = scheduled_weight(step, cfg)
        if not weight or step % cfg['interval']:
            rgb_loss.backward()
            return
        if self.scans:
            # Release the RGB graph before allocating a second rasterizer graph.
            rgb_loss.backward()
            camera, samples = self.scanner(step)
            pkg = render(camera, self.model, pipe, background, return_plane=True, return_depth_normal=False)
        else:
            samples = self.projected(camera)
        loss, valid = relative_loss(pkg, samples, cfg, camera)
        # Preserve the mean depth weight when rendering one scanner face every N RGB updates.
        objective = loss * (weight * cfg['interval'])
        if not self.scans:
            objective = objective + rgb_loss
        objective.backward()
        self.totals += [1, len(samples[0]), valid, loss.detach().item()*valid]
        if step % 500 == 0:
            updates, samples_seen, used, error_sum = self.totals
            message = f'{step},{weight:.6g},{int(updates)},{int(samples_seen)},{int(used)},{error_sum/max(1,used):.6g}'
            with self.history.open('a') as stream:
                if stream.tell() == 0:
                    stream.write('step,weight,updates,candidates,accepted,mean_relative_huber\n')
                stream.write(message+'\n')
            print(f'LiDAR depth step {step:,}: {int(used):,}/{int(samples_seen):,} accepted, '
                  f'relative Huber {error_sum/max(1,used):.5f}, weight {weight:.3g}', flush=True)
            if not used:
                raise RuntimeError('No usable LiDAR depth in this logging window. Check cloud/scan alignment and coverage.')
            self.totals.fill(0)
