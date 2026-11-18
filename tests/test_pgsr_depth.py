import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.pgsr_depth import DepthSupervisor, depth_config, face_basis, front_samples, relative_loss, scheduled_weight
from scripts.pgsr_depth_eval import compare, gradient_error
from scripts.run_orange4f_pgsr import protocol
from test_region_ablation import region_source


def test_projected_front_support_and_depth_units():
    v, u = torch.meshgrid(torch.arange(9), torch.arange(9), indexing='ij')
    xyz = torch.stack(((u-4.5)/9*2, (v-4.5)/9*2, torch.full_like(u, 2)), -1).reshape(-1, 3)
    hole = (v == 4) & (u == 4)
    points = torch.cat([xyz[~hole.flatten()], xyz*2, -xyz])
    pixels, z = front_samples(points, 9, 9, 9, 9, 1, True, 1000, 0)
    assert 40 not in pixels and len(z) == 80
    torch.testing.assert_close(z, torch.full_like(z, 2))
    # A scanner observing a real hit through the hole keeps it, with the nearest z on other rays.
    pixels, z = front_samples(points, 9, 9, 9, 9, 1, False, 1000, 0)
    assert len(set(pixels.tolist())) == 81 and z[pixels == 40].item() == 4


def make_package(values):
    z = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    normal = torch.zeros(3, 1, len(values))
    normal[2] = -1
    return z, dict(plane_depth=z.view(1,1,-1), rendered_alpha=torch.ones(1,1,len(values)),
                   rendered_normal=normal)


def camera(n):
    return SimpleNamespace(image_width=n, Cx=n/2, Cy=.5, Fx=n, Fy=n)


def test_depth_loss_gradient_and_exclusions(tmp_path):
    cfg = depth_config(tmp_path, None)
    z, pkg = make_package([2.5, 1.5, 2, float('nan'), 3, 4])
    pkg['rendered_alpha'][0,0,4] = .49
    pkg['rendered_normal'][:,:,5] = 0
    loss, n = relative_loss(pkg, (torch.arange(6), torch.full((6,), 2.)), cfg, camera(6))
    assert n == 3
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([1/6, -1/6, 0, 0, 0, 0]))
    empty, n = relative_loss(pkg, (torch.tensor([3,4,5]), torch.ones(3)), cfg, camera(6))
    assert n == 0 and torch.isfinite(empty) and empty.item() == 0


def write_manifest(tmp_path):
    pose = np.eye(4)
    pose[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    pose[:3,3] = [2,3,4]
    points = np.concatenate([np.array([[.4,-.3,3.]]) @ face_basis(i).T for i in range(6)])
    np.save(tmp_path/'scan.npy', points)
    manifest = tmp_path/'lidar_scans.json'
    manifest.write_text(json.dumps(dict(format='scanner_xyz_v1', scans=[dict(xyz='scan.npy', scanner_to_world=pose.tolist())])))
    return manifest, points, pose


def test_scanner_targets_reproject_from_actual_origin(tmp_path, monkeypatch):
    manifest, points, pose = write_manifest(tmp_path)
    cfg = depth_config(tmp_path, None)
    assert cfg['mode'] == 'scanner' and cfg['manifest'] == str(manifest)
    def stub_camera(uid, r, t, fovx, fovy, w, h, path, name, *args, **kwargs):
        return SimpleNamespace(R=r, T=t, image_name=name, width=w)
    monkeypatch.setitem(sys.modules, 'scene.cameras', SimpleNamespace(Camera=stub_camera))
    d = DepthSupervisor(cfg, SimpleNamespace(gaussians=None, model_path=tmp_path))
    d.scanner(7004)
    assert len(d.cache[0]) == 6
    for cam, (px, depth) in d.cache[0]:
        i = int(cam.image_name.split('_')[-1])
        np.testing.assert_allclose(-cam.R @ cam.T, pose[:3,3])
        hit_world = pose[:3,:3] @ points[i] + pose[:3,3]
        hit_cam = cam.R.T @ hit_world + cam.T
        np.testing.assert_allclose(depth, hit_cam[2])
        assert np.linalg.norm(hit_cam) > depth.item()
        uv = np.floor(256*hit_cam[:2]/hit_cam[2] + 256.5).astype(int)
        assert px.item() == uv[1]*512 + uv[0]
    np.save(tmp_path/'scan.npy', points*2)
    assert depth_config(tmp_path, manifest)['scans_sha256'] != cfg['scans_sha256']
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='scanner_xyz_v1'):
        depth_config(tmp_path, None)


@pytest.mark.parametrize('mode', ['scanner','projected'])
def test_matched_warmup_and_gradient_path(mode, tmp_path):
    cfg = depth_config(tmp_path, None)
    cfg.update(mode=mode, interval=4 if mode == 'scanner' else 1)
    d = DepthSupervisor({**cfg, 'mode':'projected'}, SimpleNamespace(gaussians=None, model_path=tmp_path))
    d.cfg = cfg
    cam = camera(1)
    z, pkg = make_package([2.5])
    samples = (torch.tensor([0]), torch.tensor([2.]))
    d.projected = lambda _: samples
    d.scans = [1] if mode == 'scanner' else []
    d.scanner = lambda _: (cam, samples)
    rgb = torch.tensor(.5, requires_grad=True)
    for step in (7000, 8000, 9000):
        rgb.grad = None
        z.grad = None
        d.backward(rgb.square(), step, cam, pkg, None, None, lambda *args, **kwargs:pkg)
        assert rgb.grad.item() == 1
        if step == 7000:
            assert z.grad is None
        else:
            assert z.grad.item() == pytest.approx(.5*scheduled_weight(step,cfg)*cfg['interval'])
    assert scheduled_weight(7000,cfg) == 0 and scheduled_weight(9000,cfg) == .05
    assert scheduled_weight(30000,cfg) == .05 and scheduled_weight(50000,cfg) == .01
    # A retry from a checkpoint drops depth logs for the abandoned suffix.
    DepthSupervisor({**cfg, 'mode':'projected'}, SimpleNamespace(gaussians=None, model_path=tmp_path), 8000)
    with d.history.open() as stream:
        assert [r['step'] for r in csv.DictReader(stream)] == ['8000']


def test_protocol_and_paired_quality(tmp_path, region_source):
    scene, cloud, manifest = region_source
    suite = tmp_path/'suite'
    fixed = suite/'3dgs_fixed_lidar'
    fixed.mkdir(parents=True)
    (fixed/'cfg_args').write_text(f'Namespace(source_path={str(scene)!r}, images="images")')
    (fixed/'input.ply').symlink_to(cloud)
    (fixed/'keyframes.json').write_bytes(manifest.read_bytes())
    args = SimpleNamespace(suite=suite, variant='guided', source=None, output=None, scan_manifest=None)
    control, _, _, split = protocol(args)
    args.depth_supervision = True
    trial, _, _, _ = protocol(args)
    assert 'depth_supervision' not in control and trial['depth_supervision']['mode'] == 'projected'
    assert control['model'] != trial['model'] and control['fingerprints'] == trial['fingerprints']
    from pathlib import Path
    for run, offset in [(control, 0), (trial, 1)]:
        root = Path(run['model'])
        metrics = root/'psnr_all_frames/iteration_30000'
        metrics.mkdir(parents=True)
        (root/'dataset').mkdir()
        (root/'pgsr_run.json').write_text(json.dumps(run))
        (root/'dataset/split.json').write_text(json.dumps(split))
        (metrics/'summary.json').write_text('{}')
        with (metrics/'per_frame.csv').open('w') as out:
            w = csv.writer(out)
            w.writerow(('image_name','split','psnr_db'))
            for group, names in split.items():
                for name in names:
                    w.writerow((name,'train' if group == 'train' else 'held-out',20+offset))
    compare(Path(trial['model']), Path(control['model']), 30000)
    result = json.loads((metrics/'depth_comparison.json').read_text())
    assert result['splits']['held-out']['delta_db'] == 1 and result['splits']['held-out']['frames'] == 3
    gt = np.indices((10,10)).sum(axis=0) % 2
    assert gradient_error(gt, gt) == 0 and gradient_error(gt, np.zeros_like(gt)) == 1
