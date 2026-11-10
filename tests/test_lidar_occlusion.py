"""Independent geometry, arbitrary-view visibility, workflow, and scoped stopping."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.lidar_occlusion import fit_surfels, load_geometry, prepare_geometry, save_geometry, surfel_triangles
from utils.occlusion_depth import LidarDepthCache, clip_vertices
from utils.occlusion_culling import conservative_keep
from scripts import run_occlusion, stop_render_jobs
from scripts.occlusion_options import training_paths, prepare_training_geometry
from test_occlusion import camera


def plane():
    x, y = np.meshgrid(np.arange(-1, 1.001, .04), np.arange(-1, 1.001, .04))
    return np.stack((x.ravel(), y.ravel(), np.full(x.size, 2.)), axis=1).astype(np.float32)


def reference_rasterize(context, pos, tri, resolution, grad_db):
    """Small CPU triangle reference for the independent camera/raster integration."""
    h, w = resolution
    vertices = pos[0, :, :3]/pos[0, :, 3:]
    y, x = torch.meshgrid((torch.arange(h)+.5)*2/h-1, (torch.arange(w)+.5)*2/w-1, indexing='ij')
    result = torch.zeros(1, h, w, 4)
    depth = torch.full((h, w), float('inf'))
    for idx, face in enumerate(tri):
        a, b, c = vertices[face.long()]
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-10:
            continue
        u = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        v = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        z = u*a[2] + v*b[2] + (1-u-v)*c[2]
        hit = (u >= -1e-6) & (v >= -1e-6) & (u+v <= 1+1e-6) & (z >= -1) & (z <= 1) & (z < depth)
        depth[hit] = z[hit]
        result[0, :, :, 2][hit] = z[hit]
        result[0, :, :, 3][hit] = idx+1
    return result, None


class LidarOcclusionTests(unittest.TestCase):
    def test_local_surface_normals_radius_and_open_hole(self):
        points = plane()
        points = points[(np.abs(points[:, 0]) > .3) | (np.abs(points[:, 1]) > .3)]
        arrays, metadata = fit_surfels(points, voxel_size=.03, max_radius=.08, batch_size=131)
        self.assertGreater(metadata['trusted_fraction'], .7)
        np.testing.assert_allclose(np.abs(arrays['normals'][:, 2]), 1, atol=1e-5)
        self.assertTrue(np.all(arrays['radii'] <= .08))
        self.assertTrue(np.all(np.max(np.abs(arrays['centers'][:, :2]), axis=1)-arrays['radii'] > .2))
        # Thin lines do not become opaque surfaces.
        line = np.stack((np.linspace(-1, 1, 100), np.zeros(100), np.ones(100)), axis=1)
        with self.assertRaisesRegex(ValueError, 'No supported planar'):
            fit_surfels(line, voxel_size=.01)

    def test_geometry_reuse_and_source_settings_identity(self):
        from plyfile import PlyData, PlyElement
        with tempfile.TemporaryDirectory() as tmp:
            points = plane()
            vertex = np.empty(len(points), dtype=[('x','f4'),('y','f4'),('z','f4')])
            for i, axis in enumerate('xyz'):
                vertex[axis] = points[:, i]
            cloud, asset = Path(tmp)/'input.ply', Path(tmp)/'occlusion_geometry.npz'
            PlyData([PlyElement.describe(vertex, 'vertex')]).write(cloud)
            metadata = prepare_geometry(cloud, asset, voxel_size=.03)
            stamp = asset.stat().st_mtime_ns
            again = prepare_geometry(cloud, asset, voxel_size=.03)
            self.assertEqual(metadata['source_sha256'], again['source_sha256'])
            self.assertEqual(stamp, asset.stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError, 'different inputs/settings'):
                prepare_geometry(cloud, asset, voxel_size=.04)
            arrays, saved = load_geometry(asset)
            self.assertEqual(saved['surfels'], len(arrays['centers']))

    def test_octagons_stay_on_local_surface_and_inside_radius(self):
        centers = torch.tensor([[0., 0., 2.], [2., 0., 0.]])
        normals = torch.tensor([[0., 0., 1.], [1., 0., 0.]])
        radii = torch.tensor([.1, .2])
        vertices, faces = surfel_triangles(centers, normals, radii)
        offsets = vertices.reshape(2, 9, 3)-centers[:, None]
        torch.testing.assert_close((offsets*normals[:, None]).sum(-1), torch.zeros(2, 9))
        self.assertTrue((offsets.norm(dim=-1) <= radii[:, None]+1e-6).all())
        self.assertEqual(tuple(faces.shape), (16, 3))
        self.assertLess(int(faces.max()), len(vertices))

    def test_novel_camera_recomputes_first_surface_and_swaps_visibility(self):
        fake_dr = SimpleNamespace(RasterizeCudaContext=lambda **kw: object(), rasterize=reference_rasterize)
        modules = {'nvdiffrast': SimpleNamespace(torch=fake_dr), 'nvdiffrast.torch': fake_dr}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            asset = Path(tmp)/'geometry.npz'
            save_geometry(asset, [[0, 0, 2], [0, 0, 4]], [[0, 0, 1]]*2, [3, 3])
            provider = LidarDepthCache(asset, Path(tmp)/'cache', edge_pixels=0)
            front, back = camera(32), camera(32)
            projection = back.full_proj_transform.clone()
            back.world_view_transform = torch.diag(torch.tensor([-1., 1., -1., 1.]))
            back.world_view_transform[3, 2] = 6
            back.full_proj_transform = back.world_view_transform@projection
            xyz = torch.tensor([[0., 0., 2.], [0., 0., 4.]])
            scales, rotations = torch.full_like(xyz, .01), torch.tensor([[1., 0, 0, 0]]).repeat(2, 1)
            masks = []
            for cam in (front, back):
                tiles = provider.tiles(cam, torch.device('cpu'))
                torch.testing.assert_close(tiles, torch.full((2, 2), 2.), atol=.0001, rtol=0)
                masks.append(conservative_keep(cam, xyz, scales, rotations, tiles).tolist())
            self.assertEqual(masks, [[True, False], [False, True]])
            self.assertEqual(len(list(provider.directory.glob('*.npy'))), 2)

    def test_training_geometry_uses_exact_saved_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SimpleNamespace(model_path=tmp, occlusion_mode='lidar', occlusion_geometry='',
                occlusion_cache='', occlusion_points='', occlusion_voxel_size=.05, occlusion_max_radius=.1,
                occlusion_neighbors=24, occlusion_confidence=.6)
            training_paths(dataset)
            self.assertEqual(dataset.occlusion_geometry, str(Path(tmp)/'occlusion_geometry.npz'))
            with patch('utils.lidar_occlusion.prepare_geometry') as prepare:
                prepare_training_geometry(dataset)
            self.assertEqual(prepare.call_args.args[0], str(Path(tmp)/'input.ply'))

    def test_workflow_native_training_then_all_frame_metrics_then_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source/sparse').mkdir(parents=True)
            for name in ['images.txt', 'cameras.txt']:
                (root/'source/sparse'/name).touch()
            args = run_occlusion.parser().parse_args(['--source', str(root/'source'), '--model', str(root/'model'), '--dry-run'])
            plan = run_occlusion.build_plan(args)
            names = [s['name'] for s in plan['stages']]
            self.assertEqual(names, ['train', 'trained_on_psnr', 'trained_on_images', 'trained_off_psnr', 'trained_off_images'])
            command = plan['stages'][0]['argv']
            self.assertEqual(command[0], 'train.py')
            self.assertIn('lidar', command)
            self.assertIn('--antialiasing', command)
            self.assertNotIn('--train_test_exp', command)
            self.assertNotIn('proxy', ' '.join(command).lower())
            self.assertIn('--in-process', plan['stages'][2]['argv'])
            self.assertFalse((root/'model').exists())

    def test_workflow_invocation_stays_in_process_and_preserves_errors(self):
        argv, stdout = sys.argv, sys.stdout
        with patch('runpy.run_path', side_effect=SystemExit(0)) as run:
            run_occlusion.invoke(['train.py', '--iterations', '2'])
            self.assertEqual(run.call_args.kwargs['run_name'], '__main__')
        self.assertIs(sys.argv, argv)
        self.assertIs(sys.stdout, stdout)
        with patch('runpy.run_path', side_effect=RuntimeError('training failure')):
            with self.assertRaisesRegex(RuntimeError, 'training failure'):
                run_occlusion.invoke(['train.py'])

    def test_stop_scope_excludes_other_gpus_and_other_jobs(self):
        gpu = '0, GPU-a\n1, GPU-b\n2, GPU-c\n'
        processes = 'GPU-a, 10\nGPU-b, 11\nGPU-c, 12\nGPU-a, 13\nGPU-c, 13\n'
        self.assertEqual(stop_render_jobs.gpu_processes(gpu, processes, [0, 1]), {10:{0}, 11:{1}})
        self.assertTrue(stop_render_jobs.matches_job(['python', '/repo/scripts/render.py', '--runs', 'proxygs_stride3_original'], ['proxygs_stride3_original']))
        self.assertFalse(stop_render_jobs.matches_job(['python', '/repo/train.py', '--model', 'unrelated'], ['proxygs_stride3_original']))
        self.assertFalse(stop_render_jobs.matches_job(['python', '/repo/scripts/render.py', '--runs', 'unrelated'], ['proxygs_stride3_original']))

    def test_stop_signals_only_revalidated_process_via_pidfd(self):
        row = dict(pid=10, gpus=[0], argv=['python', 'render.py', '--runs', 'proxygs_stride3_original'], start='123')
        with patch.object(stop_render_jobs.subprocess, 'check_output', side_effect=['0, GPU-a\n', 'GPU-a, 10\n']), \
                patch.object(stop_render_jobs, 'candidates', side_effect=[[row], [row], []]), \
                patch.object(stop_render_jobs.os, 'pidfd_open', return_value=99) as opened, \
                patch.object(stop_render_jobs.os, 'close') as closed, \
                patch.object(stop_render_jobs.signal, 'pidfd_send_signal') as send, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stop_render_jobs.main(['--gpus', '0', '1', '--stop']), 0)
        opened.assert_called_once_with(10)
        send.assert_called_once_with(99, stop_render_jobs.signal.SIGTERM)
        closed.assert_called_once_with(99)


if __name__ == '__main__':
    unittest.main()
