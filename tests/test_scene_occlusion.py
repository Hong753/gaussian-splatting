"""Real BVH/TSDF tests: wall, doorway, camera-z units, original IDs, and cross-view fusion."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts.prepare_da3_occlusion import save_npz
from scripts.prepare_scene_occlusion import main, precompute_views
from scripts.occlusion_options import apply_overrides
from utils.lidar_occlusion import file_digest
from utils.scene_occlusion import FORMAT, MeshDepthCache, TriangleRaycaster, classify_lidar, load_scene, mesh_arrays
from utils.sparse_lidar import frame_filename
from utils.occlusion_culling import conservative_keep
from test_occlusion import camera


def quad(x0=-3., x1=3., z=2.):
    return np.array([[x0, -3, z], [x1, -3, z], [x1, 3, z], [x0, 3, z]], np.float32), np.array([[0, 1, 2], [0, 2, 3]], np.int32)


def cloud(path, points):
    rows = np.zeros(len(points), dtype=[(k, 'f4') for k in 'xyz'])
    for i, key in enumerate('xyz'):
        rows[key] = points[:, i]
    PlyData([PlyElement.describe(rows, 'vertex')]).write(str(path))


def manifest(root, vertices=None, faces=None):
    if vertices is None:
        vertices, faces = quad()
    cloud(root/'input.ply', vertices)
    save_npz(root/'mesh.npz', vertices=vertices, faces=faces)
    report = dict(format=FORMAT, complete=True, mesh='mesh.npz', mesh_sha256=file_digest(root/'mesh.npz'),
        point_cloud_sha256=file_digest(root/'input.ply'), source=str(root), relative_margin=.05,
        training_images=['a.jpg'])
    (root/'scene.json').write_text(json.dumps(report))
    return root/'scene.json'


@unittest.skipUnless(importlib.util.find_spec('open3d'), 'Install requirements-scene-occlusion.txt for real mesh tests')
class SceneOcclusionTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[64., 0, 63.5], [0, 64., 63.5], [0, 0, 1]], np.float32)
        self.e = np.eye(4, dtype=np.float32)

    def test_wall_blocks_background_and_preserves_surface_front_and_ids(self):
        ray = TriangleRaycaster(*quad())
        points = np.array([[np.nan, 0, 2], [0, 0, 2], [0, 0, 4], [0, 0, 1], [0, 0, -2], [40, 0, 2]], np.float32)
        result = classify_lidar(points, self.k, self.e, 128, 128, ray, chunk=2)
        self.assertEqual(result['visible_ids'].tolist(), [1])
        self.assertEqual(result['occluded_ids'].tolist(), [2])
        self.assertEqual(result['in_front_ids'].tolist(), [3])
        self.assertEqual(result['unknown_ids'].tolist(), [])

    def test_doorway_does_not_close_or_hide_visible_room(self):
        left, lf = quad(-3, -.25)
        right, rf = quad(.25, 3)
        ray = TriangleRaycaster(np.concatenate((left, right)), np.concatenate((lf, rf+4)))
        points = np.array([[0, 0, 4], [1, 0, 4], [-1, 0, 4]], np.float32)
        result = classify_lidar(points, self.k, self.e, 128, 128, ray)
        self.assertEqual(result['unknown_ids'].tolist(), [0])
        self.assertEqual(result['occluded_ids'].tolist(), [1, 2])
        rear, rear_faces = quad(z=4)
        ray = TriangleRaycaster(np.concatenate((left, right, rear)), np.concatenate((lf, rf+4, rear_faces+8)))
        result = classify_lidar(points, self.k, self.e, 128, 128, ray)
        self.assertEqual(result['visible_ids'].tolist(), [0])

    def test_depth_is_camera_z_at_oblique_pixels_and_rotated_translated_pose(self):
        rotation = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32)
        origin = np.array([10., 4, -2], np.float32)
        e = np.eye(4, dtype=np.float32)
        e[:3, :3], e[:3, 3] = rotation, -rotation@origin
        vertices, faces = quad()
        ray = TriangleRaycaster(vertices@rotation+origin, faces)
        depth = ray.depth(self.k, e, 128, 128)
        np.testing.assert_allclose(depth, 2., atol=2e-6)
        points = np.array([[1., 0, 2], [1., 0, 4]], np.float32)@rotation+origin
        result = classify_lidar(points, self.k, e, 128, 128, ray)
        self.assertEqual(result['visible_ids'].tolist(), [0])
        self.assertEqual(result['occluded_ids'].tolist(), [1])

    def test_mesh_depth_cache_culls_gaussians_and_supports_novel_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = manifest(root)
            provider = MeshDepthCache(scene)
            cam = camera(128)
            points = torch.tensor([[0., 0, 2], [0., 0, 4], [0., 0, 1]])
            keep = conservative_keep(cam, points, torch.full_like(points, .02),
                torch.tensor([[1., 0, 0, 0]]).repeat(3, 1), provider.tiles(cam, 'cpu'))
            self.assertEqual(keep.tolist(), [True, False, True])
            moved = camera(128, translation=(0, 0, -.5))
            self.assertAlmostEqual(float(provider.raw_depth(moved, 'cpu')[64, 64]), 1.5, places=5)
            frame = provider.diagnostic_frame(cam, 'cpu')
            self.assertEqual(frame['report']['kind'], FORMAT)
            self.assertAlmostEqual(float(frame['depth'][64, 64]), 2., places=5)
            self.assertGreater(float(provider.tiles(cam, 'cpu')[4, 4]), 2.)
            spec = {'type': '3dgs', 'cfg': {}}
            apply_overrides(spec, SimpleNamespace(occlusion='on', occlusion_geometry=scene))
            self.assertEqual(spec['cfg']['occlusion_mode'], 'mesh')

    def test_mesh_and_cloud_identity_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = manifest(root)
            cloud(root/'different.ply', np.array([[1., 0, 0]], np.float32))
            with self.assertRaisesRegex(ValueError, 'LiDAR cloud differ'):
                load_scene(scene, root/'different.ply')
            save_npz(root/'mesh.npz', vertices=np.ones((3, 3)), faces=np.array([[0, 1, 2]]))
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_scene(scene)

    def test_real_tsdf_fuses_complementary_views_and_rejects_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth_dir, output = root/'depth', root/'scene'
            depth_dir.mkdir()
            h, w = 64, 64
            yy, xx = np.indices((h, w))
            k = np.array([[32., 0, 31.5], [0, 32., 31.5], [0, 0, 1]], np.float32)
            points = np.stack(((xx.ravel()-31.5)/16, (yy.ravel()-31.5)/16, np.full(xx.size, 2.)), axis=1)
            cloud(root/'input.ply', points)
            names = ['left.jpg', 'right.jpg', 'both.jpg']
            for i, name in enumerate(names):
                valid = xx > 28 if i == 1 else xx < 36
                save_npz(depth_dir/'frames'/frame_filename(name), depth=np.full((h, w), 2., np.float32),
                    valid=valid, intrinsics=k, extrinsics=self.e)
            config = dict(version=2, complete=False, images=names, source=str(root),
                point_cloud_sha256=file_digest(root/'input.ply'))
            (depth_dir/'metadata.json').write_text(json.dumps(config))
            command = ['fuse', '--depth-dir', str(depth_dir), '--point-cloud', str(root/'input.ply'),
                '--output', str(output), '--voxel-size', '.05', '--truncation', '.15', '--threads', '1']
            with self.assertRaisesRegex(ValueError, 'COMPLETE'):
                main(command)
            with contextlib.redirect_stdout(io.StringIO()):
                main(command+['--output',str(root/'preview'),'--allow-partial'])
            preview_report,_=load_scene(root/'preview/scene.json')
            self.assertTrue(preview_report['diagnostic_only'])
            from scripts.occlusion_options import prepare_training_geometry
            dataset=SimpleNamespace(occlusion_mode='mesh',occlusion_geometry=str(root/'preview/scene.json'),
                model_path=str(root),source_path=str(root))
            with self.assertRaisesRegex(ValueError,'diagnostic preview'):
                prepare_training_geometry(dataset)
            config['complete'] = True
            (depth_dir/'metadata.json').write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                main(command)
            report, mesh = load_scene(output/'scene.json')
            self.assertEqual(report['integrated_frames'], 3)
            self.assertGreater(report['triangles'], 100)
            ray = TriangleRaycaster(*mesh_arrays(mesh))
            depth = ray.depth(k, self.e, w, h)
            np.testing.assert_allclose(depth[10:-10, 10:-10], 2., atol=.035)
            result = classify_lidar(np.array([[-.75, 0, 4], [.75, 0, 4]], np.float32), k, self.e, w, h, ray)
            self.assertEqual(result['occluded_ids'].tolist(), [0, 1])
            with contextlib.redirect_stdout(io.StringIO()):
                main(command)  # Reuse exact input snapshot.

    def test_visibility_precompute_preserves_ids_and_partial_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = manifest(root)
            points = np.array([[0, 0, 2], [0, 0, 4], [0, 0, 1]], np.float32)
            cloud(root/'input.ply', points)
            report = json.loads(scene.read_text())
            report['point_cloud_sha256'] = file_digest(root/'input.ply')
            scene.write_text(json.dumps(report))
            records = {}
            for name in ('a.jpg', 'b.jpg'):
                Image.new('RGB', (128, 128)).save(root/name)
                records[name] = dict(name=name, image_path=str(root/name), R=np.eye(3).tolist(), T=[0., 0., 0.],
                    fov_x=np.pi/2, fov_y=np.pi/2)
            args = SimpleNamespace(scene=scene, point_cloud=root/'input.ply', output=None,
                split='all', image='a.jpg', resolution=-1, margin=.05, device='cpu', threads=1)
            with patch('scripts.prepare_scene_occlusion.read_colmap_records', return_value=records):
                with contextlib.redirect_stdout(io.StringIO()):
                    precompute_views(args)
                    metadata = json.loads((root/'visibility/metadata.json').read_text())
                    self.assertFalse(metadata['complete'])
                    args.image = None
                    precompute_views(args)
            metadata = json.loads((root/'visibility/metadata.json').read_text())
            self.assertTrue(metadata['complete'])
            with np.load(root/'visibility/frames'/frame_filename('a.jpg'), allow_pickle=False) as data:
                self.assertEqual(data['visible_ids'].tolist(), [0])
                self.assertEqual(data['occluded_ids'].tolist(), [1])
                self.assertEqual(data['in_front_ids'].tolist(), [2])
            if importlib.util.find_spec('viser'):
                import socket
                from scripts import inspect_scene_visibility as viewer
                with socket.socket() as connection:
                    connection.bind(('127.0.0.1', 0))
                    port = connection.getsockname()[1]
                with patch.object(viewer, 'time', SimpleNamespace(sleep=Mock(side_effect=KeyboardInterrupt))):
                    with patch('scripts.view_comparisons.read_colmap_records', return_value=records):
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(viewer.main(['--scene', str(scene), '--point-cloud', str(root/'input.ply'),
                                '--image', 'a.jpg', '--host', '127.0.0.1', '--port', str(port)]), 0)

    def test_mesh_workflow_keeps_all_frame_psnr_and_videos_and_rejects_stale_visibility(self):
        from scripts import run_occlusion
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = manifest(root)
            (root/'sparse').mkdir()
            for name in ('images.txt', 'cameras.txt'):
                (root/'sparse'/name).touch()
            baseline = root/'base'
            (baseline/'point_cloud/iteration_30000').mkdir(parents=True)
            (baseline/'point_cloud/iteration_30000/point_cloud.ply').touch()
            (baseline/'cfg_args').write_text('Namespace(model_type="3dgs", antialiasing=True)')
            keyframes = baseline/'keyframes.json'
            keyframes.write_text(json.dumps({'selected_images': ['a.jpg']}))
            args = run_occlusion.parser().parse_args(['--source', str(root), '--model', str(root/'model'),
                '--keyframes', str(keyframes), '--baseline', str(baseline), '--scene-geometry', str(scene),
                '--position-warmup-steps', '7000', '--position-ramp-steps', '2000', '--position-lr-scale', '.02'])
            with self.assertRaisesRegex(ValueError, 'Finish whole-scene'):
                run_occlusion.build_plan(args)
            (root/'visibility').mkdir()
            visibility = root/'visibility/metadata.json'
            visibility.write_text(json.dumps(dict(complete=True, scene_sha256='stale', images=['a.jpg'])))
            with self.assertRaisesRegex(ValueError, 'different scene'):
                run_occlusion.build_plan(args)
            visibility.write_text(json.dumps(dict(complete=True, scene_sha256=file_digest(scene), images=['a.jpg'])))
            plan = run_occlusion.build_plan(args)
            train = plan['stages'][0]['argv']
            self.assertEqual(train[train.index('--occlusion_mode')+1], 'mesh')
            self.assertEqual(train[train.index('--occlusion_geometry')+1], str(scene))
            self.assertEqual(plan['training_schedule']['position_warmup_steps'], 7000)
            self.assertEqual(plan['stages'][1]['argv'][0], 'scripts/eval_all_frames.py')
            self.assertEqual([s['name'] for s in plan['stages'][-2:]], ['random_video', 'actual_video'])
            self.assertIn('fixed scene mesh culling', plan['stages'][-1]['argv'][plan['stages'][-1]['argv'].index('--labels')+2])


if __name__ == '__main__':
    unittest.main()
