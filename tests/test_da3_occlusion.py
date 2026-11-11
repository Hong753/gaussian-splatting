import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.da3_occlusion import (VERSION, camera_matrices, reliable_pixels, fit_scale,
                                 cross_view_support, training_tiles, TrainingDepthCache)
from utils.occlusion_depth import camera_signature
from utils.sparse_lidar import frame_filename
from scripts.prepare_da3_occlusion import front_points, windows, save_npz
from scripts import run_da3_occlusion, prepare_da3_occlusion


def camera(size=128):
    return SimpleNamespace(image_name='train.jpg', image_width=size, image_height=size,
        FoVx=math.pi/2, FoVy=math.pi/2, znear=.01, zfar=100., world_view_transform=torch.eye(4))


class DA3Tests(unittest.TestCase):
    def test_preparation_end_to_end_with_fake_depth_predictor(self):
        """Exercise actual projection, alignment, checks, cache, previews and reuse."""
        from PIL import Image
        from plyfile import PlyData, PlyElement
        from utils.occlusion_depth import record_camera
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            root, records = Path(tmp), {}
            for i in range(3):
                name = f'{i}.png'
                Image.new('RGB', (128, 128), (100+i, 120, 130)).save(root/name)
                records[name] = dict(name=name, image_path=str(root/name), R=np.eye(3).tolist(),
                    T=[.02*i, 0, 0], width=128, height=128, fov_x=math.pi/2, fov_y=math.pi/2)
            names = list(records)
            keyframes = root/'keyframes.json'
            keyframes.write_text(json.dumps(dict(selected_images=names)))
            x, y = np.meshgrid(np.linspace(-2, 2, 160), np.linspace(-2, 2, 160))
            vertex = np.zeros(x.size, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            vertex['x'], vertex['y'], vertex['z'] = x.ravel(), y.ravel(), 2
            cloud = root/'input.ply'
            PlyData([PlyElement.describe(vertex, 'vertex')]).write(cloud)
            calls = []
            class FakeModel:
                @classmethod
                def from_pretrained(cls, *args, **kwargs): return cls()
                def to(self, device): return self
                def eval(self): return self
                def inference(self, images, **kwargs):
                    calls.append(kwargs)
                    assert len(images) == 3 and kwargs['infer_gs'] is False
                    return SimpleNamespace(depth=np.ones((3, 128, 128), np.float32),
                        conf=np.ones((3, 128, 128), np.float32), intrinsics=kwargs['intrinsics'])
            real_to = torch.Tensor.to
            def cpu_to(tensor, *args, **kwargs):
                args = tuple('cpu' if isinstance(a, str) and a == 'cuda' else a for a in args)
                return real_to(tensor, *args, **kwargs)
            args = ['--source', tmp, '--keyframes', str(keyframes), '--point-cloud', str(cloud),
                    '--output', str(root/'output'), '--process-res', '140']
            with ExitStack() as stack:
                stack.enter_context(patch.object(prepare_da3_occlusion, 'read_colmap_records', return_value=records))
                stack.enter_context(patch('torch.cuda.is_available', return_value=True))
                stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
                stack.enter_context(patch.dict(sys.modules, {
                    'depth_anything_3.api': SimpleNamespace(DepthAnything3=FakeModel),
                    'huggingface_hub': SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha='fixed-revision'))}))
                self.assertEqual(prepare_da3_occlusion.main(args+['--inspect-views', '1']), 0)
                partial = json.loads((root/'output/metadata.json').read_text())
                self.assertFalse(partial['complete'])
                self.assertEqual(partial['summary']['frames'], 1)
                cams = [record_camera(records[n], -1, 'cpu') for n in names]
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    TrainingDepthCache(root/'output', root, cams, -1, point_cloud=cloud)
                diagnostic = TrainingDepthCache(root/'output', root, cams, -1,
                                                point_cloud=cloud, allow_partial=True)
                self.assertEqual(diagnostic.available_images, {names[1]})
                frame = diagnostic.diagnostic_frame(cams[1], 'cpu')
                self.assertIn('anchors', frame)
                self.assertGreater(frame['valid'].mean(), .1)
                copied_cloud = root/'copied_input.ply'
                copied_cloud.write_bytes(cloud.read_bytes())
                copied_args = args[:]
                copied_args[copied_args.index('--point-cloud')+1] = str(copied_cloud)
                self.assertEqual(prepare_da3_occlusion.main(copied_args+['--inspect-views', '1']), 0)
                self.assertEqual(prepare_da3_occlusion.main(args), 0)
                self.assertEqual(prepare_da3_occlusion.main(args), 0)
            self.assertEqual(len(calls), 2)  # One preview, one full batch; both reused.
            metadata = json.loads((root/'output/metadata.json').read_text())
            self.assertTrue(metadata['complete'])
            self.assertGreater(metadata['summary']['mean_reliable_tile_fraction'], .1)
            self.assertTrue((root/'output/index.html').is_file())
            cams = [record_camera(records[n], -1, 'cpu') for n in names]
            provider = TrainingDepthCache(root/'output', root, cams, -1, point_cloud=cloud)
            self.assertTrue(torch.isfinite(provider.tiles(cams[0], 'cpu')).any())

    def test_explicit_tiles_block_gradients_but_default_render_does_not(self):
        from utils.graphics_utils import getProjectionMatrix
        class Rasterizer:
            def __init__(self, raster_settings): pass
            def __call__(self, **kwargs):
                value = sum(v.sum() for v in kwargs.values() if v is not None)*.001
                return value.expand(3, 8, 8), torch.ones(len(kwargs['means3D']), dtype=torch.int32), value.expand(1, 8, 8)
        spec = importlib.util.spec_from_file_location('_da3_renderer', Path(__file__).resolve().parents[1]/'gaussian_renderer/__init__.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'diff_gaussian_rasterization': SimpleNamespace(
            GaussianRasterizationSettings=lambda **kw: SimpleNamespace(**kw), GaussianRasterizer=Rasterizer),
            'scene.gaussian_model': SimpleNamespace(GaussianModel=object)}):
            spec.loader.exec_module(module)
        c = camera()
        c.full_proj_transform = getProjectionMatrix(.01, 100., c.FoVx, c.FoVy).T
        c.camera_center = torch.zeros(3)
        pc = SimpleNamespace(get_xyz=torch.tensor([[0., 0., 1.], [0., 0., 3.]], requires_grad=True),
            get_scaling=torch.full((2, 3), .03, requires_grad=True),
            get_rotation=torch.tensor([[1., 0., 0., 0.]]*2, requires_grad=True),
            get_opacity=torch.ones(2, 1, requires_grad=True), get_features=torch.ones(2, 1, 3, requires_grad=True), active_sh_degree=0)
        pipe = SimpleNamespace(occlusion_mode='off', occlusion_margin=.05, occlusion_sigma=3., occlusion_chunk_size=1024,
            train_depth_occlusion_dir='present-but-not-used-by-evaluation', antialiasing=True,
            compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        real_zeros = torch.zeros_like
        def cpu_zeros(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return real_zeros(*args, **kwargs)
        with patch.object(module.torch, 'zeros_like', side_effect=cpu_zeros):
            result = module.render(c, pc, pipe, torch.zeros(3), occlusion_tiles=torch.full((8, 8), 2.))
            plain = module.render(c, pc, pipe, torch.zeros(3))
        self.assertEqual(result['occlusion_keep'].tolist(), [True, False])
        self.assertNotIn('occlusion_keep', plain)
        self.assertEqual(plain['radii'].tolist(), [1, 1])
        result['render'].sum().backward()
        for name in ('get_xyz', 'get_scaling', 'get_rotation', 'get_opacity', 'get_features'):
            self.assertEqual(getattr(pc, name).grad[1].count_nonzero().item(), 0)

    def test_scale_with_occluded_outliers_and_spatial_holdout(self):
        yy, xx = np.indices((128, 128))
        depth = (2+xx/128+yy/256).astype(np.float32)
        u, v = xx.ravel()[::3], yy.ravel()[::3]
        z = depth[v, u]*2.5
        z[::7] *= 3  # Behind-surface points must not bias the scale.
        scale, report = fit_scale(depth, np.ones_like(depth, bool), u, v, z)
        self.assertTrue(report['accepted'])
        self.assertAlmostEqual(scale, 2.5, places=5)
        self.assertGreater(report['holdout_anchors'], 30)
        # A spatially varying prediction cannot be repaired by one global scale.
        broken = depth*(.5+xx/128)
        scale, report = fit_scale(broken, np.ones_like(depth, bool), u, v, z)
        self.assertIsNone(scale)
        self.assertFalse(report['accepted'])

    def test_sparse_or_local_anchors_rejected(self):
        depth = np.ones((128, 128), np.float32)
        y, x = np.indices(depth.shape)
        scale, _ = fit_scale(depth, np.ones_like(depth, bool), x[:16].ravel(), y[:16].ravel(), np.ones(16*128)*2)
        self.assertIsNone(scale)

    def test_confidence_and_depth_boundaries(self):
        depth = np.full((64, 64), 2., np.float32)
        depth[:, 32:] = 5
        conf = np.ones_like(depth)
        conf[10:20, 10:20] = 0
        valid = reliable_pixels(depth, conf)
        self.assertFalse(valid[:, 31:33].any())
        self.assertFalse(valid[10:20, 10:20].any())
        self.assertTrue(valid[24, 20])

    def test_native_projection_frontmost_z_not_euclidean_range(self):
        record = dict(R=np.eye(3), T=[0, 0, 0], fov_x=math.pi/2, fov_y=math.pi/2)
        k, e = camera_matrices(record, 128, 128)
        points = torch.tensor([[0, 0, 2.], [0, 0, 5.], [1, 0, 2.], [0, 0, -1.]])
        u, v, z = front_points(points, k, e, 128, 128, 1)
        self.assertEqual(z.tolist(), [2, 2])
        self.assertEqual(u.tolist(), [64, 96])
        self.assertTrue(np.all(v == 64))

    def test_multiview_support_uses_camera_transform(self):
        d = np.full((64, 64), 2., np.float32)
        valid = np.ones_like(d, bool)
        k = np.array([[32, 0, 31.5], [0, 32, 31.5], [0, 0, 1]], np.float32)
        e = np.eye(4, dtype=np.float32)
        other = e.copy()
        other[2, 3] = -.5
        correct = np.full_like(d, 1.5)
        count = cross_view_support(d, valid, k, e, [(correct, valid, k, other)])
        self.assertEqual(count[32, 32], 1)
        wrong = cross_view_support(d, valid, k, e, [(d, valid, k, other)])
        self.assertEqual(wrong[32, 32], 0)

    def test_tile_resampling_keeps_holes_and_uncertainty(self):
        c = camera()
        k = np.array([[32, 0, 31.75], [0, 32, 31.75], [0, 0, 1]], np.float32)
        d = np.full((64, 64), 2., np.float32)
        valid = np.ones_like(d, bool)
        valid[30:34, 30:34] = False
        tiles = training_tiles(d, valid, k, c, .1, 1)
        self.assertTrue(np.isinf(tiles[3:5, 3:5]).all())
        self.assertAlmostEqual(float(tiles[1, 1]), 2.2, places=5)

    def test_cache_rejects_eval_images_pose_changes_and_cloud_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = camera()
            metadata = dict(version=VERSION, complete=True, source=tmp, resolution=-1, edge_pixels=1,
                images=[c.image_name], camera_signatures={c.image_name: camera_signature(c, 1)})
            (root/'metadata.json').write_text(json.dumps(metadata))
            save_npz(root/'frames'/frame_filename(c.image_name), image=c.image_name, tiles=np.full((8, 8), 2., np.float32))
            cache = TrainingDepthCache(root, tmp, [c], -1)
            self.assertEqual(cache.tiles(c, 'cpu').shape, (8, 8))
            heldout = SimpleNamespace(**{**vars(c), 'image_name': 'heldout.jpg'})
            with self.assertRaises(ValueError):
                TrainingDepthCache(root, tmp, [heldout], -1)
            changed = camera()
            changed.world_view_transform[3, 0] = .1
            with self.assertRaises(ValueError):
                TrainingDepthCache(root, tmp, [changed], -1).tiles(changed, 'cpu')
            cloud = root/'input.ply'
            cloud.write_text('different')
            with self.assertRaises(ValueError):
                TrainingDepthCache(root, tmp, [c], -1, point_cloud=cloud)

    def test_windows_never_add_heldout_views_and_cover_each_core_once(self):
        names = [f'train_{i}' for i in range(11)]
        chunks = list(windows(names))
        self.assertEqual([n for _, core in chunks for n in core], names)
        self.assertTrue(all(set(batch) <= set(names) and set(core) <= set(batch) for batch, core in chunks))
        self.assertTrue(all(len(batch) <= 8 for batch, _ in chunks))

    def test_workflow_preserves_schedule_and_evaluates_without_target_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'metadata.json').write_text(json.dumps(dict(complete=True, source=tmp, resolution=-1, edge_pixels=1, images=['a', 'b', 'c'])))
            keyframes = root/'keyframes.json'
            keyframes.write_text(json.dumps(dict(selected_images=['a', 'b', 'c'])))
            args = run_da3_occlusion.parser().parse_args(['--source', tmp, '--model', str(root/'model'),
                '--keyframes', str(keyframes), '--depth-dir', tmp])
            raw = dict(stages=[dict(name='train', argv=['train.py', '--occlusion_mode', 'lidar',
                '--occlusion_voxel_size', '.05', '--occlusion_max_radius', '.1']),
                dict(name='trained_on_psnr', argv=['scripts/eval_all_frames.py', '--occlusion', 'saved']),
                dict(name='trained_off_psnr', argv=['skip']), dict(name='baseline_on_psnr', argv=['skip'])])
            with patch.object(run_da3_occlusion.run_occlusion, 'build_plan', return_value=raw):
                plan = run_da3_occlusion.build_plan(args)
            self.assertEqual((args.position_warmup_steps, args.position_ramp_steps, args.position_lr_scale), (7000, 2000, .02))
            self.assertEqual(plan['render_occlusion'], 'off')
            self.assertEqual(len(plan['stages']), 2)
            cmd = plan['stages'][0]['argv']
            self.assertEqual(cmd[cmd.index('--occlusion_mode')+1], 'off')
            self.assertIn('--train_depth_occlusion_dir', cmd)
            self.assertNotIn('--lidar_depth_weight', cmd)


if __name__ == '__main__':
    unittest.main()
