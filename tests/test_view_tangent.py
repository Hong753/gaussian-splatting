"""Physical surface/visibility cases and native Gaussian population mutations."""
import json
import argparse
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.view_tangent import (DEFAULTS, camera_intrinsics, eligible_parents,
                                match_visible_surface, options, surface_frame, tangent_children)
import test_surface_experiment
from test_occlusion import camera
from test_lidar_occlusion import reference_rasterize


class ViewTangentTests(unittest.TestCase):
    def frame(self, cam=None, z=2.):
        cam = cam or camera(64)
        k = camera_intrinsics(cam, 'cpu')
        depth = torch.full((64, 64), z)
        return cam, surface_frame(depth, torch.ones_like(depth, dtype=torch.bool), k)

    def test_foreground_only_even_when_background_is_rendered(self):
        cam, frame = self.frame()
        xyz = torch.tensor([[0., 0, 2.], [0, 0, 4.], [0, 0, 1.], [20, 0, 2.]])
        idx, refs = match_visible_surface(cam, xyz, torch.arange(4), frame, chunk_size=1)
        self.assertEqual(idx.tolist(), [0])
        torch.testing.assert_close(refs['normal'], torch.tensor([[0., 0, 1.]]))
        hidden, _ = match_visible_surface(cam, xyz, torch.tensor([False, True, True, True]), frame)
        self.assertEqual(len(hidden), 0)

    def test_holes_depth_edges_and_nonplanarity_do_not_seed_growth(self):
        cam = camera(64)
        k = camera_intrinsics(cam, 'cpu')
        depth = torch.full((64, 64), 2.)
        depth[:, 32:] = 4
        valid = torch.ones_like(depth, dtype=torch.bool)
        valid[20, 20] = False
        frame = surface_frame(depth, valid, k)
        self.assertEqual(frame['support'][16, 31].item(), 0)
        self.assertEqual(frame['support'][20, 20].item(), 0)
        self.assertEqual(frame['support'][21, 21].item(), 0)
        self.assertGreater(frame['support'][16, 16].item(), 0)
        depth = torch.full((64, 64), 2.)
        depth[29, 29] = 2.04  # Below jump threshold, but off the central plane.
        frame = surface_frame(depth, torch.ones_like(valid), k)
        self.assertEqual(frame['support'][32, 32].item(), 0)

    def test_tilted_plane_and_rotated_camera_use_camera_z(self):
        cam = camera(64)
        cam.world_view_transform = torch.tensor([[0., 0, -1, 0], [0, 1, 0, 0],
                                                 [1, 0, 0, 0], [0, 0, 5, 1.]])
        k = camera_intrinsics(cam, 'cpu')
        y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        n = torch.tensor([.1, -.1, 1.]); n /= n.norm()
        rays = torch.stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], torch.ones_like(x)), -1)
        depth = 2./(rays*n).sum(-1)
        frame = surface_frame(depth, torch.ones_like(depth, dtype=torch.bool), k)
        cam_point = rays[32, 32]*depth[32, 32]
        world_point = (cam_point-cam.world_view_transform[3, :3])@cam.world_view_transform[:3, :3].T
        idx, refs = match_visible_surface(cam, world_point[None], torch.tensor([0]), frame)
        self.assertEqual(idx.tolist(), [0])
        torch.testing.assert_close(refs['anchor'][0], world_point, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(refs['normal'][0], n@cam.world_view_transform[:3, :3].T, atol=1e-5, rtol=1e-5)

    def test_children_preserve_plane_distance_and_remain_in_measured_disk(self):
        n = torch.tensor([1., 2, 3]); n /= n.norm()
        refs = dict(anchor=torch.zeros(50, 3), normal=n.expand(50, 3),
                    support=torch.full((50,), .08))
        parent = .01*n.expand(50, 3)
        children = tangent_children(parent, 20*torch.randn_like(parent), refs)
        distance = (children*n).sum(-1)
        torch.testing.assert_close(distance, torch.full((50,), .01), atol=1e-7, rtol=1e-5)
        self.assertTrue(((children-distance[:, None]*n).norm(dim=1) <= .072001).all())
        self.assertTrue(((children-parent).norm(dim=1) > 0).all())

    def model(self):
        model = test_surface_experiment.SurfaceExperimentTests().model()
        model.use_absgs = False
        model.initialize_view_surface()
        model._xyz.data[:] = torch.tensor([[0., 0, 2.], [.2, 0, 2.]])
        model._scaling.data[:] = math.log(.005)
        model.xyz_gradient_accum = torch.zeros(2, 1)
        model.denom = torch.zeros(2, 1)
        model.view_tangent_options = dict(DEFAULTS, view_tangent_max_points=4)
        cam, frame = self.frame()
        matched = model.observe_view_surface(cam, torch.arange(2), frame)
        self.assertEqual(matched.tolist(), [0, 1])
        model.xyz_gradient_accum[:] = torch.tensor([[4.], [2.]])
        model.denom[:] = 2
        return model

    def densify(self, model, **kwargs):
        real_zeros = torch.zeros
        def cpu_zeros(*a, **kw):
            if kw.get('device') == 'cuda': kw['device'] = 'cpu'
            return real_zeros(*a, **kw)
        with patch.object(torch, 'zeros', side_effect=cpu_zeros):
            return model.densify_and_prune(.1, .005, 1., kwargs.get('max_screen_size'), torch.ones(len(model.get_xyz)))

    def test_mixed_clone_split_population_and_checkpoint_refs(self):
        model = self.model()
        model._scaling.data[1] = math.log(.02)  # Split, while first point clones.
        initial = model.get_xyz.detach().clone()
        stats = self.densify(model)
        self.assertEqual((stats['cloned_parents'], stats['split_parents'], stats['points_after']), (1, 1, 4))
        self.assertEqual(len(model.surface_refs['tolerance']), 4)
        torch.testing.assert_close(model.get_xyz[:, 2], torch.full((4,), 2.))
        self.assertTrue((model.get_xyz[1]-initial[0]).norm().item() > 0)
        self.assertEqual(model.denom.count_nonzero().item(), 0)
        self.assertEqual(model.capture()[13], 'view_tangent')
        self.assertEqual(len(model.capture()[12]['anchor']), 4)
        model.tmp_radii = torch.ones(4)
        model.prune_points(torch.tensor([False, True, False, False]))
        self.assertEqual(len(model.surface_refs['support']), 3)
        self.assertEqual(len(model.optimizer.param_groups[0]['params'][0]), 3)

    def test_only_highest_gradient_parent_grows_with_point_budget(self):
        model = self.model()
        model.view_tangent_options['view_tangent_max_points'] = 3
        stats = self.densify(model)
        self.assertEqual((stats['selected_parents'], stats['points_after']), (1, 3))
        self.assertLess((model.get_xyz[-1]-model.get_xyz[0]).norm().item(), .1)
        model.xyz_gradient_accum[:] = 100
        model.denom[:] = 5
        stats = self.densify(model)
        self.assertEqual((stats['selected_parents'], stats['points_after']), (0, 3))
        self.assertEqual(model.denom.count_nonzero().item(), 0)

    def test_unsupported_or_drifted_parents_do_not_grow(self):
        for kind in ('unknown', 'single_observation', 'normal_drift', 'tangent_drift'):
            model = self.model()
            if kind == 'unknown': model.surface_refs['confidence'].zero_()
            if kind == 'single_observation': model.denom[:] = 1
            if kind == 'normal_drift': model._xyz.data[:, 2] += 1
            if kind == 'tangent_drift': model._xyz.data[:, 0] += 1
            with self.subTest(kind=kind):
                stats = self.densify(model)
                self.assertEqual(stats['selected_parents'], 0)
                self.assertEqual(len(model.get_xyz), 2)

    def test_pruning_retains_pre_growth_max_radii(self):
        model = self.model()
        model.max_radii2D[:] = torch.tensor([50., 1.])
        stats = self.densify(model, max_screen_size=20)
        self.assertEqual(stats['selected_parents'], 1)
        self.assertTrue((model.get_xyz[:, 0] > .1).all())

    def test_changed_surface_association_resets_previous_statistics(self):
        model = self.model()
        model.surface_refs['normal'][0] = torch.tensor([1., 0, 0])
        cam, frame = self.frame()
        model.observe_view_surface(cam, torch.arange(2), frame)
        self.assertEqual(model.denom[:, 0].tolist(), [0, 2])
        self.assertEqual(model.xyz_gradient_accum[:, 0].tolist(), [0, 2])

    def test_restore_retains_dynamic_references_and_accumulated_observations(self):
        from arguments import OptimizationParams
        model = self.model()
        self.densify(model)
        model.xyz_gradient_accum[:] = .25
        model.denom[:] = 3
        restored = self.model()
        restored._exposure = torch.nn.Parameter(torch.zeros(1, 3, 4))
        opt = OptimizationParams(argparse.ArgumentParser(add_help=False))
        opt.surface_mode = 'view_tangent'
        real_zeros = torch.zeros
        def cpu_zeros(*a, **kw):
            if kw.get('device') == 'cuda': kw['device'] = 'cpu'
            return real_zeros(*a, **kw)
        with patch.object(torch, 'zeros', side_effect=cpu_zeros):
            restored.restore(model.capture(), opt)
        self.assertEqual(len(restored.get_xyz), 4)
        for key in model.surface_refs:
            torch.testing.assert_close(restored.surface_refs[key], model.surface_refs[key])
        torch.testing.assert_close(restored.denom, model.denom)
        torch.testing.assert_close(restored.xyz_gradient_accum, model.xyz_gradient_accum)

    def test_da3_surface_uses_raw_depth_own_intrinsics_and_trusted_mask(self):
        from utils.da3_occlusion import VERSION, TrainingDepthCache
        from utils.occlusion_depth import camera_signature
        from utils.sparse_lidar import frame_filename
        from scripts.prepare_da3_occlusion import save_npz
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cam = camera(128)
            cam.image_name = 'train.jpg'
            metadata = dict(version=VERSION, complete=True, source=tmp, resolution=-1, edge_pixels=1,
                images=[cam.image_name], camera_signatures={cam.image_name: camera_signature(cam, 1)})
            (root/'metadata.json').write_text(json.dumps(metadata))
            # DA3 has a different pixel grid; matching must use its stored K.
            k = np.array([[32, 0, 28.5], [0, 32, 31.5], [0, 0, 1]], np.float32)
            valid = np.ones((64, 64), bool)
            valid[15:20, 15:20] = False
            save_npz(root/'frames'/frame_filename(cam.image_name), image=cam.image_name,
                tiles=np.full((8, 8), 2.2, np.float32), depth=np.full((64, 64), 2., np.float32),
                valid=valid, intrinsics=k)
            cache = TrainingDepthCache(root, tmp, [cam], -1)
            frame = cache.surface(cam, 'cpu')
            self.assertEqual(frame['depth'][32, 32].item(), 2.)
            self.assertEqual(frame['support'][16, 16].item(), 0.)
            idx, refs = match_visible_surface(cam, torch.tensor([[0., 0, 2.], [0, 0, 2.2]]), torch.arange(2), frame)
            self.assertEqual(idx.tolist(), [0])
            self.assertEqual(frame['intrinsic'][0, 2].item(), 28.5)

    def test_lidar_provider_returns_raw_first_hit_not_tile_maximum(self):
        from utils.lidar_occlusion import save_geometry
        from utils.occlusion_depth import LidarDepthCache
        fake = SimpleNamespace(RasterizeCudaContext=lambda **kw: object(), rasterize=reference_rasterize)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'nvdiffrast': SimpleNamespace(torch=fake), 'nvdiffrast.torch': fake}):
            path = Path(tmp)/'geometry.npz'
            save_geometry(path, [[0, 0, 2], [0, 0, 4]], [[0, 0, 1]]*2, [10, 10])
            cache = LidarDepthCache(path)
            frame = cache.surface(camera(64), 'cpu')
            self.assertAlmostEqual(frame['depth'][32, 32].item(), 2, places=4)
            self.assertGreater(frame['support'][32, 32].item(), 0)
            cache.surface_memory.clear()
            with patch.object(cache, 'renderer', side_effect=AssertionError('should reuse disk depth')):
                again = cache.surface(camera(64), 'cpu')
            torch.testing.assert_close(again['depth'], frame['depth'])

    def test_workflow_enables_mode_and_keeps_all_frame_psnr_and_both_videos(self):
        from scripts import run_occlusion
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source/sparse').mkdir(parents=True)
            for name in ('images.txt', 'cameras.txt'): (root/'source/sparse'/name).touch()
            baseline = root/'base'
            (baseline/'point_cloud/iteration_30000').mkdir(parents=True)
            (baseline/'point_cloud/iteration_30000/point_cloud.ply').touch()
            (baseline/'cfg_args').write_text('Namespace(model_type="3dgs", antialiasing=True)')
            args = run_occlusion.parser().parse_args(['--source', str(root/'source'), '--model', str(root/'model'),
                '--baseline', str(baseline), '--view-tangent', '--gpu', '2',
                '--position-warmup-steps', '7000', '--position-ramp-steps', '2000', '--position-lr-scale', '.02'])
            plan = run_occlusion.build_plan(args)
            train = plan['stages'][0]['argv']
            self.assertEqual(train[train.index('--surface_mode')+1], 'view_tangent')
            self.assertEqual(plan['training_schedule']['densification_phase_offset'], 7000)
            self.assertEqual(plan['gpu'], '2')
            self.assertEqual(plan['stages'][1]['argv'][0], 'scripts/eval_all_frames.py')
            self.assertEqual([s['name'] for s in plan['stages'][-2:]], ['random_video', 'actual_video'])
            self.assertIn('source', plan['stages'][-1]['argv'])

    def test_shell_launcher_activates_native_environment_and_selects_gpu_2(self):
        script = Path(__file__).resolve().parents[1]/'scripts/run_view_tangent.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shell = root/'conda.sh'
            trace = root/'trace'
            shell.write_text('conda() { export TEST_ACTIVE_ENV="$2"; }\n'
                'python() { printf "%s|%s\\n" "$TEST_ACTIVE_ENV" "$*" >> "$TEST_TRACE"; return 0; }\n')
            env = dict(os.environ, DA3_CONDA_SH=str(shell), TEST_TRACE=str(trace), DA3_BASELINE='/baseline')
            for source in ('lidar', 'da3'):
                completed = subprocess.run(['bash', str(script), '2', source, '--dry-run'], env=env,
                    capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                text = trace.read_text().splitlines()[-1]
                self.assertTrue(text.startswith('3dgs|-u scripts/run_'))
                self.assertIn('--gpu 2', text)
                self.assertIn('--view-tangent --dry-run', text)
                self.assertIn('viewtangent_warm7k_lr50', text)
                if source == 'da3': self.assertIn('--depth-dir', text)


if __name__ == '__main__':
    unittest.main()
