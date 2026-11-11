"""Foreground opacity gradients, bounded surface matching and runnable plans."""
import argparse
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from arguments import OptimizationParams
from scripts import run_occlusion, run_soft_surface
from utils.soft_surface import (DEFAULTS, SoftSurfaceRegularizer, options, ramp,
    normal_attraction, render_surface_opacity, surface_opacity_loss)
from utils.view_tangent import camera_intrinsics, surface_frame, match_visible_surface
from utils.warmup_schedule import position_lr_step
from test_occlusion import camera
from test_scene_occlusion import manifest
from test_view_comparisons import dataset, checkpoint


class Cloud:
    def __init__(self, xyz):
        self._xyz = torch.nn.Parameter(torch.tensor(xyz))
        n = len(xyz)
        self._opacity = torch.nn.Parameter(torch.full((n,1), -2.2))
        self._scaling = torch.nn.Parameter(torch.full((n,3), math.log(.01)))
        self._rotation = torch.nn.Parameter(torch.tensor([[1.,0,0,0]]*n))
    get_xyz = property(lambda self: self._xyz)
    get_scaling = property(lambda self: self._scaling.exp())
    get_rotation = property(lambda self: self._rotation)
    get_opacity = property(lambda self: self._opacity.sigmoid())


class SoftSurfaceTests(unittest.TestCase):
    def frame(self):
        cam = camera(64)
        cam.full_proj_transform = torch.eye(4)
        cam.camera_center = torch.zeros(3)
        depth = torch.full((64,64), 2.)
        return cam, surface_frame(depth, torch.ones_like(depth, dtype=torch.bool), camera_intrinsics(cam, 'cpu'))

    def test_surface_opacity_reduces_rear_supervision_without_hard_gate(self):
        def rear_grad(alpha):
            colors = torch.tensor([.8, .1], requires_grad=True)
            rgb = alpha*colors[0]+(1-alpha)*.5*colors[1]
            return torch.autograd.grad(rgb, colors)[0][1]
        self.assertGreater(rear_grad(.1), rear_grad(.98))
        self.assertGreater(rear_grad(.98), 0)
        logit = torch.tensor(-2.2, requires_grad=True)
        loss = surface_opacity_loss(logit.sigmoid()[None], torch.tensor([True]), .98)
        self.assertLess(torch.autograd.grad(loss, logit)[0], 0)
        self.assertEqual(surface_opacity_loss(torch.tensor([.99]), torch.tensor([True]), .98), 0)

    def test_auxiliary_pass_keeps_full_transmittance_but_detaches_geometry(self):
        cam, _ = self.frame()
        pc = Cloud([[0.,0,1.],[0,0,2.],[0,0,4.]])
        captured = {}
        class Rasterizer:
            def __init__(self, raster_settings):
                captured['settings'] = raster_settings
            def __call__(self, **kw):
                captured.update(kw)
                a = kw['opacities'][:,0]
                transmittance = torch.cat((a.new_ones(1), (1-a).cumprod(0)[:-1]))
                image = ((a*transmittance)[:,None]*kw['colors_precomp']).sum(0)[:,None,None]
                return image, None, None
        module = SimpleNamespace(GaussianRasterizationSettings=SimpleNamespace, GaussianRasterizer=Rasterizer)
        with patch.dict('sys.modules', diff_gaussian_rasterization=module):
            opacity = render_surface_opacity(cam, pc, SimpleNamespace(debug=False, antialiasing=True), torch.tensor([1]))
            grads = torch.autograd.grad(surface_opacity_loss(opacity, torch.ones_like(opacity, dtype=torch.bool), .98),
                [pc._opacity, pc._xyz, pc._scaling, pc._rotation], allow_unused=True)
        self.assertEqual(len(captured['means3D']), 3)
        self.assertEqual(captured['colors_precomp'][:,0].tolist(), [0.,1.,0.])
        self.assertTrue(captured['settings'].antialiasing)
        self.assertEqual(captured['settings'].bg.count_nonzero(), 0)
        self.assertTrue(all(not captured[k].requires_grad for k in ('means3D','means2D','scales','rotations')))
        self.assertGreater(grads[0][0], 0)  # Unassociated front floater is discouraged.
        self.assertLess(grads[0][1], 0)     # Real surface gains opacity.
        self.assertEqual(grads[0][2], 0)    # Farther room gains none.
        self.assertTrue(all(g is None for g in grads[1:]))

    def test_only_nearby_surfaces_attract_and_gradient_is_normal(self):
        cam, frame = self.frame()
        pc = Cloud([[0.,0,2.02],[0,0,4.],[0,0,1.]])
        indices, refs = match_visible_surface(cam, pc.get_xyz, torch.arange(3), frame, depth_abs=.1, depth_relative=0.)
        self.assertEqual(indices.tolist(), [0])
        loss = normal_attraction(pc.get_xyz, indices, refs, .1, 100)
        grad = torch.autograd.grad(loss, pc._xyz)[0]
        self.assertGreater(grad[0,2], 0)
        torch.testing.assert_close(grad[:,:2], torch.zeros(3,2))
        torch.testing.assert_close(grad[1:], torch.zeros(2,3))
        # Test a tilted normal separately: tangential displacement receives no pull.
        normal = torch.tensor([[1.,2.,3.]])
        normal /= normal.norm()
        point = torch.tensor([[.02,-.01,.01]], requires_grad=True)
        value = normal_attraction(point, torch.tensor([0]), dict(anchor=torch.zeros_like(point), normal=normal), .1, 1)
        gradient = torch.autograd.grad(value, point)[0]
        torch.testing.assert_close(gradient-(gradient*normal).sum(-1,keepdim=True)*normal, torch.zeros_like(gradient), atol=1e-6, rtol=0)

    def regularizer(self, frame):
        reg = SoftSurfaceRegularizer.__new__(SoftSurfaceRegularizer)
        reg.options, reg.warmup = dict(DEFAULTS), 7000
        reg.provider = SimpleNamespace(surface=lambda *a: frame)
        return reg

    def test_warmup_new_children_and_unknown_surface(self):
        cam, frame = self.frame()
        pc = Cloud([[0.,0,2.02],[0,0,4.]])
        reg = self.regularizer(frame)
        captured = []
        def opacity_pass(camera, cloud, pipe, indices):
            captured.append(indices.tolist())
            return cloud.get_opacity[indices].mean().expand(64,64)
        with patch('utils.soft_surface.render_surface_opacity', side_effect=opacity_pass):
            loss, stats = reg.loss(cam, pc, None, torch.arange(2), 7000)
            grad = torch.autograd.grad(loss, pc._xyz, allow_unused=True)[0]
            self.assertIsNone(grad)
            self.assertEqual(stats['position_weight'], 0)
            loss, stats = reg.loss(cam, pc, None, torch.arange(2), 8000)
            self.assertGreater(torch.autograd.grad(loss, pc._xyz)[0][0,2], 0)
            # Appended children are matched from current positions, not cached IDs.
            pc = Cloud([[0.,0,2.02],[0,0,4.],[.01,0,2.01],[0,0,3.]])
            _, stats = reg.loss(cam, pc, None, torch.arange(4), 8000)
            self.assertEqual(captured[-1], [0,2])
            self.assertEqual(stats['surface_gaussians'], 2)
            cam.alpha_mask = torch.zeros(1,64,64)
            loss, stats = reg.loss(cam, pc, None, torch.arange(4), 8000)
            self.assertFalse(loss.requires_grad)
            self.assertEqual(stats['captured_gaussians'], 0)
            self.assertEqual(stats['reference_fraction'], 0)

    def test_depth_discontinuity_and_hole_do_not_reward_opacity(self):
        cam, _ = self.frame()
        depth = torch.full((64,64), 2.); depth[:,32:] = 4.
        frame = surface_frame(depth, torch.ones_like(depth, dtype=torch.bool), camera_intrinsics(cam, 'cpu'))
        pc = Cloud([[0.,0,2.],[0,0,4.]])
        with patch('utils.soft_surface.render_surface_opacity', side_effect=AssertionError('Unsupported boundary rendered')):
            loss, stats = self.regularizer(frame).loss(cam, pc, None, torch.arange(2), 8000)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(stats['surface_gaussians'], 0)

    def test_config_and_schedule_validation(self):
        defaults = OptimizationParams(argparse.ArgumentParser())
        self.assertEqual(options(defaults), DEFAULTS)
        for updates in (dict(soft_surface_capture_distance=.01), dict(soft_surface_opacity_target=1.),
                        dict(soft_surface_memory_views=0), dict(soft_surface_opacity_weight=float('nan'))):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                options(updates)
        self.assertEqual(ramp(7000,7000,1000), 0)
        self.assertEqual(ramp(7500,7000,1000), .5)
        self.assertEqual(position_lr_step(7000,7000,2000), (0.,0))
        self.assertEqual(position_lr_step(8000,7000,2000), (.5,0))

    def test_reference_identity_and_disk_cache_are_checked_without_loading_all_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = manifest(root)
            cfg = SimpleNamespace(**dict(DEFAULTS, soft_surface_geometry=str(scene)), position_warmup_steps=7000)
            data = SimpleNamespace(model_path=str(root), source_path=str(root))
            reg = SoftSurfaceRegularizer(cfg, data, [SimpleNamespace(image_name='a.jpg')])
            self.assertEqual(reg.provider.memory_views, 4)
            self.assertEqual(len(reg.provider.surface_memory), 0)
            self.assertIsNone(reg.provider.renderer)
            self.assertTrue(str(reg.provider.directory).startswith(str(root/'occlusion_depth_cache')))
            with self.assertRaisesRegex(ValueError, 'training cameras differ'):
                SoftSurfaceRegularizer(cfg, data, [SimpleNamespace(image_name='wrong.jpg')])
            with (root/'input.ply').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'initialization LiDAR cloud differ'):
                SoftSurfaceRegularizer(cfg, data, [SimpleNamespace(image_name='a.jpg')])

    def test_workflow_without_hard_visibility_cache_and_mesh_free_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, keyframes, names, selected = dataset(root)
            scene = manifest(root)
            report = json.loads(scene.read_text())
            report.update(source=str(source), training_images=selected)
            scene.write_text(json.dumps(report))
            # Unrelated partial hard-culling ID cache must not gate this ablation.
            (root/'visibility').mkdir()
            (root/'visibility/metadata.json').write_text('interrupted hard-culling cache')
            base = root/'baseline'; base.mkdir(); checkpoint(base,30000)
            (base/'cfg_args').write_text('Namespace(model_type="3dgs", antialiasing=True)')
            argv = ['--source',str(source),'--model',str(root/'soft'),'--baseline',str(base),
                '--keyframes',str(keyframes),'--scene-geometry',str(scene)]
            args = run_soft_surface.parser().parse_args(argv)
            plan = run_occlusion.build_plan(args)
            cmd = plan['stages'][0]['argv']
            self.assertEqual(cmd[cmd.index('--occlusion_mode')+1], 'off')
            self.assertNotIn('--surface_mode', cmd)
            self.assertNotIn('--start_checkpoint', cmd)
            self.assertIn('--soft_surface_geometry', cmd)
            self.assertEqual(plan['gpu'], '0')
            self.assertEqual(plan['training_schedule']['position_warmup_steps'], 7000)
            self.assertEqual(plan['training_schedule']['position_ramp_steps'], 2000)
            self.assertAlmostEqual(plan['training_schedule']['position_lr_init'], 3.2e-6)
            self.assertAlmostEqual(plan['training_schedule']['position_lr_final'], 3.2e-8)
            self.assertEqual([s['name'] for s in plan['stages']],
                ['train','trained_off_psnr','trained_off_images','baseline_off_psnr','baseline_off_images','random_video','actual_video'])
            for stage in plan['stages'][1:]:
                command = stage['argv']
                self.assertEqual(command[command.index('--occlusion')+1], 'off')
                if stage['name'].endswith('_images'):
                    self.assertEqual(command[command.index('--split')+1], 'both')
                if stage['name'].endswith('_psnr'):
                    self.assertEqual(command[0], 'scripts/eval_all_frames.py')
                if stage['name'].endswith('_video'):
                    self.assertIn('soft surface opacity', ' '.join(command))
                    self.assertNotIn('culling during training', ' '.join(command))
            for conflicting in (['--view-tangent'], ['--prepare-visibility'], ['--inference-occlusion','saved']):
                with self.subTest(conflicting=conflicting), self.assertRaises(ValueError):
                    run_occlusion.build_plan(run_soft_surface.parser().parse_args(argv+conflicting))
            checkpoint(root/'soft',30000)
            cfg = dict(soft_surface_geometry=str(scene), occlusion_mode='off', occlusion_inference_mode='off',
                **plan['training_schedule'])
            (root/'soft/cfg_args').write_text(repr(argparse.Namespace(**cfg)))
            scene.unlink()
            evaluation = run_occlusion.build_plan(run_soft_surface.parser().parse_args(argv+['--evaluate-only']))
            self.assertFalse(any(s['name']=='train' for s in evaluation['stages']))
            self.assertAlmostEqual(evaluation['training_schedule']['position_lr_init'], 3.2e-6)


if __name__ == '__main__':
    unittest.main()
