"""Fixed centers across adaptive activation, Adam, pose updates and resume."""
import argparse
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from plyfile import PlyData, PlyElement

import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
import test_lidar_activation as activation_tests
from test_pose_culling import measured_mesh
from test_pose_refinement import differentiable_image
from utils import lidar_offsets as offsets, lidar_activation as activation
from utils.fixed_lidar import audit, configure_training
from scripts import run_local_scale, tune_local_scale, run_soft_surface_long as periodic


class FixedAnchorTests(unittest.TestCase):
    prepare = activation_tests.ActivationTests.prepare
    model = activation_tests.ActivationTests.model
    step = activation_tests.ActivationTests.step
    grow = activation_tests.ActivationTests.grow

    def setUp(self):
        activation_tests.ActivationTests.setUp(self)
        self.opt.lidar_offsets = json.dumps(offsets.FIXED)

    def test_centers_stay_exact_through_growth_and_pose_steps_appearance_learns(self):
        model = self.model()
        xyz = model._xyz.detach().clone()
        appearance = model._features_dc.detach().clone()
        initial_active = int(activation.active_mask(model).sum())
        for iteration in range(1, 9):
            row = model.pose_refiner.update(self.cam, model, None, torch.zeros(3), iteration,
                                            renderer=differentiable_image)
            self.step(model, iteration)
            if 2 < iteration <= 7:
                self.grow(model, iteration)
            self.assertEqual(model.update_learning_rate(iteration), 0.)
            self.assertFalse(model._xyz.requires_grad)
            self.assertIsNone(model._xyz.grad)
            self.assertNotIn(model._xyz, model.optimizer.state)
            self.assertTrue(torch.equal(xyz, model._xyz))
            stats = audit(model, iteration)
            self.assertTrue(stats['positions_unchanged'])
            self.assertEqual(stats['offset_max'], 0.)
            self.assertEqual(stats['offset_radius_max'], 0.)
        self.assertGreater(int(activation.active_mask(model).sum()), initial_active)
        self.assertFalse(torch.equal(appearance, model._features_dc))
        self.assertEqual(offsets.regularization(model).item(), 0.)

    def test_resume_and_ply_keep_exact_original_active_coordinates(self):
        model = self.model()
        for iteration in range(1, 5):
            self.step(model, iteration)
            if iteration > 2: self.grow(model, iteration)
        stream = io.BytesIO()
        torch.save(model.capture(), stream)
        stream.seek(0)
        other = fixed.load_model()(0)
        other._exposure = torch.nn.Parameter(model._exposure.detach().clone())
        with fixed.cpu_allocations():
            other.restore(torch.load(stream, weights_only=True), self.opt)
        self.assertFalse(other._xyz.requires_grad)
        self.assertNotIn(other._xyz, other.optimizer.state)
        for iteration in range(5, 9):
            self.step(model, iteration)
            self.step(other, iteration)
            if iteration <= 7:
                self.assertEqual(self.grow(model, iteration), self.grow(other, iteration))
        self.assertTrue(torch.equal(other._xyz, offsets.state(other)['anchors']))
        for name in ('_xyz', '_opacity', '_features_dc', '_scaling'):
            self.assertTrue(torch.equal(getattr(model, name), getattr(other, name)))
        with patch.dict(other.save_ply.__globals__, {'PlyData': PlyData, 'PlyElement': PlyElement}):
            other.save_ply(str(self.root/'fixed.ply'))
        vertices = PlyData.read(self.root/'fixed.ply')['vertex']
        exported = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
        np.testing.assert_array_equal(exported, self.xyz[activation.active_mask(other).numpy()])

    def test_reset_is_rejected_before_touching_opacity_or_its_moments(self):
        model = self.model()
        self.step(model, 4)
        before = model._opacity.detach().clone()
        momentum = deepcopy(model.optimizer.state[model._opacity])
        with self.assertRaisesRegex(RuntimeError, 'Opacity resets are disabled'):
            model.reset_opacity()
        self.assertTrue(torch.equal(before, model._opacity))
        for key, value in momentum.items():
            self.assertTrue(torch.equal(value, model.optimizer.state[model._opacity][key]))

    def test_resume_rejects_changed_policy_or_moved_fixed_center(self):
        model = self.model()
        self.step(model, 4)
        saved = deepcopy(model.capture())
        changed = deepcopy(self.opt)
        changed.lidar_offsets = json.dumps({**offsets.DEFAULTS, 'warmup':2, 'ramp':2, 'end':8})
        with fixed.cpu_allocations(), self.assertRaisesRegex(ValueError, 'offset policy differs'):
            discrete.raw_model(self.xyz).restore(deepcopy(saved), changed)
        saved[1][0, 0] += .0001
        with fixed.cpu_allocations(), self.assertRaisesRegex(ValueError, 'Saved LiDAR positions'):
            discrete.raw_model(self.xyz).restore(saved, self.opt)
        with torch.no_grad(): model._xyz[0, 0] += .0001
        with self.assertRaisesRegex(RuntimeError, 'LiDAR offset audit failed'):
            audit(model, 4)


class FixedAnchorPlanTests(unittest.TestCase):
    setUp = fixed.FixedLidarPlanTests.setUp

    def test_only_position_policy_changes_from_latest_depth_ablation(self):
        discrete.write_cloud(self.ply, np.array([[i*.01, 0, 2] for i in range(16)], np.float32))
        run_local_scale.prepare(self.argv)
        root = self.root/'geometry'
        root.mkdir()
        geometry = measured_mesh(root)
        from utils.lidar_occlusion import file_digest
        report = json.loads(geometry.read_text())
        report.update(source=str(self.source), point_cloud_sha256=file_digest(self.ply))
        geometry.write_text(json.dumps(report))
        common = ['--reference-model', str(self.model), '--gpu', '0',
            '--ceiling-multiplier', '15', '--target-multiplier', '3', '--schedule-mode', 'joint',
            '--pose-refinement', '--lidar-offsets', '--pose-culling', '--scene-geometry', str(geometry),
            '--lidar-activation', '--depth-decay']
        baseline, frozen = self.root/'bounded', self.root/'fixed_positions'
        tune_local_scale.prepare(common+['--model', str(baseline)])
        argv = common+['--model', str(frozen), '--fixed-lidar-positions']
        tune_local_scale.prepare(argv)
        tune_local_scale.prepare(argv)  # Repeated preparation must be idempotent.
        old, new = periodic.load_plan(baseline), periodic.load_plan(frozen)
        self.assertEqual(old['frames'], new['frames'])
        self.assertEqual(new['iterations'], [10000, 20000, 30000])
        for key in ('pose_refinement', 'pose_culling', 'lidar_activation', 'distance_updates',
                    'training_schedule', 'ceiling_multiplier', 'target_multiplier', 'opacity_resets'):
            self.assertEqual(new['plan'][key], old['plan'][key], key)
        self.assertFalse(new['plan']['opacity_resets'])
        self.assertEqual(new['plan']['position_mode'], 'fixed')
        self.assertEqual(new['plan']['lidar_offsets'], offsets.FIXED)
        command = new['plan']['stages'][0]['argv']
        from arguments import ModelParams, OptimizationParams
        parser = argparse.ArgumentParser()
        mp, op = ModelParams(parser), OptimizationParams(parser)
        parsed, _ = parser.parse_known_args(command[1:])
        dataset, opt = mp.extract(parsed), op.extract(parsed)
        configure_training(opt, dataset)
        self.assertEqual(opt.densify_until_iter, 0)
        self.assertEqual(opt.position_lr_init, 0.)
        self.assertEqual(offsets.requested(opt), offsets.FIXED)
        self.assertEqual(dataset.occlusion_inference_mode, 'off')


if __name__ == '__main__': unittest.main()
