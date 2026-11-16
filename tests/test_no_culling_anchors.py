"""Mesh-free activation, frozen reserves, fixed XYZ and runnable GPU 1 plans."""
import argparse
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_discrete_lidar as discrete
import test_fixed_lidar as fixed
from test_fixed_anchor_experiment import FixedAnchorTests
from utils import lidar_activation as activation, lidar_offsets as offsets
from utils.fixed_lidar import configure_training, project_scales
from utils.occlusion_culling import masked_adam_step
from utils.distance_updates import policy, weights, adam_step
from scripts import run_local_scale, tune_local_scale, run_soft_surface_long as periodic


class NoCullingAnchorTests(FixedAnchorTests):
    def setUp(self):
        super().setUp()
        self.opt.pose_culling = ''
        self.pipe = SimpleNamespace(occlusion_mode='off')
        self.no_cutoff = patch('utils.ray_occlusion.pixel_cutoff',
                               side_effect=AssertionError('No depth lookup is allowed'))
        self.no_cutoff.start()
        self.addCleanup(self.no_cutoff.stop)

    def test_front_and_back_candidates_survive_without_mesh_or_depth_lookup(self):
        xyz = torch.tensor([[0., 0., 2.], [0., 0., 30.], [50., 0., 2.], [0., 0., -.1]])
        model = SimpleNamespace(fixed_lidar_state={'position_offsets': {'anchors': xyz}})
        self.assertEqual(activation.visible_candidates(model, torch.arange(4), self.camera, self.pipe).tolist(),
                         [True, True, False, False])

    def test_native_radii_mask_preserves_inactive_rows_and_both_adam_moments(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                model = self.model()
                self.step(model, 3)  # Existing nonzero momenta must stay frozen too.
                active = activation.active_mask(model).clone()
                package = {'radii': torch.ones(len(active), dtype=torch.int32)}
                hidden = int(active.nonzero()[0])
                package['radii'][hidden] = 0
                keep = activation.optimization_mask(model, package)
                self.assertTrue(torch.equal(keep, active & (package['radii'] > 0)))
                before = [p.detach().clone() for g in model.optimizer.param_groups for p in g['params']]
                moments = deepcopy(model.optimizer.state_dict())
                loss = -model.get_opacity.sum() - model._features_dc.sum()
                loss.backward()
                if weighted:
                    opt = deepcopy(self.opt)
                    opt.distance_weighting = True
                    opt.distance_metric = 'depth'
                    opt.distance_decay = 'exponential'
                    opt.distance_normalization = 'relative'
                    factors = weights(model._xyz, self.camera.camera_center, policy(opt),
                        world_view_transform=self.camera.world_view_transform, keep=keep)
                    adam_step(model.optimizer, factors, keep=keep, chunk_size=4)
                else:
                    masked_adam_step(model.optimizer, keep, chunk_size=4)
                project_scales(model)
                model.optimizer.zero_grad(set_to_none=True)
                for old, group in zip(before, model.optimizer.param_groups):
                    self.assertTrue(torch.equal(old[~keep], group['params'][0][~keep]))
                for key, state in moments['state'].items():
                    for name in ('exp_avg', 'exp_avg_sq'):
                        self.assertTrue(torch.equal(state[name][~keep], model.optimizer.state_dict()['state'][key][name][~keep]))
                self.assertTrue(torch.equal(model._xyz, offsets.state(model)['anchors']))

    def test_resume_cannot_silently_enable_culling(self):
        model = self.model()
        self.step(model, 4)
        changed = deepcopy(self.opt)
        from utils.pose_culling import DEFAULTS
        changed.pose_culling = json.dumps(DEFAULTS)
        with fixed.cpu_allocations(), self.assertRaisesRegex(ValueError, 'pose-culling policy'):
            discrete.raw_model(self.xyz).restore(deepcopy(model.capture()), changed)


class NoCullingPlanTests(unittest.TestCase):
    setUp = fixed.FixedLidarPlanTests.setUp

    def test_gpu1_plan_has_no_mesh_inputs_or_culling_and_retains_every_10k_evaluation(self):
        discrete.write_cloud(self.ply, np.array([[i*.01, 0, 2] for i in range(16)], np.float32))
        run_local_scale.prepare(self.argv)
        target = self.root/'gpu1_noculling'
        args = ['--reference-model', str(self.model), '--model', str(target), '--gpu', '1',
            '--ceiling-multiplier', '15', '--target-multiplier', '3', '--schedule-mode', 'joint',
            '--pose-refinement', '--lidar-offsets', '--fixed-lidar-positions', '--lidar-activation', '--depth-decay',
            '--debug-preview-every', '250', '--debug-preview-train', '1', '--debug-preview-test', '3']
        with patch('utils.pose_culling.geometry_inputs', side_effect=AssertionError('No mesh input is allowed')):
            tune_local_scale.prepare(args)
            tune_local_scale.prepare(args)
        data = periodic.load_plan(target)
        plan = data['plan']
        self.assertEqual(data['gpu'], '1')
        self.assertEqual(data['iterations'], [10000, 20000, 30000])
        self.assertEqual(data['frames'], sorted(self.names))
        self.assertFalse(plan['hard_culling'])
        self.assertFalse(plan['opacity_resets'])
        self.assertNotIn('pose_culling', plan)
        self.assertNotIn('scene_geometry', plan)
        self.assertEqual(plan['lidar_offsets'], offsets.FIXED)
        self.assertEqual(plan['lidar_activation'], activation.DEFAULTS)
        self.assertEqual(plan['distance_updates']['normalization'], 'relative')
        self.assertEqual(plan['pose_refinement']['every'], 1)
        self.assertEqual(plan['training_schedule']['cap']['final_multiplier'], 15.)
        command = plan['stages'][0]['argv']
        self.assertEqual(command[command.index('--debug_preview_every')+1], '250')
        self.assertEqual(command[command.index('--debug_preview_train')+1], '1')
        self.assertEqual(command[command.index('--debug_preview_test')+1], '3')
        for flag in ('--pose_culling', '--occlusion_geometry', '--train_depth_occlusion_dir'):
            self.assertNotIn(flag, command)
        self.assertEqual(command[command.index('--occlusion_mode')+1], 'off')
        self.assertEqual(command[command.index('--occlusion_inference_mode')+1], 'off')
        from arguments import ModelParams, OptimizationParams
        parser = argparse.ArgumentParser()
        mp, op = ModelParams(parser), OptimizationParams(parser)
        parsed, _ = parser.parse_known_args(command[1:])
        dataset, opt = mp.extract(parsed), op.extract(parsed)
        configure_training(opt, dataset)
        self.assertEqual(opt.densify_until_iter, 0)
        self.assertEqual(opt.pose_culling, '')
        self.assertEqual(activation.requested(opt), activation.DEFAULTS)


if __name__ == '__main__': unittest.main()
