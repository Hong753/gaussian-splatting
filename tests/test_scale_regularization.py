"""Gradient direction, soft-bound semantics, optimizer state and checkpoint fidelity."""
import contextlib
import io
import json
import math
from unittest.mock import patch
import unittest

import torch

from utils.fixed_lidar import (audit, project_scales, scale_regularization,
    sample_scale_gradients, added_scale_gradient_stats, validate_control)
from test_fixed_lidar import model, options, cpu_allocations
import test_fixed_lidar as fixed_tests
from scripts import run_fixed_lidar as runner
from scripts import run_soft_surface_long as periodic


def reg_options(ratio=2., world=.05, weight=.01):
    opt = options(ratio, world)
    opt.scale_control = 'regularize'
    opt.scale_regularization_weight = weight
    return opt


class ScaleRegularizationTests(unittest.TestCase):
    def test_penalty_is_zero_below_target_and_gradients_shrink_only_exceeding_axes(self):
        m = model(reg_options())
        delta = torch.tensor([[-.4, 0., .2], [.8, -.1, 0.]])
        with torch.no_grad():
            m._scaling.copy_(m.fixed_lidar_state['log_scale_cap']+delta)
        loss = scale_regularization(m)
        self.assertAlmostEqual(float(loss), .01*(.2**2+.8**2)/6, places=8)
        loss.backward()
        torch.testing.assert_close(m._scaling.grad, .01*2/6*delta.clamp_min(0), atol=1e-9, rtol=1e-5)
        self.assertIsNone(m._xyz.grad)
        self.assertIsNone(m._opacity.grad)
        self.assertIsNone(m._rotation.grad)
        self.assertIsNone(m._features_dc.grad)
        with torch.no_grad():
            m._scaling.copy_(m.fixed_lidar_state['log_scale_cap']-1)
        m.optimizer.zero_grad(set_to_none=True)
        scale_regularization(m).backward()
        self.assertEqual(float(scale_regularization(m)), 0.)
        self.assertEqual(int(torch.count_nonzero(m._scaling.grad)), 0)

    def test_regularizer_derivative_matches_finite_difference(self):
        m = model(reg_options())
        m._scaling = torch.nn.Parameter(m._scaling.detach().double())
        m.fixed_lidar_state['log_scale_cap'] = m.fixed_lidar_state['log_scale_cap'].double()
        with torch.no_grad():
            m._scaling.copy_(m.fixed_lidar_state['log_scale_cap']+.4)
        exact = torch.autograd.grad(scale_regularization(m), m._scaling)[0][0,0].item()
        with torch.no_grad():
            m._scaling[0,0] += 1e-5
            plus = float(scale_regularization(m))
            m._scaling[0,0] -= 2e-5
            minus = float(scale_regularization(m))
        self.assertAlmostEqual(exact, (plus-minus)/2e-5, places=10)

    def test_all_points_penalty_cannot_be_avoided_by_lowering_opacity(self):
        m = model(reg_options())
        with torch.no_grad():
            m._scaling[1] = m.fixed_lidar_state['log_scale_cap'][1]+math.log(2)
            m._opacity.fill_(-100)
        penalty = scale_regularization(m)
        self.assertGreater(float(penalty), 0.)
        penalty.backward()
        self.assertGreater(float(m._scaling.grad[1].min()), 0.)
        self.assertIsNone(m._opacity.grad)

    def test_same_initial_bounds_then_no_projection_or_moment_resets(self):
        m = model(reg_options())
        torch.testing.assert_close(m.get_scaling[1], torch.full((3,), .05), atol=1e-8, rtol=0)
        with torch.no_grad():
            m._scaling[0,0] = m.fixed_lidar_state['log_scale_cap'][0,0]+.6
        (-m._scaling.sum()+scale_regularization(m)).backward()
        m.optimizer.step()
        before = m._scaling.detach().clone()
        moments = {k: m.optimizer.state[m._scaling][k].clone() for k in ('exp_avg', 'exp_avg_sq')}
        project_scales(m)
        self.assertTrue(torch.equal(before, m._scaling))
        for key, value in moments.items():
            self.assertTrue(torch.equal(m.optimizer.state[m._scaling][key], value))
        report = audit(m, 1)
        self.assertEqual(report['scale_control'], 'regularize')
        self.assertFalse(report['scales_within_bounds'])
        self.assertTrue(report['scales_finite'])
        self.assertGreater(report['above_target_fraction'], 0.)

    def test_regularizer_only_adam_updates_reduce_excess_without_moving_or_deleting_points(self):
        m = model(reg_options())
        xyz = m.get_xyz.detach().clone()
        with torch.no_grad():
            m._scaling.copy_(m.fixed_lidar_state['log_scale_cap']+.5)
        initial = float(scale_regularization(m))
        for _ in range(100):
            scale_regularization(m).backward()
            m.optimizer.step(); project_scales(m); m.optimizer.zero_grad(set_to_none=True)
        self.assertLess(float(scale_regularization(m)), initial*.2)
        self.assertTrue(torch.equal(m.get_xyz, xyz))
        self.assertNotIn(m._xyz, m.optimizer.state)
        self.assertEqual(len(m.get_xyz), 2)

    def test_serialized_resume_keeps_excess_scales_exactly_and_rejects_wrong_policy(self):
        opt = reg_options()
        m = model(opt)
        with torch.no_grad():
            m._scaling.copy_(m.fixed_lidar_state['log_scale_cap']+.4)
        scale_regularization(m).backward(); m.optimizer.step()
        stream = io.BytesIO(); torch.save(m.capture(), stream); stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = model(opt)
        with cpu_allocations():
            restored.restore(state, opt)
        self.assertTrue(torch.equal(restored._scaling, m._scaling))
        self.assertFalse(audit(restored, 2)['scales_within_bounds'])
        torch.testing.assert_close(restored.optimizer.state[restored._scaling]['exp_avg'],
                                   m.optimizer.state[m._scaling]['exp_avg'], atol=0, rtol=0)
        with cpu_allocations(), self.assertRaisesRegex(ValueError, 'scale-control mode differs'):
            model(options(2.,.05)).restore(state, options(2.,.05))
        other = reg_options(weight=.1)
        with cpu_allocations(), self.assertRaisesRegex(ValueError, 'regularization weight differs'):
            model(other).restore(state, other)

    def test_gradient_log_measures_added_regularizer_without_changing_gradients(self):
        m = model(reg_options())
        with torch.no_grad():
            m._scaling.copy_(m.fixed_lidar_state['log_scale_cap']+.4)
        (.2*m._scaling.sum()).backward()
        sample = sample_scale_gradients(m, torch.tensor([[0],[1]]))
        expected = torch.autograd.grad(scale_regularization(m), m._scaling)[0]
        scale_regularization(m).backward()
        before = m._scaling.grad.clone()
        stats = added_scale_gradient_stats(m, sample)
        self.assertAlmostEqual(stats['regularizer_scale_grad_norm'], float(expected.norm()), places=7)
        self.assertGreater(stats['regularizer_to_rgb_scale_grad_ratio'], 0.)
        self.assertTrue(torch.equal(before, m._scaling.grad))
        empty = sample_scale_gradients(m, torch.zeros(2, dtype=torch.bool))
        self.assertIsNone(added_scale_gradient_stats(m, empty)['regularizer_to_rgb_scale_grad_ratio'])

    def test_invalid_policy_and_weight_are_rejected(self):
        for mode, weight in (('other',.01),('regularize',0),('regularize',-1),('regularize',float('nan'))):
            with self.subTest(mode=mode,weight=weight), self.assertRaises(ValueError):
                validate_control(mode, weight)


class ScaleRegularizationPlanTests(unittest.TestCase):
    setUp = fixed_tests.FixedLidarPlanTests.setUp

    def test_separate_gpu1_plan_and_diagnostics_accept_a_soft_target_exceedance(self):
        runner.prepare(self.argv+['--scale-control','regularize','--max-scale-world','.05','--gpu','1'])
        data = periodic.load_plan(self.model)
        self.assertEqual(data['gpu'], '1')
        self.assertEqual(data['iterations'], list(range(10000,100001,10000)))
        self.assertEqual(data['plan']['scale_control'], 'regularize')
        self.assertEqual(data['plan']['scale_regularization_weight'], .01)
        command = data['plan']['stages'][0]['argv']
        self.assertEqual(command[command.index('--scale_control')+1], 'regularize')
        (self.model/'fixed_lidar_training.jsonl').write_text(json.dumps(dict(iteration=10000,
            positions_unchanged=True, scales_finite=True, scales_within_bounds=False, gaussians=3))+'\n')
        (self.model/'scale_regularization_training.jsonl').write_text(json.dumps(dict(iteration=10000,
            scale_regularization_loss=.001, regularizer_scale_grad_norm=.001,
            regularizer_to_rgb_scale_grad_ratio=.1))+'\n')
        diagnostic = periodic.experiment_diagnostics(data, 0, 10000)
        self.assertTrue(diagnostic['passed']); self.assertEqual(diagnostic['warnings'], [])
        with patch.object(runner, 'execute_plan') as execute:
            periodic.train(self.model)
        self.assertEqual(execute.call_args[0][0].gpu, '1')
        self.assertEqual(execute.call_args[0][1], data['plan'])

    def test_original_saved_hard_plan_still_rebuilds_without_new_arguments(self):
        args = runner.parser().parse_args(self.argv)
        old = vars(args).copy()
        del old['scale_control']; del old['scale_regularization_weight']
        from types import SimpleNamespace
        self.assertEqual(runner.build_plan(args), runner.build_plan(SimpleNamespace(**old)))


if __name__ == '__main__':
    unittest.main()
