"""Actual cache preprocessing, optimizer population changes, and resume invariants."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData, PlyElement
import torch

import test_fixed_lidar as fixed
from arguments import ModelParams, OptimizationParams
from utils.discrete_lidar_cache import prepare, read_metadata
from utils.discrete_lidar import adapt, audit, configure_training, candidate_children
from scripts import run_discrete_lidar as runner, run_soft_surface_long as periodic


def write_cloud(path, xyz):
    values = np.zeros(len(xyz), dtype=[(k, 'f4') for k in ('x','y','z','nx','ny','nz')]+
                                      [(k,'u1') for k in ('red','green','blue')])
    for col, key in enumerate(('x','y','z')):
        values[key] = xyz[:, col]
    values['red'], values['green'], values['blue'] = 100, 150, 200
    PlyData([PlyElement.describe(values, 'vertex')]).write(path)


def options(cache):
    parser = argparse.ArgumentParser()
    params = OptimizationParams(parser)
    return params.extract(parser.parse_args(['--discrete_lidar', '--discrete_lidar_cache', str(cache),
        '--discrete_lidar_max_new', '4', '--discrete_lidar_radius', '.045']))


def raw_model(xyz):
    m = fixed.load_model()(0)
    n = len(xyz)
    def par(value):
        return torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32).clone())
    m._xyz = par(xyz)
    m._scaling = par(np.full((n, 3), math.log(.02)))
    m._features_dc = par(np.full((n, 1, 3), .3))
    m._features_rest = par(np.empty((n, 0, 3)))
    m._opacity = par(np.zeros((n, 1)))
    m._rotation = par(np.tile([1, 0, 0, 0], (n, 1)))
    m._exposure = par(np.eye(3, 4)[None])
    m.pretrained_exposures = None
    m.spatial_lr_scale = 1000.
    m.max_radii2D = torch.zeros(n)
    return m


class DiscreteLidarTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.xyz = np.array([[i*.01, 0, 2] for i in range(16)]+[[0,0,2]], np.float32)
        self.cloud, self.cache = self.root/'input.ply', self.root/'cache'
        write_cloud(self.cloud, self.xyz)
        with contextlib.redirect_stdout(io.StringIO()):
            self.meta = prepare(self.cloud, self.cache, fraction=.25, voxel=.08, neighbors=5, workers=1, chunk=3)
        self.opt = options(self.cache)

    def model(self, opt=None):
        m = raw_model(self.xyz)
        with fixed.cpu_allocations():
            m.training_setup(opt or self.opt)
        return m

    def test_cache_deduplicates_exact_coordinates_preserves_coverage_and_reuses_files(self):
        self.assertEqual((self.meta['original_count'], self.meta['count'], self.meta['initial_count']), (17,16,4))
        xyz = np.load(self.cache/'xyz.npy')
        original_ids = np.load(self.cache/'source_ids.npy')
        np.testing.assert_array_equal(xyz, self.xyz[original_ids])
        selected = np.load(self.cache/'initial_ids.npy')
        self.assertEqual(len(np.unique(np.floor(xyz[selected, 0]/.08))), 2)
        neighbor = np.load(self.cache/'neighbors.npy')
        self.assertTrue(np.all(neighbor != np.arange(16)[:, None]))
        distances = np.linalg.norm(xyz[neighbor]-xyz[:, None], axis=-1)
        expected = np.sqrt(np.maximum((distances[:, :3]**2).mean(axis=1), 1e-7))
        np.testing.assert_allclose(expected, np.load(self.cache/'sigma.npy'), rtol=2e-6)
        mtime = (self.cache/'neighbors.npy').stat().st_mtime_ns
        with contextlib.redirect_stdout(io.StringIO()):
            reused = prepare(self.cloud, self.cache, fraction=.25, voxel=.08, neighbors=5, workers=1)
        self.assertEqual(reused, self.meta)
        self.assertEqual(mtime, (self.cache/'neighbors.npy').stat().st_mtime_ns)
        with self.assertRaisesRegex(ValueError, 'different inputs/settings'):
            prepare(self.cloud, self.cache, fraction=.5, voxel=.08, neighbors=5)
        with (self.cache/'sigma.npy').open('ab') as stream:
            stream.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'cache changed'):
            read_metadata(self.cache)

    def test_adam_then_growth_and_scale_pruning_preserve_centers_survivor_moments_and_fresh_child_state(self):
        m = self.model()
        old_ids = m.discrete_lidar_state['active_ids'].clone()
        self.assertEqual(len(old_ids), 4)
        self.assertFalse(m._xyz.requires_grad)
        self.assertLessEqual(float(m.get_scaling.max()), .02500001)
        loss = -m._features_dc.sum() - m._opacity.sum() + m._scaling.square().sum()
        loss.backward(); m.optimizer.step(); m.optimizer.zero_grad(set_to_none=True)
        colors = m._features_dc.detach().clone()
        moments = m.optimizer.state[m._features_dc]['exp_avg'].clone()
        with torch.no_grad():
            m._scaling[0, 0] = math.log(.06)
        m.xyz_gradient_accum.fill_(1.)
        m.denom.fill_(1.)
        with fixed.cpu_allocations():
            result = adapt(m, 100, .0002)
        self.assertEqual(result['scale_pruned'], 1)
        self.assertGreater(result['added'], 0)
        self.assertLessEqual(result['added'], self.opt.discrete_lidar_max_new)
        ids = m.discrete_lidar_state['active_ids']
        torch.testing.assert_close(ids[:3], old_ids[1:], atol=0, rtol=0)
        self.assertNotIn(int(old_ids[0]), ids.tolist())  # No immediate reset/reseed of the pruned coordinate.
        torch.testing.assert_close(m._features_dc[:3], colors[1:], atol=0, rtol=0)
        after = m.optimizer.state[m._features_dc]['exp_avg']
        torch.testing.assert_close(after[:3], moments[1:], atol=0, rtol=0)
        self.assertEqual(int(torch.count_nonzero(after[3:])), 0)
        self.assertEqual(int(torch.count_nonzero(m.xyz_gradient_accum)), 0)
        self.assertTrue(audit(m, 100)['positions_in_pool'])
        self.assertEqual(audit(m, 100)['above_prune_fraction'], 0)
        # New rows render/learn on following steps, but XYZ never participates in Adam.
        anchors = m.get_xyz.detach().clone()
        for step in range(3):
            self.assertEqual(m.update_learning_rate(step), 0)
            (m._features_dc.square().mean()+m._scaling.square().mean()).backward()
            m.optimizer.step(); m.optimizer.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(anchors, m.get_xyz))
        self.assertNotIn(m._xyz, m.optimizer.state)

    def test_neighbor_radius_visibility_and_unique_candidate_rules(self):
        m = self.model()
        empty = torch.zeros(len(m.get_xyz), dtype=torch.bool)
        parent, ids = candidate_children(m, empty, .0002)
        self.assertEqual(len(ids), 0)  # No observed gradients.
        m.xyz_gradient_accum.fill_(1); m.denom.fill_(1)
        parent, ids = candidate_children(m, empty, .0002)
        self.assertGreater(len(ids), 0)
        self.assertEqual(len(ids), len(torch.unique(ids)))
        self.assertFalse(torch.isin(ids, m.discrete_lidar_state['active_ids']).any())
        distance = (m.discrete_lidar_pool['xyz'][ids]-m.get_xyz[parent]).norm(dim=1)
        self.assertLessEqual(float(distance.max()), self.opt.discrete_lidar_radius)
        m.discrete_lidar_state['policy']['radius'] = 1e-5
        self.assertEqual(len(candidate_children(m, empty, .0002)[1]), 0)

    def test_population_and_centers_cannot_be_mutated_by_native_clone_prune_or_optimizer(self):
        m = self.model()
        with self.assertRaisesRegex(RuntimeError, 'population edits'):
            m.prune_points(torch.zeros(len(m.get_xyz), dtype=torch.bool))
        with self.assertRaisesRegex(RuntimeError, 'measured candidate'):
            m.densification_postfix(*([None]*7))
        with torch.no_grad():
            m._xyz[0, 1] += .01
        with self.assertRaisesRegex(RuntimeError, 'moved off'):
            audit(m, 1)

    def test_checkpoint_resume_reloads_pool_preserves_population_and_rejects_changed_policy(self):
        m = self.model()
        (-m._features_dc.sum()).backward(); m.optimizer.step(); m.optimizer.zero_grad(set_to_none=True)
        m.xyz_gradient_accum.fill_(1); m.denom.fill_(1)
        with fixed.cpu_allocations():
            adapt(m, 600, .0002)
        stream = io.BytesIO(); torch.save(m.capture(), stream); stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = raw_model(self.xyz)
        with fixed.cpu_allocations():
            restored.restore(saved, self.opt)
        self.assertTrue(torch.equal(restored.get_xyz, m.get_xyz))
        self.assertTrue(torch.equal(restored.discrete_lidar_state['active_ids'], m.discrete_lidar_state['active_ids']))
        torch.testing.assert_close(restored.optimizer.state[restored._features_dc]['exp_avg'],
                                   m.optimizer.state[m._features_dc]['exp_avg'], atol=0, rtol=0)
        self.assertTrue(audit(restored, 601)['xyz_frozen'])
        other = options(self.cache); other.discrete_lidar_prune_sigma = .04
        with fixed.cpu_allocations(), self.assertRaisesRegex(ValueError, 'cache or policy differs'):
            raw_model(self.xyz).restore(saved, other)
        with self.assertRaisesRegex(ValueError, 'mode differs'):
            raw_model(self.xyz).restore(saved[:15], self.opt)

    def test_no_growth_when_full_pool_and_no_opacity_pruning_or_projection(self):
        full = self.root/'full'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.cloud, full, fraction=1., voxel=.08, neighbors=5, workers=1)
        m = self.model(options(full))
        m.xyz_gradient_accum.fill_(1); m.denom.fill_(1)
        with torch.no_grad():
            m._opacity.fill_(-100)
        with fixed.cpu_allocations():
            report = adapt(m, 600, .0002)
        self.assertEqual((report['added'], report['scale_pruned'], report['after']), (0,0,16))
        with torch.no_grad():
            m._scaling[0, 0] = math.log(.051)
        self.assertGreater(audit(m, 601)['above_prune_fraction'], 0)  # Oversize is allowed between events.
        with fixed.cpu_allocations():
            report = adapt(m, 700, .0002)
        self.assertEqual((report['scale_pruned'], report['after']), (1,15))

    def test_configuration_and_initial_cloud_mismatch(self):
        parser = argparse.ArgumentParser(); params = ModelParams(parser)
        data = params.extract(parser.parse_args([]))
        configure_training(self.opt, data)
        self.assertEqual((self.opt.position_lr_init, self.opt.position_lr_final), (0,0))
        self.assertTrue(data.discrete_lidar)
        self.opt.fixed_lidar = True
        with self.assertRaisesRegex(ValueError, 'without other'):
            configure_training(self.opt, data)
        m = raw_model(self.xyz)
        with torch.no_grad():
            m._xyz[0, 0] += 1
        with fixed.cpu_allocations(), self.assertRaisesRegex(ValueError, 'differ from initialization'):
            m.training_setup(options(self.cache))


class DiscretePlanTests(unittest.TestCase):
    setUp = fixed.FixedLidarPlanTests.setUp

    def test_third_plan_routes_same_all_frame_checkpoint_worker_but_own_population_policy(self):
        runner.prepare(self.argv)
        data = periodic.load_plan(self.model)
        self.assertEqual(data['experiment'], 'discrete_lidar')
        self.assertEqual(data['gpu'], '2')
        self.assertEqual(data['iterations'], list(range(10000,100001,10000)))
        command = data['plan']['stages'][0]['argv']
        self.assertIn('--discrete_lidar', command)
        self.assertNotIn('--fixed_lidar', command)
        self.assertNotIn('--max_scale_world', command)
        self.assertEqual(command[command.index('--discrete_lidar_prune_sigma')+1], '0.05')
        self.assertEqual(command[command.index('--densify_until_iter')+1], '100000')
        self.assertEqual(command[command.index('--position_lr_init')+1], '0')
        with patch.object(runner, 'execute_plan') as execute:
            periodic.train(self.model)
        self.assertEqual(execute.call_args[0][1], data['plan'])
        row = dict(iteration=10000, positions_in_pool=True, unique_coordinates=True, xyz_frozen=True, gaussians=8)
        (self.model/'discrete_lidar_training.jsonl').write_text(json.dumps(row)+'\n')
        (self.model/'discrete_lidar_growth.jsonl').write_text(json.dumps(dict(iteration=10000, added=5, scale_pruned=1))+'\n')
        diagnostic = periodic.experiment_diagnostics(data, 0, 10000)
        self.assertTrue(diagnostic['passed']); self.assertEqual(diagnostic['added'], 5)
        self.assertEqual(diagnostic['scale_pruned'], 1); self.assertEqual(diagnostic['warnings'], [])


if __name__ == '__main__':
    unittest.main()
