"""Real Adam updates and checkpoint/launcher contracts for the fixed-cloud ablation."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData, PlyElement
import torch

from arguments import ModelParams, OptimizationParams
from scripts import run_fixed_lidar as runner
from scripts import run_soft_surface_long as periodic
from utils.fixed_lidar import configure_training, audit, project_scales, validate_limits
from test_surface_experiment import load_model
from test_view_comparisons import dataset


@contextlib.contextmanager
def cpu_allocations():
    zeros = torch.zeros
    def cpu_zeros(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return zeros(*args, **kwargs)
    with patch.object(torch, 'zeros', side_effect=cpu_zeros):
        yield


def options(ratio=2., world=0.):
    parser = argparse.ArgumentParser()
    params = OptimizationParams(parser)
    return params.extract(parser.parse_args(['--fixed_lidar', '--max_scale_ratio', str(ratio),
                                            '--max_scale_world', str(world)]))


def model(opt=None):
    m = load_model()(0)
    def par(value):
        return torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
    m._xyz = par([[0,0,2],[1,0,2]])
    m._scaling = par([[math.log(.01)]*3,[math.log(.1)]*3])
    m._features_dc = par([[[.2,.4,.1]], [[.1,.8,.3]]])
    m._features_rest = torch.nn.Parameter(torch.empty(2,0,3))
    m._opacity = par([[0],[0]])
    m._rotation = par([[1,0,0,0],[1,0,0,0]])
    m._exposure = torch.nn.Parameter(torch.eye(3,4)[None])
    m.pretrained_exposures = None
    m.spatial_lr_scale = 1000.  # Large global extent must not enlarge the local cap.
    m.max_radii2D = torch.zeros(2)
    with cpu_allocations():
        m.training_setup(opt or options())
    return m


class FixedLidarTests(unittest.TestCase):
    def test_real_adam_preserves_centers_and_count_but_learns_appearance_and_bounded_scales(self):
        m = model()
        xyz, colors = m.get_xyz.detach().clone(), m._features_dc.detach().clone()
        for step in range(1, 301):
            self.assertEqual(m.update_learning_rate(step), 0.)
            # Strong pressure to enlarge every axis; change appearance/opacity too.
            loss = -m._scaling.sum() + (m._features_dc-1).square().sum() - m._opacity.sum()
            loss.backward()
            m.optimizer.step()
            project_scales(m, chunk=1)
            m.optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(m.get_scaling, torch.tensor([[.02]*3,[.2]*3]), atol=1e-7, rtol=0.)
        self.assertTrue(torch.equal(m.get_xyz, xyz))
        self.assertFalse(torch.equal(m._features_dc, colors))
        self.assertGreater(m._opacity.min().item(), 0.)
        self.assertIsNone(m._xyz.grad)
        self.assertNotIn(m._xyz, m.optimizer.state)
        self.assertEqual(audit(m, 300)['at_cap_fraction'], 1.)
        self.assertEqual(len(m.get_xyz), 2)
        # Opposite force must immediately allow contraction from the bound.
        previous = m._scaling.detach().clone()
        m._scaling.sum().backward(); m.optimizer.step(); project_scales(m)
        self.assertTrue((m._scaling < previous).all())

    def test_absolute_sigma_cap_and_ratio_below_one_apply_before_first_render(self):
        m = model(options(.5, .02))
        torch.testing.assert_close(m.get_scaling, torch.tensor([[.005]*3,[.02]*3]), atol=1e-8, rtol=0.)
        self.assertTrue(audit(m, 0)['scales_within_bounds'])

    def test_projection_preserves_unclipped_axis_moments(self):
        m = model(options(1.))
        m._scaling.sum().backward(); m.optimizer.step(); m.optimizer.zero_grad(set_to_none=True)
        state = m.optimizer.state[m._scaling]
        before = {k: state[k].clone() for k in ('exp_avg', 'exp_avg_sq')}
        with torch.no_grad():
            m._scaling[0,0] += 10
        project_scales(m, chunk=1)
        for key in before:
            self.assertEqual(state[key][0,0], 0.)
            torch.testing.assert_close(state[key][1], before[key][1])
            torch.testing.assert_close(state[key][0,1:], before[key][0,1:])

    def test_serialized_optimizer_restore_preserves_original_cap_and_fixed_mode(self):
        opt = options()
        m = model(opt)
        (-m._scaling.sum()).backward(); m.optimizer.step(); project_scales(m)
        stream = io.BytesIO(); torch.save(m.capture(), stream); stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = model(opt)
        with cpu_allocations():
            restored.restore(saved, opt)
        self.assertFalse(restored.get_xyz.requires_grad)
        self.assertTrue(torch.equal(restored.fixed_lidar_state['log_scale_cap'], m.fixed_lidar_state['log_scale_cap']))
        torch.testing.assert_close(restored.get_scaling, m.get_scaling, atol=0, rtol=0)
        self.assertTrue(audit(restored, 1)['positions_unchanged'])
        changed = options(3.)
        with cpu_allocations(), self.assertRaisesRegex(ValueError, 'limits differ'):
            model(changed).restore(saved, changed)
        with self.assertRaisesRegex(ValueError, 'mode differs'):
            restored.restore(saved[:14], opt)
        opt.fixed_lidar = False
        with self.assertRaisesRegex(ValueError, 'mode differs'):
            restored.restore(saved, opt)

    def test_prune_and_grow_cannot_silently_mutate_fixed_cloud(self):
        m = model()
        with self.assertRaisesRegex(RuntimeError, 'Pruning is disabled'):
            m.prune_points(torch.tensor([False, True]))
        with self.assertRaisesRegex(RuntimeError, 'Cloning/splitting is disabled'):
            m.densification_postfix(*([None]*7))
        with torch.no_grad():
            m._xyz[0,0] += .01
        with self.assertRaisesRegex(RuntimeError, 'audit failed'):
            audit(m, 1)

    def test_flag_overrides_all_position_and_density_schedules(self):
        parser = argparse.ArgumentParser()
        params = ModelParams(parser)
        data = params.extract(parser.parse_args([]))
        opt = options(); opt.position_ramp_steps = 2000; opt.position_warmup_steps = 7000
        configure_training(opt, data)
        self.assertEqual((opt.position_lr_init, opt.position_lr_final, opt.position_warmup_steps,
                          opt.position_ramp_steps, opt.densify_until_iter, opt.densification_phase_offset), (0,)*6)
        data.occlusion_mode = 'mesh'
        with self.assertRaisesRegex(ValueError, 'native 3DGS'):
            configure_training(opt, data)
        for ratio, world in ((0,0),(-1,0),(float('nan'),0),(2,-1),(2,float('inf'))):
            with self.subTest(ratio=ratio, world=world), self.assertRaises(ValueError):
                validate_limits(ratio, world)

    def test_exported_ply_contains_physical_bounded_scales_and_exact_centers(self):
        m = model(options(.5, .02))
        # Use the native serializer with real plyfile; only compiled CUDA
        # dependencies were stubbed by load_model for this CPU regression.
        with tempfile.TemporaryDirectory() as temp, patch.dict(m.save_ply.__globals__,
                {'PlyData': PlyData, 'PlyElement': PlyElement}):
            path = Path(temp)/'point_cloud.ply'
            m.save_ply(str(path))
            vertices = PlyData.read(path)['vertex']
            xyz = np.stack([vertices[n] for n in ('x','y','z')], axis=1)
            sigma = np.exp(np.stack([vertices['scale_'+str(i)] for i in range(3)], axis=1))
            np.testing.assert_array_equal(xyz, m.get_xyz.detach().numpy())
            np.testing.assert_allclose(sigma, [[.005]*3,[.02]*3], rtol=1e-6)


class FixedLidarPlanTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source, self.keyframes, self.names, self.selected = dataset(self.root)
        self.ply = self.root/'input.ply'
        values = np.zeros(3, dtype=[(v, 'f4') for v in ('x','y','z','nx','ny','nz')]+
                                  [(v,'u1') for v in ('red','green','blue')])
        values['z'] = 2
        PlyData([PlyElement.describe(values, 'vertex')]).write(self.ply)
        self.model = self.root/'fixed'
        self.argv = ['--source', str(self.source), '--model', str(self.model), '--keyframes', str(self.keyframes),
                     '--point-cloud', str(self.ply), '--expected-frames', '16']

    def test_fresh_100k_plan_uses_exact_cloud_and_no_density_or_soft_surface(self):
        runner.prepare(self.argv)
        data = periodic.load_plan(self.model)
        self.assertEqual(data['experiment'], 'fixed_lidar')
        self.assertEqual(data['iterations'], list(range(10000,100001,10000)))
        self.assertEqual(data['frames'], sorted(self.names))
        self.assertEqual(data['selected'], sorted(self.selected))
        self.assertEqual(data['plan']['initial_points'], 3)
        command = data['plan']['stages'][0]['argv']
        self.assertIn('--fixed_lidar', command)
        self.assertIn('--publish_checkpoint_snapshots', command)
        self.assertEqual(command[command.index('--initial_point_cloud')+1], str(self.ply))
        self.assertEqual(command[command.index('--densify_until_iter')+1], '0')
        self.assertEqual(command[command.index('--occlusion_mode')+1], 'off')
        self.assertNotIn('--soft_surface_geometry', command)
        with patch.object(runner, 'execute_plan') as execute:
            periodic.train(self.model)
            execute.assert_called_once()
        self.assertEqual(json.loads((periodic.directory(self.model)/'training.json').read_text())['state'], 'trained')

    def test_changed_lidar_prevents_training_and_partial_dataset_prevents_prepare(self):
        runner.prepare(self.argv)
        self.ply.write_bytes(self.ply.read_bytes()+b'changed')
        with patch.object(runner, 'execute_plan') as execute, self.assertRaisesRegex(ValueError, 'Input changed'):
            periodic.train(self.model)
        execute.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Expected 5009'):
            runner.prepare(self.argv[:-2])

    def test_dry_run_writes_nothing_and_existing_weights_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            runner.prepare(self.argv+['--dry-run'])
        self.assertFalse(self.model.exists())
        self.model.mkdir(); (self.model/'cfg_args').write_text('Namespace()')
        with self.assertRaisesRegex(ValueError, 'fresh model'):
            runner.prepare(self.argv)

    def test_fixed_diagnostics_do_not_report_missing_soft_surface(self):
        runner.prepare(self.argv)
        data = periodic.load_plan(self.model)
        (self.model/'fixed_lidar_training.jsonl').write_text(json.dumps(dict(iteration=10000,
            positions_unchanged=True, scales_within_bounds=True, gaussians=3))+'\n')
        result = periodic.experiment_diagnostics(data, 0, 10000)
        self.assertTrue(result['passed']); self.assertEqual(result['warnings'], [])


if __name__ == '__main__':
    unittest.main()
