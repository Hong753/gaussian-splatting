"""Actual native population updates and full launcher contract for the large-scene preset."""
import argparse
import json
import math
import unittest
from unittest.mock import patch

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts import run_native_lidar as runner, run_soft_surface_long as periodic
from utils.densification_policy import size_pruning_thresholds, opacity_reset_due
from utils.fixed_lidar import project_scales
from utils.general_utils import get_expon_lr_func
import test_native_lidar as native_tests
import test_fixed_lidar as fixed_tests


def test_preserved_radius_history_prunes_old_rows_and_identical_clones():
    for grow in (False, True):
        m = native_tests.model(native_tests.options())
        with torch.no_grad(), native_tests.cpu_allocations():
            m._scaling[:] = math.log(.001)
            m.max_radii2D[:] = torch.tensor([100., 2.])
            m.denom[:] = 1
            m.xyz_gradient_accum[0] = 1 if grow else 0
            expected = m.get_xyz[1].clone()
            m.densify_and_prune(.0002, .005, 1., 20, torch.tensor([100., 2.]),
                record_stats=True, max_world_size=0., preserve_radii=True)
        assert len(m.get_xyz) == 1
        torch.testing.assert_close(m.get_xyz[0], expected)
        assert m.last_densification_stats['pruned'] == (2 if grow else 1)
        assert not m.max_radii2D.any()


def test_split_children_require_new_radius_observations():
    m = native_tests.model(native_tests.options())
    with torch.no_grad(), native_tests.cpu_allocations():
        m._scaling[:] = torch.tensor([[math.log(2.)]*3, [math.log(.001)]*3])
        m.max_radii2D[:] = torch.tensor([100., 2.])
        m.xyz_gradient_accum[0] = 1
        m.denom[:] = 1
        m.densify_and_prune(.0002, .005, 1., 20, torch.tensor([100., 2.]),
            record_stats=True, max_world_size=0., preserve_radii=True)
    assert len(m.get_xyz) == 3
    assert m.last_densification_stats['split_parents'] == 1
    assert m.last_densification_stats['pruned'] == 0
    assert not m.max_radii2D.any()


def test_world_pruning_operates_with_screen_pruning_disabled():
    m = native_tests.model(native_tests.options())
    with torch.no_grad(), native_tests.cpu_allocations():
        m._scaling[:] = torch.tensor([[math.log(.02)]*3, [math.log(.001)]*3])
        m.max_radii2D[:] = 100
        m.denom[:] = 1
        expected = m.get_xyz[1].clone()
        m.densify_and_prune(.0002, .005, 1., 0., torch.full((2,), 100.),
            record_stats=True, max_world_size=.01, preserve_radii=True)
    assert len(m.get_xyz) == 1 and m.last_densification_stats['pruned'] == 1
    torch.testing.assert_close(m.get_xyz[0], expected)


def test_pruning_start_is_independent_of_disabled_opacity_resets():
    opt = native_tests.options()
    assert size_pruning_thresholds(opt, 3000, 100.) == (None, None)
    assert size_pruning_thresholds(opt, 3001, 100.) == (20., 10.)
    opt.prune_size_from_iter = 5000
    opt.prune_screen_size = 0
    opt.prune_world_scale_ratio = .01
    assert size_pruning_thresholds(opt, 5000, 100.) == (None, None)
    assert size_pruning_thresholds(opt, 5250, 100.) == (0, 1.)
    assert not opacity_reset_due(opt, 6000)


class LargeScenePlanTests(unittest.TestCase):
    setUp = fixed_tests.FixedLidarPlanTests.setUp

    def preset(self):
        return self.argv + ['--gpu', '0', '--resolution', '1', '--iterations', '100000',
            '--position-lr-divisor', '100', '--scaling-lr', '.001', '--percent-dense', '.0001',
            '--densify-grad-threshold', '.0001', '--densify-from-iter', '2000',
            '--densify-until-iter', '40000', '--densification-interval', '250',
            '--pruning', '--prune-world-scale-ratio', '.01', '--prune-screen-size', '0',
            '--prune-size-from-iter', '5000', '--preserve-densification-radii']

    def test_preset_parse_dispatch_real_learning_rates_and_periodic_evaluation(self):
        runner.prepare(self.preset())
        data = periodic.load_plan(self.model)
        assert data['gpu'] == '0' and data['iterations'] == list(range(10000,100001,10000))
        assert data['frames'] == sorted(self.names)
        command = data['plan']['stages'][0]['argv']
        p = argparse.ArgumentParser()
        mp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
        from utils.training_previews import add_arguments
        add_arguments(p)
        for name in ('test_iterations', 'save_iterations', 'checkpoint_iterations'):
            p.add_argument('--'+name, nargs='+', type=int)
        for name in ('disable_viewer', 'publish_checkpoint_snapshots'):
            p.add_argument('--'+name, action='store_true')
        args = p.parse_args(command[1:])
        opt = op.extract(args)
        assert args.resolution == 1 and args.debug_preview_every == 500
        assert args.antialiasing and args.occlusion_mode == args.occlusion_inference_mode == 'off'
        assert not opt.fixed_lidar and opt.scale_control == 'off'
        assert (opt.feature_lr, opt.opacity_lr, opt.rotation_lr) == (.0025, .025, .001)
        assert (opt.scaling_lr, opt.percent_dense, opt.densify_grad_threshold) == (.001, .0001, .0001)
        assert (opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval) == (2000,40000,250)
        assert opt.disable_opacity_resets and not opt.disable_pruning
        assert opt.prune_screen_size == 0 and opt.prune_world_scale_ratio == .01
        assert opt.preserve_densification_radii and opt.prune_size_from_iter == 5000
        assert opt.position_lr_max_steps == 100000
        m = native_tests.model(opt)
        with native_tests.cpu_allocations(), patch.dict(m.training_setup.__globals__, {'get_expon_lr_func':get_expon_lr_func}):
            m.training_setup(opt)
        assert math.isclose(m.update_learning_rate(100000), 1.6e-8*m.spatial_lr_scale)
        assert next(g['lr'] for g in m.optimizer.param_groups if g['name']=='scaling') == .001
        with torch.no_grad():
            m._scaling[:] = math.log(10.)
        project_scales(m)
        torch.testing.assert_close(m.get_scaling, torch.full((2,3),10.))
        with patch.object(runner,'execute_plan') as execute:
            periodic.train(self.model)
            execute.assert_called_once()
        # First growth event is 2250, not 600; evaluation follows the saved schedule.
        rows=[dict(iteration=i,points_before=5,points_after=6,pruned=1,cloned_parents=1,split_parents=1)
              for i in range(2250,10001,250)]
        (self.model/'densification.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        assert periodic.experiment_diagnostics(data,0,10000)['passed']
        assert periodic.experiment_diagnostics(data,40000,50000)['passed']

    def test_invalid_new_settings_rejected_without_creating_output(self):
        for extra in (['--scaling-lr','nan'],['--scaling-lr','0'],['--prune-screen-size','-1'],
                      ['--prune-world-scale-ratio','inf'],['--densification-interval','0'],
                      ['--densify-from-iter','16000'],['--densify-until-iter','30001']):
            with self.assertRaises(ValueError):
                runner.build_plan(runner.parser().parse_args(self.argv+extra))
        assert not self.model.exists()
