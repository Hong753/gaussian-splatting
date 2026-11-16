"""Exercise actual native optimizer/growth and the full periodic launcher contract."""
import argparse
import json
import math
import unittest
from unittest.mock import patch

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts import run_native_lidar as runner
from scripts import run_soft_surface_long as periodic
from utils.densification_policy import opacity_reset_due
from utils.fixed_lidar import configure_training, project_scales
from utils.general_utils import get_expon_lr_func
import test_fixed_lidar as fixed_tests

cpu_allocations = fixed_tests.cpu_allocations
model = fixed_tests.model


def options():
    parser = argparse.ArgumentParser()
    params = OptimizationParams(parser)
    return params.extract(parser.parse_args(['--scale_control', 'off',
        '--position_lr_init', '3.2e-6', '--position_lr_final', '3.2e-8',
        '--disable_opacity_resets', '--disable_pruning']))


def test_native_adam_moves_xyz_at_reduced_rate_without_scale_projection():
    opt = options()
    # The CPU fixture stubs native imports; restore the actual LR scheduler.
    m = model(opt)
    with cpu_allocations(), patch.dict(m.training_setup.__globals__, {'get_expon_lr_func': get_expon_lr_func}):
        m.training_setup(opt)
    assert m.fixed_lidar_state is None and m.get_xyz.requires_grad
    xyz = m.get_xyz.detach().clone()
    lr = m.update_learning_rate(1)
    expected = get_expon_lr_func(.00016*m.spatial_lr_scale, .0000016*m.spatial_lr_scale,
                                max_steps=opt.iterations)(1)/50
    assert math.isclose(lr, expected)
    (m._xyz.sum() - m._scaling.sum()).backward()
    m.optimizer.step()
    torch.testing.assert_close(m.get_xyz, xyz-lr)
    with torch.no_grad():
        m._scaling[:] = math.log(2.)
    before = m._scaling.detach().clone()
    project_scales(m)
    assert torch.equal(before, m._scaling)
    assert math.isclose(m.update_learning_rate(30000), 3.2e-8*m.spatial_lr_scale)


def test_clone_split_keeps_low_opacity_and_large_points_and_child_adam():
    m = model(options())
    with torch.no_grad(), cpu_allocations():
        m._scaling[:] = torch.tensor([[math.log(.001)]*3, [math.log(2.)]*3])
        m._opacity[:] = -20
        m.xyz_gradient_accum[:] = 1
        m.denom[:] = 1
        m.densify_and_prune(.0002, .005, 1., 20, torch.ones(2), record_stats=True, prune=False)
    assert len(m.get_xyz) == 4  # one clone, two children replacing the other parent
    stats = m.last_densification_stats
    assert (stats['cloned_parents'], stats['split_parents'], stats['pruned']) == (1, 1, 0)
    assert (m.get_opacity < .005).all()
    assert (m.get_scaling.max(dim=1).values > .1).any()
    before = m.get_xyz.detach().clone()
    m._xyz.sum().backward()
    m.optimizer.step()
    assert (m.get_xyz.detach() != before).any(dim=1).all()
    # Existing callers retain native pruning by default.
    with torch.no_grad(), cpu_allocations():
        m.xyz_gradient_accum.zero_()
        m.denom[:] = 1
        m.densify_and_prune(.0002, .005, 1., 20, torch.ones(4), record_stats=True)
    assert len(m.get_xyz) == 0 and m.last_densification_stats['pruned'] == 4


def test_reset_switch_includes_white_background_and_preserves_defaults():
    opt = options()
    assert not any(opacity_reset_due(opt, i, white) for white in (False, True) for i in range(1, 15001))
    opt.disable_opacity_resets = False
    assert opacity_reset_due(opt, 3000)
    assert opacity_reset_due(opt, 500, True)
    assert not opacity_reset_due(opt, 500)
    assert not opacity_reset_due(opt, 3000, soft_surface=True)


def test_native_pruning_removes_opacity_or_size_outliers_and_keeps_good_rows():
    for reason in ('opacity', 'size'):
        m = model(options())
        with torch.no_grad(), cpu_allocations():
            m._scaling[:] = math.log(.01)
            if reason == 'opacity':
                m._opacity[0] = -20
            else:
                m._scaling[0] = math.log(1.)
            kept = m.get_xyz[1].clone()
            m.denom[:] = 1
            m.densify_and_prune(.0002, .005, 1., 20, torch.ones(2), record_stats=True, prune=True)
        assert len(m.get_xyz) == 1 and m.last_densification_stats['pruned'] == 1
        torch.testing.assert_close(m.get_xyz[0], kept)


def test_fine_threshold_splits_instead_of_cloning_and_keeps_gradient_gate():
    for fraction, clones, splits in ((.01, 1, 0), (.0001, 0, 1)):
        opt = options()
        opt.percent_dense = fraction
        m = model(opt)
        before = m.get_xyz.detach().clone()
        with torch.no_grad(), cpu_allocations():
            # R=100, sigma=.02: old boundary is 1, new boundary is .01.
            m._scaling[:] = math.log(.02)
            m.xyz_gradient_accum[:] = torch.tensor([[.001], [0.]])
            m.denom[:] = 1
            torch.manual_seed(0)
            m.densify_and_prune(.0002, .005, 100., 20, torch.ones(2), record_stats=True, prune=True)
        stats = m.last_densification_stats
        assert (stats['cloned_parents'], stats['split_parents'], stats['pruned']) == (clones, splits, 0)
        assert math.isclose(stats['split_sigma_threshold'], fraction*100.)
        assert len(m.get_xyz) == 3
        # The zero-gradient row is preserved even above the new size threshold.
        unchanged = (m.get_xyz.detach() == before[1]).all(dim=1)
        assert unchanged.sum() == 1
        torch.testing.assert_close(m.get_scaling[unchanged], torch.full((1, 3), .02))
        if splits:
            torch.testing.assert_close(m.get_scaling[~unchanged], torch.full((2, 3), .02/1.6))
            assert not (m.get_xyz.detach() == before[0]).all(dim=1).any()


class NativeLidarPlanTests(unittest.TestCase):
    setUp = fixed_tests.FixedLidarPlanTests.setUp

    def test_fine_split_plan_only_changes_boundary_and_preserves_periodic_workflow(self):
        args = runner.parser().parse_args(self.argv+['--pruning', '--gpu', '0'])
        native = runner.build_plan(args)
        del args.percent_dense  # Existing prepared GPU 1 plans lack this option.
        assert runner.build_plan(args) == native
        runner.prepare(self.argv+['--pruning', '--gpu', '0', '--percent-dense', '0.0001'])
        data = periodic.load_plan(self.model)
        fine = data['plan']
        assert fine['stages'][0]['argv'] == native['stages'][0]['argv']+['--percent_dense', '0.0001']
        assert fine['densification']['percent_dense'] == .0001
        assert data['gpu'] == '0' and data['iterations'] == [10000, 20000, 30000]
        assert data['frames'] == sorted(self.names)
        with patch.object(runner, 'execute_plan') as execute:
            periodic.train(self.model)
            execute.assert_called_once()

    def test_invalid_split_fractions_fail_before_preparing_output(self):
        for value in ('0', '-1', 'nan', 'inf', '1.1'):
            args = runner.parser().parse_args(self.argv+['--percent-dense', value])
            with self.assertRaisesRegex(ValueError, 'percent-dense'):
                runner.build_plan(args)
        assert not self.model.exists()

    def test_pruning_only_changes_the_pruning_switch_and_dispatches(self):
        args = runner.parser().parse_args(self.argv)
        old = runner.build_plan(args)
        del args.pruning  # Older saved plans predate this option.
        assert runner.build_plan(args) == old
        runner.prepare(self.argv+['--pruning'])
        data = periodic.load_plan(self.model)
        command = data['plan']['stages'][0]['argv']
        assert data['plan']['pruning']
        assert command == [v for v in old['stages'][0]['argv'] if v != '--disable_pruning']
        p = argparse.ArgumentParser()
        op = OptimizationParams(p)
        parsed, _ = p.parse_known_args(command[1:])
        opt = op.extract(parsed)
        assert not opt.disable_pruning and opt.disable_opacity_resets
        with patch.object(runner, 'execute_plan') as execute:
            periodic.train(self.model)
            execute.assert_called_once()
        # Population reduction is expected with pruning enabled.
        row = dict(iteration=600, points_before=10, points_after=8,
                   pruned=4, cloned_parents=1, split_parents=1)
        (self.model/'densification.jsonl').write_text(json.dumps(row)+'\n')
        result = periodic.experiment_diagnostics(data, 599, 600)
        assert result['passed'] and result['pruned'] == 4

    def test_stop_matches_only_exact_native_run(self):
        from scripts.stop_reconstruction_run import matches
        command = ['python', '-u', 'scripts/run_native_lidar.py', 'train', '--model', str(self.model)]
        assert matches(command, self.model, self.root)
        assert not matches(command, self.root/'different_run', self.root)

    def test_native_plan_parses_as_training_and_dispatches_with_all_frames(self):
        runner.prepare(self.argv)
        data = periodic.load_plan(self.model)
        assert data['experiment'] == 'native_lidar' and data['gpu'] == '1'
        assert data['iterations'] == [10000, 20000, 30000]
        assert data['frames'] == sorted(self.names) and data['selected'] == sorted(self.selected)
        command = data['plan']['stages'][0]['argv']
        p = argparse.ArgumentParser()
        mp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
        from utils.training_previews import add_arguments
        add_arguments(p)
        # train.py owns these non-parameter-group arguments.
        for name in ('test_iterations', 'save_iterations', 'checkpoint_iterations'):
            p.add_argument('--'+name, nargs='+', type=int)
        p.add_argument('--disable_viewer', action='store_true')
        p.add_argument('--publish_checkpoint_snapshots', action='store_true')
        args = p.parse_args(command[1:])
        opt, dataset = op.extract(args), mp.extract(args)
        configure_training(opt, dataset)
        assert not opt.fixed_lidar and opt.scale_control == 'off'
        assert not opt.pose_refinement and not opt.distance_weighting
        assert opt.disable_pruning and opt.disable_opacity_resets
        assert opt.position_lr_init == .00016/50 and opt.position_lr_final == .0000016/50
        assert opt.position_warmup_steps == opt.position_ramp_steps == 0
        assert (opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval) == (500, 15000, 100)
        assert (opt.feature_lr, opt.opacity_lr, opt.scaling_lr, opt.rotation_lr) == (.0025, .025, .005, .001)
        assert args.debug_preview_every == 500 and args.antialiasing
        with patch.object(runner, 'execute_plan') as execute:
            periodic.train(self.model)
            execute.assert_called_once()

    def test_diagnostics_expect_growth_only_during_its_schedule(self):
        runner.prepare(self.argv)
        data = periodic.load_plan(self.model)
        assert not periodic.experiment_diagnostics(data, 0, 10000)['passed']
        rows = [dict(iteration=i, points_before=3, points_after=5,
                     pruned=0, cloned_parents=1, split_parents=1) for i in range(600, 10001, 100)]
        (self.model/'densification.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        assert periodic.experiment_diagnostics(data, 0, 10000)['passed']
        assert periodic.experiment_diagnostics(data, 20000, 30000)['passed']
