"""Frozen measured XYZ remains frozen through pruning, Adam and checkpoint restore."""
import argparse
from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts import run_native_lidar as runner, run_soft_surface_long as periodic
from utils.fixed_lidar import project_scales
import test_native_lidar as native
import test_fixed_lidar as fixed


def options():
    opt = native.options()
    opt.freeze_lidar_positions = True
    opt.scaling_lr = .001
    return opt


def appearance_step(m):
    m.optimizer.zero_grad(set_to_none=True)
    (m._features_dc.sum()+m._scaling.sum()+m._rotation.sum()+m._opacity.sum()).backward()
    assert m._xyz.grad is None
    m.optimizer.step()


def test_frozen_xyz_survives_adam_pruning_and_resume():
    for reason in ('opacity', 'world'):
        opt = options()
        m = native.model(opt)
        original = m.get_xyz.detach().clone()
        color = m._features_dc.detach().clone()
        for step in range(1, 4):
            assert m.update_learning_rate(step) == 0
            appearance_step(m)
        assert torch.equal(m.get_xyz, original)
        assert not torch.equal(color, m._features_dc)
        assert m._xyz not in m.optimizer.state
        with torch.no_grad(), native.cpu_allocations():
            m._scaling[:] = math.log(.001)
            m._opacity[:] = 0
            if reason == 'opacity':
                m._opacity[0] = -20
            else:
                m._scaling[0] = math.log(.02)
            m.xyz_gradient_accum[:] = 1
            m.denom[:] = 1
            m.densify_and_prune(.0001, .005, 1., 0., torch.ones(2),
                record_stats=True, max_world_size=.01, preserve_radii=True)
        stats = m.last_densification_stats
        assert (stats['cloned_parents'], stats['split_parents'], stats['pruned']) == (0, 0, 1)
        assert stats['positions_frozen'] and not stats['growth_enabled']
        assert not m._xyz.requires_grad
        appearance_step(m)
        torch.testing.assert_close(m.get_xyz, original[1:], atol=0, rtol=0)
        restored = native.model(opt)
        with native.cpu_allocations():
            restored.restore(deepcopy(m.capture()), opt)
        assert not restored._xyz.requires_grad and restored._xyz not in restored.optimizer.state
        appearance_step(restored)
        torch.testing.assert_close(restored.get_xyz, original[1:], atol=0, rtol=0)
        with torch.no_grad():
            restored._scaling[:] = math.log(10.)
        project_scales(restored)
        torch.testing.assert_close(restored.get_scaling, torch.full((1,3),10.))


def test_frozen_checkpoint_cannot_silently_resume_with_movable_xyz():
    m = native.model(options())
    with unittest.TestCase().assertRaisesRegex(ValueError, 'frozen LiDAR'):
        native.model(native.options()).restore(m.capture(), native.options())
    with unittest.TestCase().assertRaisesRegex(ValueError, 'frozen LiDAR'):
        m.restore(native.model(native.options()).capture()[:17], options())


class FrozenPlanTests(unittest.TestCase):
    setUp = fixed.FixedLidarPlanTests.setUp

    def test_200k_plan_and_all_frame_evaluator(self):
        argv = self.argv+['--gpu','1','--iterations','200000','--resolution','1',
            '--freeze-lidar-positions','--scaling-lr','.001','--pruning',
            '--densify-from-iter','100000','--densify-until-iter','180000',
            '--densification-interval','2000','--prune-world-scale-ratio','.01',
            '--prune-screen-size','0','--prune-size-from-iter','100000','--preserve-densification-radii']
        runner.prepare(argv)
        data = periodic.load_plan(self.model)
        assert data['gpu']=='1' and data['iterations']==list(range(10000,200001,10000))
        assert data['frames']==sorted(self.names)
        plan = data['plan']
        p = argparse.ArgumentParser()
        mp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
        parsed, _ = p.parse_known_args(plan['stages'][0]['argv'][1:])
        opt = op.extract(parsed)
        assert opt.freeze_lidar_positions and not opt.fixed_lidar
        assert opt.position_lr_init == opt.position_lr_final == 0
        assert opt.scaling_lr == .001 and opt.scale_control == 'off'
        assert not opt.disable_pruning and opt.disable_opacity_resets
        assert (opt.densify_from_iter,opt.densify_until_iter,opt.densification_interval)==(100000,180000,2000)
        assert opt.prune_size_from_iter == 100000
        assert parsed.antialiasing and parsed.resolution == 1
        assert not plan['densification']['enabled'] and plan['pruning']
        with patch.object(runner,'execute_plan') as execute:
            periodic.train(self.model)
            execute.assert_called_once()
        args = runner.parser().parse_args(argv)
        with patch.object(runner,'invoke') as smoke, patch.object(runner.fixed,'execute_plan') as execute:
            runner.execute_plan(args, plan)
            assert '--freeze-lidar-positions' in smoke.call_args.args[0]
            execute.assert_called_once_with(args, plan, smoke=False)
        assert periodic.experiment_diagnostics(data,0,100000)['passed']
        assert not periodic.experiment_diagnostics(data,100000,180000)['passed']
        rows=[dict(iteration=i,points_before=5,points_after=4,pruned=1,
                   cloned_parents=0,split_parents=0,positions_frozen=True)
              for i in range(102000,180000,2000)]
        path=self.model/'densification.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        assert periodic.experiment_diagnostics(data,100000,180000)['passed']
        assert periodic.experiment_diagnostics(data,180000,200000)['passed']
        rows[0]['positions_frozen']=False
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        assert not periodic.experiment_diagnostics(data,100000,180000)['passed']
