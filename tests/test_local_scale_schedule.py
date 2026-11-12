"""Cap/LR schedules under real Adam, serialization, and matched run preparation."""
import contextlib
import io
import json
import math
from unittest.mock import patch
import unittest

import numpy as np
import torch
from plyfile import PlyData, PlyElement

import test_local_scale as local
import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
from scripts import run_local_scale, tune_local_scale, run_soft_surface_long as periodic
from scripts.stop_reconstruction_run import matches
from utils.fixed_lidar import audit, project_scales, scale_regularization
from utils.local_scale import prepare
from utils.local_scale_schedule import GROUP_LRS, stats, validate, values


def policy(mode='joint'):
    return dict(schema=1,
        cap=dict(initial_multiplier=1.,final_multiplier=2.,start=2,end=6) if mode in ('cap','joint') else None,
        learning_rates=dict(start=2,end=8,groups={k:[v,v*.1] for k,v in GROUP_LRS.items()}) if mode in ('lr','joint') else None)


class LocalScheduleTests(unittest.TestCase):
    setUp=local.LocalScaleTests.setUp
    prepare=local.LocalScaleTests.prepare
    model=local.LocalScaleTests.model

    def scheduled_model(self, mode='joint'):
        cache=self.root/'final_caps'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.cloud,cache,workers=1,samples=25,ceiling_multiplier=2.,reuse_cache=self.cache)
        self.opt.local_scale_cache=str(cache)
        self.opt.local_scale_ceiling_multiplier=2.
        self.opt.iterations=8
        self.opt.local_scale_schedule=json.dumps(policy(mode))
        return self.model()

    def step(self, model, iteration):
        model.update_learning_rate(iteration)
        loss=(-model._scaling.sum()+(model._features_dc-1).square().sum()-model._opacity.sum()
              +model._rotation.square().sum()+scale_regularization(model))
        loss.backward()
        model.optimizer.step()
        project_scales(model)
        model.optimizer.zero_grad(set_to_none=True)

    def test_caps_increase_lrs_decrease_without_inflating_parameters_or_moments(self):
        m=self.scheduled_model()
        self.step(m,1)
        scales=m._scaling.detach().clone()
        moments={k:v.clone() for k,v in m.optimizer.state[m._scaling].items()}
        cap=m.fixed_lidar_state['log_scale_cap'].clone()
        target=m.fixed_lidar_state['log_scale_target'].clone()
        last=1.
        for iteration in range(2,9):
            self.assertEqual(m.update_learning_rate(iteration),0.)
            expected=values(policy(),iteration)
            multiplier=expected['ceiling_multiplier']
            self.assertGreaterEqual(multiplier,last); last=multiplier
            torch.testing.assert_close(m.fixed_lidar_state['log_scale_cap'],cap+math.log(multiplier),rtol=0,atol=1e-6)
            self.assertTrue(torch.equal(m._scaling,scales))
            self.assertTrue(torch.equal(m.fixed_lidar_state['log_scale_target'],target))
            for key,value in moments.items():
                self.assertTrue(torch.equal(m.optimizer.state[m._scaling][key],value))
            for group in m.optimizer.param_groups:
                self.assertEqual(group['lr'],expected['lrs'].get(group['name'],0.))
        self.assertEqual(last,2.)
        self.assertAlmostEqual(values(policy(),4)['ceiling_multiplier'],1.5)
        self.assertEqual(stats(m)['lr_scaling'],.0001)
        self.assertAlmostEqual(audit(m,8)['effective_max_scale_world'],.03)

    def test_growth_is_learned_after_release_fixed_positions_and_export_current_bounds(self):
        m=self.scheduled_model()
        initial_cap=m.fixed_lidar_state['log_scale_cap'].clone()
        xyz=m._xyz.detach().clone()
        for iteration in range(1,6):
            self.step(m,iteration)
            self.assertTrue(audit(m,iteration)['scales_within_bounds'])
        self.assertTrue((m._scaling > initial_cap).any())
        self.assertTrue(torch.equal(m._xyz,xyz)); self.assertNotIn(m._xyz,m.optimizer.state)
        current=stats(m)['current_ceiling_multiplier']
        self.assertTrue(1 < current < 2)
        with patch.dict(m.save_ply.__globals__,{'PlyData':PlyData,'PlyElement':PlyElement}):
            m.save_ply(str(self.root/'scheduled.ply'))
        vertex=PlyData.read(self.root/'scheduled.ply')['vertex']
        exported=np.column_stack([vertex['scale_'+str(i)] for i in range(3)])
        np.testing.assert_array_equal(exported,m._scaling.detach().numpy())
        self.assertTrue(np.all(exported <= m.fixed_lidar_state['log_scale_cap'].numpy()))

    def test_each_scheduled_group_rate_controls_its_actual_adam_displacement(self):
        m=self.scheduled_model()
        m._features_rest=torch.nn.Parameter(torch.ones(len(m._xyz),3,3))
        with fixed.cpu_allocations(): m.training_setup(self.opt)
        m.update_learning_rate(4)
        before={g['name']:g['params'][0].detach().clone() for g in m.optimizer.param_groups}
        for group in m.optimizer.param_groups:
            if group['name']!='xyz':
                group['params'][0].grad=torch.ones_like(group['params'][0])
        m.optimizer.step(); project_scales(m)
        for group in m.optimizer.param_groups:
            actual=before[group['name']]-group['params'][0].detach()
            expected=torch.full_like(actual,group['lr'])
            torch.testing.assert_close(actual,expected,atol=3e-7,rtol=1e-3)

    def test_resume_mid_ramp_matches_uninterrupted_adam_exactly(self):
        m=self.scheduled_model()
        for iteration in range(1,5): self.step(m,iteration)
        buf=io.BytesIO(); torch.save(m.capture(),buf); buf.seek(0)
        saved=torch.load(buf,weights_only=True)
        other=self.model()
        with fixed.cpu_allocations(): other.restore(saved,self.opt)
        self.assertEqual(stats(m),stats(other))
        for iteration in range(5,9):
            self.step(m,iteration); self.step(other,iteration)
        for attr in ('_xyz','_scaling','_rotation','_opacity','_features_dc'):
            self.assertTrue(torch.equal(getattr(m,attr),getattr(other,attr)),attr)
        self.assertEqual(stats(m),stats(other))
        torch.testing.assert_close(m.optimizer.state[m._scaling]['exp_avg'],
                                   other.optimizer.state[other._scaling]['exp_avg'],atol=0,rtol=0)

    def test_resume_rejects_changed_policy_and_schedule_cannot_rewind(self):
        m=self.scheduled_model(); self.step(m,3)
        buf=io.BytesIO(); torch.save(m.capture(),buf); buf.seek(0)
        saved=torch.load(buf,weights_only=True)
        changed=policy(); changed['learning_rates']['groups']['scaling'][0]*=2
        self.opt.local_scale_schedule=json.dumps(changed)
        other=self.model()
        with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'schedule differs'):
            other.restore(saved,self.opt)
        with self.assertRaisesRegex(ValueError,'rewind'): m.update_learning_rate(2)

    def test_cap_only_and_lr_only_keep_the_other_policy_static(self):
        for mode in ('cap','lr'):
            m=self.scheduled_model(mode)
            before_cap=m.fixed_lidar_state['log_scale_cap'].clone()
            before_lrs=[g['lr'] for g in m.optimizer.param_groups]
            m.update_learning_rate(8)
            if mode=='cap':
                self.assertEqual(before_lrs,[g['lr'] for g in m.optimizer.param_groups])
                self.assertFalse(torch.equal(before_cap,m.fixed_lidar_state['log_scale_cap']))
            else:
                self.assertTrue(torch.equal(before_cap,m.fixed_lidar_state['log_scale_cap']))
                self.assertNotEqual(before_lrs,[g['lr'] for g in m.optimizer.param_groups])

    def test_legacy_constant_run_has_no_new_schedule_state(self):
        m=self.model()
        cap=m.fixed_lidar_state['log_scale_cap'].clone()
        lrs=[g['lr'] for g in m.optimizer.param_groups]
        m.update_learning_rate(25000)
        self.assertEqual(lrs,[g['lr'] for g in m.optimizer.param_groups])
        self.assertTrue(torch.equal(cap,m.fixed_lidar_state['log_scale_cap']))
        self.assertNotIn('log_scale_cap_final',m.fixed_lidar_state)
        self.assertEqual(stats(m),{})

    def test_invalid_policies_fail_before_training(self):
        for mutate in (lambda p:p['cap'].update(end=9),
                       lambda p:p['cap'].update(start=6),
                       lambda p:p['cap'].update(initial_multiplier=3),
                       lambda p:p['learning_rates']['groups'].update(scaling=[float('nan'),.001]),
                       lambda p:p['learning_rates']['groups'].update(scaling=[.001,.01]),
                       lambda p:p['learning_rates']['groups'].update(xyz=[1.,.1])):
            cfg=policy(); mutate(cfg)
            with self.assertRaises(ValueError): validate(cfg,2.,8)


class LocalSchedulePlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_stop_helper_recognizes_only_exact_named_local_scale_workers(self):
        from pathlib import Path
        argv=['python','-u','scripts/run_local_scale.py','watch','--model',str(self.model)]
        self.assertTrue(matches(argv,self.model,Path.cwd()))
        self.assertFalse(matches(argv,self.root/'another_run',Path.cwd()))

    def test_modes_use_distinct_fresh_plans_and_all_frame_evaluations(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        run_local_scale.prepare(self.argv+['--resolution','1'])
        for mode in ('cap','lr','joint'):
            target=self.root/mode
            argv=['--reference-model',str(self.model),'--model',str(target),'--gpu','0',
                  '--schedule-mode',mode,'--lr-scaling','.0008','--lr-scaling-final','.00008']
            tune_local_scale.prepare(argv); tune_local_scale.prepare(argv)
            data=periodic.load_plan(target)
            self.assertEqual(data['frames'],sorted(self.names))
            self.assertEqual(data['iterations'],[10000,20000,30000])
            self.assertEqual(data['training_args']['resolution'],1)
            cfg=data['plan']['training_schedule']
            self.assertEqual(cfg['cap'] is not None,mode in ('cap','joint'))
            self.assertEqual(cfg['learning_rates'] is not None,mode in ('lr','joint'))
            if cfg['learning_rates']:
                self.assertEqual(cfg['learning_rates']['groups']['scaling'],[.0008,.00008])
                self.assertEqual(cfg['learning_rates']['end'],30000)
            cmd=data['plan']['stages'][0]['argv']
            self.assertEqual(json.loads(cmd[cmd.index('--local_scale_schedule')+1]),cfg)
            self.assertNotIn('--distance_weighting',cmd)
        self.assertNotIn('training_schedule',periodic.load_plan(self.model)['plan'])


if __name__=='__main__':
    unittest.main()
