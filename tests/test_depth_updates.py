"""Camera-z decay, masked Adam, adaptive-anchor resume and matched run plans."""
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest

import numpy as np
import torch

import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
import test_lidar_activation as activation_tests
from test_pose_culling import measured_mesh
from utils.distance_updates import policy,weights,adam_step,diagnostic_sample
from utils.occlusion_culling import masked_adam_step
from utils.fixed_lidar import project_scales
from utils import lidar_activation
from scripts import run_local_scale,tune_local_scale,run_soft_surface_long as periodic


def options(mode='relative'):
    opt=fixed.options()
    opt.distance_weighting=True
    opt.distance_metric='depth';opt.distance_decay='exponential'
    opt.distance_normalization=mode
    return opt


def compute(xyz,cfg,view=None,keep=None,**kwargs):
    if view is None:view=torch.eye(4,dtype=xyz.dtype,device=xyz.device)
    if keep is None:keep=torch.ones(len(xyz),device=xyz.device,dtype=torch.bool)
    return weights(xyz,torch.linalg.inv(view)[3,:3],cfg,world_view_transform=view,keep=keep,**kwargs)


class DepthWeightTests(unittest.TestCase):
    def test_absolute_exponential_matches_geometric_ratio_in_depth_not_range(self):
        xyz=torch.tensor([[0.,0,2],[20,0,2],[0,0,4],[0,0,6],[0,0,10],[0,0,20],[0,0,-1]])
        cfg=policy(options('absolute'))
        actual=compute(xyz,cfg,chunk_size=2)
        expected=torch.tensor([1,1,.5,.25,.0625,.05,0])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        depths=xyz[:-1,2]
        geometric=torch.pow(.5,((depths-2)/2).clamp_min(0)).clamp_min(.05)
        torch.testing.assert_close(actual[:-1],geometric,rtol=0,atol=0)
        self.assertFalse(actual.requires_grad)

    def test_relative_depth_gives_distant_views_the_same_weights(self):
        xyz=torch.tensor([[0.,0,2],[0,0,4],[0,0,6],[0,0,8]])
        opt=options();opt.distance_near_quantile=0.
        cfg=policy(opt);report={}
        near=compute(xyz,cfg);far=compute(xyz+torch.tensor([0.,0,10.]),cfg,diagnostics=report)
        torch.testing.assert_close(near,torch.tensor([1.,.5,.25,.125]),rtol=0,atol=0)
        torch.testing.assert_close(near,far,rtol=0,atol=0)
        self.assertEqual(report['reference_depth'],12.)
        # Hidden/reserved foreground cannot shift the per-view reference.
        mixed=torch.cat([torch.tensor([[0.,0,.3]]),xyz+torch.tensor([0.,0,10.])])
        keep=torch.tensor([False,True,True,True,True])
        torch.testing.assert_close(compute(mixed,cfg,keep=keep),torch.cat([torch.tensor([0.]),far]))

    def test_corrected_camera_rotation_and_scene_unit_rescaling(self):
        cfg=policy(options('absolute'))
        xyz=torch.tensor([[4.,0,2.],[8.,0,2.]])
        view=torch.eye(4);view[:3,:3]=torch.tensor([[0.,0,1.],[0,1,0],[-1,0,0.]])
        torch.testing.assert_close(compute(xyz,cfg,view),torch.tensor([.5,.125]))
        shifted=view.clone();shifted[3,2]=-2
        torch.testing.assert_close(compute(xyz,cfg,shifted),torch.tensor([1.,.25]))
        scaled={**cfg,'reference':20.,'half_life':20.}
        torch.testing.assert_close(compute(xyz*10,scaled,view),compute(xyz,cfg,view))

    def test_rank_is_invariant_to_depth_spacing_and_ties_get_equal_weight(self):
        cfg=policy(options('rank'))
        xyz=torch.zeros(5,3);xyz[:,2]=torch.tensor([10.,10,12,30,100])
        before=compute(xyz,cfg)
        xyz[:,2]=torch.tensor([1.,1,1.1,1.2,1.3])
        torch.testing.assert_close(compute(xyz,cfg),before,rtol=0,atol=0)
        self.assertEqual(float(before[0]),1.);self.assertEqual(float(before[1]),1.)
        self.assertTrue((before[1:]<=before[:-1]).all())
        xyz[:,2]=15.
        self.assertTrue(torch.equal(compute(xyz,cfg),torch.ones(5)))
        self.assertTrue(torch.equal(compute(xyz,cfg,keep=torch.zeros(5,dtype=torch.bool)),torch.zeros(5)))

    def test_bounded_sampling_is_deterministic_and_does_not_change_global_rng(self):
        xyz=torch.zeros(1000,3);xyz[:,2]=torch.linspace(10,20,1000)
        opt=options('rank');opt.distance_rank_samples=32;cfg=policy(opt)
        rng=torch.random.get_rng_state().clone();a={};b={}
        first=compute(xyz,cfg,diagnostics=a);second=compute(xyz,cfg,diagnostics=b,chunk_size=31)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first,second));self.assertEqual(a,b)
        self.assertEqual(a['reference_sample_count'],32)
        self.assertGreaterEqual(float(first.min()),.05-1e-7)
        self.assertLessEqual(float(first.max()),1.)

    def test_invalid_depth_policy_mask_and_weights_fail_before_mutation(self):
        for field,value in (('distance_metric','bad'),('distance_decay','bad'),('distance_half_life',0),
            ('distance_normalization','bad'),('distance_near_quantile',1.),('distance_far_quantile',.01),
            ('distance_rank_samples',0),('distance_rank_samples',32.5)):
            opt=options();setattr(opt,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):policy(opt)
        cfg=policy(options())
        with self.assertRaisesRegex(ValueError,'contribution mask'):
            weights(torch.ones(2,3),torch.zeros(3),cfg,world_view_transform=torch.eye(4))
        with self.assertRaisesRegex(ValueError,'corrected world-view'):
            weights(torch.ones(2,3),torch.zeros(3),cfg,keep=torch.ones(2,dtype=torch.bool))
        p=torch.nn.Parameter(torch.ones(2,3));p.grad=torch.ones_like(p);opt=torch.optim.Adam([p])
        for bad in (torch.tensor([1.,float('nan')]),torch.tensor([1.,-1.])):
            with self.assertRaises(ValueError):adam_step(opt,bad)
        self.assertFalse(opt.state);self.assertTrue(torch.equal(p,torch.ones(2,3)))

    def test_masked_weighted_adam_matches_weighted_moments_and_scaled_step_with_visibility_changes(self):
        p=torch.nn.Parameter(torch.zeros(5,3,dtype=torch.float64));q=torch.nn.Parameter(p.detach().clone())
        a=torch.optim.Adam([p],lr=.005,eps=1e-15);b=torch.optim.Adam([q],lr=.005,eps=1e-15)
        for i in range(8):
            w=torch.tensor([1.,.5,.25,.125,.05],dtype=p.dtype).roll(i)
            keep=torch.tensor([True,False,True,False,True]).roll(i)
            p.grad=torch.full_like(p,i+1.);q.grad=p.grad*w[:,None]
            p0=p.detach().clone();q0=q.detach().clone()
            prior=deepcopy(a.state[p])
            adam_step(a,w,chunk_size=2,keep=keep)
            masked_adam_step(b,keep,chunk_size=3)
            torch.testing.assert_close(p-p0,(q-q0)*w[:,None],atol=1e-16,rtol=1e-12)
            self.assertTrue(torch.equal(p[~keep],p0[~keep]))
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(a.state[p][key],b.state[q][key],atol=0,rtol=0)
                if prior and key!='step':self.assertTrue(torch.equal(a.state[p][key][~keep],prior[key][~keep]))

    def test_new_policy_is_saved_and_resume_rejects_changed_normalization(self):
        opt=options();m=fixed.model(opt);m._features_dc.sum().backward()
        adam_step(m.optimizer,torch.tensor([1.,.5]),keep=torch.tensor([True,False]))
        buf=io.BytesIO();torch.save(m.capture(),buf);buf.seek(0)
        other=fixed.model(opt)
        with fixed.cpu_allocations():other.restore(torch.load(buf,weights_only=True),opt)
        self.assertEqual(other.fixed_lidar_state['distance_update_policy'],policy(opt))
        bad=deepcopy(opt);bad.distance_normalization='rank'
        with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'distance update policy differs'):
            fixed.model(bad).restore(m.capture(),bad)


class AdaptiveDepthTests(activation_tests.ActivationTests):
    # Run all real-Adam reserve/growth/resume/PLY invariants with depth weighting.
    def setUp(self):
        super().setUp()
        self.opt.distance_weighting=True;self.opt.distance_metric='depth'
        self.opt.distance_decay='exponential';self.opt.distance_normalization='relative'

    def step(self,m,i):
        m.update_learning_rate(i);active=lidar_activation.active_mask(m)
        loss=-m.get_opacity.sum()-m._features_dc.sum()
        if m._xyz.requires_grad:loss=loss-m._xyz.sum()
        loss.backward()
        w=compute(m._xyz,policy(self.opt),keep=active)
        adam_step(m.optimizer,w,keep=active,chunk_size=4)
        project_scales(m);m.optimizer.zero_grad(set_to_none=True)


class DepthPlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_matched_cap15_plan_modes_keep_same_training_frames_and_pose_culling_settings(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        run_local_scale.prepare(self.argv)
        root=self.root/'geometry';root.mkdir();geometry=measured_mesh(root)
        from utils.lidar_occlusion import file_digest
        report=json.loads(geometry.read_text());report.update(source=str(self.source),point_cloud_sha256=file_digest(self.ply))
        geometry.write_text(json.dumps(report))
        base=['--reference-model',str(self.model),'--gpu','0','--ceiling-multiplier','15','--target-multiplier','3',
              '--schedule-mode','joint','--pose-refinement','--lidar-offsets','--pose-culling',
              '--scene-geometry',str(geometry),'--lidar-activation']
        baseline=self.root/'baseline';tune_local_scale.prepare(base+['--model',str(baseline)])
        original=periodic.load_plan(baseline)
        # An older prepared baseline may omit these new default fields.
        old=deepcopy(original)
        for key in ('metric','decay','half_life','normalization','near_quantile','far_quantile','rank_samples'):
            old['training_args'].pop('distance_'+key,None)
        (baseline/'periodic_evaluation/plan.json').write_text(json.dumps(old))
        tune_local_scale.prepare(base+['--model',str(baseline)])
        for mode in ('relative','rank'):
            target=self.root/mode;argv=base+['--model',str(target),'--depth-decay','--depth-mode',mode]
            tune_local_scale.prepare(argv);tune_local_scale.prepare(argv)
            data=periodic.load_plan(target);plan=data['plan']
            self.assertEqual(data['frames'],original['frames']);self.assertEqual(data['iterations'],[10000,20000,30000])
            for key in ('pose_refinement','pose_culling','lidar_offsets','lidar_activation','training_schedule','target_multiplier','ceiling_multiplier'):
                self.assertEqual(plan[key],original['plan'][key])
            self.assertEqual(plan['distance_updates']['normalization'],mode)
            self.assertEqual(plan['distance_updates']['metric'],'depth')
            command=plan['stages'][0]['argv']
            self.assertIn('--distance_weighting',command)
            self.assertEqual(command[command.index('--distance_normalization')+1],mode)
            self.assertEqual(command[command.index('--occlusion_mode')+1],'mesh')
            self.assertEqual(command[command.index('--occlusion_inference_mode')+1],'off')


if __name__=='__main__':unittest.main()
