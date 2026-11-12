"""Actual Adam/PLY/checkpoint invariants for anchored XYZ plus alternating poses."""
import contextlib
import io
import json
from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np
from plyfile import PlyData,PlyElement
import torch

import test_local_scale as local
import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
from test_pose_refinement import camera,differentiable_image
from utils import lidar_offsets as offsets
from utils.fixed_lidar import audit,project_scales,scale_regularization
from utils.local_scale import prepare,support_counts
from utils.pose_refinement import PoseRefiner,DEFAULTS as POSES
from scripts import run_local_scale,tune_local_scale,run_soft_surface_long as periodic


class LidarOffsetTests(unittest.TestCase):
    prepare=local.LocalScaleTests.prepare

    def setUp(self):
        local.LocalScaleTests.setUp(self)
        self.policy={**offsets.DEFAULTS,'warmup':2,'ramp':2,'end':8}
        self.pose_policy={**POSES,'start':2,'end':6,'max_width':32}
        self.opt.iterations=8
        self.opt.pose_refinement=json.dumps(self.pose_policy)
        self.opt.lidar_offsets=json.dumps(self.policy)
        self.opt.local_scale_ceiling_multiplier=2.5
        padded=self.root/'padded'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.cloud,padded,workers=1,samples=25,ceiling_multiplier=2.5,
                    reuse_cache=self.cache,position_padding=.005)
        self.opt.local_scale_cache=str(padded)
        source=self.root/'source/sparse/0';source.mkdir(parents=True)
        for name in ('images','cameras'): (source/(name+'.txt')).write_text('# original calibration\n')
        self.cam=camera()

    def model(self):
        m=discrete.raw_model(self.xyz)
        # Vary initial spacing to verify the radius uses unclamped initialization.
        with torch.no_grad(): m._scaling[0].fill_(np.log(.008))
        with fixed.cpu_allocations(): m.training_setup(self.opt)
        m.pose_refiner=PoseRefiner([self.cam],self.pose_policy,self.root/'source',8)
        return m

    def step(self,m,i):
        lr=m.update_learning_rate(i)
        before=m._xyz.detach().clone()
        m.pose_refiner.update(self.cam,m,None,torch.zeros(3),i,renderer=differentiable_image)
        self.assertTrue(torch.equal(before,m._xyz))
        pose=[p.detach().clone() for p in m.pose_refiner.params[self.cam.image_name]]
        loss=(m._features_dc-1).square().sum()+scale_regularization(m)
        if lr: loss=loss-m._xyz.sum()+offsets.regularization(m)
        loss.backward();m.optimizer.step();project_scales(m);m.optimizer.zero_grad(set_to_none=True)
        for a,b in zip(pose,m.pose_refiner.params[self.cam.image_name]): self.assertTrue(torch.equal(a,b))

    def test_warmup_absolute_lr_original_radius_and_anchor_prior(self):
        m=self.model();s=offsets.state(m)
        torch.testing.assert_close(s['radius'][0],torch.tensor([.002]))
        torch.testing.assert_close(s['radius'][1:],torch.full((24,1),.005))
        for i in (1,2):
            self.step(m,i)
            self.assertFalse(m._xyz.requires_grad);self.assertIsNone(m._xyz.grad)
            self.assertNotIn(m._xyz,m.optimizer.state)
            self.assertTrue(audit(m,i)['positions_unchanged'])
        self.assertAlmostEqual(m.update_learning_rate(3),.5e-5)
        self.assertAlmostEqual(m.update_learning_rate(4),1e-5) # spatial_lr_scale=1000 has no effect
        self.step(m,4)
        self.assertFalse(audit(m,4)['positions_unchanged'])
        self.assertTrue(audit(m,4)['positions_within_bounds'])
        self.assertGreater(offsets.regularization(m).item(),0)
        offsets.regularization(m).backward()
        self.assertTrue(((m._xyz-s['anchors'])*m._xyz.grad>=0).all())
        self.assertAlmostEqual(m.update_learning_rate(8),1e-6)

    def test_projection_norm_bound_clears_only_projected_rows_and_allows_inward_motion(self):
        m=self.model();self.step(m,4);s=offsets.state(m)
        with torch.no_grad():
            m._xyz[0].copy_(s['anchors'][0]+torch.tensor([1.,2.,3.]))
        momentum=m.optimizer.state[m._xyz]
        keep={k:v[1:].clone() for k,v in momentum.items() if k!='step'}
        project_scales(m)
        self.assertLessEqual(float((m._xyz[0]-s['anchors'][0]).norm()),float(s['radius'][0]))
        for k,v in keep.items():
            self.assertTrue((momentum[k][0]==0).all());self.assertTrue(torch.equal(momentum[k][1:],v))
        before=(m._xyz[0]-s['anchors'][0]).norm().item()
        offsets.regularization(m).backward();m.optimizer.step();project_scales(m)
        self.assertLess((m._xyz[0]-s['anchors'][0]).norm().item(),before)
        # FP32 large coordinates must never relax the physical bound.
        for magnitude in (100.,10000.):
            with torch.no_grad():
                s['anchors'].fill_(magnitude);m._xyz.copy_(s['anchors']+10)
            offsets.project(m,chunk=3)
            self.assertTrue(((m._xyz-s['anchors']).norm(dim=1)<=s['radius'][:,0]).all())

    def test_serialized_resume_keeps_xyz_moments_anchors_pose_state_and_next_steps(self):
        m=self.model()
        for i in range(1,5): self.step(m,i)
        buf=io.BytesIO();torch.save(m.capture(),buf);buf.seek(0);saved=torch.load(buf,weights_only=True)
        other=self.model()
        with fixed.cpu_allocations(): other.restore(saved,self.opt)
        other.pose_refiner=PoseRefiner([self.cam],self.pose_policy,self.root/'source',8,saved=other.pose_refinement_state)
        for k in ('exp_avg','exp_avg_sq'):
            self.assertTrue(torch.equal(m.optimizer.state[m._xyz][k],other.optimizer.state[other._xyz][k]))
        for i in range(5,9): self.step(m,i);self.step(other,i)
        for attr in ('_xyz','_scaling','_features_dc'):
            self.assertTrue(torch.equal(getattr(m,attr),getattr(other,attr)),attr)
        self.assertTrue(torch.equal(offsets.state(m)['anchors'],offsets.state(other)['anchors']))
        for a,b in zip(m.pose_refiner.params[self.cam.image_name],other.pose_refiner.params[self.cam.image_name]):
            self.assertTrue(torch.equal(a,b))
        with patch.dict(m.save_ply.__globals__,{'PlyData':PlyData,'PlyElement':PlyElement}):
            m.save_ply(str(self.root/'bounded.ply'))
        vertex=PlyData.read(self.root/'bounded.ply')['vertex']
        exported=np.column_stack([vertex[k] for k in ('x','y','z')])
        np.testing.assert_array_equal(exported,m._xyz.detach().numpy())
        self.assertTrue(audit(m,8)['positions_within_bounds'])

    def test_changed_policy_anchor_radius_and_missing_padding_are_rejected(self):
        m=self.model();self.step(m,4)
        for key,value in (('max_world',.006),('sigma_ratio',.3),('lr_init',2e-5)):
            saved=deepcopy(m.capture());bad=deepcopy(self.opt)
            bad.lidar_offsets=json.dumps({**self.policy,key:value})
            with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'offset policy differs'):
                self.model().restore(saved,bad)
        for key in ('anchors','radius'):
            saved=deepcopy(m.capture());saved[14]['position_offsets'][key][0]+=1
            with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'Invalid original'):
                self.model().restore(saved,self.opt)
        saved=deepcopy(m.capture())
        with torch.no_grad(): saved[1][0]+=1
        with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'Saved LiDAR positions'):
            self.model().restore(saved,self.opt)
        with self.assertRaisesRegex(ValueError,'rewind'):m.update_learning_rate(3)
        m.local_scale_audit['metadata']['config']['position_padding']=0
        with self.assertRaisesRegex(ValueError,'padding'):offsets.bind(m,self.opt)

    def test_overlap_audit_includes_pairs_that_can_approach_after_initialization(self):
        # Small caps with spacing just beyond 2*3*sigma require motion padding.
        cloud=self.root/'line.ply';xyz=np.array([[i*.0125,0,2] for i in range(16)],np.float32)
        discrete.write_cloud(cloud,xyz)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(cloud,self.root/'line_cache',world=.002,position_padding=.005,samples=16,workers=1)
        opt=deepcopy(self.opt);opt.max_scale_world=.002;opt.local_scale_ceiling_multiplier=1.
        opt.local_scale_cache=str(self.root/'line_cache')
        m=discrete.raw_model(xyz)
        with fixed.cpu_allocations():m.training_setup(opt)
        m.update_learning_rate(4)
        initial=support_counts(m)[0]
        with torch.no_grad():
            m._xyz[::2,0]+=.004;m._xyz[1::2,0]-=.004
        project_scales(m)
        ids=m.local_scale_audit['sample_ids'];count=support_counts(m)[0]
        distance=torch.cdist(m._xyz[ids],m._xyz,compute_mode='donot_use_mm_for_euclid_dist')
        radii=3*(m.get_scaling[ids].amax(1)[:,None]+m.get_scaling.amax(1)[None])
        hits=distance<=radii;hits[torch.arange(len(ids)),ids]=False
        self.assertTrue(torch.equal(count,hits.sum(1)))
        self.assertGreater(int(count.sum()),int(initial.sum()))


class LidarOffsetPlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_gpu2_matched_pose_offsets_and_all_frame_schedule(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        run_local_scale.prepare(self.argv)
        target=self.root/'gpu2'
        argv=['--reference-model',str(self.model),'--model',str(target),'--gpu','2',
              '--ceiling-multiplier','2.5','--schedule-mode','joint','--pose-refinement','--lidar-offsets']
        tune_local_scale.prepare(argv);tune_local_scale.prepare(argv)
        data=periodic.load_plan(target);plan=data['plan']
        self.assertEqual(data['gpu'],'2');self.assertEqual(data['frames'],sorted(self.names))
        self.assertEqual(data['iterations'],[10000,20000,30000])
        self.assertEqual(plan['lidar_offsets'],offsets.DEFAULTS)
        self.assertEqual(plan['effective_max_scale_world'],.0375)
        cmd=plan['stages'][0]['argv']
        self.assertEqual(json.loads(cmd[cmd.index('--lidar_offsets')+1]),offsets.DEFAULTS)
        self.assertEqual(cmd[cmd.index('--densify_until_iter')+1],'0')
        row=dict(iteration=10000,positions_unchanged=False,positions_within_bounds=True,anchors_unchanged=True,
                 scales_within_bounds=True,lidar_offset_policy=offsets.DEFAULTS)
        log=target/'fixed_lidar_training.jsonl';log.write_text(json.dumps(row)+'\n')
        self.assertTrue(periodic.experiment_diagnostics(data,0,10000)['passed'])
        row['positions_within_bounds']=False;log.write_text(json.dumps(row)+'\n')
        self.assertFalse(periodic.experiment_diagnostics(data,0,10000)['passed'])


if __name__=='__main__':unittest.main()
