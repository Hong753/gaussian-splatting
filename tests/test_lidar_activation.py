"""Reserved-anchor activation, real Adam/checkpoints/PLY and visibility checks."""
import contextlib
import io
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch
from plyfile import PlyData,PlyElement

import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
from test_lidar_offsets import LidarOffsetTests
from test_pose_culling import measured_mesh,cam
from utils import lidar_activation as activation
from utils.discrete_lidar_cache import prepare
from utils.pose_culling import DEFAULTS as CULLING,PoseMeshDepthCache
from utils.occlusion_culling import masked_adam_step
from utils.fixed_lidar import project_scales,scale_regularization,audit
from utils.lidar_offsets import regularization
from utils.local_scale import support_counts
from scripts import run_local_scale,tune_local_scale,run_soft_surface_long as periodic


class ActivationTests(unittest.TestCase):
    prepare=LidarOffsetTests.prepare

    def setUp(self):
        LidarOffsetTests.setUp(self)
        self.policy={**activation.DEFAULTS,'start':2,'end':7,'interval':1,'min_observations':1,'max_new':3}
        self.opt.pose_culling=json.dumps(CULLING)
        self.opt.lidar_activation=json.dumps(self.policy)
        self.opt.lidar_activation_cache=str(self.root/'pool')
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.cloud,self.root/'pool',workers=1,
                    **{k:self.policy[k] for k in ('fraction','voxel','neighbors','seed')})
        mesh=self.root/'geometry';mesh.mkdir()
        self.pipe=SimpleNamespace(pose_culling=CULLING,occlusion_mode='mesh',
            _occlusion_provider=PoseMeshDepthCache(measured_mesh(mesh),self.root/'depth',CULLING))
        self.camera=cam()

    def model(self):
        return LidarOffsetTests.model(self)

    def step(self,m,i):
        m.update_learning_rate(i)
        active=activation.active_mask(m)
        loss=-m.get_opacity.sum()-m._features_dc.sum()+scale_regularization(m)
        if m._xyz.requires_grad:loss=loss-m._xyz.sum()+regularization(m)
        loss.backward()
        masked_adam_step(m.optimizer,active,chunk_size=4)
        project_scales(m);m.optimizer.zero_grad(set_to_none=True)

    def grow(self,m,i,visible=True):
        m.xyz_gradient_accum.fill_(1.);m.denom.fill_(2.)
        keep=activation.active_mask(m).clone()
        if visible:
            return activation.adapt(m,self.camera,self.pipe,i,keep)
        # A genuinely farther-than-wall candidate set must fail validation.
        with patch('utils.ray_occlusion.pixel_cutoff',return_value=torch.ones(64,64)):
            return activation.adapt(m,self.camera,self.pipe,i,keep)

    def test_inactive_rows_have_zero_opacity_gradients_and_frozen_adam_state(self):
        m=self.model();active=activation.active_mask(m).clone();inactive=~active
        original={name:getattr(m,name).detach().clone() for name in ('_xyz','_scaling','_opacity','_features_dc')}
        self.assertTrue((m.get_opacity[inactive]==0).all())
        m.get_opacity.sum().backward()
        self.assertTrue((m._opacity.grad[inactive]==0).all());m.optimizer.zero_grad(set_to_none=True)
        for i in range(1,6):self.step(m,i)
        for name,value in original.items():self.assertTrue(torch.equal(getattr(m,name)[inactive],value[inactive]),name)
        for group in m.optimizer.param_groups:
            parameter=group['params'][0]
            for key in ('exp_avg','exp_avg_sq'):
                if key in m.optimizer.state.get(parameter,{}):
                    self.assertFalse(bool(m.optimizer.state[parameter][key][inactive].any()))
        self.assertTrue(audit(m,5)['inactive_positions_unchanged'])

    def test_growth_uses_unique_untouched_rows_and_preserves_existing_parameters_and_momenta(self):
        m=self.model();self.step(m,3)
        old=activation.active_mask(m).clone()
        values=[p.detach().clone() for group in m.optimizer.param_groups for p in group['params']]
        moments=deepcopy(m.optimizer.state_dict())
        event=self.grow(m,3);self.assertGreater(event['added'],0);self.assertLessEqual(event['added'],3)
        new=activation.active_mask(m)&~old
        self.assertTrue(torch.equal(m._xyz[new],m.fixed_lidar_state['position_offsets']['anchors'][new]))
        for before,group in zip(values,m.optimizer.param_groups):self.assertTrue(torch.equal(before,group['params'][0]))
        for key,state in moments['state'].items():
            for name,value in state.items():self.assertTrue(torch.equal(value,m.optimizer.state_dict()['state'][key][name]))
        before=m._features_dc.detach().clone();self.step(m,4)
        self.assertTrue((m._features_dc[new]!=before[new]).any())
        self.assertTrue(audit(m,4)['unique_anchors'])

    def test_wall_blocks_growth_and_frustum_and_unknown_depth_are_separate(self):
        m=self.model();before=activation.active_mask(m).clone()
        self.assertEqual(self.grow(m,3,visible=False)['added'],0)
        self.assertTrue(torch.equal(before,activation.active_mask(m)))
        anchors=torch.tensor([[0.,0.,2.],[0.,0.,3.],[50.,0.,2.],[0.,0.,-.1]])
        dummy=SimpleNamespace(fixed_lidar_state={'position_offsets':{'anchors':anchors}},get_xyz=anchors)
        ids=torch.arange(4)
        self.assertEqual(activation.visible_candidates(dummy,ids,self.camera,self.pipe).tolist(),[True,False,False,False])
        with patch('utils.ray_occlusion.pixel_cutoff',return_value=torch.full((64,64),float('inf'))):
            self.assertFalse(bool(activation.visible_candidates(dummy,ids,self.camera,self.pipe).any()))

    def test_resume_preserves_reserve_and_next_growth_and_export_contains_only_active(self):
        m=self.model();self.step(m,3);self.grow(m,3)
        buffer=io.BytesIO();torch.save(m.capture(),buffer);buffer.seek(0)
        other=fixed.load_model()(0);other._exposure=torch.nn.Parameter(m._exposure.detach().clone())
        with fixed.cpu_allocations():other.restore(torch.load(buffer,weights_only=True),self.opt)
        self.assertTrue(torch.equal(activation.active_mask(m),activation.active_mask(other)))
        self.assertEqual(self.grow(m,4),self.grow(other,4))
        self.step(m,5);self.step(other,5)
        for attr in ('_xyz','_scaling','_opacity','_features_dc'):
            self.assertTrue(torch.equal(getattr(m,attr),getattr(other,attr)))
        with patch.dict(m.save_ply.__globals__,{'PlyData':PlyData,'PlyElement':PlyElement}):
            m.save_ply(str(self.root/'active.ply'))
        vertex=PlyData.read(self.root/'active.ply')['vertex'];selected=activation.active_mask(m)
        self.assertEqual(len(vertex),int(selected.sum()))
        np.testing.assert_array_equal(np.column_stack([vertex[k] for k in ('x','y','z')]),m._xyz[selected].detach().numpy())
        np.testing.assert_array_equal(np.column_stack([vertex[f'scale_{i}'] for i in range(3)]),m._scaling[selected].detach().numpy())
        self.assertTrue(audit(other,5)['positions_within_bounds'])

    def test_priors_and_overlap_counts_exclude_inactive_reserve(self):
        m=self.model();active=activation.active_mask(m);m.update_learning_rate(3)
        with torch.no_grad():m._scaling.fill_(0.)
        excess=torch.relu(m._scaling-m.fixed_lidar_state['log_scale_target'])
        torch.testing.assert_close(scale_regularization(m),.01*excess[active].square().mean())
        scale_regularization(m).backward();self.assertFalse(bool(m._scaling.grad[~active].any()))
        project_scales(m)
        ids=m.local_scale_audit['sample_ids'];distance=torch.cdist(m._xyz[ids],m._xyz,compute_mode='donot_use_mm_for_euclid_dist')
        hit=distance<=3*(m.get_scaling[ids].amax(1)[:,None]+m.get_scaling.amax(1)[None])
        hit &= active[ids,None]&active[None];hit[torch.arange(len(ids)),ids]=False
        self.assertTrue(torch.equal(support_counts(m)[0],hit.sum(1)))

    def test_changed_resume_policy_or_moved_reserve_is_rejected(self):
        m=self.model();bad=deepcopy(self.opt)
        bad.lidar_activation=json.dumps({**self.policy,'max_new':4})
        with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'activation policy'):
            discrete.raw_model(self.xyz).restore(deepcopy(m.capture()),bad)
        m.update_learning_rate(3)
        with torch.no_grad():m._xyz[~activation.active_mask(m)]+=.0001
        with self.assertRaisesRegex(RuntimeError,'Inactive LiDAR anchor moved'):activation.audit(m,3)


class ActivationPlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_gpu0_cap15_target3_offsets_poses_culling_growth_and_periodic_all_frames(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        run_local_scale.prepare(self.argv)
        root=self.root/'geometry';root.mkdir();geometry=measured_mesh(root)
        from utils.lidar_occlusion import file_digest
        report=json.loads(geometry.read_text());report.update(source=str(self.source),point_cloud_sha256=file_digest(self.ply))
        geometry.write_text(json.dumps(report));target=self.root/'gpu0'
        argv=['--reference-model',str(self.model),'--model',str(target),'--gpu','0',
            '--ceiling-multiplier','15','--target-multiplier','3','--schedule-mode','joint',
            '--pose-refinement','--lidar-offsets','--pose-culling','--scene-geometry',str(geometry),'--lidar-activation']
        tune_local_scale.prepare(argv);tune_local_scale.prepare(argv)
        saved=periodic.load_plan(target);plan=saved['plan'];command=plan['stages'][0]['argv']
        self.assertEqual(saved['gpu'],'0');self.assertEqual(saved['frames'],sorted(self.names))
        self.assertEqual(saved['iterations'],[10000,20000,30000])
        self.assertAlmostEqual(plan['effective_max_scale_world'],.225)
        self.assertEqual(plan['target_multiplier'],3.)
        self.assertEqual(plan['training_schedule']['cap']['final_multiplier'],15.)
        self.assertEqual(plan['lidar_activation'],activation.DEFAULTS)
        self.assertEqual(plan['pose_refinement']['every'],1)
        self.assertEqual(command[command.index('--occlusion_inference_mode')+1],'off')
        self.assertEqual(json.loads(command[command.index('--lidar_activation')+1]),activation.DEFAULTS)


if __name__=='__main__':unittest.main()
