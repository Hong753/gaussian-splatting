"""Local geometry, real Adam/serialization, complete support counts and GPU3 plan."""
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
from scipy.spatial import cKDTree
import torch

import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
from scripts import run_local_scale as runner, run_soft_surface_long as periodic
from utils.fixed_lidar import audit, project_scales, scale_regularization
from utils.local_scale import prepare, metadata, support_counts, projected_stats


class LocalScaleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.xyz = np.array([[x*.012, y*.012, 2] for x in range(5) for y in range(5)], np.float32)
        self.cloud, self.cache = self.root/'input.ply', self.root/'cache'
        discrete.write_cloud(self.cloud, self.xyz)
        self.meta = self.prepare()
        self.opt = fixed.options(1., .015)
        self.opt.scale_control = 'hybrid'
        self.opt.local_scale_cache = str(self.cache)

    def prepare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.cloud, self.cache, workers=1, samples=25, **kwargs)

    def model(self, opt=None):
        m = discrete.raw_model(self.xyz)
        with fixed.cpu_allocations():
            m.training_setup(opt or self.opt)
        return m

    def test_knn_bounds_use_other_rows_and_match_own_ten_center_support(self):
        cap = np.load(self.cache/'ceiling.npy')
        target = np.load(self.cache/'target.npy')
        distance = cKDTree(self.xyz).query(self.xyz, k=11)[0][:,1:]
        initial = np.sqrt(np.maximum(np.mean(distance[:,:3]**2, axis=1), 1e-7))
        np.testing.assert_allclose(cap, np.minimum(np.minimum(initial, distance[:,-1]/3), .015), rtol=1e-6)
        np.testing.assert_allclose(target, np.minimum(cap, distance[:,-1]/6), rtol=1e-6)
        self.assertTrue(np.all(target < cap))
        self.assertEqual(self.meta['audit_samples_omitted'], 0)
        before = (self.cache/'ceiling.npy').stat().st_mtime_ns
        self.assertEqual(self.prepare(), self.meta)
        self.assertEqual((self.cache/'ceiling.npy').stat().st_mtime_ns, before)

    def test_distant_rooms_do_not_enlarge_local_scales_and_scene_units_scale_covariantly(self):
        original = np.load(self.cache/'ceiling.npy')
        larger = self.root/'larger.ply'
        discrete.write_cloud(larger, np.concatenate([self.xyz, self.xyz+[100,100,0]]))
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(larger, self.root/'large_cache', workers=1, samples=10)
        np.testing.assert_array_equal(np.load(self.root/'large_cache/ceiling.npy')[:25], original)
        metric = self.root/'rescaled.ply'
        discrete.write_cloud(metric, self.xyz*10)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(metric, self.root/'rescaled_cache', workers=1, samples=10, world=.15)
        np.testing.assert_allclose(np.load(self.root/'rescaled_cache/ceiling.npy'), original*10, rtol=1e-6)

    def test_penalty_and_projection_allow_coverage_band_and_have_separate_gradient_roles(self):
        m = self.model(); state = m.fixed_lidar_state
        middle = (state['log_scale_target']+state['log_scale_cap'])/2
        with torch.no_grad():
            m._scaling.copy_(middle.expand_as(m._scaling))
        project_scales(m)
        self.assertTrue(torch.equal(m._scaling, middle.expand_as(m._scaling)))
        scale_regularization(m).backward()
        self.assertTrue((m._scaling.grad > 0).all())
        self.assertIsNone(m._xyz.grad); self.assertIsNone(m._opacity.grad)
        before = m._scaling.detach().clone()
        m.optimizer.step(); project_scales(m); m.optimizer.zero_grad(set_to_none=True)
        self.assertTrue((m._scaling < before).all())
        with torch.no_grad():
            m._scaling.copy_(state['log_scale_target']-.5)
        scale_regularization(m).backward()
        self.assertEqual(torch.count_nonzero(m._scaling.grad), 0)

    def test_real_adam_growth_pressure_freezes_xyz_and_projects_exported_axes(self):
        m = self.model(); original = m._xyz.detach().clone(); colors = m._features_dc.detach().clone()
        for iteration in range(1,31):
            self.assertEqual(m.update_learning_rate(iteration), 0)
            (-m._scaling.sum()+scale_regularization(m)+(m._features_dc-1).square().sum()).backward()
            m.optimizer.step(); project_scales(m); m.optimizer.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(m._xyz, original)); self.assertFalse(torch.equal(colors,m._features_dc))
        self.assertNotIn(m._xyz, m.optimizer.state)
        self.assertTrue(audit(m,30)['scales_within_bounds'])
        self.assertEqual(len(m._xyz), 25)
        self.assertTrue(torch.equal(m.optimizer.state[m._scaling]['exp_avg'], torch.zeros_like(m._scaling)))
        # Direction can reverse immediately after hitting the hard ceiling.
        previous = m._scaling.detach().clone()
        m._scaling.sum().backward(); m.optimizer.step(); project_scales(m)
        self.assertTrue((m._scaling < previous).all())
        with patch.dict(m.save_ply.__globals__, {'PlyData':PlyData, 'PlyElement':PlyElement}):
            m.save_ply(str(self.root/'point_cloud.ply'))
        vertex = PlyData.read(self.root/'point_cloud.ply')['vertex']
        logs = np.column_stack([vertex['scale_'+str(i)] for i in range(3)])
        np.testing.assert_array_equal(logs, m._scaling.detach().numpy())
        self.assertTrue(np.all(logs <= m.fixed_lidar_state['log_scale_cap'].numpy()))

    def test_checkpoint_roundtrip_keeps_bounds_moments_and_rejects_other_geometry(self):
        m = self.model()
        scale_regularization(m).backward(); m.optimizer.step(); project_scales(m)
        buffer = io.BytesIO(); torch.save(m.capture(),buffer); buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = self.model()
        with fixed.cpu_allocations():
            restored.restore(saved, self.opt)
        self.assertTrue(torch.equal(restored._scaling, m._scaling))
        for key in ('log_scale_cap','log_scale_target'):
            self.assertTrue(torch.equal(restored.fixed_lidar_state[key], m.fixed_lidar_state[key]))
        torch.testing.assert_close(restored.optimizer.state[restored._scaling]['exp_avg'],
                                   m.optimizer.state[m._scaling]['exp_avg'], rtol=0, atol=0)
        shifted = self.root/'shifted.ply'
        discrete.write_cloud(shifted, self.xyz+[0,0,1])
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(shifted,self.root/'shifted_cache',workers=1,samples=25)
        self.opt.local_scale_cache = str(self.root/'shifted_cache')
        with fixed.cpu_allocations(), self.assertRaisesRegex(ValueError,'Checkpoint local-scale cache differs'):
            restored.restore(saved,self.opt)
        with self.assertRaisesRegex(ValueError,'ordered initialization'):
            self.model()

    def test_cached_diagnostics_equal_all_pairs_for_anisotropic_rotated_varying_scales(self):
        m = self.model()
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            m._scaling.add_(torch.log(.2+.8*torch.rand(m._scaling.shape, generator=generator)))
            m._rotation.copy_(torch.nn.functional.normalize(torch.randn(m._rotation.shape, generator=generator),dim=1))
        samples = m.local_scale_audit['sample_ids']
        sphere, centers = support_counts(m,chunk=3)
        # Independent full matrix calculation in double precision.
        from scipy.spatial.transform import Rotation
        xyz = m._xyz.detach().double().numpy(); scales=m.get_scaling.detach().double().numpy()
        quat=m._rotation.detach().double().numpy()
        rotations=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
        for slot, i in enumerate(samples.tolist()):
            delta=xyz-xyz[i]
            sphere_hits=(np.linalg.norm(delta,axis=1) <= 3*(scales[i].max()+scales.max(1)))
            center_hits=(((delta @ rotations[i])/scales[i])**2).sum(1) <= 9
            sphere_hits[i]=False; center_hits[i]=False
            self.assertEqual(int(sphere[slot]),int(sphere_hits.sum()))
            self.assertEqual(int(centers[slot]),int(center_hits.sum()))
        self.assertLessEqual(audit(m,1)['support_sphere_fraction_over_k'],1.)

    def test_pair_budget_omits_whole_samples_never_partial_counts(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data=prepare(self.cloud,self.root/'budget',workers=1,samples=25,pair_budget=30)
        self.assertGreater(data['audit_samples_omitted'],0)
        self.opt.local_scale_cache=str(self.root/'budget')
        m=self.model()
        ids=m.local_scale_audit['sample_ids']; sphere,_=support_counts(m)
        dist=torch.cdist(m._xyz[ids],m._xyz,compute_mode='donot_use_mm_for_euclid_dist')
        radii=3*(m.get_scaling[ids].amax(1)[:,None]+m.get_scaling.amax(1)[None])
        hits=dist<=radii; hits[torch.arange(len(ids)),ids]=False
        self.assertTrue(torch.equal(sphere,hits.sum(1)))

    def test_cache_corruption_changed_policy_reordered_input_and_coincident_rows_fail(self):
        with self.assertRaisesRegex(ValueError,'inputs/settings differ'):
            self.prepare(neighbors=8)
        with self.assertRaisesRegex(ValueError,'ordered initialization'):
            bad=discrete.raw_model(self.xyz[::-1].copy())
            with fixed.cpu_allocations():
                bad.training_setup(self.opt)
        with (self.cache/'target.npy').open('ab') as stream:
            stream.write(b'bad')
        with self.assertRaisesRegex(ValueError,'cache changed'):
            metadata(self.cache)
        duplicate=self.root/'duplicate.ply'
        discrete.write_cloud(duplicate,np.tile(self.xyz[0],(11,1)))
        with self.assertRaisesRegex(ValueError,'zero K-th'):
            prepare(duplicate,self.root/'bad_cache',workers=1)

    def test_projected_radius_diagnostics_skip_frustum_culled_and_are_radii_not_diameters(self):
        stats=projected_stats(torch.tensor([0,0,2,4,8],dtype=torch.int32))
        self.assertEqual(stats['sampled_visible_radius_px_count'],3)
        self.assertEqual(stats['visible_radius_px_median'],4.)
        self.assertEqual(stats['visible_radius_px_max'],8.)
        self.assertEqual(projected_stats(torch.zeros(4,dtype=torch.int32)),dict(sampled_visible_radius_px_count=0))


class LocalScalePlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_gpu3_plan_dispatch_snapshot_schedule_and_all_frame_evaluation(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        runner.prepare(self.argv)
        data=periodic.load_plan(self.model)
        self.assertEqual(data['gpu'],'3'); self.assertEqual(data['experiment'],'local_scale')
        self.assertEqual(data['iterations'],list(range(10000,100001,10000)))
        self.assertEqual(data['frames'],sorted(self.names))
        self.assertEqual(data['plan']['initial_points'],16)
        command=data['plan']['stages'][0]['argv']
        self.assertEqual(command[command.index('--scale_control')+1],'hybrid')
        self.assertEqual(command[command.index('--max_scale_world')+1],'0.015')
        self.assertEqual(command[command.index('--position_lr_init')+1],'0')
        self.assertEqual(command[command.index('--densify_until_iter')+1],'0')
        with patch.object(runner,'execute_plan') as execute:
            periodic.train(self.model)
        self.assertEqual(execute.call_args[0][0].gpu,'3')
        self.assertEqual(execute.call_args[0][1],data['plan'])

    def test_hybrid_report_requires_hard_bounds_and_reports_regularizer(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        runner.prepare(self.argv)
        data=periodic.load_plan(self.model)
        row=dict(iteration=10000,positions_unchanged=True,scales_finite=True,scales_within_bounds=False)
        path=self.model/'fixed_lidar_training.jsonl'
        path.write_text(json.dumps(row)+'\n')
        self.assertFalse(periodic.experiment_diagnostics(data,0,10000)['passed'])
        row.update(scales_within_bounds=True,local_scale_audit_samples_omitted=1)
        path.write_text(json.dumps(row)+'\n')
        (self.model/'scale_regularization_training.jsonl').write_text(json.dumps(dict(iteration=10000,
            scale_regularization_loss=.003,regularizer_scale_grad_norm=.001))+'\n')
        report=periodic.experiment_diagnostics(data,0,10000)
        self.assertTrue(report['passed']); self.assertEqual(report['scale_control'],'hybrid')
        self.assertGreater(report['regularizer']['scale_regularization_loss']['max'],0)
        self.assertTrue(any('whole overlap-audit' in w for w in report['warnings']))

    def test_invalid_local_settings_fail_before_training(self):
        for flags in ([], ['--neighbors','2'], ['--overlap-factor','.5'], ['--scale-control','clamp']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                runner.build_plan(runner.parser().parse_args(self.argv+flags))


if __name__ == '__main__':
    unittest.main()
