"""Matched cap relaxation, cached distances, expanded audits and pressure diagnostics."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial import cKDTree
from types import SimpleNamespace

import test_local_scale as local
import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
from utils.local_scale import prepare,support_counts,pressure_stats
from utils.fixed_lidar import project_scales,audit
from scripts import run_local_scale,tune_local_scale,run_soft_surface_long as periodic


class LocalScaleTuningTests(unittest.TestCase):
    setUp=local.LocalScaleTests.setUp
    prepare=local.LocalScaleTests.prepare
    model=local.LocalScaleTests.model

    def test_relaxing_two_and_two_half_caps_keeps_target_and_initialization(self):
        original=self.model()
        base_cap=original.fixed_lidar_state['log_scale_cap'].exp()
        base_target=original.fixed_lidar_state['log_scale_target']
        for multiplier in (2.,2.5):
            cache=self.root/f'cap{multiplier}'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(self.cloud,cache,workers=1,samples=25,ceiling_multiplier=multiplier,reuse_cache=self.cache)
            self.opt.local_scale_cache=str(cache)
            self.opt.local_scale_ceiling_multiplier=multiplier
            m=self.model()
            torch.testing.assert_close(m._scaling,original._scaling,rtol=0,atol=1e-6)
            torch.testing.assert_close(m.fixed_lidar_state['log_scale_cap'].exp(),base_cap*multiplier,rtol=1e-6,atol=0)
            torch.testing.assert_close(m.fixed_lidar_state['log_scale_target'],base_target,rtol=0,atol=0)
            with torch.no_grad():
                m._scaling.copy_(m.fixed_lidar_state['log_scale_cap']+.1)
            project_scales(m)
            self.assertTrue((m.get_scaling > base_cap).all())
            self.assertTrue(audit(m,1)['scales_within_bounds'])
            self.assertAlmostEqual(audit(m,1)['effective_max_scale_world'],.015*multiplier)

    def test_cached_reuse_skips_knn_queries_and_rebuilds_complete_larger_supports(self):
        # A larger patch leaves neighbors beyond the old broad-phase radius.
        self.xyz=np.array([[x*.012,y*.012,2] for x in range(9) for y in range(9)],np.float32)
        self.cloud=self.root/'larger_input.ply';self.cache=self.root/'larger_base'
        discrete.write_cloud(self.cloud,self.xyz)
        self.meta=self.prepare()
        tree=cKDTree(self.xyz)
        wrapped=SimpleNamespace(query_ball_point=tree.query_ball_point,
                                query=lambda *a,**k: (_ for _ in ()).throw(AssertionError('KNN recomputed')))
        cache=self.root/'larger'
        with patch('scipy.spatial.cKDTree',return_value=wrapped),contextlib.redirect_stdout(io.StringIO()):
            data=prepare(self.cloud,cache,workers=1,samples=25,ceiling_multiplier=2.5,reuse_cache=self.cache)
        self.assertGreater(data['audit_candidate_pairs'],self.meta['audit_candidate_pairs'])
        self.opt.local_scale_cache=str(cache);self.opt.local_scale_ceiling_multiplier=2.5
        m=self.model()
        with torch.no_grad():
            m._scaling.copy_(m.fixed_lidar_state['log_scale_cap'])
        ids=m.local_scale_audit['sample_ids']
        radius=3*(m.get_scaling[ids].amax(1)[:,None]+m.get_scaling.amax(1)[None])
        distance=torch.cdist(m._xyz[ids],m._xyz,compute_mode='donot_use_mm_for_euclid_dist')
        hit=distance<=radius;hit[torch.arange(len(ids)),ids]=False
        self.assertTrue(torch.equal(support_counts(m)[0],hit.sum(1)))

    def test_target_can_be_relaxed_separately_and_multiplier_resume_must_match(self):
        cache=self.root/'both'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.cloud,cache,workers=1,samples=25,ceiling_multiplier=2.,target_multiplier=1.5,reuse_cache=self.cache)
        self.opt.local_scale_cache=str(cache);self.opt.local_scale_ceiling_multiplier=2.;self.opt.local_scale_target_multiplier=1.5
        m=self.model()
        np.testing.assert_allclose(m.fixed_lidar_state['log_scale_target'].exp().numpy().ravel(),
                                   np.load(self.cache/'target.npy')*1.5,rtol=1e-6)
        restored=self.model()
        with fixed.cpu_allocations():
            restored.restore(m.capture(),self.opt)
        self.opt.local_scale_ceiling_multiplier=2.5
        with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'multiplier differs'):
            restored.restore(m.capture(),self.opt)

    def test_growth_pressure_excludes_untouched_points_and_checks_gradient_direction(self):
        m=self.model()
        with torch.no_grad():
            m._scaling.copy_(m.fixed_lidar_state['log_scale_cap'])
        # At their caps: row 0 wants growth, row 1 wants shrinkage, row 2 had no RGB gradient.
        sample=(torch.tensor([0,1,2]),torch.tensor([[-1.,0,0],[1.,1,1],[0.,0,0]]))
        stats=pressure_stats(m,sample)
        self.assertEqual(stats['sampled_rgb_active_gaussians'],2)
        self.assertEqual(stats['active_at_cap_fraction'],1.)
        self.assertEqual(stats['active_growth_at_cap_fraction'],.5)
        m._scaling.grad=torch.zeros_like(m._scaling)
        m._scaling.grad[:3]=sample[1]
        m._scaling.grad[0,0]=.1 # Regularizer cancels the outward RGB request.
        net=pressure_stats(m,sample)
        self.assertEqual(net['active_net_growth_at_cap_fraction'],0.)
        self.assertGreater(net['active_regularizer_to_rgb_scale_grad_ratio'],0.)
        empty=pressure_stats(m,(sample[0],torch.zeros_like(sample[1])))
        self.assertEqual(empty,dict(sampled_rgb_active_gaussians=0))


class LocalTuningPlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_matched_short_tune_uses_original_gpu3_inputs_and_fresh_gpu0_destination(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        run_local_scale.prepare(self.argv+['--resolution','1'])
        replacement=self.root/'tune'
        args=['--reference-model',str(self.model),'--model',str(replacement),'--gpu','0']
        tune_local_scale.prepare(args)
        tune_local_scale.prepare(args)
        data=periodic.load_plan(replacement)
        self.assertEqual(data['gpu'],'0')
        self.assertEqual(data['iterations'],[10000,20000,30000])
        self.assertEqual(data['frames'],sorted(self.names))
        self.assertEqual(data['training_args']['resolution'],1)
        self.assertEqual(data['plan']['effective_max_scale_world'],.03)
        self.assertEqual(data['plan']['ceiling_multiplier'],2.)
        self.assertEqual(data['plan']['target_multiplier'],1.)
        command=data['plan']['stages'][0]['argv']
        self.assertNotIn('--distance_weighting',command)
        self.assertEqual(command[command.index('--position_lr_init')+1],'0')
        with patch.object(run_local_scale,'execute_plan') as execute:
            periodic.train(replacement)
        execute.assert_called_once()
        self.assertEqual(periodic.load_plan(self.model)['gpu'],'3')

    def test_inspection_distinguishes_global_caps_from_active_growth_pressure(self):
        self.model.mkdir()
        (self.model/'fixed_lidar_training.jsonl').write_text(json.dumps(dict(iteration=10000,at_cap_fraction=.8))+'\n')
        (self.model/'scale_regularization_training.jsonl').write_text(json.dumps(dict(iteration=10000,
            active_growth_at_cap_fraction=.1,regularizer_to_rgb_scale_grad_ratio=.2))+'\n'+ '{incomplete')
        with contextlib.redirect_stdout(io.StringIO()):
            result=tune_local_scale.inspect(self.model)
        self.assertEqual(result['latest']['at_cap_fraction'],.8)
        self.assertEqual(result['active_growth_at_cap_fraction_recent_median'],.1)


if __name__=='__main__':
    unittest.main()
