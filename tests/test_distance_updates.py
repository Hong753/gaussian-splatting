"""Optimizer parity/attenuation, fixed-cloud resume and exact restart scope."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_fixed_lidar as fixed
from utils.distance_updates import weights,policy,adam_step,diagnostic_sample
from utils.fixed_lidar import project_scales,audit
from scripts import run_fixed_lidar,run_distance_weighted as runner,run_soft_surface_long as periodic
from scripts import stop_reconstruction_run as stop


def options():
    opt=fixed.options()
    opt.distance_weighting=True
    return opt


class DistanceUpdateTests(unittest.TestCase):
    def test_distance_falloff_is_metric_and_translation_invariant(self):
        xyz=torch.tensor([[0.,0,0],[2,0,0],[4,0,0],[8,0,0],[100,0,0]])
        config=policy(options())
        result=weights(xyz,torch.zeros(3),config,chunk_size=2)
        torch.testing.assert_close(result,torch.tensor([1.,1.,.25,.0625,.05]),rtol=0,atol=0)
        offset=torch.tensor([1.,3.,9.])
        self.assertTrue(torch.equal(result,weights(xyz+offset,offset,config)))
        indices,stats=diagnostic_sample(xyz,torch.zeros(3),result,torch.tensor([[1],[3]]))
        self.assertEqual(stats['sampled_visible_gaussians'],2)
        self.assertEqual(stats['distance_median'],5.)
        self.assertEqual(stats['weight_min'],.0625)

    def test_unit_weights_match_native_adam_parameters_and_moments(self):
        generator=torch.Generator().manual_seed(12)
        p=torch.nn.Parameter(torch.randn(19,3,5,generator=generator,dtype=torch.float64))
        q=torch.nn.Parameter(p.detach().clone())
        a=torch.optim.Adam([p],lr=.008,betas=(.8,.95),eps=1e-9,foreach=False)
        b=torch.optim.Adam([q],lr=.008,betas=(.8,.95),eps=1e-9,foreach=False)
        for step in range(15):
            grad=torch.randn(p.shape,generator=generator,dtype=p.dtype)
            if step%3==0:
                grad[2:8]=0
            p.grad=grad.clone();q.grad=grad.clone()
            adam_step(a,torch.ones(19,dtype=p.dtype),chunk_size=4);b.step()
            torch.testing.assert_close(p,q,atol=1e-13,rtol=1e-12)
            for key in ('exp_avg','exp_avg_sq','step'):
                torch.testing.assert_close(a.state[p][key],b.state[q][key],atol=0,rtol=0)

    def test_weights_reduce_actual_steps_and_feed_weighted_gradients_to_moments(self):
        p=torch.nn.Parameter(torch.zeros(4,3,dtype=torch.float64))
        q=torch.nn.Parameter(p.detach().clone())
        a=torch.optim.Adam([p],lr=.01,eps=1e-15,foreach=False)
        b=torch.optim.Adam([q],lr=.01,eps=1e-15,foreach=False)
        w=torch.tensor([1.,.25,.05,0.],dtype=p.dtype)
        for step in range(12):
            g=torch.full_like(p,1.+step)
            if step==7:
                g.zero_() # Residual Adam momentum must also be attenuated.
            p.grad=g.clone();q.grad=g*w[:,None]
            p0=p.detach().clone();q0=q.detach().clone()
            report=adam_step(a,w,chunk_size=2,sample_indices=torch.tensor([1]));b.step()
            torch.testing.assert_close(p-p0,(q-q0)*w[:,None],atol=1e-14,rtol=1e-12)
            torch.testing.assert_close(a.state[p]['exp_avg'],b.state[q]['exp_avg'],atol=0,rtol=0)
            self.assertAlmostEqual(report['group0_0_applied_to_proposal_ratio'],.25,places=12)
        self.assertTrue(torch.equal(p[3],torch.zeros(3,dtype=p.dtype)))

    def test_gradient_scaling_alone_would_cancel_but_explicit_step_weight_does_not(self):
        p=torch.nn.Parameter(torch.zeros(3,1));q=torch.nn.Parameter(torch.zeros(3,1))
        a=torch.optim.Adam([p],lr=.01,eps=1e-15);b=torch.optim.Adam([q],lr=.01,eps=1e-15)
        w=torch.tensor([1.,.25,.05])
        p.grad=torch.ones_like(p);q.grad=w[:,None].clone()
        adam_step(a,w);b.step()
        torch.testing.assert_close(q,torch.full_like(q,-.01))
        torch.testing.assert_close(p,-.01*w[:,None])

    def test_fixed_cloud_appearance_learns_while_positions_and_scale_bounds_survive(self):
        opt=options();m=fixed.model(opt)
        original=m._xyz.detach().clone();color=m._features_dc.detach().clone()
        for step in range(1,21):
            (-m._scaling.sum()+(m._features_dc-1).square().sum()).backward()
            w=weights(m._xyz,torch.zeros(3),policy(opt))
            adam_step(m.optimizer,w,chunk_size=1);project_scales(m);m.optimizer.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(m._xyz,original));self.assertNotIn(m._xyz,m.optimizer.state)
        self.assertFalse(torch.equal(m._features_dc,color))
        self.assertTrue(audit(m,20)['scales_within_bounds'])
        self.assertEqual(audit(m,20)['distance_update_policy'],policy(opt))
        with self.assertRaisesRegex(RuntimeError,'Pruning is disabled'):
            m.prune_points(torch.ones(2,dtype=torch.bool))

    def test_checkpoint_preserves_weighted_moments_and_rejects_policy_changes(self):
        opt=options();m=fixed.model(opt)
        m._features_dc.sum().backward()
        adam_step(m.optimizer,torch.tensor([1.,.25]));project_scales(m)
        buffer=io.BytesIO();torch.save(m.capture(),buffer);buffer.seek(0)
        saved=torch.load(buffer,weights_only=True)
        restored=fixed.model(opt)
        with fixed.cpu_allocations():
            restored.restore(saved,opt)
        torch.testing.assert_close(restored._features_dc,m._features_dc,rtol=0,atol=0)
        torch.testing.assert_close(restored.optimizer.state[restored._features_dc]['exp_avg'],
                                   m.optimizer.state[m._features_dc]['exp_avg'],rtol=0,atol=0)
        for changed in (fixed.options(),options()):
            changed.distance_reference=4.
            with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'distance update policy differs'):
                restored.restore(saved,changed)
        # Legacy unweighted fixed-cloud checkpoints cannot silently resume in the new optimizer.
        old=fixed.model().capture()
        with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'distance update policy differs'):
            restored.restore(old,opt)

    def test_invalid_config_and_optimizer_variants_fail_before_mutation(self):
        for field,value in (('distance_reference',0),('distance_power',-1),('distance_min_weight',0),
                            ('distance_min_weight',1.1),('distance_chunk_size',0),('fixed_lidar',False),
                            ('scale_control','regularize')):
            opt=options();setattr(opt,field,value)
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                policy(opt)
        p=torch.nn.Parameter(torch.ones(3,2));p.grad=torch.ones_like(p)
        opt=torch.optim.Adam([p],weight_decay=.01)
        with self.assertRaisesRegex(ValueError,'default Adam'):
            adam_step(opt,torch.ones(3))
        self.assertEqual(len(opt.state),0);self.assertTrue(torch.equal(p,torch.ones(3,2)))


class DistancePlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_restart_inherits_original_cap_resolution_frames_and_schedule(self):
        run_fixed_lidar.prepare(self.argv+['--max-scale-world','.015','--resolution','1','--iterations','60000'])
        replacement=self.root/'distance'
        runner.prepare(['--previous-model',str(self.model),'--model',str(replacement),'--gpu','0'])
        data=periodic.load_plan(replacement)
        self.assertEqual(data['experiment'],'distance_weighted')
        self.assertEqual(data['iterations'],list(range(10000,60001,10000)))
        self.assertEqual(data['frames'],sorted(self.names))
        self.assertEqual(data['plan']['max_scale_world'],.015)
        self.assertEqual(data['training_args']['resolution'],1)
        self.assertEqual(data['plan']['distance_updates']['reference'],2.)
        self.assertEqual(data['plan']['max_scale_ratio'],2.)
        with patch.object(runner,'execute_plan') as execute:
            periodic.train(replacement)
        execute.assert_called_once()
        self.assertEqual(execute.call_args[0][1],data['plan'])
        (replacement/'fixed_lidar_training.jsonl').write_text(json.dumps(dict(iteration=10000,
            positions_unchanged=True,scales_within_bounds=True,distance_update_policy=data['plan']['distance_updates']))+'\n')
        self.assertFalse(periodic.experiment_diagnostics(data,0,10000)['passed'])
        (replacement/'distance_updates_training.jsonl').write_text(json.dumps(dict(iteration=10000,weight_median=.25))+'\n')
        self.assertTrue(periodic.experiment_diagnostics(data,0,10000)['passed'])

    def test_wrong_gpu_and_changed_source_fail_before_replacement_is_prepared(self):
        run_fixed_lidar.prepare(self.argv)
        replacement=self.root/'distance'
        argv=['--previous-model',str(self.model),'--model',str(replacement)]
        with self.assertRaisesRegex(ValueError,'requested GPU'):
            runner.prepare(argv+['--gpu','1'])
        self.assertFalse(replacement.exists())
        self.ply.write_bytes(self.ply.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'input changed'):
            runner.prepare(argv)
        self.assertFalse(replacement.exists())

    def test_identical_preflight_retry_is_allowed_but_changed_settings_are_not(self):
        run_fixed_lidar.prepare(self.argv)
        replacement=self.root/'distance'
        argv=['--previous-model',str(self.model),'--model',str(replacement)]
        runner.prepare(argv)
        runner.prepare(argv)
        with self.assertRaisesRegex(ValueError,'different settings'):
            runner.prepare(argv+['--reference','4'])


class StopRunTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.model=self.root/'old'
        folder=self.model/'periodic_evaluation';folder.mkdir(parents=True)
        (folder/'plan.json').write_text(json.dumps(dict(model=str(self.model),gpu='0')))

    def test_matching_requires_python_entrypoint_and_exact_model_argument(self):
        self.assertTrue(stop.matches(['python','-u','scripts/run_fixed_lidar.py','train','--model',str(self.model)],self.model,self.root))
        self.assertTrue(stop.matches(['python3','scripts/run_fixed_lidar.py','watch','--model','old'],self.model,self.root))
        self.assertFalse(stop.matches(['python','scripts/run_fixed_lidar.py','train','--model',str(self.model)+'_other'],self.model,self.root))
        self.assertFalse(stop.matches(['python','other.py','--model',str(self.model)],self.model,self.root))
        self.assertFalse(stop.matches(['python','-c','run_fixed_lidar.py','--model',str(self.model)],self.model,self.root))

    def test_pidfd_signal_revalidates_identity_and_preserves_other_runs(self):
        row=dict(pid=101,start='555',argv=['python','run_fixed_lidar.py','train','--model',str(self.model)])
        with patch.object(stop,'candidates',side_effect=[[row],[row],[]]), \
                patch.object(stop.os,'pidfd_open',return_value=77),patch.object(stop.os,'close') as close, \
                patch.object(stop.signal,'pidfd_send_signal') as send,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stop.main(['--model',str(self.model),'--gpu','0','--stop']),0)
        send.assert_called_once_with(77,stop.signal.SIGTERM);close.assert_called_once_with(77)
        with patch.object(stop,'candidates',side_effect=[[row],[dict(row,start='999')],[]]), \
                patch.object(stop.os,'pidfd_open',return_value=77),patch.object(stop.os,'close'), \
                patch.object(stop.signal,'pidfd_send_signal') as send,contextlib.redirect_stdout(io.StringIO()):
            stop.main(['--model',str(self.model),'--stop'])
        send.assert_not_called()

    def test_gpu_mismatch_never_scans_or_signals(self):
        with patch.object(stop,'candidates') as scan,self.assertRaisesRegex(ValueError,'requested GPU/model'):
            stop.main(['--model',str(self.model),'--gpu','1','--stop'])
        scan.assert_not_called()

    def test_proc_inventory_excludes_other_models_users_and_zombies(self):
        proc=self.root/'proc';proc.mkdir()
        for pid,model,state in ((10,self.model,'S'),(11,self.root/'another','S'),(12,self.model,'Z')):
            directory=proc/str(pid);directory.mkdir()
            (directory/'cmdline').write_bytes(b'\0'.join(v.encode() for v in
                ['python3','-u','scripts/run_fixed_lidar.py','watch','--model',str(model)])+b'\0')
            fields=[state]+['0']*18+['12345']+['0']*3
            (directory/'stat').write_text(f'{pid} (python3) '+ ' '.join(fields))
            (directory/'cwd').symlink_to(self.root,target_is_directory=True)
        found=stop.candidates(self.model,proc=proc)
        self.assertEqual([row['pid'] for row in found],[10])
        self.assertEqual(found[0]['start'],'12345')
        self.assertEqual(stop.candidates(self.model,proc=proc,uid=stop.os.getuid()+1),[])


if __name__=='__main__':
    unittest.main()
