"""Unconstrained native scales and exact, non-destructive culling restarts."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from plyfile import PlyData, PlyElement

import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
from test_pose_culling import measured_mesh
from scripts import run_fixed_lidar, run_local_scale, tune_local_scale, run_soft_surface_long as periodic
from scripts import restart_culling_previews as restart
from scripts.serve_training_previews import preview_directory
from utils.fixed_lidar import audit, project_scales, scale_regularization
from utils.training_previews import create_previewer, DEFAULTS


class UnconstrainedTests(unittest.TestCase):
    def test_native_initialization_adam_export_and_resume_are_never_capped(self):
        opt = fixed.options(.5, .0001)  # Deliberately tiny references must have NO effect in off mode.
        opt.scale_control = 'off'
        m = fixed.model(opt)
        torch.testing.assert_close(m.get_scaling, torch.tensor([[.01]*3, [.1]*3]))
        xyz = m.get_xyz.detach().clone()
        for _ in range(5):
            (-m._scaling.sum() - m._opacity.sum()).backward()
            m.optimizer.step(); m.optimizer.zero_grad(set_to_none=True)
        scales = m._scaling.detach().clone()
        moments = deepcopy(m.optimizer.state[m._scaling])
        project_scales(m, initializing=True)
        torch.testing.assert_close(m._scaling, scales, atol=0, rtol=0)
        for key, value in moments.items():
            torch.testing.assert_close(m.optimizer.state[m._scaling][key], value, atol=0, rtol=0)
        self.assertEqual(scale_regularization(m).item(), 0.)
        self.assertGreater(m.get_scaling.max().item(), .1)
        self.assertTrue(torch.equal(xyz, m.get_xyz))
        self.assertIsNone(m._xyz.grad)
        self.assertNotIn(m._xyz, m.optimizer.state)
        self.assertIsNone(audit(m, 5)['scales_within_bounds'])
        restored = fixed.model(opt)
        with fixed.cpu_allocations():
            restored.restore(deepcopy(m.capture()), opt)
        torch.testing.assert_close(restored.get_scaling, m.get_scaling, atol=0, rtol=0)
        bad = deepcopy(opt); bad.scale_control = 'clamp'
        with fixed.cpu_allocations(), self.assertRaisesRegex(ValueError, 'scale-control mode'):
            restored.restore(deepcopy(m.capture()), bad)
        with self.assertRaises(RuntimeError): m.reset_opacity()
        with self.assertRaises(RuntimeError): m.prune_points(torch.zeros(2, dtype=torch.bool))
        with self.assertRaises(RuntimeError): m.densification_postfix(*([None]*7))
        import tempfile
        with tempfile.TemporaryDirectory() as temp, patch.dict(m.save_ply.__globals__, {'PlyData':PlyData,'PlyElement':PlyElement}):
            path = Path(temp)/'point_cloud.ply'; m.save_ply(str(path)); rows = PlyData.read(path)['vertex'].data
            exported = np.stack([rows[f'scale_{i}'] for i in range(3)], axis=1)
            np.testing.assert_array_equal(exported, scales.numpy())


class PlanTests(unittest.TestCase):
    setUp = fixed.FixedLidarPlanTests.setUp

    def test_gpu2_plan_retains_native_lrs_and_all_frame_evaluation(self):
        run_fixed_lidar.prepare(self.argv + ['--gpu','2','--scale-control','off','--iterations','30000'])
        data = periodic.load_plan(self.model); command = data['plan']['stages'][0]['argv']
        self.assertEqual(data['gpu'], '2')
        self.assertEqual(data['iterations'], [10000,20000,30000])
        self.assertEqual(data['frames'], sorted(self.names))
        for key, value in (('--scale_control','off'), ('--densify_until_iter','0'),
                           ('--position_lr_init','0'), ('--debug_preview_every','500'), ('--occlusion_mode','off')):
            self.assertEqual(command[command.index(key)+1], value)
        for key in ('--local_scale_schedule','--pose_refinement','--lidar_activation','--distance_weighting',
                    '--scaling_lr','--opacity_lr','--rotation_lr','--feature_lr'):
            self.assertNotIn(key, command)
        self.assertFalse(data['plan']['opacity_resets'])
        (self.model/'fixed_lidar_training.jsonl').write_text(json.dumps(dict(iteration=1000,
            positions_unchanged=True, scales_finite=True, scales_within_bounds=None))+'\n')
        self.assertTrue(periodic.experiment_diagnostics(data,0,10000)['passed'])

    def culling_run(self):
        discrete.write_cloud(self.ply, np.array([[i*.01,0,2] for i in range(16)], np.float32))
        run_local_scale.prepare(self.argv)
        meshroot = self.root/'geometry'; meshroot.mkdir(); geometry = measured_mesh(meshroot)
        report = json.loads(geometry.read_text())
        report.update(source=str(self.source), point_cloud_sha256=restart.digest(self.ply))
        geometry.write_text(json.dumps(report))
        target = self.root/'gpu0'
        tune_local_scale.prepare(['--reference-model',str(self.model),'--model',str(target),'--gpu','0',
            '--ceiling-multiplier','15','--target-multiplier','3','--schedule-mode','joint','--pose-refinement',
            '--lidar-offsets','--fixed-lidar-positions','--lidar-activation','--depth-decay',
            '--pose-culling','--scene-geometry',str(geometry)])
        return periodic.load_plan(target)

    def test_restart_copies_exact_policies_and_reuses_caches_without_touching_original(self):
        data = self.culling_run(); original = deepcopy(data)
        with patch.object(restart.stopper, 'candidates', return_value=[{'pid':123}]):
            self.assertEqual(restart.choose_source(self.root,'0')['model'], data['model'])
        receipt = restart.prepare_restart(data, self.root/'restarted')
        new = periodic.load_plan(receipt['model'])
        for key in ('pose_refinement','lidar_offsets','lidar_activation','distance_updates','training_schedule',
                    'pose_culling','local_scale_cache','lidar_activation_cache','max_scale_world','scale_control'):
            self.assertEqual(new['plan'][key], data['plan'][key])
        self.assertEqual(new['frames'], data['frames'])
        self.assertEqual(new['iterations'], data['iterations'])
        self.assertEqual(new['training_args']['debug_preview_every'],500)
        self.assertEqual(periodic.load_plan(data['model']), original)
        with self.assertRaisesRegex(ValueError,'fresh output'):
            restart.prepare_restart(data, data['model'])
        with self.assertRaisesRegex(ValueError,'requested GPU'):
            restart.choose_source(self.root,'1', Path(data['model']))

    def test_ambiguous_or_changed_inputs_never_stop_any_process(self):
        data = self.culling_run()
        with patch.object(restart.stopper, 'candidates', return_value=[]), self.assertRaisesRegex(ValueError,'found 0'):
            restart.choose_source(self.root,'0')
        restart.prepare_restart(data,self.root/'another')
        with patch.object(restart.stopper,'candidates',return_value=[{'pid':123}]), self.assertRaisesRegex(ValueError,'found 2'):
            restart.choose_source(self.root,'0')
        self.ply.write_bytes(self.ply.read_bytes()+b'changed')
        with patch.object(restart.stopper,'main') as stop, self.assertRaisesRegex(ValueError,'input changed'):
            restart.main(['--model',data['model'],'--stop','--receipt',str(self.root/'receipt.json')])
        stop.assert_not_called()

    def test_matching_preview_variants_use_the_same_frames_with_distinct_pipeline_modes(self):
        cams = [SimpleNamespace(image_name='train.jpg', original_image=torch.full((3,4,4), .3))]
        test = [SimpleNamespace(image_name='test.jpg', original_image=torch.full((3,4,4), .3))]
        scene = SimpleNamespace(model_path=str(self.model),getTrainCameras=lambda:cams,getTestCameras=lambda:test)
        args = SimpleNamespace(source_path=str(self.source))
        off = create_previewer(scene,args,DEFAULTS)
        on = create_previewer(scene,args,DEFAULTS,subdirectory='debug_previews_training')
        self.assertEqual(off.manifest,on.manifest)
        modes=[]
        def render(c, model, pipe, *a, **kw):
            modes.append(pipe.occlusion_mode)
            return {'render':c.original_image*.9}
        for observer, mode in ((off,'off'),(on,'mesh')):
            snap=observer.capture(1,None,render,SimpleNamespace(occlusion_mode=mode),torch.zeros(3))
            self.assertEqual(snap['occlusion_mode'],mode)
        self.assertEqual(modes,['off','off','mesh','mesh'])
        self.assertEqual(preview_directory(model=self.model,variant='training'), on.root)


if __name__ == '__main__': unittest.main()
