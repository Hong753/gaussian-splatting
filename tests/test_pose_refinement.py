"""Pose geometry, native-optimizer transactions, immutable source and snapshots."""
from copy import deepcopy
import io
import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.pose_refinement import (DEFAULTS,PoseRefiner,rotation_quaternion,rotation_matrix,quaternion_product,
    corrected_transform,camera_with_transform,source_fingerprints)
from utils.checkpoint_snapshots import atomic_json,publish_snapshot,read_snapshot
from scripts import pose_options,run_local_scale,tune_local_scale,run_soft_surface_long as periodic
from test_view_comparisons import dataset
import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
import test_periodic_soft_surface as periodic_tests


def camera(name='frame.jpg', center=(.1,-.1,0.), angles=(.1,-.2,.05)):
    R=rotation_matrix(rotation_quaternion(torch.tensor(angles,dtype=torch.float32)))
    C=torch.tensor(center,dtype=torch.float32)
    w2c=torch.eye(4);w2c[:3,:3]=R;w2c[:3,3]=-R@C
    return SimpleNamespace(image_name=name,world_view_transform=w2c.T.contiguous(),projection_matrix=torch.eye(4),
        full_proj_transform=w2c.T.contiguous(),camera_center=C,original_image=torch.full((3,32,32),.4),
        alpha_mask=torch.ones(1,32,32),image_height=32,image_width=32,FoVx=1.,FoVy=1.)


def differentiable_image(cam,gaussians,pipe,bg,t,r,quaternion=None):
    return (t.sum()+r.sum()*2+.2).expand(3,cam.image_height,cam.image_width)


class PoseTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.source,self.manifest,self.names,self.selected=dataset(self.root)
        self.cameras=[camera(name) for name in self.selected]
        self.cfg={**DEFAULTS,'start':0,'end':12,'max_width':32,'prior_weight':1e-7}
        self.refiner=PoseRefiner(self.cameras,self.cfg,self.source,12)
        self.pipe=SimpleNamespace(antialiasing=True)
        self.bg=torch.zeros(3)

    def step(self,refiner=None,iteration=1,cam=None,renderer=differentiable_image):
        return (refiner or self.refiner).update(cam or self.cameras[0],None,self.pipe,self.bg,iteration,renderer=renderer)

    def test_zero_rotation_has_finite_correct_jacobian(self):
        vector=torch.zeros(3,dtype=torch.float64,requires_grad=True)
        matrix=rotation_matrix(rotation_quaternion(vector))
        torch.testing.assert_close(matrix,torch.eye(3,dtype=torch.float64),rtol=0,atol=0)
        self.assertTrue(torch.autograd.gradcheck(lambda v:rotation_matrix(rotation_quaternion(v)),(vector,),eps=1e-6))

    def test_pose_changes_mean_covariance_and_world_view_direction_consistently(self):
        cam=self.cameras[0]
        t=torch.tensor([.01,-.02,.005]);r=torch.tensor([.003,-.002,.001])
        w2c,center=corrected_transform(cam,t,r)
        original=cam.world_view_transform.T
        expected_R=rotation_matrix(rotation_quaternion(r))@original[:3,:3]
        torch.testing.assert_close(center,cam.camera_center+original[:3,:3].T@t)
        xyz=torch.tensor([[.3,.2,2.],[-.4,.7,3.]])
        via_world=(torch.cat((xyz,torch.ones(2,1)),1)@w2c.T)[:,:3]
        torch.testing.assert_close(via_world,(xyz-center)@expected_R.T)
        from utils.pose_refinement import base_quaternion
        q=quaternion_product(rotation_quaternion(r),base_quaternion(cam))
        torch.testing.assert_close(rotation_matrix(q),expected_R,atol=1e-6,rtol=1e-6)
        qg=rotation_quaternion(torch.tensor([.1,.3,-.2]));S=torch.diag(torch.tensor([.01,.03,.05]).square())
        Rg=rotation_matrix(qg);new=rotation_matrix(quaternion_product(q,qg))
        torch.testing.assert_close(new@S@new.T,expected_R@(Rg@S@Rg.T)@expected_R.T,atol=1e-8,rtol=1e-5)
        posed=camera_with_transform(cam,w2c)
        self.assertIs(posed.original_image,cam.original_image)
        self.assertFalse(torch.equal(posed.world_view_transform,cam.world_view_transform))

    def test_updates_are_bounded_train_only_and_source_cameras_stay_unchanged(self):
        fingerprints=source_fingerprints(self.source)
        original=self.cameras[0].world_view_transform.clone()
        other=self.refiner.params[self.cameras[1].image_name]
        for step in range(1,13):
            row=self.step(iteration=step)
            self.assertLessEqual(row['translation_norm'],.03+1e-7)
            self.assertLessEqual(row['rotation_degrees'],.5+1e-6)
        self.assertGreater(self.refiner.counts[self.cameras[0].image_name]['accepted'],0)
        self.assertTrue(all(torch.count_nonzero(p)==0 for p in other))
        self.assertNotIn(self.cameras[1].image_name,self.refiner.optimizers)
        self.assertTrue(torch.equal(original,self.cameras[0].world_view_transform))
        self.assertEqual(fingerprints,source_fingerprints(self.source))
        held=camera('not-a-training-view.jpg')
        self.assertIs(self.refiner.camera(held),held)
        fresh=PoseRefiner(self.cameras,self.cfg,self.source,12)
        with self.assertRaisesRegex(ValueError,'Only training'): self.step(fresh,cam=held)

    def test_worsening_proposal_restores_offsets_and_adam_state(self):
        # Forward is physically t/r; backward deliberately points the wrong way.
        def wrong(cam,g,p,b,t,r,quaternion=None):
            value=t.sum()+2*r.sum()
            surrogate=2*value.detach()-value
            return (surrogate+.2).expand(3,32,32)
        row=self.step(renderer=wrong)
        self.assertFalse(row['accepted'])
        name=self.cameras[0].image_name
        self.assertTrue(all(torch.count_nonzero(p)==0 for p in self.refiner.params[name]))
        self.assertEqual(len(self.refiner.optimizer(name).state),0)

    def test_projection_clamps_vector_norm_and_clears_only_selected_pose_moments(self):
        self.step()
        name=self.cameras[0].image_name
        with torch.no_grad():
            for param in self.refiner.params[name]: param.fill_(1.)
        self.assertTrue(self.refiner.project(name))
        for param,limit in zip(self.refiner.params[name],(.03,math.radians(.5))):
            self.assertAlmostEqual(float(param.norm()),limit,places=7)
            self.assertTrue((self.refiner.optimizer(name).state[param]['exp_avg']==0).all())

    def test_serialized_resume_matches_uninterrupted_pose_optimizer(self):
        for i in range(1,4): self.step(iteration=i)
        buffer=io.BytesIO();torch.save(self.refiner.capture(),buffer);buffer.seek(0)
        saved=torch.load(buffer,weights_only=True)
        other=PoseRefiner(self.cameras,self.cfg,self.source,12,saved=saved)
        for i in range(4,9):
            self.assertEqual(self.step(iteration=i),self.step(other,iteration=i))
        for name in self.selected:
            for a,b in zip(self.refiner.params[name],other.params[name]): self.assertTrue(torch.equal(a,b))
        changed={**self.cfg,'translation_lr':.002}
        with self.assertRaisesRegex(ValueError,'differs'):
            PoseRefiner(self.cameras,changed,self.source,12,saved=saved)

    def test_native_gaussian_checkpoint_carries_pose_state_and_rejects_disabling_it(self):
        from utils.local_scale import prepare
        import contextlib
        xyz=np.array([[x*.01,y*.01,2] for y in range(4) for x in range(4)],np.float32)
        cloud=self.root/'input.ply';discrete.write_cloud(cloud,xyz)
        cache=self.root/'local_cache'
        with contextlib.redirect_stdout(io.StringIO()): prepare(cloud,cache,samples=16,workers=1)
        opt=fixed.options(1.,.015);opt.scale_control='hybrid';opt.local_scale_cache=str(cache)
        opt.pose_refinement=json.dumps(self.cfg)
        model=discrete.raw_model(xyz)
        with fixed.cpu_allocations(): model.training_setup(opt)
        self.step();model.pose_refiner=self.refiner
        buffer=io.BytesIO();torch.save(model.capture(),buffer);buffer.seek(0)
        saved=torch.load(buffer,weights_only=True)
        other=discrete.raw_model(xyz)
        with fixed.cpu_allocations(): other.restore(saved,opt)
        loaded=PoseRefiner(self.cameras,self.cfg,self.source,12,saved=other.pose_refinement_state)
        for a,b in zip(loaded.params[self.selected[0]],self.refiner.params[self.selected[0]]):
            self.assertTrue(torch.equal(a,b))
        opt.pose_refinement=''
        with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'pose-refinement mode'):
            other.restore(saved,opt)

    def cache_spec(self,iteration=10):
        model=self.root/'model';model.mkdir(exist_ok=True)
        (model/'keyframes.json').write_bytes(self.manifest.read_bytes())
        self.refiner.export(model/f'point_cloud/iteration_{iteration}',iteration)
        return dict(path=model,type='3dgs',cfg=dict(source_path=str(self.source),pose_refinement=self.cfg),camera_poses='refined')

    def test_checkpoint_pose_cache_applies_only_to_training_and_detects_source_changes(self):
        self.step()
        spec=self.cache_spec();cache=pose_options.load_cache(spec,10)
        posed=pose_options.apply_camera(self.cameras[0],cache)
        torch.testing.assert_close(posed.world_view_transform,self.refiner.camera(self.cameras[0]).world_view_transform)
        held=camera('heldout.jpg');self.assertIs(pose_options.apply_camera(held,cache),held)
        original={**spec,'camera_poses':'original'};self.assertIsNone(pose_options.load_cache(original,10))
        path=next(iter(self.refiner.fingerprints));Path(path).write_text(Path(path).read_text()+'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'calibration'): pose_options.load_cache(spec,10)

    def test_cache_transform_cannot_disagree_with_claimed_bounded_offsets(self):
        spec=self.cache_spec();path=spec['path']/'point_cloud/iteration_10/poses.json'
        data=json.loads(path.read_text());data['frames'][self.selected[0]]['world_to_camera'][0][3]+=10
        atomic_json(path,data)
        with self.assertRaisesRegex(ValueError,'does not match'): pose_options.load_cache(spec,10)

    def test_cache_cannot_silently_relax_saved_pose_bounds(self):
        spec=self.cache_spec();path=spec['path']/'point_cloud/iteration_10/poses.json'
        data=json.loads(path.read_text());data['policy']['max_translation']=1.
        atomic_json(path,data)
        with self.assertRaisesRegex(ValueError,'differs'): pose_options.load_cache(spec,10)

    def test_publication_includes_immutable_matching_pose_sidecar(self):
        spec=self.cache_spec();model=spec['path'];(model/'cfg_args').write_text('Namespace()')
        pc=SimpleNamespace(get_xyz=torch.zeros(1,3),pose_refiner=self.refiner,capture=lambda:('gaussians',self.refiner.capture()))
        def save(iteration):
            folder=model/f'point_cloud/iteration_{iteration}';folder.mkdir(exist_ok=True,parents=True)
            (folder/'point_cloud.ply').write_text('fake gaussian data')
            self.refiner.export(folder,iteration)
        scene=SimpleNamespace(model_path=str(model),gaussians=pc,save=save)
        snap=publish_snapshot(scene,10)
        self.assertEqual(snap,read_snapshot(model,10))
        self.assertEqual(snap['pose_cache'],'point_cloud/iteration_10/poses.json')
        pose=model/snap['pose_cache'];pose.write_text(pose.read_text().replace('never optimized','never OPTIMIZED'))
        with self.assertRaisesRegex(ValueError,'fingerprint'): read_snapshot(model,10)


class PosePlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_additional_run_matches_joint_scale_recipe_and_evaluates_all_frames(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        run_local_scale.prepare(self.argv)
        destination=self.root/'pose_trial'
        args=['--reference-model',str(self.model),'--model',str(destination),'--gpu','0','--schedule-mode','joint','--pose-refinement']
        tune_local_scale.prepare(args)
        state=periodic.directory(destination)/'training.json'
        atomic_json(state,dict(state='failed',error='pose CUDA preflight'))
        tune_local_scale.prepare(args)
        self.assertEqual(json.loads(state.read_text())['state'],'prepared')
        data=periodic.load_plan(destination)
        self.assertEqual(data['plan']['pose_refinement'],DEFAULTS)
        self.assertEqual(data['iterations'],[10000,20000,30000])
        self.assertEqual(data['frames'],sorted(self.names))
        command=data['plan']['stages'][0]['argv']
        self.assertEqual(json.loads(command[command.index('--pose_refinement')+1]),DEFAULTS)
        self.assertIn('--local_scale_schedule',command)
        self.assertEqual(command[command.index('--position_lr_init')+1],'0')
        self.assertTrue(any(path.endswith('images.txt') for path in data['input_fingerprints']))
        # A real partially trained run must not silently restart from scratch.
        (destination/'cfg_args').write_text('Namespace()')
        atomic_json(state,dict(state='failed',error='training error'))
        with self.assertRaisesRegex(ValueError,'fresh model directory'):
            tune_local_scale.prepare(args)


class PoseEvaluationTests(unittest.TestCase):
    setUp=periodic_tests.PeriodicTests.setUp
    prepare=periodic_tests.PeriodicTests.prepare

    def test_original_and_cached_protocols_are_both_complete_and_resume_checked(self):
        data=self.prepare(10000)
        data['plan']['pose_refinement']=DEFAULTS
        snapshot=dict(iteration=10000,snapshot_id='pose_snapshot',gaussians=10,
            files={'point_cloud/iteration_10000/poses.json':dict(sha256='pose-hash')},
            pose_cache='point_cloud/iteration_10000/poses.json')
        calls=[]
        def invoke(command):
            refined='--camera-poses' in command
            calls.append((command[0],refined))
            if command[0].endswith('render.py'):
                output=Path(command[command.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
                (output/'index.html').write_text('GT | render | error')
                return
            output=self.model/'psnr_all_frames/iteration_10000'
            if refined: output=output/'poses_refined'
            output.mkdir(parents=True,exist_ok=True)
            summary=dict(iteration=10000,model=str(self.model),source=str(self.source),model_type='3dgs',
                antialiasing=True,occlusion=dict(occlusion_mode='off'),camera_poses='refined' if refined else 'original',
                counts=dict(all=16,selected=6,unselected=10),mean_psnr_db=dict(all=24. if refined else 23.,
                    selected=26. if refined else 24.,unselected=22.))
            if refined: summary['pose_cache']=dict(sha256='pose-hash')
            atomic_json(output/'summary.json',summary)
            with (output/'per_frame.csv').open('w') as stream:
                writer=csv.writer(stream);writer.writerow(['image_name','selected_for_training','psnr_db'])
                writer.writerows((name,int(name in data['selected']),23.) for name in data['frames'])
        with patch.object(periodic,'read_snapshot',return_value=snapshot),patch.object(periodic.run_occlusion,'invoke',side_effect=invoke):
            result=periodic.evaluate(data,snapshot,0)
        self.assertEqual(calls,[('scripts/eval_all_frames.py',False),('scripts/render.py',False),
                               ('scripts/eval_all_frames.py',True),('scripts/render.py',True)])
        self.assertEqual(result['mean_psnr_db']['selected'],24.)
        self.assertEqual(result['refined_poses']['mean_psnr_db']['selected'],26.)
        self.assertIsNotNone(periodic.completed_result(self.model,10000,snapshot))
        periodic.write_history(data,[result])
        history=json.loads((periodic.directory(self.model)/'history.json').read_text())
        self.assertEqual(history['completed'][0]['refined_selected_psnr_db'],26.)
        Path(result['refined_poses']['previews']).unlink()
        self.assertIsNone(periodic.completed_result(self.model,10000,snapshot))


if __name__=='__main__':
    unittest.main()
