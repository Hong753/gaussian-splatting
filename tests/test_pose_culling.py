"""Real CPU mesh rays/cache and training-plan checks; CUDA is checked on host."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.pose_culling import DEFAULTS, PoseMeshDepthCache, validate, statistics
from utils.pose_refinement import PoseRefiner,DEFAULTS as POSES
from scripts import occlusion_options,run_local_scale,tune_local_scale,run_soft_surface_long as periodic
from test_scene_occlusion import manifest,quad
from test_occlusion import camera
from test_pose_refinement import differentiable_image
import test_fixed_lidar as fixed
import test_discrete_lidar as discrete
from test_lidar_offsets import LidarOffsetTests


def measured_mesh(root,vertices=None,faces=None):
    path=manifest(root,vertices,faces)
    data=json.loads(path.read_text());data.update(geometry_source='full_initialization_lidar',relative_margin=.25)
    path.write_text(json.dumps(data))
    return path


def cam(shift=0.,size=64):
    value=camera(size,translation=(0,0,-shift))
    value.image_name='a.jpg';value.projection_matrix=torch.linalg.solve(value.world_view_transform,value.full_proj_transform)
    value.camera_center=value.world_view_transform.inverse()[3,:3]
    value.alpha_mask=torch.ones(1,size,size);value.original_image=torch.full((3,size,size),.4)
    return value


class PoseCullingTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)

    def provider(self,vertices=None,faces=None):
        self.path=measured_mesh(self.root,vertices,faces)
        return PoseMeshDepthCache(self.path,self.root/'cache',DEFAULTS)

    def test_wall_tolerance_ignores_legacy_margin_and_preserves_foreground(self):
        provider=self.provider();frame=provider.frame(cam(),'cpu')
        self.assertAlmostEqual(float(frame['depth'][32,32]),2.,places=5)
        self.assertAlmostEqual(float(frame['cutoff'][32,32]),2.06,places=5)
        self.assertTrue(frame['pose_mask'][32,32])
        self.assertTrue(torch.isinf(frame['cutoff'][0,32]))
        # On-wall / slightly behind inside tolerance / far behind.
        self.assertEqual((torch.tensor([2.,2.04,4.]) > frame['cutoff'][32,32]).tolist(),[False,False,True])

    def test_doorway_unknown_rays_are_never_filled_and_edges_are_conservative(self):
        left,lf=quad(-3,-.25);right,rf=quad(.25,3)
        provider=self.provider(np.concatenate([left,right]),np.concatenate([lf,rf+4]))
        frame=provider.frame(cam(),'cpu')
        self.assertTrue(torch.isinf(frame['cutoff'][32,32]))
        self.assertFalse(frame['pose_mask'][32,32])
        self.assertAlmostEqual(float(frame['cutoff'][32,16]),2.06,places=5)

    def test_pose_proposals_memory_only_bounded_and_frozen_cache_replaces_per_view(self):
        provider=self.provider();frame=provider.frame(cam(),'cpu')
        files=list(provider.directory.glob('*.npz'));self.assertEqual(len(files),1)
        before=files[0].read_bytes()
        for i in range(1,8):
            view=cam(i*.05);view.occlusion_transient=True
            posed=provider.frame(view,'cpu')
            self.assertAlmostEqual(float(posed['depth'][32,32]),2-i*.05,places=5)
            self.assertLessEqual(len(provider.memory),2)
        self.assertEqual(files[0].read_bytes(),before)
        frozen=cam(.43);new=provider.frame(frozen,'cpu')
        self.assertEqual(len(list(provider.directory.glob('*.npz'))),1)
        self.assertNotEqual(frame['signature'],new['signature'])
        restored=PoseMeshDepthCache(self.path,self.root/'cache',DEFAULTS)
        with patch('utils.scene_occlusion.MeshDepthRenderer',side_effect=AssertionError('Should load disk')):
            self.assertAlmostEqual(float(restored.frame(frozen,'cpu')['depth'][32,32]),1.57,places=5)

    def test_relative_tolerance_is_bounded_and_predicted_supplements_unused(self):
        vertices,faces=quad(-100,100,z=50.);vertices[:,1]*=40
        provider=self.provider(vertices,faces)
        self.assertAlmostEqual(float(provider.frame(cam(),'cpu')['cutoff'][32,32]),50.15,places=4)
        for update in ({'absolute_margin':-.1},{'memory_views':0},{'max_margin':.01},{'edge_pixels':1.5}):
            with self.assertRaises(ValueError):validate({**DEFAULTS,**update})
        data=json.loads(self.path.read_text());data['geometry_source']='predicted_depth';self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'measured LiDAR'):
            PoseMeshDepthCache(self.path,self.root/'new',DEFAULTS)

    def test_pose_objective_mask_fixed_for_proposal_and_empty_support_skips(self):
        provider=self.provider();camera=cam()
        source=self.root/'source/sparse/0';source.mkdir(parents=True)
        for name in ('images','cameras'):(source/(name+'.txt')).write_text('# original\n')
        refiner=PoseRefiner([camera],{**POSES,'start':0,'end':12,'max_width':64,'prior_weight':1e-7},self.root/'source',12)
        pipe=SimpleNamespace(pose_culling=dict(DEFAULTS),occlusion_mode='mesh',_occlusion_provider=provider)
        row=refiner.update(camera,None,pipe,torch.zeros(3),1,renderer=differentiable_image)
        self.assertTrue(row['accepted']);self.assertEqual(row['pose_pixels'],60*60)
        self.assertFalse(provider.directory.exists())
        self.assertTrue(refiner.camera(camera).occlusion_transient)
        refiner.iteration=13;self.assertFalse(refiner.camera(camera).occlusion_transient)
        pipe.pose_culling={**DEFAULTS,'pose_min_pixels':100000}
        row=refiner.update(camera,None,pipe,torch.zeros(3),13,renderer=differentiable_image)
        self.assertIsNone(row) # Frozen pose phase.
        refiner.iteration=1
        row=refiner.update(camera,None,pipe,torch.zeros(3),2,renderer=differentiable_image)
        self.assertEqual(row['skipped'],'insufficient_mesh_interior')

    def test_saved_inference_off_and_explicit_on_restore_same_policy(self):
        provider=self.provider()
        cfg=dict(occlusion_mode='mesh',occlusion_backend='ray',occlusion_geometry=str(self.path),
                 occlusion_cache=str(self.root/'cache'),occlusion_margin=0.,pose_culling=dict(DEFAULTS),occlusion_inference_mode='off')
        pipe=occlusion_options.configure_pipeline(SimpleNamespace(),cfg);pipe._occlusion_provider=provider
        disabled=occlusion_options.inference_pipeline(pipe,cfg)
        self.assertEqual(disabled.occlusion_mode,'off');self.assertFalse(hasattr(disabled,'_occlusion_provider'))
        spec=occlusion_options.apply_overrides(dict(type='3dgs',cfg=cfg),SimpleNamespace(occlusion='on'))
        self.assertEqual(spec['cfg']['pose_culling'],DEFAULTS);self.assertEqual(spec['cfg']['occlusion_mode'],'mesh')
        package=dict(occlusion_fragment_flags=torch.tensor([1,2,3,0]),raster_radii=torch.tensor([2,2,2,0]))
        stats=statistics(package,cam(),SimpleNamespace(get_xyz=torch.zeros(4,3)),pipe,1)
        self.assertEqual(stats['contributing'],2);self.assertAlmostEqual(stats['fully_blocked_candidate_fraction'],1/3)


class CullingCheckpointTests(LidarOffsetTests):
    # Exercise the existing real-Adam offset/pose/checkpoint tests with the new
    # policy attached as well. Rasterization is separately tested on CUDA.
    def setUp(self):
        super().setUp();self.opt.pose_culling=json.dumps(DEFAULTS)

    def test_changed_culling_policy_cannot_resume(self):
        m=self.model();self.step(m,4);saved=deepcopy(m.capture())
        bad=deepcopy(self.opt);bad.pose_culling=json.dumps({**DEFAULTS,'absolute_margin':.1})
        with fixed.cpu_allocations(),self.assertRaisesRegex(ValueError,'pose-culling policy'):
            discrete.raw_model(self.xyz).restore(saved,bad)


class CullingPlanTests(unittest.TestCase):
    setUp=fixed.FixedLidarPlanTests.setUp

    def test_gpu3_plan_preserves_cap5_offsets_and_all_frame_evaluation(self):
        discrete.write_cloud(self.ply,np.array([[i*.01,0,2] for i in range(16)],np.float32))
        run_local_scale.prepare(self.argv)
        meshroot=self.root/'geometry';meshroot.mkdir();geometry=measured_mesh(meshroot)
        from utils.lidar_occlusion import file_digest
        report=json.loads(geometry.read_text());report.update(source=str(self.source),point_cloud_sha256=file_digest(self.ply))
        geometry.write_text(json.dumps(report))
        target=self.root/'gpu3'
        args=['--reference-model',str(self.model),'--model',str(target),'--gpu','3','--ceiling-multiplier','5',
              '--schedule-mode','joint','--pose-refinement','--lidar-offsets','--pose-culling','--scene-geometry',str(geometry)]
        tune_local_scale.prepare(args);tune_local_scale.prepare(args)
        data=periodic.load_plan(target);plan=data['plan'];command=plan['stages'][0]['argv']
        self.assertEqual(data['gpu'],'3');self.assertEqual(data['frames'],sorted(self.names))
        self.assertEqual(data['iterations'],[10000,20000,30000]);self.assertEqual(plan['effective_max_scale_world'],.075)
        self.assertEqual(command[command.index('--occlusion_mode')+1],'mesh')
        self.assertEqual(command[command.index('--occlusion_inference_mode')+1],'off')
        self.assertEqual(command[command.index('--densify_until_iter')+1],'0')
        self.assertEqual(plan['pose_culling'],DEFAULTS)
        for name in (geometry,meshroot/'mesh.npz'):self.assertIn(str(name),data['input_fingerprints'])
        import argparse
        from arguments import ModelParams, OptimizationParams
        from utils.fixed_lidar import configure_training
        parser=argparse.ArgumentParser();mp=ModelParams(parser);op=OptimizationParams(parser)
        parsed,_=parser.parse_known_args(command[1:]);dataset,opt=mp.extract(parsed),op.extract(parsed)
        configure_training(opt,dataset)
        self.assertEqual(dataset.pose_culling,DEFAULTS);self.assertEqual(opt.densify_until_iter,0)
        self.assertEqual(opt.pose_culling_reference[str(geometry)],file_digest(geometry))
        self.assertEqual(occlusion_options.configure_pipeline(SimpleNamespace(),dataset).pose_culling,DEFAULTS)


class CullingPreviewTests(unittest.TestCase):
    def test_three_column_report_uses_identical_names_and_targets(self):
        from scripts.pose_culling_preview import combine
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for mode in ('off','on'):
                path=root/mode/'train/model_0';path.mkdir(parents=True)
                record=dict(gt='gt/a.png',image='renders/a.png',error='errors/a.png',comparison='comparisons/a.jpg',
                            psnr_db=23.,target_sha256='same')
                (path/'frames.json').write_text(json.dumps({'a.jpg':record}))
            combine(root/'off',root/'on',root/'pair',root/'model',10000)
            summary=json.loads((root/'pair/train/summary.json').read_text())
            self.assertEqual([m['label'] for m in summary['models']],['Culling OFF','Culling ON'])
            self.assertEqual(len(summary['frames'][0]['renders']),2)
            self.assertTrue((root/'pair/index.html').is_file())
            record['target_sha256']='different'
            (root/'on/train/model_0/frames.json').write_text(json.dumps({'a.jpg':record}))
            with self.assertRaisesRegex(ValueError,'ground truth'):
                combine(root/'off',root/'on',root/'pair',root/'model',10000)


if __name__=='__main__':unittest.main()
