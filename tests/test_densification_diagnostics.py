"""Large-scene clone classification, real growth counters, and viewer/API contracts."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

from utils.densification_diagnostics import (training_extent,classify_sizes,project_centers,
    roi_pixels,footprint_roi,center_frustum,size_report,rotations,ellipsoid_segments,projected_ellipses,growth_window_stats)
from utils.graphics_utils import getProjectionMatrix
from scripts.densification_inspector_backend import DensificationInspector
from scripts.inspect_densification import build_viewer,sample_ids
from test_surface_experiment import SurfaceExperimentTests


class DensificationDiagnosticsTests(unittest.TestCase):
    def test_same_large_footprint_changes_branch_only_with_global_extent(self):
        sigma=np.array([.1,.1,.4]);radii=np.array([80,0,80])
        np.testing.assert_array_equal(classify_sizes(sigma,radii,.01*50,20),[2,0,2])
        np.testing.assert_array_equal(classify_sizes(sigma,radii,.01*5,20),[3,0,3])
        # Equality belongs to the native clone branch; radius==threshold is not large.
        np.testing.assert_array_equal(classify_sizes(np.array([.5,.51,.5]),np.array([21,21,20]),.5,20),[2,3,1])

    def test_extent_uses_selected_camera_centers_including_rotated_poses(self):
        r=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        records={str(i):dict(R=r.T,T=-r@np.array([x,0,0])) for i,x in enumerate((0,2,100))}
        self.assertAlmostEqual(training_extent(records,['0','1']),1.1,places=6)
        self.assertGreater(training_extent(records,['0','1','2']),60)

    def test_native_projection_roi_and_display_sampling_do_not_drop_statistics(self):
        view=np.eye(4,dtype=np.float32)
        projection=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).numpy().T
        points=np.array([[0.,0,2],[2,0,2],[0,0,-2]],np.float32)
        projected=project_centers(points,view,projection,100,100,chunk=1)
        np.testing.assert_allclose(projected[0],[49.5,49.5,2],atol=1e-5)
        np.testing.assert_array_equal(center_frustum(projected,100,100),[True,False,False])
        rectangle=roi_pixels((.4,.4,.6,.6),100,100)
        # Center lies outside the crop but its large footprint overlaps it.
        np.testing.assert_array_equal(footprint_roi(projected,np.array([5,50,100]),rectangle),[True,True,False])
        self.assertEqual(len(sample_ids(np.arange(10000),100)),100)
        self.assertEqual(len(sample_ids(np.arange(10000),0)),10000)

    def test_rotated_ellipsoids_and_projected_contours_have_correct_physical_sizes(self):
        q=np.array([[math.sqrt(.5),0,0,math.sqrt(.5)]])
        scales=np.array([[.1,.2,.3]])
        r=rotations(q)
        np.testing.assert_allclose(r[0]@np.array([1,0,0]),[0,1,0],atol=1e-14)
        rings=ellipsoid_segments(np.array([[0.,0,2]]),scales,q)
        np.testing.assert_allclose(np.max(np.abs(rings.reshape(-1,3)-[0,0,2]),axis=0),[.6,.3,.9],atol=1e-6)
        contours=projected_ellipses(np.array([[0.,0,2]]),np.array([[.1,.1,.1]]),np.array([[1,0,0,0]]),
                                    np.eye(4),100,100,math.pi/2,math.pi/2)
        expected=3*math.sqrt((50*.1/2)**2+.3)
        np.testing.assert_allclose(np.linalg.norm(contours[0],axis=1),expected,atol=1e-6)

    def test_real_growth_keeps_colocated_clones_and_reports_gradient_gating(self):
        model=SurfaceExperimentTests().model()
        model.surface_mode='off';model.surface_refs=None;model.use_absgs=False
        model._scaling.data[0]=math.log(.005);model._scaling.data[1]=math.log(.02)
        model.xyz_gradient_accum=torch.ones(2,1);model.denom=torch.ones(2,1)
        model.max_radii2D=torch.tensor([80.,30.])
        initial=model._xyz.detach().clone()
        original_zeros=torch.zeros
        def cpu_zeros(*a,**kw):
            if kw.get('device')=='cuda':kw['device']='cpu'
            return original_zeros(*a,**kw)
        with patch.object(torch,'zeros',side_effect=cpu_zeros),patch.object(torch.cuda,'empty_cache'):
            model.densify_and_prune(.1,.005,1.,None,torch.tensor([80,30]),record_stats=True)
        stats=model.last_densification_stats
        self.assertEqual((stats['cloned_parents'],stats['split_parents'],stats['points_after']),(1,1,4))
        self.assertEqual(stats['clone_parents_radius_gt20_in_window'],1)
        # Native clone remains exactly at its parent with exactly the same scale.
        self.assertEqual(int((model.get_xyz==initial[0]).all(dim=1).sum()),2)
        self.assertEqual(stats['eligible_clone_parents'],1)
        no_grad=growth_window_stats(torch.tensor([[.005]*3,[.02]*3]),torch.zeros(2,1),torch.zeros(2,1),
                                   torch.tensor([80,30]),1.,.01,.1,.1)
        self.assertEqual(no_grad['eligible_clone_parents']+no_grad['eligible_split_parents'],0)

    def backend(self,root):
        backend=DensificationInspector.__new__(DensificationInspector)
        backend.args=SimpleNamespace(model=root,ellipsoids=5,probe_gradients=False,scene=None)
        backend.iteration=30000;backend.ply=root/'point_cloud.ply';backend.extent=50.
        backend.percent_dense=.01;backend.cutoff=.5;backend.opt={};backend.cfg={}
        backend.records={'wall.jpg':{}};backend.names=['wall.jpg'];backend.spec={'splits':{'train':{'wall.jpg'}}}
        backend.points=np.array([[0,0,z] for z in (2.,3.,4.)],np.float32)
        backend.lidar=backend.points.copy();backend.lidar_colors=None
        backend.max_sigma=np.array([.01,.1,.1]);backend.opacities=np.array([1.,.9,.9])
        backend.model=SimpleNamespace(get_xyz=torch.tensor(backend.points),get_scaling=torch.full((3,3),.1),
            get_rotation=torch.tensor([[1.,0,0,0]]*3))
        backend.pipe=SimpleNamespace(occlusion_mode='off',antialiasing=True)
        backend.bg=torch.ones(3);backend.output=root/'diagnostic'
        camera=SimpleNamespace(FoVx=math.pi/2,FoVy=math.pi/2,image_width=16,image_height=16)
        backend.frame=dict(name='wall.jpg',camera=camera,view=np.eye(4),width=16,height=16,
            pred=np.full((3,16,16),.5),target=np.full((3,16,16),.6),radii=np.array([5,30,40]),
            projected=np.array([[7.5,7.5,z] for z in (2.,3.,4.)]),gradient=None,
            frustum=np.ones(3,bool),lidar_frustum=np.ones(3,bool),c2w=np.eye(4),radius=None)
        return backend

    def fake_render(self,camera,model,pipe,bg,override_color):
        self.assertEqual(pipe.occlusion_mode,'off')
        self.assertEqual(bg.count_nonzero(),0)
        # First (small) Gaussian is opaque; large rear candidates contribute zero.
        alpha=torch.tensor([1.,.9,.9]);trans=torch.cat((torch.ones(1),(1-alpha).cumprod(0)[:-1]))
        pixel=((alpha*trans)[:,None]*override_color).sum(0)
        return {'render':pixel[:,None,None].expand(3,16,16)}

    def test_native_compositing_distinguishes_large_occluded_candidates_from_contributors(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=self.backend(Path(tmp))
            with patch.dict('sys.modules',gaussian_renderer=SimpleNamespace(render=self.fake_render)):
                result=backend.analyze('wall.jpg')
            report=result['report']
            self.assertEqual(report['size_branch_counts']['large_clone_size'],2)
            self.assertEqual(report['compositing_mass_fraction']['large_clone_size'],0.)
            self.assertEqual(report['compositing_mass_fraction']['other'],1.)
            self.assertFalse(report['densification_window_open_at_checkpoint'])
            folder,archive=backend.export(result)
            self.assertEqual(json.loads((folder/'summary.json').read_text())['total_gaussians'],3)
            with ZipFile(archive) as bundle:
                self.assertIn('contributions.png',bundle.namelist())
                self.assertIn('largest_footprints.csv',bundle.namelist())
                self.assertIn('summary.json',bundle.namelist())

    def test_real_viser_api_builds_all_layers_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=self.backend(Path(tmp))
            args=SimpleNamespace(host='127.0.0.1',port=0,image='wall.jpg',large_radius=20.,roi=(0,0,1,1),
                                 scene=None,max_points=2)
            with patch.dict('sys.modules',gaussian_renderer=SimpleNamespace(render=self.fake_render)):
                server=build_viewer(backend,args)
            try:
                # Creation uses the installed Viser API, not a substitute GUI.
                self.assertGreater(len(server.gui._gui_input_handle_from_uuid),10)
            finally:
                server.stop()


if __name__=='__main__':unittest.main()
