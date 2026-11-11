"""Ray visibility regressions and integration; actual CUDA checks live in smoke_ray_occlusion.py."""
import contextlib
import importlib.util
import io
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch

from utils.ray_occlusion import classify_points_depth,reference_composite
from utils.occlusion_depth import pixel_bounds,tile_maxima
from utils.occlusion_culling import conservative_keep
from test_occlusion import camera


class RayTests(unittest.TestCase):
    def test_lidar_overlay_preserves_original_rows_and_rgb(self):
        from plyfile import PlyData,PlyElement
        from scripts.ray_inspector_backend import read_lidar_cloud
        xyz=np.array([[0,0,2],[np.nan,0,3],[1,0,4]],np.float32)
        rgb=np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            for color_type in (None,'u1','f4'):
                with self.subTest(color_type=color_type):
                    fields=[(key,'f4') for key in ('x','y','z')]
                    if color_type:fields.extend((key,color_type) for key in ('red','green','blue'))
                    rows=np.empty(3,dtype=fields)
                    for column,key in enumerate(('x','y','z')):rows[key]=xyz[:,column]
                    if color_type:
                        for column,key in enumerate(('red','green','blue')):
                            rows[key]=rgb[:,column]/255 if color_type=='f4' else rgb[:,column]
                    path=Path(directory)/'cloud.ply'
                    PlyData([PlyElement.describe(rows,'vertex')]).write(path)
                    points,colors=read_lidar_cloud(path)
                    np.testing.assert_equal(points,xyz)
                    if color_type:np.testing.assert_array_equal(colors,rgb)
                    else:self.assertIsNone(colors)

    def test_cuda_viewer_states_are_disjoint_and_use_image_candidate_denominator(self):
        from scripts.ray_inspector_backend import fragment_layers
        flags=np.array([0,1,2,3,0])
        radii=np.array([0,3,3,3,2])
        ids,report=fragment_layers(flags,radii)
        self.assertEqual(ids['contributing_ids'].tolist(),[1])
        self.assertEqual(ids['fragment_partial_ids'].tolist(),[3])
        shown=np.concatenate([ids[k] for k in ('fragment_blocked_ids','fragment_partial_ids',
                                              'contributing_ids','fragment_inactive_ids')])
        self.assertEqual(sorted(shown.tolist()),list(range(5)))
        self.assertEqual(report['fully_blocked_fraction_in_image'],.25)
        # Densified children are rows in the current checkpoint, not original
        # LiDAR IDs. They are classified without a fixed initial population.
        grown,growth=fragment_layers(np.r_[flags,[2,1]],np.r_[radii,[2,2]])
        self.assertEqual(grown['fragment_blocked_ids'].tolist(),[2,5])
        self.assertEqual(grown['contributing_ids'].tolist(),[1,6])
        self.assertEqual(growth['fully_blocked_fraction_in_image'],2/6)
        _,empty=fragment_layers(np.zeros(3,int),np.zeros(3,int))
        self.assertEqual(empty['fully_blocked_fraction_in_image'],0.)

    def test_smoke_finiteness_check_identifies_backend_and_parameter(self):
        from scripts.smoke_ray_occlusion import require_finite
        require_finite([('_xyz',torch.zeros(2,3))],'AA=False native gradient')
        for invalid in (float('nan'),float('inf')):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(AssertionError,r'AA=False native gradient: _xyz has 6/6 non-finite values'):
                    require_finite([('_xyz',torch.full((2,3),invalid))],'AA=False native gradient')

    def test_depth_reduction_materializes_gradients_before_native_backward(self):
        # Native C++ indexes depth gradients by pixel, without tensor strides.
        # A sum supplies H*W logical elements backed by just one scalar.
        seen=[]
        class RawRaster(torch.autograd.Function):
            @staticmethod
            def forward(ctx,means):
                ctx.shape=means.shape
                return means.sum().expand(3,8,8).clone(),means.sum().expand(1,8,8).clone()
            @staticmethod
            def backward(ctx,rgb_grad,depth_grad):
                seen.append(depth_grad)
                if not depth_grad.is_contiguous():
                    raise AssertionError(f'Raw depth pointer received strides {depth_grad.stride()}')
                return torch.ones(ctx.shape)*(rgb_grad.sum()+depth_grad.sum())
        class Rasterizer:
            def __init__(self,raster_settings):self.ray=hasattr(raster_settings,'occlusion_depth')
            def __call__(self,**kwargs):
                rgb,depth=RawRaster.apply(kwargs['means3D'])
                result=(rgb,torch.ones(2,dtype=torch.int32),depth)
                return result+(torch.ones(2,dtype=torch.int32),) if self.ray else result
        stub=SimpleNamespace(GaussianRasterizationSettings=lambda **kw:SimpleNamespace(**kw),GaussianRasterizer=Rasterizer)
        spec=importlib.util.spec_from_file_location('_depth_stride_renderer',Path(__file__).resolve().parents[1]/'gaussian_renderer/__init__.py')
        module=importlib.util.module_from_spec(spec)
        zeros_like=torch.zeros_like
        def cpu_zeros_like(value,**kwargs):
            kwargs.pop('device',None)
            return zeros_like(value,**kwargs)
        with patch.dict(sys.modules,{'diff_gaussian_rasterization':stub,'diff_gaussian_rasterization_occlusion':stub,
                                    'scene.gaussian_model':SimpleNamespace(GaussianModel=object)}):
            spec.loader.exec_module(module)
            pc=SimpleNamespace(get_xyz=torch.full((2,3),.01,requires_grad=True),get_opacity=torch.ones(2,1),
                get_scaling=torch.ones(2,3),get_rotation=torch.ones(2,4),get_features=torch.ones(2,1,3),active_sh_degree=0)
            cam=camera(8);cam.camera_center=torch.zeros(3)
            pipe=SimpleNamespace(occlusion_backend='ray',occlusion_mode='off',occlusion_margin=0.,debug=False,
                antialiasing=True,compute_cov3D_python=False,convert_SHs_python=False)
            for ray in (False,True):
                for layout in ('sum','transpose','rgb_only','rgb_and_depth'):
                    with self.subTest(ray=ray,layout=layout),patch.object(module.torch,'zeros_like',side_effect=cpu_zeros_like):
                        out=module.render(cam,pc,pipe,torch.zeros(3),**({'occlusion_depth':torch.full((8,8),2.)} if ray else {}))
                        rgb_gradient=0.
                        if layout=='sum':
                            gradient,=torch.autograd.grad(out['depth'].sum()*.01,(pc.get_xyz,))
                            expected=torch.full_like(out['depth'],.01)
                        elif layout=='transpose':
                            expected=torch.arange(64,dtype=torch.float32).reshape(1,8,8).transpose(1,2)
                            gradient,=torch.autograd.grad(out['depth'],(pc.get_xyz,),grad_outputs=expected)
                        else:
                            pixels=torch.ones(8,8);pixels[3,4]=0
                            loss=(out['render']*pixels).sum()
                            expected=torch.zeros_like(out['depth'])
                            if layout=='rgb_and_depth':
                                loss=loss+out['depth'].sum()*.01
                                expected.fill_(.01)
                            gradient,=torch.autograd.grad(loss,(pc.get_xyz,))
                            rgb_gradient=3*pixels.sum()
                        torch.testing.assert_close(seen[-1],expected)
                        torch.testing.assert_close(gradient,torch.ones_like(pc.get_xyz)*(expected.sum()+rgb_gradient))

    def test_hole_changes_only_its_ray_and_blocked_loss_has_zero_background_gradient(self):
        alpha=torch.full((2,16,16),.2,requires_grad=True)
        colors=torch.tensor([[.8,.1,.1],[.1,.9,.2]],requires_grad=True)
        z=torch.tensor([1.,4.]);wall=torch.full((16,16),2.)
        bg=torch.zeros(3)
        blocked=reference_composite(alpha,colors,z,wall,bg)
        hole=wall.clone();hole[8,8]=float('inf')
        partial=reference_composite(alpha,colors,z,hole,bg)
        difference=(partial-blocked).abs().sum(0)
        self.assertEqual(int((difference>0).sum()),1)
        self.assertGreater(float(difference[8,8]),0)
        pixels=torch.ones(16,16);pixels[8,8]=0
        gradients=torch.autograd.grad((partial*pixels).sum(),(alpha,colors))
        self.assertEqual(int(gradients[0][1].count_nonzero()),0)
        self.assertEqual(int(gradients[1][1].count_nonzero()),0)
        self.assertGreater(int(gradients[1][0].count_nonzero()),0)

    def test_depth_hole_no_longer_disables_neighboring_surface_pixels(self):
        depth=torch.full((256,256),2.);depth[120,120]=float('inf')
        bound=pixel_bounds(depth,1)
        self.assertEqual(float(bound[127,127]),2.)
        self.assertTrue(torch.isinf(bound[120,120]))
        keep=conservative_keep(camera(),torch.tensor([[0.,0.,4.]]),torch.full((1,3),.02),
            torch.tensor([[1.,0,0,0]]),tile_maxima(depth,1))
        self.assertTrue(bool(keep[0])) # Regression: old whole-splat gate cannot reject it.

    def test_point_labels_partition_original_rows_and_use_camera_z(self):
        k=np.array([[32.,0,31.5],[0,32.,31.5],[0,0,1]],np.float32)
        depth=torch.full((64,64),2.)
        points=np.array([[0,0,2],[0,0,4],[0,0,1],[0,0,-1],[100,0,2],[np.nan,0,1]],np.float32)
        labels=classify_points_depth(points,k,np.eye(4,dtype=np.float32),depth,chunk_size=2)
        self.assertEqual(labels['on_surface_ids'].tolist(),[0])
        self.assertEqual(labels['culled_ids'].tolist(),[1])
        self.assertEqual(labels['in_front_ids'].tolist(),[2])
        self.assertEqual(labels['outside_ids'].tolist(),[3,4,5])
        depth[32,32]=float('inf')
        labels=classify_points_depth(points,k,np.eye(4,dtype=np.float32),depth)
        self.assertEqual(labels['unknown_ids'].tolist(),[0,1,2])
        self.assertEqual(len(labels['frustum_ids'])+len(labels['outside_ids']),len(points))
        # Rotation and translation preserve the same camera rays and point IDs.
        rotation=np.array([[0.,0,1],[0,1,0],[-1,0,0]],np.float32)
        origin=np.array([10.,4.,-2.],np.float32)
        e=np.eye(4,dtype=np.float32);e[:3,:3]=rotation;e[:3,3]=-rotation@origin
        transformed=classify_points_depth(points[:3]@rotation+origin,k,e,torch.full((64,64),2.))
        self.assertEqual(transformed['culled_ids'].tolist(),[1])

    def test_native_render_routes_full_pixel_depth_and_real_contribution_flags(self):
        seen=[]
        class Rasterizer:
            def __init__(self,raster_settings):self.settings=raster_settings
            def __call__(self,**kwargs):
                seen.append((self.settings,kwargs))
                zero=kwargs['means3D'].sum()*0+kwargs['means2D'].sum()*0
                return zero.expand(3,8,8),torch.ones(3,dtype=torch.int32),zero.expand(1,8,8),torch.tensor([1,2,3])
        stub=SimpleNamespace(GaussianRasterizationSettings=lambda **kw:SimpleNamespace(**kw),GaussianRasterizer=Rasterizer)
        spec=importlib.util.spec_from_file_location('_ray_renderer',Path(__file__).resolve().parents[1]/'gaussian_renderer/__init__.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'diff_gaussian_rasterization':stub,'diff_gaussian_rasterization_occlusion':stub,
                                    'scene.gaussian_model':SimpleNamespace(GaussianModel=object)}):
            spec.loader.exec_module(module)
            pc=SimpleNamespace(get_xyz=torch.ones(3,3,requires_grad=True),get_opacity=torch.ones(3,1),
                get_scaling=torch.ones(3,3),get_rotation=torch.ones(3,4),get_features=torch.ones(3,1,3),active_sh_degree=0)
            cam=camera(8);cam.camera_center=torch.zeros(3)
            pipe=SimpleNamespace(occlusion_backend='ray',occlusion_mode='off',occlusion_margin=.05,debug=False,
                antialiasing=True,compute_cov3D_python=False,convert_SHs_python=False)
            depth=torch.full((8,8),2.)
            out=module.render(cam,pc,pipe,torch.zeros(3),occlusion_depth=depth,separate_sh=True)
        self.assertEqual(out['occlusion_keep'].tolist(),[True,False,True])
        self.assertEqual(out['occlusion_culled'].tolist(),[False,True,False])
        self.assertEqual(out['radii'].tolist(),[1,0,1])
        self.assertEqual(out['viewspace_points'].shape,(3,3))
        torch.testing.assert_close(seen[0][0].occlusion_depth,depth+.05)
        self.assertNotIn('dc',seen[0][1])

    @unittest.skipUnless(importlib.util.find_spec('viser'),'viser required')
    def test_viewer_creates_all_requested_layers_with_real_viser_api(self):
        from scripts import inspect_ray_occlusion as viewer
        from test_scene_occlusion import quad
        vertices,faces=quad()
        points=np.array([[0,0,2],[0,0,4],[0,0,-2]],np.float32)
        ids={key+'_ids':np.array(value) for key,value in dict(frustum=[0,1],culled=[1],on_surface=[0],
            in_front=[],unknown=[],outside=[2]).items()}
        pixels=np.zeros((64,64,3),np.uint8)
        result=dict(name='a.jpg',k=np.array([[32,0,31.5],[0,32,31.5],[0,0,1]]),c2w=np.eye(4),
            groups={'LiDAR':ids},counts={'LiDAR':dict(total=3,frustum=2,culled=1,on_surface=1)},
            report=dict(integrated_frames=38,depth_coverage=.8),images={'gt':('Actual',pixels),'depth':('Depth',pixels)})
        fake=SimpleNamespace(vertices=vertices,faces=faces,names=['a.jpg'],point_sets={'LiDAR':points},analyze=lambda _:result)
        with socket.socket() as connection:
            connection.bind(('127.0.0.1',0));port=connection.getsockname()[1]
        with patch('scripts.ray_inspector_backend.RayInspector',return_value=fake),patch.object(viewer,'time',SimpleNamespace(sleep=Mock(side_effect=KeyboardInterrupt))):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(viewer.main(['--scene','fixture.json','--point-cloud','fixture.ply','--image','a.jpg',
                    '--host','127.0.0.1','--port',str(port)]),0)

    @unittest.skipUnless(importlib.util.find_spec('viser'),'viser required')
    def test_checkpoint_viewer_keeps_lidar_and_gaussians_independent(self):
        import viser
        from scripts import inspect_ray_occlusion as viewer
        from scripts.ray_inspector_backend import fragment_layers
        from test_scene_occlusion import quad
        vertices,faces=quad()
        points=np.array([[0,0,2],[0,0,4],[0,0,-2],[.1,0,3]],np.float32)
        ids,fragments=fragment_layers(np.array([1,2,0,3]),np.array([2,2,0,2]))
        pixels=np.zeros((64,64,3),np.uint8)
        result=dict(name='a.jpg',k=np.array([[32,0,31.5],[0,32,31.5],[0,0,1]]),c2w=np.eye(4),
            groups={'LiDAR':{},'Gaussians':ids},counts={'LiDAR':{'total':4},'Gaussians':{'total':4}},
            report=dict(integrated_frames=0,depth_coverage=1.,fragments=fragments,
                        checkpoint=dict(iteration=30000)),images={'gt':('Actual',pixels),'on':('ON',pixels),'off':('OFF',pixels)})
        fake=SimpleNamespace(vertices=vertices,faces=faces,names=['a.jpg','b.jpg'],
            point_sets={'LiDAR':points,'Gaussians':points},analyze=Mock(side_effect=lambda name:dict(result,name=name)))
        with socket.socket() as connection:
            connection.bind(('127.0.0.1',0));port=connection.getsockname()[1]
        with contextlib.redirect_stdout(io.StringIO()):
            server=viser.ViserServer(host='127.0.0.1',port=port)
            controls,clouds,meshes,buttons,markdowns={},{},{},{},{}
            def capture(method,handles):
                def call(*args,**kwargs):
                    handle=method(*args,**kwargs)
                    handles[args[0]]=handle
                    return handle
                return call
            def visible_gaussians():
                return {name.rsplit('/',1)[1] for name,handle in clouds.items()
                        if name.startswith('/points/Gaussians/') and handle.visible}
            def exercise_controls(_):
                self.assertTrue(controls['Show LiDAR'].value)
                self.assertTrue(controls['Show Gaussians'].value)
                self.assertNotIn('Point set',controls)
                self.assertFalse(meshes['/surface'].visible)
                self.assertEqual(visible_gaussians(),viewer.default_layers('Gaussians'))
                np.testing.assert_allclose(clouds['/points/LiDAR/raw'].points,points,atol=.001)
                np.testing.assert_array_equal(clouds['/points/Gaussians/contributing'].points,points[[0]])
                np.testing.assert_allclose(clouds['/points/Gaussians/fragment_partial'].points,points[[3]],atol=.001)
                self.assertIn('CUDA contributing: 2',markdowns[''].content)
                # Toggling either cloud must retain the other's visible nodes.
                original_green=clouds['/points/Gaussians/contributing']
                controls['Show LiDAR'].value=False
                self.assertFalse(clouds['/points/LiDAR/raw'].visible)
                self.assertEqual(visible_gaussians(),viewer.default_layers('Gaussians'))
                self.assertIs(clouds['/points/Gaussians/contributing'],original_green)
                controls['Show LiDAR'].value=True
                controls['Show Gaussians'].value=False
                self.assertTrue(clouds['/points/LiDAR/raw'].visible)
                self.assertEqual(visible_gaussians(),set())
                controls['Surface display (visual only)'].value='wireframe'
                self.assertTrue(meshes['/surface'].wireframe)
                self.assertTrue(meshes['/surface'].visible)
                buttons['Show contributing Gaussians only'].value=True
                self.assertFalse(meshes['/surface'].visible)
                self.assertFalse(clouds['/points/LiDAR/raw'].visible)
                self.assertEqual(visible_gaussians(),{'contributing','fragment_partial'})
                self.assertEqual(fake.analyze.call_count,1)  # Display controls never rerender/reclassify.
                raw=clouds['/points/LiDAR/raw']
                controls['Camera'].value='b.jpg'
                self.assertEqual(fake.analyze.call_count,2)
                self.assertIs(clouds['/points/LiDAR/raw'],raw)  # Static layer retained between frames.
                self.assertIsNot(clouds['/points/Gaussians/contributing'],original_green)
                self.assertEqual(visible_gaussians(),{'contributing','fragment_partial'})
                raise KeyboardInterrupt
            try:
                with patch('viser.ViserServer',return_value=server), \
                     patch.object(server.gui,'add_dropdown',side_effect=capture(server.gui.add_dropdown,controls)), \
                     patch.object(server.gui,'add_checkbox',side_effect=capture(server.gui.add_checkbox,controls)), \
                     patch.object(server.gui,'add_button',side_effect=capture(server.gui.add_button,buttons)), \
                     patch.object(server.gui,'add_markdown',side_effect=capture(server.gui.add_markdown,markdowns)), \
                     patch.object(server.scene,'add_point_cloud',side_effect=capture(server.scene.add_point_cloud,clouds)), \
                     patch.object(server.scene,'add_mesh_simple',side_effect=capture(server.scene.add_mesh_simple,meshes)), \
                     patch('scripts.ray_inspector_backend.RayInspector',return_value=fake), \
                     patch.object(viewer,'time',SimpleNamespace(sleep=exercise_controls)):
                    self.assertEqual(viewer.main(['--scene','fixture.json','--point-cloud','fixture.ply',
                        '--model','checkpoint','--image','a.jpg','--host','127.0.0.1','--port',str(port)]),0)
            finally:
                server.stop()


if __name__=='__main__':unittest.main()
