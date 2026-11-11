"""Measured walls must block without DA3; openings and native coordinates survive."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from utils.lidar_surface import build_lidar_mesh
from utils.ray_occlusion import classify_points_depth, projected_lidar_depth, first_surface_audit
from utils.scene_occlusion import MeshDepthCache, TriangleRaycaster, classify_lidar, load_scene, scene_raycaster
from utils.lidar_occlusion import file_digest
from scripts.prepare_da3_occlusion import save_npz
from test_scene_occlusion import cloud, manifest, quad
from test_occlusion import camera


def sampled_wall(z=2., extent=1., spacing=.05, doorway=False):
    x,y=np.meshgrid(np.arange(-extent,extent+spacing/2,spacing),np.arange(-extent,extent+spacing/2,spacing))
    points=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,z)))
    return points[~((np.abs(points[:,0])<.3)&(np.abs(points[:,1])<.6))] if doorway else points


@unittest.skipUnless(importlib.util.find_spec('open3d'),'Real mesh reconstruction requires Open3D')
class LidarSurfaceTests(unittest.TestCase):
    def test_measured_wall_blocks_without_predicted_depth_and_keeps_doorway(self):
        points=np.concatenate((sampled_wall(doorway=True),sampled_wall(4.,2.,.1)))
        vertices,faces,ids,missing,report=build_lidar_mesh(points,block_size=1.5,max_block_points=1000,threads=1)
        np.testing.assert_array_equal(vertices,points[ids].astype(np.float32))
        self.assertGreater(report['meshed_representative_fraction'],.95)
        self.assertGreater(report['blocks'],1)
        ray=TriangleRaycaster(vertices,faces)
        origins=np.zeros(3)
        # Pixel rays at seams and between measured samples, not just mesh vertices.
        xx,yy=np.meshgrid(np.linspace(-.9,.9,73),np.linspace(-.9,.9,71))
        directions=np.column_stack((xx.ravel()/2,yy.ravel()/2,np.ones(xx.size)))
        hit=ray.cast(origins,directions)
        solid=(np.abs(xx.ravel())>.4)|(np.abs(yy.ravel())>.7)
        doorway=(np.abs(xx.ravel())<.2)&(np.abs(yy.ravel())<.5)
        self.assertGreater(np.mean(np.abs(hit[solid]-2)<1e-4),.995)
        np.testing.assert_allclose(hit[doorway],4.,atol=1e-4)
        k=np.array([[32,0,31.5],[0,32,31.5],[0,0,1]])
        probes=np.array([[np.nan,0,1],[.7,0,2],[1.4,0,4],[0,0,4],[0,0,-1]])
        labels=classify_lidar(probes,k,np.eye(4),64,64,ray,.05,0)
        self.assertEqual(labels['occluded_ids'].tolist(),[2])
        self.assertEqual(labels['visible_ids'].tolist(),[1,3])

    def test_parallel_walls_not_averaged_and_transformed_geometry_stays_aligned(self):
        points=np.concatenate((sampled_wall(),sampled_wall(2.12)))
        vertices,faces,_,_,_=build_lidar_mesh(points,threads=1)
        # Disallow triangles that tilt across the two parallel sheets.
        self.assertTrue((np.ptp(vertices[faces,2],axis=1)<1e-4).all())
        ray=TriangleRaycaster(vertices,faces)
        np.testing.assert_allclose(ray.cast(np.zeros(3),np.array([[.1,.15,1.],[-.15,.1,1.]])),2.,atol=1e-4)
        rotation=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
        origin=np.array([10.,4.,-2.])
        transformed=sampled_wall()@rotation+origin
        v,f,_,_,_=build_lidar_mesh(transformed,threads=1)
        ray=TriangleRaycaster(v,f)
        np.testing.assert_allclose(ray.cast(origin,np.array([[.1,.15,1.],[-.15,.1,1.]])@rotation),2.,atol=1e-4)

    def test_lidar_has_priority_over_da3_on_known_rays_in_cache_and_exact_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            v,f=quad(-3,-.25)
            path=manifest(root,v,f)
            report=json.loads(path.read_text())
            sv,sf=quad(z=1.) # Wrong foreground prediction must not override known LiDAR.
            save_npz(root/'supplement.npz',vertices=sv,faces=sf)
            report.update(supplement_mesh='supplement.npz',supplement_mesh_sha256=file_digest(root/'supplement.npz'),
                          supplement_policy='missing_lidar_rays_only')
            path.write_text(json.dumps(report))
            provider=MeshDepthCache(path)
            depth=provider.raw_depth(camera(128),'cpu')
            self.assertAlmostEqual(float(depth[64,32]),2.,places=4)
            self.assertAlmostEqual(float(depth[64,96]),1.,places=4)
            ray=scene_raycaster(report,root/'mesh.npz')
            np.testing.assert_allclose(ray.cast(np.zeros(3),np.array([[-.5,0,1],[.5,0,1]])),[2,1],atol=1e-4)
            (root/'supplement.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'supplementary'):
                load_scene(path)

    def test_builder_is_training_ready_without_da3_cache_and_reuses_only_matching_inputs(self):
        from scripts.prepare_lidar_scene import main
        from scripts.occlusion_options import prepare_training_geometry
        from types import SimpleNamespace
        from utils.occlusion_depth import record_camera
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);points=sampled_wall()
            cloud(root/'input.ply',points)
            (root/'keyframes.json').write_text(json.dumps({'selected_images':['a.jpg']}))
            Image.new('RGB',(64,64)).save(root/'a.jpg')
            record=dict(name='a.jpg',image_path=str(root/'a.jpg'),R=np.eye(3).tolist(),T=[0.,0.,0.],
                        fov_x=np.pi/2,fov_y=np.pi/2)
            argv=['--source',str(root),'--keyframes',str(root/'keyframes.json'),'--point-cloud',str(root/'input.ply'),
                  '--output',str(root/'scene'),'--threads','1']
            with patch('scripts.prepare_lidar_scene.read_colmap_records',return_value={'a.jpg':record}),contextlib.redirect_stdout(io.StringIO()):
                main(argv)
                main(argv)
                with self.assertRaisesRegex(ValueError,'changed'):
                    main(argv+['--voxel-size','.04'])
            report,_=load_scene(root/'scene/scene.json',root/'input.ply',root)
            self.assertEqual(report['geometry_source'],'full_initialization_lidar')
            self.assertFalse(report['diagnostic_only'])
            self.assertEqual(report['relative_margin'],0.)
            dataset=SimpleNamespace(occlusion_mode='mesh',occlusion_geometry=str(root/'scene/scene.json'),
                model_path=str(root),source_path=str(root),keyframe_manifest=str(root/'keyframes.json'))
            prepare_training_geometry(dataset,[record_camera(record,-1,'cpu')])


class SurfaceDiagnosticsTests(unittest.TestCase):
    def test_yellow_band_independent_of_culling_margin_and_all_classes_disjoint(self):
        points=np.array([[0,0,5],[0,0,5.25],[0,0,6],[0,0,4],[0,0,-1]])
        k=np.array([[32,0,31.5],[0,32,31.5],[0,0,1]])
        labels=classify_points_depth(points,k,np.eye(4),torch.full((64,64),5.),.05,.1,surface_band=.03)
        self.assertEqual(labels['on_surface_ids'].tolist(),[0])
        self.assertEqual(labels['uncertain_ids'].tolist(),[1])
        self.assertEqual(labels['culled_ids'].tolist(),[2])
        self.assertEqual(labels['in_front_ids'].tolist(),[3])
        combined=np.concatenate([v for k,v in labels.items() if k!='frustum_ids'])
        self.assertEqual(sorted(combined.tolist()),list(range(len(points))))

    def test_foreground_audit_exposes_missing_and_farther_mesh_without_filling_sparse_depth(self):
        k=np.array([[2,0,1],[0,2,1],[0,0,1]])
        points=np.array([[0,0,4],[0,0,2],[1,0,2],[-1,0,2],[0,0,-1],[np.nan,0,1]])
        depth=projected_lidar_depth(points,k,np.eye(4),3,3,'cpu',chunk_size=2).numpy()
        self.assertEqual(depth[1,1],2.)
        self.assertEqual(np.isfinite(depth).sum(),3)
        mesh=np.full((3,3),np.inf,np.float32);mesh[1,1]=4.;mesh[1,2]=2.
        report,image=first_surface_audit(depth,mesh,np.zeros((3,3,3),np.uint8))
        self.assertEqual(report['counts'],dict(agreement=1,missing_mesh=1,mesh_behind_lidar=1,mesh_in_front_of_lidar=0))
        self.assertAlmostEqual(report['foreground_miss_fraction_of_measured'],2/3)
        np.testing.assert_array_equal(image[1,1],[245,55,55])


if __name__=='__main__':
    unittest.main()
