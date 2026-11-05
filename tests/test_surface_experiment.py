import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.surface_geometry import estimate_patches, points_digest


def stub_module(name, **attrs):
    module = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(module, k, v)
    return module


def rotation(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=1).reshape(-1,3,3)


def load_model():
    modules = {
        'utils.general_utils': stub_module('utils.general_utils', inverse_sigmoid=lambda x: x,
                    get_expon_lr_func=lambda *a, **kw: lambda _: 0, build_rotation=rotation,
                    strip_symmetric=lambda x: x, build_scaling_rotation=lambda x, y: x),
        'utils.system_utils': stub_module('utils.system_utils', mkdir_p=lambda x: None),
        'utils.sh_utils': stub_module('utils.sh_utils', RGB2SH=lambda x: x),
        'utils.graphics_utils': stub_module('utils.graphics_utils', BasicPointCloud=object),
        'plyfile': stub_module('plyfile', PlyData=object, PlyElement=object),
        'simple_knn': stub_module('simple_knn'),
        'simple_knn._C': stub_module('simple_knn._C', distCUDA2=lambda x: x),
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('test_gaussian_model', ROOT / 'scene/gaussian_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.GaussianModel


class SurfaceExperimentTests(unittest.TestCase):
    def test_pca_plane_and_line_confidence(self):
        x, y = np.meshgrid(np.arange(8), np.arange(8))
        pts = np.stack((x.ravel(), y.ravel(), np.ones(x.size)), axis=1).astype(np.float32)
        batches = list(estimate_patches(pts, neighbors=12, batch=13, workers=1))
        normals = np.concatenate([x[1] for x in batches])
        confidence = np.concatenate([x[2] for x in batches])
        self.assertGreater(np.abs(normals[:, 2]).min(), .999)
        self.assertGreater(confidence.min(), .99)
        line = np.stack((np.arange(32), np.zeros(32), np.zeros(32)), axis=1).astype(np.float32)
        self.assertEqual(np.concatenate([x[2] for x in estimate_patches(line, neighbors=8, workers=1)]).max(), 0)
        self.assertNotEqual(points_digest(pts), points_digest(pts[::-1]))

    def model(self):
        Model = load_model()
        m = Model(0)
        def par(x): return torch.nn.Parameter(torch.tensor(x, dtype=torch.float32))
        m._xyz = par([[0,0,0], [5,0,0]])
        m._features_dc = par([[[0,0,0]], [[0,0,0]]])
        m._features_rest = par([[], []]).reshape(2,0,1)
        m._features_rest = torch.nn.Parameter(m._features_rest)
        m._opacity = par([[0], [0]])
        m._scaling = par([[0,0,-2], [0,0,0]])
        m._rotation = par([[1,0,0,0],[1,0,0,0]])
        m.surface_mode = 'constrained'
        m.surface_refs = {'anchor':m._xyz.detach().clone(), 'normal':torch.tensor([[0.,0,1],[0,0,1]]),
                          'confidence':torch.tensor([1.,0]), 'support':torch.tensor([100.,100.])}
        m.percent_dense = .01
        m.tmp_radii = torch.ones(2)
        m.max_radii2D = torch.zeros(2)
        m.xyz_gradient_accum = torch.zeros((2,2))
        m.denom = torch.zeros((2,1))
        m.use_absgs = True
        m.optimizer = torch.optim.Adam([{'params':[getattr(m, attr)],'name':name} for name,attr in
             [('xyz','_xyz'),('f_dc','_features_dc'),('f_rest','_features_rest'),
              ('opacity','_opacity'),('scaling','_scaling'),('rotation','_rotation')]])
        return m

    def test_initialize_surface_and_reject_wrong_cloud(self):
        m = self.model()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / 'metadata.json').write_text(json.dumps({'points': 2,
                'points_sha256_float32': points_digest(m.get_xyz.detach().numpy())}))
            np.save(path / 'normals.npy', np.array([[1,0,0], [0,0,1]], np.float32))
            np.save(path / 'confidence.npy', np.array([.9,.2], np.float32))
            np.save(path / 'support.npy', np.array([1,1], np.float32))
            m.initialize_surface(path, 'constrained', .6, .2)
            axes = rotation(m._rotation)[:, :, 2]
            self.assertTrue(torch.allclose(axes[0], torch.tensor([1.,0,0]), atol=1e-6))
            self.assertAlmostEqual(m.get_scaling[0,2].item(), np.exp(-2)*.2, places=6)
            self.assertTrue(torch.allclose(m._rotation[1], torch.tensor([1.,0,0,0])))
            m._xyz.data[0,0] += 1
            with self.assertRaises(ValueError):
                m.initialize_surface(path, 'constrained')

    def test_tangent_split_and_prune_references(self):
        m = self.model()
        original_zeros = torch.zeros
        def cpu_zeros(*a, **kw):
            if kw.get('device') == 'cuda': kw['device'] = 'cpu'
            return original_zeros(*a, **kw)
        torch.manual_seed(4)
        with mock.patch.object(torch, 'zeros', side_effect=cpu_zeros):
            m.densify_and_split(torch.tensor([[1.],[0.]]), .5, 1, N=2)
        self.assertEqual(len(m.get_xyz), 3)
        self.assertTrue(torch.allclose(m.get_xyz[1:,2], torch.zeros(2)))
        self.assertTrue(torch.equal(m.surface_refs['confidence'], torch.tensor([0.,1.,1.])))
        self.assertEqual(m.xyz_gradient_accum.shape, (3, 2))
        self.assertTrue(torch.allclose(m._scaling[1:,2], torch.full((2,),-2.)))
        m.prune_points(torch.tensor([False, True, False]))
        self.assertEqual(len(m.surface_refs['anchor']), len(m.get_xyz))
        self.assertTrue(torch.equal(m.surface_refs['confidence'], torch.tensor([0.,1.])))

    def test_checkpoint_restores_surface_refs(self):
        m = self.model()
        m._exposure = torch.nn.Parameter(torch.zeros(1, 3, 4))
        m.spatial_lr_scale = 1
        args = types.SimpleNamespace(surface_mode='constrained', percent_dense=.01, absgs=True,
            position_lr_init=.001, position_lr_final=.00001, position_lr_delay_mult=.01,
            position_lr_max_steps=100, feature_lr=.01, opacity_lr=.01, scaling_lr=.01,
            rotation_lr=.01, exposure_lr_init=.01, exposure_lr_final=.001,
            exposure_lr_delay_steps=0, exposure_lr_delay_mult=0, iterations=100)
        state = m.capture()
        restored = self.model()
        restored._exposure = torch.nn.Parameter(torch.zeros(1, 3, 4))
        real_zeros = torch.zeros
        def cpu_zeros(*a, **kw):
            if kw.get('device') == 'cuda': kw['device'] = 'cpu'
            return real_zeros(*a, **kw)
        with mock.patch.object(torch, 'zeros', side_effect=cpu_zeros):
            restored.restore(state, args)
        self.assertTrue(torch.equal(restored.surface_refs['normal'], m.surface_refs['normal']))
        self.assertEqual(restored.surface_mode, 'constrained')
        args.surface_mode = 'tangent'
        with self.assertRaises(ValueError):
            restored.restore(state, args)

    def test_surface_loss_penalizes_normal_motion_and_not_tangent(self):
        m = self.model()
        m.surface_refs['support'][:] = 10
        m._xyz.data[0,0] = 1
        loss = m.surface_loss(torch.tensor([[0]]), 1, 0, 0, 100)
        self.assertAlmostEqual(loss.item(), 0.)
        m._xyz.data[0,2] = 2
        loss = m.surface_loss(torch.tensor([[0]]), 1, 0, 0, 100)
        self.assertAlmostEqual(loss.item(), .04, places=6)
        loss.backward()
        self.assertGreater(m._xyz.grad[0,2].item(), 0)
        self.assertEqual(m._xyz.grad[0,0].item(), 0)


if __name__ == '__main__':
    unittest.main()
