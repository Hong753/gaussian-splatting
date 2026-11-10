"""Physical occlusion cases, gradient routing, optimizer freezing and cache fidelity."""
import importlib.util
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.occlusion_options import apply_overrides, metric_directory
from utils.occlusion_culling import conservative_keep, depth_pyramid, rectangle_maximum, masked_adam_step
from utils.occlusion_depth import ProxyDepthCache, camera_signature, mesh_clip_vertices, tile_maxima
from utils.graphics_utils import getProjectionMatrix


def camera(size=256, translation=(0, 0, 0)):
    view = torch.eye(4)
    view[3, :3] = torch.tensor(translation)
    projection = getProjectionMatrix(.01, 100., math.pi/2, math.pi/2).T
    return SimpleNamespace(image_width=size, image_height=size, FoVx=math.pi/2, FoVy=math.pi/2,
        znear=.01, zfar=100., world_view_transform=view, full_proj_transform=view @ projection)


class OcclusionTests(unittest.TestCase):
    def test_wall_front_hidden_near_surface_and_large_extent(self):
        points = torch.tensor([[0., 0., 1.], [0., 0., 3.], [0., 0., 2.05], [0., 0., 3.], [0., 0., -.1]])
        scales = torch.full_like(points, .02)
        scales[3, 2] = .5
        rotations = torch.tensor([[1., 0, 0, 0]]).repeat(len(points), 1)
        keep = conservative_keep(camera(), points, scales, rotations, torch.full((16, 16), 2.), chunk_size=2)
        self.assertEqual(keep.tolist(), [True, False, True, True, True])

    def test_holes_and_partial_footprints_are_retained(self):
        points = torch.tensor([[-1., 0., 3.], [0., 0., 3.], [1., 0., 3.]])
        tiles = torch.full((16, 16), 2.)
        tiles[:, 8:] = float("inf")
        keep = conservative_keep(camera(), points, torch.full_like(points, .02),
                                  torch.tensor([[1., 0, 0, 0]]).repeat(3, 1), tiles)
        self.assertEqual(keep.tolist(), [False, True, True])
        keep = conservative_keep(camera(), points, torch.full_like(points, .02),
                                  torch.tensor([[1., 0, 0, 0]]).repeat(3, 1), tiles*float("inf"))
        self.assertTrue(keep.all())

    def test_camera_transform_and_rotated_gaussian_depth_extent(self):
        cam = camera(translation=(0, 0, -10))
        points = torch.tensor([[0., 0., 13.], [0., 0., 13.]])
        scales = torch.tensor([[.5, .01, .01], [.5, .01, .01]])
        rotations = torch.tensor([[1., 0, 0, 0], [math.sqrt(.5), 0, math.sqrt(.5), 0]])
        keep = conservative_keep(cam, points, scales, rotations, torch.full((16, 16), 2.))
        self.assertEqual(keep.tolist(), [False, True])

    def test_mip_query_never_underestimates_any_covered_depth(self):
        torch.manual_seed(7)
        tiles = torch.rand(17, 29)*10
        tiles[5, 8] = float("inf")
        atlas, metadata = depth_pyramid(tiles)
        for _ in range(80):
            xs = torch.randint(0, 29, (2,)).sort().values
            ys = torch.randint(0, 17, (2,)).sort().values
            lo, hi = torch.tensor([[xs[0], ys[0]]]), torch.tensor([[xs[1], ys[1]]])
            bound = rectangle_maximum(atlas, metadata, lo, hi)[0]
            exact = tiles[ys[0]:ys[1]+1, xs[0]:xs[1]+1].max()
            self.assertGreaterEqual(float(bound), float(exact))

    def test_tile_depth_treats_holes_as_unknown_and_erodes_edges(self):
        depth = torch.ones(48, 48)*2
        depth[24, 24] = float("nan")
        tiles = tile_maxima(depth, 1)
        self.assertTrue(torch.isinf(tiles).all())  # Outer padding + center hole cover all 3x3 tiles.
        depth[:] = 2
        self.assertEqual(float(tile_maxima(depth, 1)[1, 1]), 2.)

    def test_mesh_projection_orientation_and_depth(self):
        cam = camera()
        vertices = torch.tensor([[0., -.5, 2.], [0., .5, 2.], [0., 0., .01], [0., 0., 100.]])
        clip = mesh_clip_vertices(vertices, cam)
        ndc = clip[:, :3]/clip[:, 3:]
        self.assertGreater(float(ndc[0, 1]), 0) # Image top -> GL top before output flip.
        self.assertLess(float(ndc[1, 1]), 0)
        torch.testing.assert_close(ndc[2:, 2], torch.tensor([-1., 1.]), atol=1e-6, rtol=1e-6)

    def test_masked_adam_matches_default_for_active_rows_and_freezes_momentum(self):
        torch.manual_seed(9)
        a = torch.nn.Parameter(torch.randn(7, 5, dtype=torch.float64))
        b = torch.nn.Parameter(a.detach().clone())
        reference, masked = torch.optim.Adam([a], lr=.03, eps=1e-15), torch.optim.Adam([b], lr=.03, eps=1e-15)
        for _ in range(4):
            gradient = torch.randn_like(a)
            a.grad, b.grad = gradient.clone(), gradient.clone()
            reference.step()
            masked_adam_step(masked, torch.ones(7, dtype=torch.bool), chunk_size=3)
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        keep = torch.tensor([True, False, True, False, True, False, True])
        snapshot = [b.detach().clone(), masked.state[b]["exp_avg"].clone(), masked.state[b]["exp_avg_sq"].clone()]
        b.grad = torch.ones_like(b)
        masked_adam_step(masked, keep, chunk_size=2)
        for actual, old in zip([b, masked.state[b]["exp_avg"], masked.state[b]["exp_avg_sq"]], snapshot):
            self.assertTrue(torch.equal(actual[~keep], old[~keep]))
        self.assertFalse(torch.equal(b[keep], snapshot[0][keep]))
        # Upstream densification replaces parameter tensors and drops their gradients.
        replacement = torch.nn.Parameter(torch.zeros(11, 5))
        masked.param_groups[0]["params"] = [replacement]
        masked_adam_step(masked, keep) # No stale-index access when gradient is absent.

    def test_cache_reuse_pose_invalidation_and_invalid_cache_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp)/"mesh.ply"
            mesh.write_text("fixture")
            cache = ProxyDepthCache(mesh, Path(tmp)/"cache", edge_pixels=0, memory_views=0)
            cam = camera(32)
            with patch("utils.occlusion_depth.MeshDepthRenderer", return_value=lambda c: torch.ones(c.image_height, c.image_width)*2) as renderer:
                cache.tiles(cam, torch.device("cpu"))
                self.assertEqual(renderer.call_count, 1)
            other = ProxyDepthCache(mesh, Path(tmp)/"cache", edge_pixels=0, memory_views=0)
            with patch("utils.occlusion_depth.MeshDepthRenderer", side_effect=AssertionError("Must load cache")):
                self.assertTrue(torch.equal(other.tiles(cam, torch.device("cpu")), torch.ones(2, 2)*2))
            moved = camera(32, translation=(1, 0, 0))
            self.assertNotEqual(camera_signature(cam, 0), camera_signature(moved, 0))
            self.assertNotEqual(camera_signature(cam, 0), camera_signature(camera(64), 0))
            np.save(other.directory/(camera_signature(cam, 0)+".npy"), np.zeros((2, 2)))
            with self.assertRaisesRegex(ValueError, "Invalid proxy depth"):
                other.tiles(cam, torch.device("cpu"))

    def test_render_overrides_keep_baseline_metrics_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp)/"mesh.ply"
            mesh.touch()
            spec = dict(path=Path(tmp), type="3dgs", cfg={})
            apply_overrides(spec, SimpleNamespace(occlusion="on", occlusion_mesh=mesh))
            self.assertEqual(spec["cfg"]["occlusion_mode"], "proxy")
            target = metric_directory(tmp, 30000, spec["render_variant"])
            self.assertNotEqual(target, metric_directory(tmp, 30000))
            cfg = dict(spec["cfg"])
            spec = dict(path=Path(tmp), type="3dgs", cfg=cfg)
            apply_overrides(spec, SimpleNamespace(occlusion="off"))
            self.assertEqual(spec["render_variant"], "occlusion_off")

    def test_collector_discloses_render_override_separately(self):
        from scripts.collect_ablation_results import main
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/"baseline"
            model.mkdir()
            (model/"cfg_args").write_text("Namespace(model_type='3dgs', antialiasing=True)")
            for variant, value, mode in (("", 20., "off"), ("occlusion_on_test", 23., "proxy")):
                folder = metric_directory(model, 30000, variant)
                folder.mkdir(parents=True)
                (folder/"summary.json").write_text(json.dumps(dict(iteration=30000, antialiasing=True,
                    counts=dict(all=10, selected=3, unselected=7),
                    mean_psnr_db=dict(all=value, selected=value, unselected=value),
                    occlusion=dict(occlusion_mode=mode), render_variant=variant)))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["--root", tmp])
            self.assertIn("| off | on | occlusion_on_test | 23.000 |", output.getvalue())
            self.assertIn("| off | off | saved | 20.000 |", output.getvalue())

    def test_renderer_routes_gradients_and_full_length_visibility(self):
        calls, seen_settings = [], []
        class Rasterizer:
            def __init__(self, raster_settings):
                self.settings = raster_settings
                seen_settings.append(raster_settings)
            def __call__(self, **kwargs):
                calls.append(kwargs)
                self.assert_aa = self.settings.antialiasing
                value = sum(t.sum() for t in kwargs.values() if t is not None)*.001
                return value.expand(3, 8, 8), torch.ones(len(kwargs["means3D"]), dtype=torch.int32), value.expand(1, 8, 8)
        mocks = {"diff_gaussian_rasterization": SimpleNamespace(GaussianRasterizationSettings=lambda **kw: SimpleNamespace(**kw), GaussianRasterizer=Rasterizer),
                 "scene.gaussian_model": SimpleNamespace(GaussianModel=object)}
        module_spec = importlib.util.spec_from_file_location("_occlusion_renderer", Path(__file__).resolve().parents[1]/"gaussian_renderer/__init__.py")
        module = importlib.util.module_from_spec(module_spec)
        with patch.dict(sys.modules, mocks):
            module_spec.loader.exec_module(module)
        zeros_like = torch.zeros_like
        def cpu_zeros(*args, **kwargs):
            kwargs["device"] = "cpu"
            return zeros_like(*args, **kwargs)
        for separate in (True, False):
            for keep in (torch.tensor([True, False]), torch.tensor([False, False])):
                with self.subTest(separate=separate, keep=keep.tolist()):
                    def leaf(*shape): return torch.ones(shape, requires_grad=True)
                    pc = SimpleNamespace(get_xyz=leaf(2,3), get_opacity=leaf(2,1), get_scaling=leaf(2,3), get_rotation=leaf(2,4),
                        get_features=leaf(2,1,3), get_features_dc=leaf(2,1,3), get_features_rest=leaf(2,1,3), active_sh_degree=0)
                    cam = camera(8)
                    cam.camera_center = torch.zeros(3)
                    pipe = SimpleNamespace(occlusion_mode="proxy", antialiasing=True, debug=False, compute_cov3D_python=False, convert_SHs_python=False)
                    before = len(calls)
                    with patch.object(module.torch, "zeros_like", side_effect=cpu_zeros), patch("utils.occlusion_culling.visibility_mask", return_value=keep):
                        result = module.render(cam, pc, pipe, torch.zeros(3), separate_sh=separate)
                    result["render"].sum().backward()
                    self.assertEqual(len(calls)-before, int(keep.any()))
                    self.assertTrue(seen_settings[-1].antialiasing)
                    self.assertEqual(result["radii"].tolist(), keep.to(torch.int).tolist())
                    self.assertTrue(torch.equal(pc.get_xyz.grad[~keep], torch.zeros_like(pc.get_xyz.grad[~keep])))
                    self.assertTrue(torch.equal(result["viewspace_points"].grad[~keep], torch.zeros_like(result["viewspace_points"].grad[~keep])))


if __name__ == "__main__":
    unittest.main()
