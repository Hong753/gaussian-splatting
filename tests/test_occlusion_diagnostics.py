import importlib.util
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from utils.occlusion_diagnostics import (COLORS, classify, display_indices, depth_points, depth_rgb,
    projected_overlay, write_colored_ply, nearest_ray_point, write_report, reference_on_image,
    reference_panels, reason_counts)
from utils.occlusion_culling import conservative_keep
from scripts.occlusion_inspector_backend import gradient_statistics, probe_verdict, InspectorBackend, PARAMETERS
from test_occlusion import camera


def synthetic_view():
    """Small explicitly synthetic two-layer scene for the actual viewer API."""
    cam = camera(128)
    cam.image_name = 'SYNTHETIC wall test'
    x, y = np.meshgrid(np.linspace(-.35, .35, 13), np.linspace(-.35, .35, 13))
    front = np.stack((x.ravel(), y.ravel(), np.full(x.size, 1.5)), 1)
    back = np.stack((2*x.ravel(), 2*y.ravel(), np.full(x.size, 3.)), 1)
    xyz = np.concatenate((front, back, [[20, 20, 1]])).astype(np.float32)
    rotations = torch.tensor([[1., 0, 0, 0]]).repeat(len(xyz), 1)
    tiles = torch.full((8, 8), 2.)
    keep = conservative_keep(cam, torch.from_numpy(xyz), torch.full((len(xyz), 3), .015), rotations, tiles).numpy()
    off = np.full(len(xyz), 3); off[-1] = 0
    on = np.where(keep, off, 0)
    labels, counts = classify(keep, on, off)
    rgb = np.full((128, 128, 3), 35, np.uint8)
    k = np.array([[64, 0, 63.5], [0, 64, 63.5], [0, 0, 1]])
    overlay = projected_overlay(xyz, labels, np.eye(4), k, rgb)
    images = {key: (key+' (SYNTHETIC)', overlay if key == 'centers' else rgb)
              for key in ('gt', 'culling_on', 'culling_off', 'difference', 'centers', 'depth')}
    summary = dict(image=cam.image_name, counts=counts, culled_fraction_all=counts['culled']/len(xyz),
        culled_fraction_raster_candidates=counts['culled_with_raster_footprint']/counts['raster_candidates_without_culling'],
        reliable_tile_fraction=1., gradient_probe=None)
    surface = depth_points(np.full((128, 128), 2.), k, np.eye(4), stride=5)
    return xyz, dict(name=cam.image_name, camera=cam, c2w=np.eye(4), tiles=tiles, labels=labels,
                    radii_on=on, radii_off=off, images=images, summary=summary, surface=surface)


class OcclusionDiagnosticTests(unittest.TestCase):
    def test_partition_and_actual_candidate_denominator(self):
        keep = np.array([True, False, True, False])
        labels, counts = classify(keep, [4, 0, 0, 0], [4, 5, 0, 0])
        self.assertEqual(labels.tolist(), [1, 2, 0, 2])
        self.assertEqual((counts['total'], counts['culled'], counts['culled_with_raster_footprint']), (4, 2, 1))
        self.assertEqual(sum(counts[k] for k in ('active', 'culled', 'other')), 4)
        with self.assertRaises(RuntimeError): classify(keep, [4, 2, 0, 0], [4, 5, 0, 0])
        with self.assertRaises(RuntimeError): classify(keep, [3, 0, 0, 0], [4, 5, 0, 0])

    def test_no_sampling_by_default_and_complete_export_with_ids(self):
        labels = np.array([1, 2, 0, 2, 1], np.uint8)
        xyz = np.arange(15).reshape(5, 3).astype(np.float32)
        self.assertEqual(display_indices(labels, 2).tolist(), [1, 3])
        self.assertEqual(len(display_indices(labels, 2, 1)), 1)
        with tempfile.TemporaryDirectory() as tmp:
            from plyfile import PlyData
            path = Path(tmp)/'points.ply'
            self.assertEqual(write_colored_ply(path, xyz, labels, chunk=2), 4)
            data = PlyData.read(path)['vertex']
            self.assertEqual(data['gaussian_id'].tolist(), [0, 1, 3, 4])
            self.assertEqual(data['label'].tolist(), [1, 2, 2, 1])
            np.testing.assert_equal(data['red'], COLORS[labels[[0, 1, 3, 4]], 0])

    def test_depth_geometry_uses_camera_z_and_c2w(self):
        depth = np.array([[2., np.inf], [0., 4.]])
        k = np.eye(3)
        pose = np.eye(4); pose[:3, 3] = [10, 20, 30]
        points = depth_points(depth, k, pose, 1)
        np.testing.assert_allclose(points, [[10, 20, 32], [14, 24, 34]])
        color, _ = depth_rgb([[2., np.inf]], 32, 16)
        self.assertTrue((color[:, 16:] == 0).all())

    def test_details_are_from_the_same_conservative_predicate(self):
        xyz = torch.tensor([[0., 0, 1.], [0, 0, 3.], [0, 0, 2.05]])
        scales = torch.full_like(xyz, .02)
        rot = torch.tensor([[1., 0, 0, 0]]).repeat(3, 1)
        tiles = torch.full((8, 8), 2.)
        plain = conservative_keep(camera(128), xyz, scales, rot, tiles)
        detailed, info = conservative_keep(camera(128), xyz, scales, rot, tiles, return_details=True, chunk_size=1)
        torch.testing.assert_close(plain, detailed)
        self.assertEqual(info['center_uv'].shape, (3, 2))
        self.assertGreater(info['nearest_support_z'][1].item(), info['surface_bound_z'][1].item()+.05)
        with self.assertRaises(ValueError):
            conservative_keep(camera(128), xyz[:0], scales[:0], rot[:0], tiles, return_details=True)

    def test_backward_verdict_nontrivial_and_zero_hidden_gradients(self):
        hidden = torch.tensor([False, True, True])
        on = gradient_statistics(['_xyz'], [torch.tensor([[1., 0, 0], [0, 0, 0], [0, 0, 0]])], hidden, chunk=1)
        off = gradient_statistics(['_xyz'], [torch.ones(3, 3)], hidden)
        self.assertEqual(on['culled_nonzero_gaussians'], 0)
        self.assertEqual(off['culled_nonzero_gaussians'], 2)
        self.assertTrue(probe_verdict(on, off, 2).startswith('PASS'))
        self.assertTrue(probe_verdict(off, off, 2).startswith('FAILED'))
        zero = gradient_statistics(['_xyz'], [torch.zeros(3, 3)], hidden)
        self.assertTrue(probe_verdict(zero, zero, 2).startswith('INCONCLUSIVE'))
        self.assertTrue(probe_verdict(on, off, 0).startswith('INCONCLUSIVE'))

    def test_report_and_non_zbuffered_overlay_show_hidden_centers(self):
        xyz, view = synthetic_view()
        self.assertGreater(view['summary']['counts']['active'], 0)
        self.assertGreater(view['summary']['counts']['culled'], 0)
        self.assertTrue((view['images']['centers'][1][..., 0] > 100).any())
        with tempfile.TemporaryDirectory() as tmp:
            write_report(tmp, view['summary'], view['images'])
            self.assertTrue((Path(tmp)/'index.html').is_file())
            report = json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(report['counts']['total'], len(xyz))
            self.assertEqual(len(list(Path(tmp).glob('*.png'))), 6)

    def test_backend_backward_runs_on_and_off_without_changing_weights(self):
        backend = InspectorBackend.__new__(InspectorBackend)
        backend.torch, backend.is_da3, backend.cfg = torch, False, {}
        backend.parameters = [torch.nn.Parameter(torch.full((3, 1), .01), requires_grad=False) for _ in PARAMETERS]
        backend.model, backend.background = object(), torch.zeros(3)
        backend.pipe, backend.off_pipe = object(), object()
        before = [p.detach().clone() for p in backend.parameters]
        keep = torch.tensor([True, False, True])
        def render(cam, model, pipe, bg, use_trained_exp, **kwargs):
            self.assertFalse(use_trained_exp)
            enabled = pipe is backend.pipe
            self.assertEqual('occlusion_tiles' in kwargs, enabled)
            selected = keep if enabled else torch.ones_like(keep)
            image = sum(p[selected].sum() for p in backend.parameters).expand(3, 16, 16)
            result = dict(render=image, radii=selected.int()*3)
            if enabled: result['occlusion_keep'] = keep
            return result
        backend.render = render
        cam = SimpleNamespace(original_image=torch.zeros(3, 16, 16), alpha_mask=None)
        on_image, on_radius, mask, on_grad = backend.render_state(cam, torch.ones(1, 1), True, True)
        off_image, off_radius, _, off_grad = backend.render_state(cam, torch.ones(1, 1), False, True, ~keep)
        self.assertGreater((off_image-on_image).abs().mean().item(), 0)
        self.assertEqual(classify(mask, on_radius, off_radius)[0].tolist(), [1, 2, 1])
        self.assertEqual(on_grad['culled_nonzero_gaussians'], 0)
        self.assertEqual(off_grad['culled_nonzero_gaussians'], 1)
        for old, parameter in zip(before, backend.parameters):
            torch.testing.assert_close(old, parameter)
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_ray_picker_maps_to_original_checkpoint_row(self):
        xyz = np.array([[2, 0, 2], [0, 0, -1], [0, 0, 4], [3, 1, 1.]])
        self.assertEqual(nearest_ray_point(xyz, np.array([1, 2, 3]), [0, 0, 0], [0, 0, 1], chunk=1), 2)

    def test_dense_wall_culls_back_layer_where_floor_patches_leave_unknown(self):
        from utils.occlusion_depth import tile_maxima
        xyz = torch.tensor([[0., 0, 3.], [0., 0, 1.], [0., 0, .1]])
        scale = torch.full_like(xyz, .02)
        rot = torch.tensor([[1., 0, 0, 0]]).repeat(3, 1)
        sparse = torch.full((128, 128), float('inf')); sparse[96:120] = 2
        dense = torch.full_like(sparse, 2.)
        kept_sparse, a = conservative_keep(camera(128), xyz, scale, rot, tile_maxima(sparse), return_details=True)
        kept_dense, b = conservative_keep(camera(128), xyz, scale, rot, tile_maxima(dense), return_details=True)
        self.assertEqual(kept_sparse.tolist(), [True, True, True])
        self.assertEqual(kept_dense.tolist(), [False, True, True])
        reasons = reason_counts(kept_sparse, a['reliable'], a['surface_bound_z'], [3, 3, 3])
        self.assertEqual(reasons['missing_depth_coverage'], 2)
        self.assertEqual(reasons['outside_conservative_test'], 1)
        self.assertEqual(sum(reasons.values()), 3)
        reasons = reason_counts(kept_dense, b['reliable'], b['surface_bound_z'], [3, 3, 3])
        self.assertEqual(reasons['culled'], 1)
        self.assertEqual(reasons['overlaps_depth_or_margin'], 1)
        self.assertEqual(reasons['missing_depth_coverage'], 0)

    def test_reference_panels_use_intrinsics_and_keep_rejected_depth_distinct(self):
        frame = dict(depth=np.array([[2., 4.], [3., 5.]], np.float32),
            valid=np.array([[True, False], [True, True]]), intrinsics=np.eye(3))
        # Deliberately offset principal point; naive resizing would miss this.
        native = np.array([[1., 0, 1.], [0, 1., 0], [0, 0, 1.]])
        raw, valid = reference_on_image(frame, native, 2, 3)
        np.testing.assert_allclose(raw, [[0, 2, 4], [0, 3, 5]])
        self.assertFalse(valid[0, 2]); self.assertTrue(valid[0, 1])
        frame['anchors'] = dict(u=np.array([1, 2, 2]), v=np.array([0, 1, 0]), z=np.array([2, 9, 4.]))
        panels, stats = reference_panels(frame, native, np.full((2, 3, 3), 100, np.uint8), [[np.inf]])
        self.assertEqual(stats['lidar_anchors'], dict(projected=3, agree=1, disagree=1, untrusted=1))
        self.assertEqual(stats['trusted_pixel_fraction'], .5)
        self.assertEqual(stats['usable_tile_pixel_fraction'], 0)
        self.assertIn('anchors', panels)

    def test_two_references_use_same_checkpoint_and_do_not_overwrite_reports(self):
        from utils.occlusion_depth import tile_maxima
        backend = InspectorBackend.__new__(InspectorBackend)
        cam = camera(128)
        cam.image_name = 'test.png'
        cam.original_image, cam.alpha_mask = torch.zeros(3, 128, 128), None
        xyz = torch.nn.Parameter(torch.tensor([[0., 0, 1.], [0., 0, 3.]]), requires_grad=False)
        backend.model = SimpleNamespace(get_xyz=xyz,
            _scaling=torch.nn.Parameter(torch.full((2, 3), .02).log(), requires_grad=False),
            _rotation=torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]]).repeat(2, 1), requires_grad=False))
        backend.parameters = [xyz, torch.nn.Parameter(torch.ones(2, 1), requires_grad=False),
            backend.model._scaling, backend.model._rotation,
            torch.nn.Parameter(torch.ones(2, 1), requires_grad=False),
            torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)]
        backend.xyz, backend.cfg, backend.background = xyz.detach().numpy(), {}, torch.zeros(3)
        backend.pipe = SimpleNamespace(occlusion_margin=.05, occlusion_sigma=3., occlusion_chunk_size=1, antialiasing=True)
        backend.off_pipe = object()
        backend.torch, backend.current, backend.views = torch, None, {}
        backend.source_key, backend.names, backend.iteration = 'da3', [cam.image_name], 7000
        backend.saved_reference, backend.checkpoint_sha256 = 'LiDAR patches', 'synthetic-checkpoint'
        backend.camera = lambda name: cam
        before = [p.detach().clone() for p in backend.parameters]
        class Provider:
            def __init__(self, dense):
                self.depth = torch.full((128, 128), 2. if dense else float('inf'))
            def tiles(self, camera, device): return tile_maxima(self.depth)
            def diagnostic_frame(self, camera, device):
                return dict(depth=self.depth.numpy(), valid=torch.isfinite(self.depth).numpy(),
                    intrinsics=np.array([[64., 0, 63.5], [0, 64., 63.5], [0, 0, 1.]]))
        backend.providers = {'saved': Provider(False), 'da3': Provider(True)}
        backend.source_options = {'saved': 'Saved LiDAR patches', 'da3': 'Diagnostic LiDAR + DA3'}
        calls = []
        def render(camera, model, pipe, bg, use_trained_exp, **kwargs):
            tiles = kwargs.get('occlusion_tiles')
            calls.append(tiles is not None)
            keep = (torch.ones(2, dtype=torch.bool) if tiles is None else conservative_keep(
                cam, xyz.detach(), model._scaling.detach().exp(), model._rotation.detach(), tiles))
            value = sum(p[keep].sum() for p in backend.parameters)*.001
            result = dict(render=value.expand(3, 128, 128), radii=keep.int()*3)
            if tiles is not None: result['occlusion_keep'] = keep
            return result
        backend.render = render
        with tempfile.TemporaryDirectory() as tmp:
            backend.root, backend.ply = Path(tmp), Path(tmp)/'immutable.ply'
            result = backend.analyze(cam.image_name, True, 'da3')
            self.assertEqual(result['labels'].tolist(), [1, 2])
            self.assertEqual(backend.views['saved']['labels'].tolist(), [1, 1])
            self.assertEqual(result['summary']['candidate_reasons']['culled'], 1)
            self.assertEqual(result['summary']['gradient_probe']['on']['culled_nonzero_gaussians'], 0)
            self.assertEqual(result['summary']['gradient_probe']['off']['culled_nonzero_gaussians'], 1)
            self.assertEqual(len(result['summary']['reference_comparison']), 2)
            self.assertTrue(result['summary']['diagnostic_override'])
            self.assertFalse(backend.views['saved']['summary']['diagnostic_override'])
            self.assertEqual(len(list(Path(tmp).rglob('summary.json'))), 2)
            count = len(calls)
            backend.analyze(cam.image_name, False, 'saved')
            self.assertEqual(len(calls), count)  # UI reference changes reuse the same state.
        for old, param in zip(before, backend.parameters):
            torch.testing.assert_close(param, old)
            self.assertFalse(param.requires_grad)

    @unittest.skipUnless(importlib.util.find_spec('viser'), 'optional viewer dependency')
    def test_real_viser_layer_visibility_cap_and_ply_export(self):
        import viser
        from scripts.inspect_occlusion import InspectorViewer
        xyz, result = synthetic_view()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            result['directory'] = Path(tmp)
            backend = SimpleNamespace(names=[result['name']], args=SimpleNamespace(image=None),
                xyz=xyz, current=None, iteration=7000, torch=torch,
                source_options={'saved': 'Synthetic saved reference'}, source_key='saved',
                model=SimpleNamespace(get_xyz=torch.from_numpy(xyz), _scaling=torch.full((len(xyz), 3), .015).log(),
                    _rotation=torch.tensor([[1., 0, 0, 0]]).repeat(len(xyz), 1)),
                pipe=SimpleNamespace(occlusion_margin=.05, occlusion_sigma=3., occlusion_chunk_size=131072))
            backend.gaussian_details = lambda index: InspectorBackend.gaussian_details(backend, index)
            def analyze(name, check, source):
                backend.current = result
                return result
            backend.analyze = analyze
            server = viser.ViserServer(host='127.0.0.1', port=port, verbose=False)
            try:
                viewer = InspectorViewer(server, backend)
                thread = viewer.inspect(result['name']); thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
                self.assertTrue(viewer.ready)
                self.assertEqual(len(viewer.nodes['active'].points), result['summary']['counts']['active'])
                self.assertEqual(len(viewer.nodes['culled'].points), result['summary']['counts']['culled'])
                self.assertNotIn('other', viewer.nodes)
                viewer.limit.value = 10; viewer.update_layers()
                self.assertEqual(len(viewer.nodes['culled'].points), 10)
                viewer.layers['active'].value = False; viewer.update_layer('active')
                self.assertNotIn('active', viewer.nodes)
                viewer.inspect_gaussian(200).join(timeout=10)
                self.assertIn('selected', viewer.nodes)
                self.assertIn('ray', viewer.nodes)
                self.assertIn('Culled:', viewer.details.content)
                viewer.export_ply().join(timeout=10)
                from plyfile import PlyData
                data = PlyData.read(Path(tmp)/'active_and_culled.ply')['vertex']
                self.assertEqual(len(data), result['summary']['counts']['active']+result['summary']['counts']['culled'])
            finally:
                server.stop()


if __name__ == '__main__':
    unittest.main()
