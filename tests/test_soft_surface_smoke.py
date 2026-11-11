"""Smoke-test oracles must tolerate depth quantization, not real tangent forces."""
import math
import unittest

import torch

from scripts.smoke_soft_surface import check_normal_pull
from utils.soft_surface import normal_attraction
from utils.view_tangent import camera_intrinsics, match_visible_surface, surface_frame
from test_occlusion import camera


class SoftSurfaceSmokeTests(unittest.TestCase):
    def quantized_wall(self, ulps):
        cam = camera(64)
        near, far, wall = cam.znear, cam.zfar, 2.
        ndc = torch.tensor((far+near)/(far-near)-2*far*near/(far-near)/wall, dtype=torch.float32)
        shifted = ndc.clone()
        for _ in range(ulps):
            shifted = torch.nextafter(shifted, torch.full_like(shifted, float('inf')))
        raster_z = ndc.expand(64,64).clone()
        raster_z[:,32:] = shifted
        depth = 2*near*far/(far+near-raster_z*(far-near))
        frame = surface_frame(depth, torch.isfinite(depth), camera_intrinsics(cam, 'cpu'))
        xyz = torch.tensor([[.03125,.03125,2.02],[.0625,.0625,4.04]], requires_grad=True)
        ids, refs = match_visible_surface(cam, xyz, torch.arange(2), frame, depth_abs=.1, depth_relative=0.)
        self.assertEqual(ids.tolist(), [0])
        return depth, xyz, ids, refs

    def test_quantized_depth_reproduces_false_world_xy_failure(self):
        for ulps in (2,3,4,8):
            with self.subTest(ndc_ulps=ulps):
                depth, xyz, ids, refs = self.quantized_wall(ulps)
                self.assertLess((depth-2).abs().max(), 2e-4)
                gradient = torch.autograd.grad(normal_attraction(xyz, ids, refs, .1, 100), xyz)[0]
                self.assertGreater(gradient[0,:2].abs().max(), 1e-5)  # Original smoke falsely failed.
                metrics = check_normal_pull(gradient, xyz, ids, refs)
                self.assertLess(metrics['relative_tangent_gradient_max'], 1e-6)
                self.assertEqual(metrics['unassociated_gradient_max'], 0.)

    def test_actual_tangent_force_wrong_direction_and_background_updates_fail(self):
        _, xyz, ids, refs = self.quantized_wall(3)
        normal = refs['normal'][0]
        # Construct the tangent geometrically instead of assuming a world axis.
        tangent = torch.cross(normal, normal.new_tensor([0.,1.,0.]), dim=-1)
        tangent /= tangent.norm()
        correct = torch.autograd.grad(normal_attraction(xyz, ids, refs, .1, 100), xyz)[0]
        bad_tangent = correct.clone(); bad_tangent[0] += .01*tangent
        wrong_direction = -correct
        hidden_update = correct.clone(); hidden_update[1,2] = .01
        nonfinite = correct.clone(); nonfinite[0,0] = float('nan')
        for gradients, message in ((bad_tangent, 'tangential'), (wrong_direction, 'toward the plane'),
                                   (hidden_update, 'Unassociated'), (nonfinite, 'Non-finite'),
                                   (torch.zeros_like(correct), 'toward the plane')):
            with self.subTest(message=message), self.assertRaisesRegex(AssertionError, message):
                check_normal_pull(gradients, xyz, ids, refs)

    def test_exact_and_rotated_planes_with_points_on_both_sides(self):
        for normal in (torch.tensor([0.,0.,1.]), torch.tensor([1.,2.,3.])/math.sqrt(14)):
            with self.subTest(normal=normal.tolist()):
                xyz = torch.stack((.02*normal, -.03*normal, 2*normal)).requires_grad_()
                ids = torch.tensor([0,1])
                refs = dict(anchor=torch.zeros(2,3), normal=normal.expand(2,3))
                gradient = torch.autograd.grad(normal_attraction(xyz, ids, refs, .1, 100), xyz)[0]
                metrics = check_normal_pull(gradient, xyz, ids, refs)
                self.assertLess(metrics['relative_tangent_gradient_max'], 1e-6)
                if normal[2] == 1:
                    torch.testing.assert_close(gradient[:,:2], torch.zeros(3,2), atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
