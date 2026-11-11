#!/usr/bin/env python3
"""Actual CUDA test of soft surface opacity, gradient attenuation and attraction."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_normal_pull(gradient, xyz, indices, refs):
    """Check the local plane, not world x/y, including with quantized depth.

    A finite-precision depth image can tilt an otherwise axis-aligned wall
    normal. Correct normal-directed gradients then have nonzero x/y entries.
    """
    import torch
    gradient, xyz = gradient.detach(), xyz.detach()
    normal, anchor = refs['normal'].detach(), refs['anchor'].detach()
    if not all(torch.isfinite(value).all() for value in (gradient, xyz, normal, anchor)):
        raise AssertionError('Non-finite position gradient or surface reference')
    torch.testing.assert_close(normal.norm(dim=-1), normal.new_ones(len(indices)), atol=1e-6, rtol=0.)
    selected = gradient[indices]
    normal_component = (selected*normal).sum(-1)
    tangent = selected-normal_component[:,None]*normal
    lengths = selected.norm(dim=-1)
    tolerance = 32*torch.finfo(gradient.dtype).eps*lengths.clamp_min(1.)
    signed_distance = ((xyz[indices]-anchor)*normal).sum(-1)
    other = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
    other[indices] = False
    other_max = gradient[other].abs().max().item() if other.any() else 0.
    metrics = dict(reference_normals=normal.cpu().tolist(),
        position_gradients=selected.cpu().tolist(), signed_distances=signed_distance.cpu().tolist(),
        tangent_gradient_max=tangent.norm(dim=-1).max().item(),
        relative_tangent_gradient_max=(tangent.norm(dim=-1)/lengths.clamp_min(1e-12)).max().item(),
        unassociated_gradient_max=other_max)
    if (tangent.norm(dim=-1) > tolerance).any():
        raise AssertionError('Position pull has a tangential component: '+json.dumps(metrics))
    displaced = signed_distance.abs() > 1e-7
    if not (normal_component[displaced]*signed_distance[displaced] > 0).all():
        raise AssertionError('Gradient descent does not pull toward the plane: '+json.dumps(metrics))
    if other_max != 0:
        raise AssertionError('Unassociated Gaussians received position gradients: '+json.dumps(metrics))
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='0')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Soft surface smoke requires the real CUDA rasterizers')
    from gaussian_renderer import render
    from utils.graphics_utils import getProjectionMatrix
    from utils.scene_occlusion import MeshDepthRenderer
    from utils.soft_surface import render_surface_opacity, surface_opacity_loss, normal_attraction
    from utils.view_tangent import camera_intrinsics, surface_frame, match_visible_surface

    view = torch.eye(4, device='cuda')
    camera = SimpleNamespace(image_width=64, image_height=64, FoVx=math.pi/2, FoVy=math.pi/2,
        znear=.01, zfar=100.,
        world_view_transform=view, full_proj_transform=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T,
        camera_center=torch.zeros(3, device='cuda'))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'wall.npz'
        np.savez(path, vertices=np.array([[-3,-3,2],[3,-3,2],[3,3,2],[-3,3,2]], np.float32),
            faces=np.array([[0,1,2],[0,2,3]], np.int32))
        depth = MeshDepthRenderer(path, torch.device('cuda'))(camera)
    depth_atol = 2e-4
    torch.testing.assert_close(depth, torch.full_like(depth, 2.), atol=depth_atol, rtol=0.,
        msg='CUDA wall depth differs from the analytic z=2 plane')
    depth_error = (depth-2.).abs().max().item()
    frame = surface_frame(depth, torch.isfinite(depth), camera_intrinsics(camera, 'cuda'))

    class Cloud:
        active_sh_degree = 0
        def __init__(self):
            def param(data):
                return torch.nn.Parameter(torch.tensor(data, dtype=torch.float32, device='cuda'))
            self._xyz = param([[.03125,.03125,2.02],[.0625,.0625,4.04]])
            self._scaling = param([[math.log(.25)]*3,[math.log(.5)]*3])
            self._rotation = param([[1,0,0,0]]*2)
            self._opacity = param([[-2.2],[0.]])
            self._colors = param([[.8,.2,.1],[.1,.8,.2]])
        get_xyz = property(lambda self: self._xyz)
        get_scaling = property(lambda self: self._scaling.exp())
        get_rotation = property(lambda self: torch.nn.functional.normalize(self._rotation, dim=-1))
        get_opacity = property(lambda self: self._opacity.sigmoid())

    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
        debug=True, antialiasing=True, occlusion_mode='off')
    bg = torch.zeros(3, device='cuda')
    pc = Cloud()
    indices, refs = match_visible_surface(camera, pc.get_xyz, torch.arange(2, device='cuda'), frame,
        depth_abs=.1, depth_relative=0.)
    assert indices.tolist() == [0], 'Background room must not enter surface association'
    # Depth accuracy and gradient direction are separate contracts. Bound the
    # normal's angular error by the allowed endpoint depth errors over the
    # six-pixel finite-difference baseline (surface_frame radius = 3).
    span = 6*2./max(frame['intrinsic'][0,0].item(), frame['intrinsic'][1,1].item())
    normal_tolerance = 2*math.sqrt(2)*depth_atol/span
    normal_error = (refs['normal']-refs['normal'].new_tensor([0.,0.,1.])).norm(dim=-1).max().item()
    assert normal_error <= normal_tolerance, (
        f'Wall normal error {normal_error:.6g} exceeds depth-precision bound {normal_tolerance:.6g}; '
        f'normal={refs["normal"].tolist()}, depth_error={depth_error:.6g}')
    valid = torch.zeros((64,64), dtype=torch.bool, device='cuda')
    valid[32,32] = True
    opacity = render_surface_opacity(camera, pc, pipe, indices)
    loss = surface_opacity_loss(opacity, valid, .98)
    params = [pc._opacity, pc._xyz, pc._scaling, pc._rotation, pc._colors]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    assert torch.isfinite(grads[0]).all()
    assert grads[0][0] < 0 and grads[0][1] == 0, 'Only the associated foreground should gain opacity'
    assert all(g is None for g in grads[1:]), 'Opacity objective must not move/inflate/recolor splats'
    # The exact analytic plane must still produce exactly zero x/y gradients.
    exact_refs = dict(anchor=pc._xyz.new_tensor([[0.,0.,2.]]),
        normal=pc._xyz.new_tensor([[0.,0.,1.]]))
    analytic_gradient = torch.autograd.grad(normal_attraction(pc.get_xyz, indices, exact_refs, .1, 100), pc._xyz)[0]
    analytic_metrics = check_normal_pull(analytic_gradient, pc.get_xyz, indices, exact_refs)
    torch.testing.assert_close(analytic_gradient[0,:2], torch.zeros_like(analytic_gradient[0,:2]), atol=0., rtol=0.)
    attraction = normal_attraction(pc.get_xyz, indices, refs, .1, 100)
    gradient = torch.autograd.grad(attraction, pc._xyz)[0]
    measured_metrics = check_normal_pull(gradient, pc.get_xyz, indices, refs)
    print('CUDA soft surface position checks: '+json.dumps(dict(
        wall_depth_error=depth_error, wall_normal_error=normal_error,
        analytic_plane=analytic_metrics, rasterized_plane=measured_metrics)), flush=True)
    assert not surface_opacity_loss(opacity, torch.zeros_like(valid), .98).requires_grad

    def background_color_gradient():
        image = render(camera, pc, pipe, bg, override_color=pc._colors)['render']
        assert torch.isfinite(image).all()
        return float(torch.autograd.grad(image[0,32,32], pc._colors)[0][1,0])

    before_gradient = background_color_gradient()
    before_alpha = float(opacity[32,32].detach())
    optimizer = torch.optim.Adam([pc._opacity], lr=.15)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        coverage = render_surface_opacity(camera, pc, pipe, indices)
        surface_opacity_loss(coverage, valid, .98).backward()
        assert torch.isfinite(pc._opacity.grad).all()
        optimizer.step()
    after_alpha = float(render_surface_opacity(camera, pc, pipe, indices)[32,32].detach())
    after_gradient = background_color_gradient()
    assert after_alpha > before_alpha+.3
    assert 0 < after_gradient < .5*before_gradient, 'Soft opacity must attenuate, not hard-gate, rear gradients'
    report = dict(passed=True, cuda_device=torch.cuda.get_device_name(), torch=torch.__version__,
        antialiasing=True, mesh_first_hit_correct=True, background_excluded_from_surface=True,
        opacity_only_auxiliary_gradients=True, normal_only_position_pull=True, unknown_pixels_skipped=True,
        wall_depth_max_error=depth_error, wall_normal_error=normal_error,
        analytic_position_check=analytic_metrics, rasterized_position_check=measured_metrics,
        surface_alpha_before=before_alpha, surface_alpha_after=after_alpha,
        rear_color_gradient_before=before_gradient, rear_color_gradient_after=after_gradient,
        rgb_render_without_mesh=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
