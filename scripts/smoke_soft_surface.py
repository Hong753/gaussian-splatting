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
    torch.testing.assert_close(depth[32,32], depth.new_tensor(2.), atol=2e-4, rtol=0.)
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
    valid = torch.zeros((64,64), dtype=torch.bool, device='cuda')
    valid[32,32] = True
    opacity = render_surface_opacity(camera, pc, pipe, indices)
    loss = surface_opacity_loss(opacity, valid, .98)
    params = [pc._opacity, pc._xyz, pc._scaling, pc._rotation, pc._colors]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    assert torch.isfinite(grads[0]).all()
    assert grads[0][0] < 0 and grads[0][1] == 0, 'Only the associated foreground should gain opacity'
    assert all(g is None for g in grads[1:]), 'Opacity objective must not move/inflate/recolor splats'
    attraction = normal_attraction(pc.get_xyz, indices, refs, .1, 100)
    movement = torch.autograd.grad(attraction, pc._xyz)[0]
    assert movement[0,2] > 0 and movement[1].abs().max() == 0
    assert movement[0,:2].abs().max() < 1e-5
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
