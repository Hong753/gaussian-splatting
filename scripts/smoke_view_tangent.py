#!/usr/bin/env python3
"""Exercise real CUDA AA rendering, visibility gradients, and tangent growth."""
import argparse
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from arguments import OptimizationParams
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scripts.occlusion_options import configure_pipeline
    from utils.graphics_utils import getProjectionMatrix
    from utils.sh_utils import RGB2SH
    from utils.view_tangent import camera_intrinsics, surface_frame

    if not torch.cuda.is_available():
        raise RuntimeError('Activate the native CUDA-enabled 3dgs environment')
    torch.manual_seed(17)
    cam = SimpleNamespace(image_width=128, image_height=128, FoVx=math.pi/2, FoVy=math.pi/2,
        world_view_transform=torch.eye(4, device='cuda'),
        full_proj_transform=getProjectionMatrix(.01, 100., math.pi/2, math.pi/2).cuda().T,
        camera_center=torch.zeros(3, device='cuda'))
    model = GaussianModel(0)
    par = lambda a: torch.nn.Parameter(torch.tensor(a, dtype=torch.float32, device='cuda'))
    model._xyz = par([[-.2, 0, 2], [.2, .1, 2], [0, 0, 4]])
    model._scaling = par([[math.log(.02)]*3, [math.log(.06)]*3, [math.log(.02)]*3])
    model._rotation = par([[1, 0, 0, 0]]*3)
    model._opacity = par([[0]]*3)
    model._features_dc = torch.nn.Parameter(RGB2SH(torch.tensor([[.8,.2,.1], [.1,.8,.2], [.1,.2,.8]], device='cuda'))[:, None])
    model._features_rest = torch.nn.Parameter(torch.empty(3, 0, 3, device='cuda'))
    model._exposure = torch.nn.Parameter(torch.zeros(1, 3, 4, device='cuda'))
    model.spatial_lr_scale = 1
    model.max_radii2D = torch.zeros(3, device='cuda')
    model.initialize_view_surface()
    opt = OptimizationParams(argparse.ArgumentParser(add_help=False))
    opt.surface_mode, opt.percent_dense, opt.view_tangent_max_points = 'view_tangent', .04, 6
    model.training_setup(opt)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=True)
    configure_pipeline(pipe, dict(occlusion_mode='off'))
    depth = torch.full((128, 128), 2., device='cuda')
    frame = surface_frame(depth, torch.ones_like(depth, dtype=torch.bool), camera_intrinsics(cam, 'cuda'))
    tiles, bg = torch.full((8, 8), 2., device='cuda'), torch.zeros(3, device='cuda')
    for _ in range(2):
        result = render(cam, model, pipe, bg, occlusion_tiles=tiles)
        result['render'].square().sum().backward()
        assert result['occlusion_keep'].tolist() == [True, True, False]
        assert model._features_dc.grad[2].count_nonzero().item() == 0
        indices = model.observe_view_surface(cam, result['visibility_filter'], frame)
        assert indices.tolist() == [0, 1]
        model.add_densification_stats(result['viewspace_points'], indices)
        model.optimizer.zero_grad(set_to_none=True)
    assert model.xyz_gradient_accum[2].item() == 0 and model.denom[:, 0].tolist() == [2, 2, 0]
    original = model.get_xyz.detach().clone()
    stats = model.densify_and_prune(0., .005, 1., None, result['radii'])
    assert (stats['cloned_parents'], stats['split_parents'], stats['points_after']) == (1, 1, 5), stats
    # Retained clone parent, untouched hidden point, then clone and split children.
    torch.testing.assert_close(model.get_xyz[1], original[2])
    torch.testing.assert_close(model.get_xyz[2:, 2], torch.full((3,), 2., device='cuda'))
    assert (model.get_xyz[2]-original[0]).norm().item() > 0
    assert len(model.surface_refs['anchor']) == len(model.get_xyz)
    assert all(len(group['params'][0]) == len(model.get_xyz) for group in model.optimizer.param_groups)
    ordinary = render(cam, model, pipe, bg)
    assert 'occlusion_keep' not in ordinary
    torch.cuda.synchronize()
    print('CUDA view-tangent smoke passed: AA gradients, hidden-row exclusion, bounded clone/split, ordinary novel-view render')


if __name__ == '__main__':
    main()
