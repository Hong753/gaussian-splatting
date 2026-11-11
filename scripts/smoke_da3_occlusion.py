#!/usr/bin/env python3
"""Native CUDA rasterizer check for explicit training-depth tiles and gradients."""
import argparse
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='3')
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from scripts.occlusion_options import configure_pipeline
    from utils.graphics_utils import getProjectionMatrix
    from utils.sh_utils import RGB2SH
    from utils.occlusion_culling import masked_adam_step
    view = torch.eye(4, device='cuda')
    cam = SimpleNamespace(image_width=128, image_height=128, FoVx=math.pi/2, FoVy=math.pi/2,
        world_view_transform=view, full_proj_transform=getProjectionMatrix(.01, 100., math.pi/2, math.pi/2).cuda().T,
        camera_center=torch.zeros(3, device='cuda'))
    pc = GaussianModel(0)
    parameter = lambda x: torch.nn.Parameter(torch.tensor(x, dtype=torch.float32, device='cuda'))
    pc._xyz = parameter([[0, 0, 1], [0, 0, 3]])
    pc._scaling = parameter([[math.log(.03)]*3]*2)
    pc._rotation = parameter([[1, 0, 0, 0]]*2)
    pc._opacity = parameter([[0]]*2)
    pc._features_dc = torch.nn.Parameter(RGB2SH(torch.tensor([[.8,.2,.1], [.1,.8,.2]], device='cuda'))[:,None,:])
    pc._features_rest = torch.nn.Parameter(torch.empty(2, 0, 3, device='cuda'))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=True)
    configure_pipeline(pipe, dict(occlusion_mode='off'))
    # A local wall covers both projected footprints. Unrelated unknown tiles
    # outside this wall used to poison the coarse depth lookup and keep both.
    tiles, bg = torch.full((8, 8), float('inf'), device='cuda'), torch.zeros(3, device='cuda')
    tiles[3:5, 3:5] = 2.
    result = render(cam, pc, pipe, bg, occlusion_tiles=tiles)
    assert result['occlusion_keep'].tolist() == [True, False]
    result['render'].square().sum().backward()
    for name in ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc'):
        gradient = getattr(pc, name).grad
        assert gradient is not None and gradient[1].count_nonzero().item() == 0, name
    assert pc._features_dc.grad[0].count_nonzero().item() > 0
    hole = tiles.clone()
    hole[3, 4] = float('inf')
    with torch.no_grad():
        uncertain = render(cam, pc, pipe, bg, occlusion_tiles=hole)
        assert uncertain['occlusion_keep'].tolist() == [True, True]
    # Ordinary evaluation ignores the training depth map unless explicitly passed.
    ordinary = render(cam, pc, pipe, bg)
    assert 'occlusion_keep' not in ordinary and ordinary['radii'][1].item() > 0
    assert not torch.allclose(ordinary['render'], result['render'])
    opt = torch.optim.Adam([pc._features_dc], lr=.01)
    pc._features_dc.grad = torch.ones_like(pc._features_dc)
    opt.step()  # Establish nonzero momentum for both rows.
    before = pc._features_dc.detach().clone()
    momentum = opt.state[pc._features_dc]['exp_avg'].clone()
    masked_adam_step(opt, result['occlusion_keep'])
    assert torch.equal(before[1], pc._features_dc[1])
    assert torch.equal(momentum[1], opt.state[pc._features_dc]['exp_avg'][1])
    print('CUDA depth-map smoke passed: exact wall footprint lookup, internal-hole retention, '
          'masked rendering/gradients/Adam, evaluation without target depth')


if __name__ == '__main__':
    main()
