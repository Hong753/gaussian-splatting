#!/usr/bin/env python3
"""Exercise CUDA mesh depth, AA culling, tangent growth and mesh-free inference."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def exercise(backend, directory):
    import numpy as np
    import torch
    from arguments import OptimizationParams
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scripts.occlusion_options import configure_pipeline, inference_pipeline
    from utils.graphics_utils import getProjectionMatrix
    from utils.lidar_occlusion import file_digest
    from utils.occlusion_culling import masked_adam_step
    from utils.scene_occlusion import FORMAT, MeshDepthCache
    from utils.sh_utils import RGB2SH

    # Two connected triangles, rendered by the actual CUDA mesh provider used
    # in training. A safety margin affects clipping, never the tangent plane.
    mesh = directory/'mesh.npz'
    np.savez_compressed(mesh,
        vertices=np.array([[-3,-3,2], [3,-3,2], [3,3,2], [-3,3,2]], np.float32),
        faces=np.array([[0,1,2], [0,2,3]], np.int32))
    scene = directory/'scene.json'
    scene.write_text(json.dumps(dict(format=FORMAT, complete=True, mesh=mesh.name,
        mesh_sha256=file_digest(mesh), relative_margin=.05)))
    provider = MeshDepthCache(scene)
    torch.manual_seed(17)
    cam = SimpleNamespace(image_width=128, image_height=128, FoVx=math.pi/2, FoVy=math.pi/2,
        znear=.01, zfar=100., world_view_transform=torch.eye(4, device='cuda'),
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
    configure_pipeline(pipe, dict(occlusion_mode='mesh', occlusion_backend=backend, occlusion_geometry=str(scene)))
    pipe._occlusion_provider = provider
    frame = provider.surface(cam, 'cuda')
    torch.testing.assert_close(frame['depth'][64, 64], torch.tensor(2., device='cuda'), atol=5e-5, rtol=0)
    assert provider.pixel_depth(cam, 'cuda')[64, 64] > 2.09
    assert list(provider.directory.glob('*.depth.npz')), 'Mesh depth was not persisted'
    bg = torch.zeros(3, device='cuda')
    for _ in range(2):
        result = render(cam, model, pipe, bg)
        result['render'].square().sum().backward()
        assert torch.isfinite(result['render']).all()
        assert result['occlusion_keep'].tolist() == [True, True, False]
        if backend == 'ray':
            assert result['occlusion_fragment_flags'].tolist() == [1, 1, 2]
        for group in model.optimizer.param_groups:
            grad = group['params'][0].grad
            assert grad is not None and torch.isfinite(grad).all(), group['name']
            assert grad[2].count_nonzero().item() == 0, group['name']
        indices = model.observe_view_surface(cam, result['visibility_filter'], frame)
        assert indices.tolist() == [0, 1]
        model.add_densification_stats(result['viewspace_points'], indices)
        model.optimizer.zero_grad(set_to_none=True)
    assert model.xyz_gradient_accum[2].item() == 0 and model.denom[:, 0].tolist() == [2, 2, 0]
    original = model.get_xyz.detach().clone()
    # A close camera must not destroy the measured wall merely because its
    # Gaussians have large projected footprints. Old tangent pruning erased it.
    model.max_radii2D[:2] = 64
    stats = model.densify_and_prune(0., .005, 1., 20, result['radii'])
    assert stats['pruned_screen_size'] == 0 and stats['large_screen_retained'] == 1
    assert (stats['cloned_parents'], stats['split_parents'], stats['points_after']) == (1, 1, 5), stats
    # Retained clone parent, untouched hidden point, then clone and split children.
    torch.testing.assert_close(model.get_xyz[1], original[2])
    torch.testing.assert_close(model.get_xyz[2:, 2], torch.full((3,), 2., device='cuda'))
    assert (model.get_xyz[2]-original[0]).norm().item() > 0
    refs = model.surface_refs
    delta = model.get_xyz[2:]-refs['anchor'][2:]
    distance = (delta*refs['normal'][2:]).sum(-1, keepdim=True)
    assert ((delta-distance*refs['normal'][2:]).norm(dim=1) <= .90001*refs['support'][2:]).all()
    assert len(refs['anchor']) == len(model.get_xyz)
    assert all(len(group['params'][0]) == len(model.get_xyz) for group in model.optimizer.param_groups)

    # Population changes must remain compatible with the next ray render and
    # masked Adam update. The hidden Gaussian is still frozen after growth.
    hidden = {g['name']: g['params'][0][1].detach().clone() for g in model.optimizer.param_groups}
    after = render(cam, model, pipe, bg)
    after['render'].square().sum().backward()
    assert not after['occlusion_keep'][1]
    masked_adam_step(model.optimizer, after['occlusion_keep'])
    model.optimizer.zero_grad(set_to_none=True)
    for group in model.optimizer.param_groups:
        assert torch.isfinite(group['params'][0]).all(), group['name']
        torch.testing.assert_close(group['params'][0][1], hidden[group['name']], atol=0, rtol=0)
    ordinary_pipe = inference_pipeline(pipe, dict(occlusion_mode='mesh', occlusion_backend=backend,
                                                 occlusion_inference_mode='off'))
    ordinary = render(cam, model, ordinary_pipe, bg)
    assert 'occlusion_keep' not in ordinary and torch.isfinite(ordinary['render']).all()
    torch.cuda.synchronize()
    return dict(backend=backend, antialiasing=True, cuda_mesh_first_hit=True,
        raw_surface_without_culling_margin=True, disk_depth_cache=True,
        blocked_gradients_zero=True, hidden_parents_excluded=True,
        bounded_tangent_clone_split=True, large_projected_surface_survives=True, optimizer_after_growth=True,
        mesh_free_inference=True, growth=stats, passed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--backend', choices=('whole', 'ray'), default='ray')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Activate the native CUDA-enabled 3dgs environment')
    print(f'CUDA view-tangent smoke: mesh + {args.backend} culling + AA', flush=True)
    with tempfile.TemporaryDirectory(prefix='mesh-tangent-smoke-') as temporary:
        result = exercise(args.backend, Path(temporary))
    result.update(cuda_device=torch.cuda.get_device_name(), torch=torch.__version__)
    if args.output:
        from scripts.view_comparisons import atomic_json
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
