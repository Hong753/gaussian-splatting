#!/usr/bin/env python3
"""Tiny native CUDA AA render/backward with fixed XYZ and bounded scale updates."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='0')
    p.add_argument('--output', type=Path)
    p.add_argument('--scale-control', choices=('clamp', 'regularize'), default='clamp')
    args = p.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.fixed_lidar import audit, project_scales, scale_regularization
    from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
    if not torch.cuda.is_available():
        raise RuntimeError('Native fixed LiDAR smoke requires CUDA')
    opt_parser = argparse.ArgumentParser()
    params = OptimizationParams(opt_parser)
    opt = params.extract(opt_parser.parse_args(['--fixed_lidar', '--max_scale_ratio', '1.0',
        '--scale_control', args.scale_control,
        '--position_lr_init', '0', '--position_lr_final', '0', '--densify_until_iter', '0']))
    cloud = BasicPointCloud(points=np.array([[-.15,-.1,2],[.15,-.1,2],[-.15,.1,2],[.15,.1,2]], np.float32),
        colors=np.array([[.7,.2,.1],[.2,.6,.2],[.2,.3,.8],[.8,.7,.2]], np.float32),
        normals=np.zeros((4,3), np.float32))
    model = GaussianModel(0)
    model.create_from_pcd(cloud, [SimpleNamespace(image_name='smoke')], 100.)
    model.training_setup(opt)
    original = model.get_xyz.detach().clone()
    original_color = model._features_dc.detach().clone()
    if args.scale_control == 'regularize':
        # Deliberate excess: test the sign of the differentiable penalty and
        # verify projection is disabled after initialization.
        with torch.no_grad():
            model._scaling[0,0] += math.log(1.2)
        gradient = torch.autograd.grad(scale_regularization(model), model._scaling)[0]
        assert gradient[0,0] > 0 and torch.count_nonzero(gradient) == 1
        before = model._scaling.detach().clone()
        project_scales(model)
        assert torch.equal(before, model._scaling), 'Soft penalty must not silently hard-clamp scales'
    camera = SimpleNamespace(image_width=48, image_height=48, FoVx=math.pi/2, FoVy=math.pi/2,
        world_view_transform=torch.eye(4, device='cuda'),
        full_proj_transform=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T,
        camera_center=torch.zeros(3, device='cuda'))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=True, antialiasing=True, occlusion_mode='off')
    for step in range(1, 5):
        model.update_learning_rate(step)
        image = render(camera, model, pipe, torch.zeros(3, device='cuda'))['render']
        assert torch.isfinite(image).all()
        # RGB exercises native backward; the added outward scale pressure ensures
        # the hard constraint is tested even when the photometric gradient shrinks.
        ((image-.6).square().mean() - model._scaling.mean() + scale_regularization(model)).backward()
        assert model._xyz.grad is None
        assert model._features_dc.grad is not None and model._features_dc.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
        model.optimizer.step()
        project_scales(model)
        model.optimizer.zero_grad(set_to_none=True)
        audit(model, step)
    assert torch.equal(original, model.get_xyz)
    assert not torch.equal(original_color, model._features_dc)
    assert model._xyz not in model.optimizer.state
    if args.scale_control == 'regularize':
        assert not audit(model, 4)['scales_within_bounds'], 'Synthetic outward force must be allowed past soft target'
        assert model.optimizer.state[model._scaling]['exp_avg'][0,0] != 0, 'Soft mode must preserve Adam moments'
    restored = GaussianModel(0)
    # Exposure metadata is separate in native checkpoints; match the real Scene
    # initialization before restoring the Gaussian/optimizer state.
    restored._exposure = torch.nn.Parameter(model._exposure.detach().clone())
    restored.restore(model.capture(), opt)
    torch.testing.assert_close(restored.get_scaling, model.get_scaling, atol=0, rtol=0)
    assert torch.equal(restored.get_xyz, original)
    result = dict(passed=True, cuda_device=torch.cuda.get_device_name(), torch=torch.__version__,
        native_aa_forward_backward=True, fixed_centers=True, xyz_adam_state_absent=True,
        appearance_updated=True, scale_bound_enforced=args.scale_control == 'clamp',
        scale_penalty_gradient_checked=args.scale_control == 'regularize', checkpoint_round_trip=True,
        **audit(model, 4))
    if args.output:
        from utils.checkpoint_snapshots import atomic_json
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
