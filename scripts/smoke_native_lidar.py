#!/usr/bin/env python3
"""Check native AA RGB gradients, reduced XYZ updates, and unpruned clone/split."""
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
    p.add_argument('--gpu', default='1')
    p.add_argument('--output', type=Path)
    p.add_argument('--percent-dense', type=float, default=.01)
    p.add_argument('--position-lr-divisor', type=float, default=50.)
    p.add_argument('--scaling-lr', type=float, default=.005)
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--preserve-densification-radii', action='store_true')
    args = p.parse_args(argv)
    if not math.isfinite(args.percent_dense) or not 0 < args.percent_dense <= 1:
        p.error('percent-dense must be finite and in (0, 1]')
    if (not math.isfinite(args.position_lr_divisor) or args.position_lr_divisor < 1 or
            not math.isfinite(args.scaling_lr) or args.scaling_lr <= 0 or args.iterations < 1):
        p.error('Invalid learning rate or duration')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.fixed_lidar import project_scales
    from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
    if not torch.cuda.is_available():
        raise RuntimeError('Native LiDAR smoke requires the CUDA-enabled 3dgs environment')
    opt_parser = argparse.ArgumentParser()
    params = OptimizationParams(opt_parser)
    opt = params.extract(opt_parser.parse_args(['--scale_control', 'off',
        '--position_lr_init', str(.00016/args.position_lr_divisor),
        '--position_lr_final', str(.0000016/args.position_lr_divisor),
        '--scaling_lr', str(args.scaling_lr), '--iterations', str(args.iterations),
        '--position_lr_max_steps', str(args.iterations),
        '--disable_opacity_resets', '--disable_pruning', '--percent_dense', str(args.percent_dense)]))
    cloud = BasicPointCloud(points=np.array([[-.15,-.1,2],[.15,-.1,2],[-.15,.1,2],[.15,.1,2]], np.float32),
        colors=np.array([[.7,.2,.1],[.2,.6,.2],[.2,.3,.8],[.8,.7,.2]], np.float32),
        normals=np.zeros((4,3), np.float32))
    model = GaussianModel(0)
    model.create_from_pcd(cloud, [SimpleNamespace(image_name='smoke')], 100.)
    model.training_setup(opt)
    original = model.get_xyz.detach().clone()
    camera = SimpleNamespace(image_width=48, image_height=48, FoVx=math.pi/2, FoVy=math.pi/2,
        world_view_transform=torch.eye(4, device='cuda'),
        full_proj_transform=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T,
        camera_center=torch.zeros(3, device='cuda'))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=True, antialiasing=True, occlusion_mode='off')
    for step in range(1, 4):
        assert model.update_learning_rate(step) > 0
        image = render(camera, model, pipe, torch.zeros(3, device='cuda'))['render']
        assert torch.isfinite(image).all()
        (image-.6).square().mean().backward()
        assert model._xyz.grad is not None and model._xyz.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for g in model.optimizer.param_groups for p in g['params'])
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(original, model.get_xyz)
    # Force both native growth branches and an opacity that would be pruned.
    with torch.no_grad():
        model._scaling[:2] = math.log(opt.percent_dense*.1)
        model._scaling[2:] = math.log(opt.percent_dense*10.)
        model._opacity[0] = -20.
        before_scale = model._scaling.clone()
        project_scales(model)
        assert torch.equal(before_scale, model._scaling)
        model.xyz_gradient_accum[:] = 1
        model.denom[:] = 1
        model.densify_and_prune(.0002, .005, 1., 20, torch.ones(4, device='cuda'),
                               record_stats=True, prune=False, preserve_radii=args.preserve_densification_radii)
    assert len(model.get_xyz) == 8
    assert model.last_densification_stats['cloned_parents'] == 2
    assert model.last_densification_stats['split_parents'] == 2
    assert model.last_densification_stats['pruned'] == 0
    assert (model.get_opacity < .005).any()
    # Newly created rows participate in subsequent Adam updates.
    before = model.get_xyz.detach().clone()
    model._xyz.sum().backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    assert (model.get_xyz.detach() != before).any(dim=1).all()
    # Native pruning removes the two deliberately transparent rows while
    # retaining the supported rows. Opacity resets are a separate switch.
    with torch.no_grad():
        model.xyz_gradient_accum.zero_()
        model.denom[:] = 1
        model.densify_and_prune(.0002, .005, 100., None,
            torch.ones(len(model.get_xyz), device='cuda'), record_stats=True, prune=True,
            preserve_radii=args.preserve_densification_radii)
    assert len(model.get_xyz) == 6 and model.last_densification_stats['pruned'] == 2
    assert (model.get_opacity >= .005).all()
    restored = GaussianModel(0)
    restored._exposure = torch.nn.Parameter(model._exposure.detach().clone())
    restored.restore(model.capture(), opt)
    torch.testing.assert_close(restored.get_xyz, model.get_xyz, atol=0, rtol=0)
    assert restored.get_xyz.requires_grad and restored.fixed_lidar_state is None
    if args.preserve_densification_radii:
        with torch.no_grad():
            # Regression: a 100px measured radius must survive bookkeeping until
            # the 20px screen check, while supported small rows remain present.
            model._scaling[:] = math.log(.001)
            model._opacity[:] = 0
            model.xyz_gradient_accum.zero_()
            model.denom[:] = 1
            model.max_radii2D[:] = 2
            model.max_radii2D[0] = 100
            model.densify_and_prune(.0002, .005, 1., 20, torch.ones(6, device='cuda'),
                record_stats=True, max_world_size=0., preserve_radii=True)
            assert len(model.get_xyz) == 5 and model.last_densification_stats['pruned'] == 1
            assert not model.max_radii2D.any()
            # Disable screen deletion independently; a world-size outlier is
            # still pruned, while other large screen footprints are kept.
            model._scaling[0] = math.log(.02)
            model.denom[:] = 1
            model.max_radii2D[:] = 100
            model.densify_and_prune(.0002, .005, 1., 0., torch.ones(5, device='cuda'),
                record_stats=True, max_world_size=.01, preserve_radii=True)
            assert len(model.get_xyz) == 4 and model.last_densification_stats['pruned'] == 1
    result = dict(passed=True, cuda_device=torch.cuda.get_device_name(), torch=torch.__version__,
        native_aa_rgb_xyz_gradients=True, reduced_lr_updates_xyz=True, no_scale_projection=True,
        clone_and_split=True, pruning_disabled_path_checked=True, native_opacity_pruning=True,
        children_optimized=True, percent_dense=opt.percent_dense,
        position_lr_divisor=args.position_lr_divisor, scaling_lr=opt.scaling_lr,
        radius_history_regression_checked=args.preserve_densification_radii,
        checkpoint_round_trip=True)
    if args.output:
        from utils.checkpoint_snapshots import atomic_json
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
