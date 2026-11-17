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


def check_empty_rendering(render, device='cuda', antialiasing=True):
    """A globally empty model differs from a nonempty, invisible camera view."""
    import torch
    from utils.graphics_utils import getProjectionMatrix
    camera=SimpleNamespace(image_width=8,image_height=8,FoVx=math.pi/2,FoVy=math.pi/2,
        world_view_transform=torch.eye(4,device=device),
        full_proj_transform=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).to(device).T,
        camera_center=torch.zeros(3,device=device))
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,antialiasing=antialiasing,
                         debug=True,occlusion_mode='off')
    for n in (0,2):
        xyz=torch.tensor([[0.,0.,-2.]]*n,device=device).reshape(n,3).requires_grad_()
        values=dict(get_xyz=xyz,get_features=torch.zeros(n,16,3,device=device,requires_grad=True),
            get_opacity=torch.full((n,1),.1,device=device,requires_grad=True),
            get_scaling=torch.full((n,3),.01,device=device,requires_grad=True),
            get_rotation=torch.tensor([[1.,0.,0.,0.]]*n,device=device).reshape(n,4).requires_grad_())
        pc=SimpleNamespace(**values,active_sh_degree=3,max_sh_degree=3)
        bg=torch.tensor([.1,.2,.3],device=device)
        output=render(camera,pc,pipe,bg)
        assert output['radii'].shape==(n,) and not output['radii'].any()
        torch.testing.assert_close(output['render'],bg[:,None,None].expand(3,8,8))
        output['render'].sum().backward()
        for value in values.values():
            assert value.grad is not None and value.grad.shape==value.shape
            assert torch.isfinite(value.grad).all() and not value.grad.any()
    return dict(empty_cloud_sh_backward=True,nonempty_invisible_view_backward=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='1')
    p.add_argument('--output', type=Path)
    p.add_argument('--percent-dense', type=float, default=.01)
    p.add_argument('--position-lr-divisor', type=float, default=50.)
    p.add_argument('--scaling-lr', type=float, default=.005)
    p.add_argument('--feature-lr', type=float, default=.0025)
    p.add_argument('--opacity-lr', type=float, default=.025)
    p.add_argument('--rotation-lr', type=float, default=.001)
    p.add_argument('--opacity-resets', action='store_true')
    p.add_argument('--antialiasing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--preserve-densification-radii', action='store_true')
    p.add_argument('--freeze-lidar-positions', action='store_true')
    p.add_argument('--region-loss', action='store_true')
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
    empty_checks=check_empty_rendering(render, antialiasing=args.antialiasing)
    opt_parser = argparse.ArgumentParser()
    params = OptimizationParams(opt_parser)
    opt = params.extract(opt_parser.parse_args(['--scale_control', 'off',
        '--position_lr_init', str(.00016/args.position_lr_divisor),
        '--position_lr_final', str(.0000016/args.position_lr_divisor),
        '--scaling_lr', str(args.scaling_lr), '--iterations', str(args.iterations),
        '--feature_lr',str(args.feature_lr),'--opacity_lr',str(args.opacity_lr),
        '--rotation_lr',str(args.rotation_lr),
        '--position_lr_max_steps', str(args.iterations),
        *([] if args.opacity_resets else ['--disable_opacity_resets']),
        '--disable_pruning', '--percent_dense', str(args.percent_dense),
        *(['--freeze_lidar_positions'] if args.freeze_lidar_positions else [])]))
    cloud = BasicPointCloud(points=np.array([[-.15,-.1,2],[.15,-.1,2],[-.15,.1,2],[.15,.1,2]], np.float32),
        colors=np.array([[.7,.2,.1],[.2,.6,.2],[.2,.3,.8],[.8,.7,.2]], np.float32),
        normals=np.zeros((4,3), np.float32))
    model = GaussianModel(0)
    model.create_from_pcd(cloud, [SimpleNamespace(image_name='smoke')], 100.)
    model.training_setup(opt)
    expected_rates=dict(xyz=0. if args.freeze_lidar_positions else opt.position_lr_init*100.,
        f_dc=opt.feature_lr,f_rest=opt.feature_lr/20,opacity=opt.opacity_lr,
        scaling=opt.scaling_lr,rotation=opt.rotation_lr)
    for group in model.optimizer.param_groups:
        assert math.isclose(group['lr'],expected_rates[group['name']])
    original = model.get_xyz.detach().clone()
    original_color = model._features_dc.detach().clone()
    camera = SimpleNamespace(image_width=48, image_height=48, FoVx=math.pi/2, FoVy=math.pi/2,
        world_view_transform=torch.eye(4, device='cuda'),
        full_proj_transform=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T,
        camera_center=torch.zeros(3, device='cuda'))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=True, antialiasing=args.antialiasing, occlusion_mode='off')
    for step in range(1, 4):
        lr = model.update_learning_rate(step)
        assert (lr == 0) if args.freeze_lidar_positions else (lr > 0)
        image = render(camera, model, pipe, torch.zeros(3, device='cuda'))['render']
        assert torch.isfinite(image).all()
        if args.region_loss:
            from utils.region_loss import photometric_terms
            mask=torch.zeros(1,48,48,device='cuda',dtype=torch.bool)
            mask[:,12:36,12:36]=True
            l1, similarity=photometric_terms(image,torch.full_like(image,.6),mask)
            loss=.8*l1+.2*(1-similarity)
            pixels=torch.autograd.grad(loss,image,retain_graph=True)[0]
            assert (pixels[:,~mask[0]]==0).all() and pixels[:,mask[0]].abs().sum()>0
            loss.backward()
        else:
            (image-.6).square().mean().backward()
        if args.freeze_lidar_positions:
            assert model._xyz.grad is None and not model._xyz.requires_grad
        else:
            assert model._xyz.grad is not None and model._xyz.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for g in model.optimizer.param_groups for p in g['params'])
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
    if args.freeze_lidar_positions:
        assert torch.equal(original, model.get_xyz)
        assert not torch.equal(original_color, model._features_dc)
        assert model._xyz not in model.optimizer.state
        with torch.no_grad():
            model._scaling[:] = math.log(.001)
            model._scaling[1] = math.log(.02)
            model._opacity[0] = -20
            model.xyz_gradient_accum[:] = 1  # Even high gradients cannot trigger growth.
            model.denom[:] = 1
            model.densify_and_prune(.0001, .005, 1., 0., torch.ones(4, device='cuda'),
                record_stats=True, max_world_size=.01, preserve_radii=True)
        assert len(model.get_xyz) == 2
        assert model.last_densification_stats['cloned_parents'] == model.last_densification_stats['split_parents'] == 0
        assert model.last_densification_stats['pruned'] == 2 and model.last_densification_stats['positions_frozen']
        assert torch.equal(model.get_xyz, original[2:]) and not model._xyz.requires_grad
        restored = GaussianModel(0)
        restored._exposure = torch.nn.Parameter(model._exposure.detach().clone())
        restored.restore(model.capture(), opt)
        restored._features_dc.sum().backward()
        restored.optimizer.step()
        assert not restored._xyz.requires_grad and restored._xyz not in restored.optimizer.state
        assert torch.equal(restored.get_xyz, original[2:])
        if args.opacity_resets:
            # Exercise the actual parameter replacement and Adam state path.
            from utils.densification_policy import opacity_reset_due
            assert [i for i in range(1,15000) if opacity_reset_due(opt,i)]==[3000,6000,9000,12000]
            restored.reset_opacity()
            assert (restored.get_opacity <= .010001).all()
            assert torch.equal(restored.get_xyz, original[2:])
            restored.optimizer.zero_grad(set_to_none=True)
            restored._opacity.sum().backward();restored.optimizer.step()
            assert torch.isfinite(restored.get_opacity).all() and not restored._xyz.requires_grad
        with torch.no_grad():
            from utils.densification_policy import PruningWouldEmptyScene
            restored._scaling[:]=math.log(.001)
            restored._opacity[:]=0
            restored.max_radii2D[:]=100
            before=restored.get_xyz.clone()
            restored.densify_and_prune(.0002,.005,1.,20,torch.ones(2,device='cuda'),
                record_stats=True,protect_nonempty=True)
            assert torch.equal(before,restored.get_xyz) and not restored.max_radii2D.any()
            restored._opacity[:]=-20
            try:
                restored.densify_and_prune(.0002,.005,1.,20,torch.ones(2,device='cuda'),
                    record_stats=True,protect_nonempty=True)
            except PruningWouldEmptyScene as exc:
                assert exc.report['prune_opacity_candidates']==2
                assert torch.equal(before,restored.get_xyz)
            else: raise AssertionError('Pruning deleted the entire fixed cloud')
        result = dict(passed=True, cuda_device=torch.cuda.get_device_name(), torch=torch.__version__,
            native_rgb_gradients=True, antialiasing=args.antialiasing, fixed_centers=True, appearance_updated=True,
            opacity_and_world_pruning=True, no_cloning_or_splitting=True,
            checkpoint_round_trip=True, scaling_lr=opt.scaling_lr)
        result['learning_rates']=expected_rates
        result['opacity_reset_checked']=args.opacity_resets
        result.update(empty_checks,frozen_radius_reset_matches_native=True,empty_pruning_rejected=True)
        if args.region_loss:
            result['regional_rgb_loss_gradients_zero_outside_mask']=True
        if args.output:
            from utils.checkpoint_snapshots import atomic_json
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
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
        native_rgb_xyz_gradients=True, antialiasing=args.antialiasing, reduced_lr_updates_xyz=True, no_scale_projection=True,
        clone_and_split=True, pruning_disabled_path_checked=True, native_opacity_pruning=True,
        children_optimized=True, percent_dense=opt.percent_dense,
        position_lr_divisor=args.position_lr_divisor, scaling_lr=opt.scaling_lr,
        radius_history_regression_checked=args.preserve_densification_radii,
        checkpoint_round_trip=True)
    result.update(empty_checks)
    if args.output:
        from utils.checkpoint_snapshots import atomic_json
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
