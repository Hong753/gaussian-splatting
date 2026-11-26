#!/usr/bin/env python3
"""Tiny synthetic CUDA preflight; these points are NOT the scene initialization."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from models.scaffold_arguments import ScaffoldOptimizationParams
    from models.scaffold_gaussian import ScaffoldGaussianModel
    from models.scaffold_renderer import render, prefilter_voxel
    from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indexed-opacity', action='store_true')
    parser.add_argument('--offset-sh-degree', type=int, choices=(-1, 0, 1, 2, 3), default=-1)
    parser.add_argument('--offset-max-radius', type=float, default=0.,
                        help='Check an optional world-space child displacement limit (0: disabled)')
    args = parser.parse_args()
    from utils.scaffold_offset_constraints import validate_offset_radius
    try:
        validate_offset_radius(args.offset_max_radius)
    except ValueError as exc:
        parser.error(str(exc))
    check_opacity = args.indexed_opacity
    from models.scaffold_offset_sh import scaffold_model_class
    model_class = scaffold_model_class(ScaffoldGaussianModel, args.offset_sh_degree)
    if not torch.cuda.is_available():
        raise RuntimeError('Scaffold preflight requires CUDA')
    print('Scaffold synthetic preflight: 9 test anchors; real dataset loads afterwards.', flush=True)
    torch.manual_seed(4)
    xyz = np.array([[(x-1)*.2, (y-1)*.2, 2.+.02*x] for y in range(3) for x in range(3)], np.float32)
    cloud = BasicPointCloud(xyz, np.ones_like(xyz)*.5, np.zeros_like(xyz))
    camera = SimpleNamespace(image_name='synthetic', image_width=32, image_height=32,
        FoVx=math.pi/2, FoVy=math.pi/2, world_view_transform=torch.eye(4, device='cuda'),
        full_proj_transform=getProjectionMatrix(.01, 100., math.pi/2, math.pi/2).cuda().T,
        camera_center=torch.zeros(3, device='cuda'))
    pipe = SimpleNamespace(debug=True, antialiasing=False)
    bg = torch.zeros(3, device='cuda')
    for fixed in (False, True):
        parser = argparse.ArgumentParser(); params = ScaffoldOptimizationParams(parser)
        opt = params.extract(parser.parse_args(['--position_lr_init', '.00016', '--position_lr_final', '.0000016',
            '--offset_max_radius', str(args.offset_max_radius),
            *(['--freeze_anchors', '--disable_anchor_growth'] if fixed else [])]))
        model = model_class(feat_dim=8, n_offsets=2, voxel_size=.02, appearance_dim=0, update_init_factor=16)
        model.create_from_pcd(cloud, [camera], 1.)
        with torch.no_grad():
            model.mlp_opacity[2].weight.zero_(); model.mlp_opacity[2].bias.fill_(.5)
        model.training_setup(opt)
        before = model.get_anchor.detach().clone()
        if args.offset_max_radius:
            with torch.no_grad():
                # Force a real violation before testing the rendering path.
                model._offset[:, 0, 0] = 4 * args.offset_max_radius / model.get_scaling[:, 0]
            projected = model.project_offsets(record_stats=True)
            assert projected['offset_projected_children'] == len(before)
            assert projected['offset_radius_after_projection_max'] <= args.offset_max_radius * (1 + 1e-6)
        for iteration in range(1, 4):
            model.update_learning_rate(iteration)
            result = render(camera, model, pipe, bg, visible_mask=prefilter_voxel(camera, model, pipe, bg), retain_grad=True)
            weights = torch.linspace(.2, 1., 32, device='cuda')[None, None, :]
            loss = (result['render']*weights).square().mean()
            loss.backward()
            assert torch.isfinite(loss)
            if args.offset_sh_degree >= 0:
                assert model.mlp_color.coefficients.grad.abs().sum() > 0
            assert all(torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'] if p.grad is not None)
            model.training_statis(result['viewspace_points'], result['neural_opacity'],
                result['visibility_filter'], result['selection_mask'], prefilter_voxel(camera, model, pipe, bg))
            model.optimizer.step()
            model.project_offsets()
            model.optimizer.zero_grad(set_to_none=True)
        assert torch.equal(before, model.get_anchor) == fixed
        assert model._offset.abs().max() > 0
        if fixed:
            from utils.scaffold_observations import sensed_step
            visible = prefilter_voxel(camera, model, pipe, bg)
            eligible = torch.arange(len(before), device=before.device) % 2 == 0
            rows = model._offset.detach().clone()
            moment = model.optimizer.state[model._offset]['exp_avg'].clone()
            with torch.no_grad():
                expected_rgb = render(camera, model, pipe, bg, visible_mask=visible)['render']
            gated = render(camera, model, pipe, bg, visible_mask=visible, update_anchors=eligible)
            torch.testing.assert_close(gated['render'], expected_rgb)
            gated['render'].square().mean().backward()
            assert len(gated['active_parents']) and not model._offset.grad[~eligible].any()
            active = torch.zeros_like(visible)
            active[gated['active_parents']] = True
            sensed_step(model.optimizer, active)
            model.project_offsets()
            model.optimizer.zero_grad(set_to_none=True)
            assert torch.equal(model._offset[~active], rows[~active])
            assert torch.equal(model.optimizer.state[model._offset]['exp_avg'][~active], moment[~active])
        if fixed and check_opacity:
            strict = render(camera, model, pipe, bg, visible_mask=visible & eligible,
                            update_anchors=eligible, return_alpha=True)
            alpha = strict['alpha']
            assert alpha.shape == (32, 32) and torch.isfinite(alpha).all() and alpha.max() > 0
            strict_hit = alpha.detach().argmax()
            (1 - alpha.flatten()[strict_hit]).square().backward()
            grad = model.mlp_opacity[2].bias.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            assert model._scaling.grad is None and model._offset.grad is None
            model.optimizer.zero_grad(set_to_none=True)
        restored = model_class(feat_dim=8, n_offsets=2, voxel_size=.02, appearance_dim=0, update_init_factor=16)
        restored.restore(model.capture(), opt)
        assert restored.offset_max_radius == args.offset_max_radius
        if args.offset_max_radius:
            radius = torch.linalg.vector_norm(model._offset * model.get_scaling[:, None, :3], dim=-1)
            assert radius.max() <= args.offset_max_radius * (1 + 1e-6)
        model.eval(); restored.eval()
        with torch.no_grad():
            expected = render(camera, model, pipe, bg)['render']
            torch.testing.assert_close(render(camera, restored, pipe, bg)['render'], expected)
            with tempfile.TemporaryDirectory() as tmp:
                model.save_ply(str(Path(tmp)/'point_cloud.ply')); model.save_mlp_checkpoints(tmp)
                restored.load_ply(str(Path(tmp)/'point_cloud.ply')); restored.eval()
                torch.testing.assert_close(render(camera, restored, pipe, bg)['render'], expected, atol=1e-6, rtol=1e-5)
        model.train()
        with torch.no_grad():
            model.mlp_opacity[2].weight.zero_(); model.mlp_opacity[2].bias.fill_(-1.)
        empty = render(camera, model, pipe, bg, retain_grad=True)
        assert len(empty['radii']) == 0 and torch.isfinite(empty['render']).all()
        empty['render'].sum().backward()
    print(json.dumps(dict(passed=True, antialiasing=False, fixed_and_learnable_anchors=True,
        learned_offsets=True, optimizer_checkpoint=True, saved_render_parity=True, empty_view_backward=True,
        offset_max_radius=args.offset_max_radius, world_offset_projection=bool(args.offset_max_radius))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
