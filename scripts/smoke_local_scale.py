#!/usr/bin/env python3
"""Native CUDA AA smoke: local bounds, penalty, Adam projection, audit and PLY."""
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
    p.add_argument('--gpu', default='3')
    p.add_argument('--output', type=Path)
    p.add_argument('--ceiling-multiplier',type=float,default=1.)
    p.add_argument('--target-multiplier',type=float,default=1.)
    p.add_argument('--training-schedule',type=json.loads)
    args = p.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.fixed_lidar import audit, project_scales, scale_regularization
    from utils.local_scale import prepare, support_counts, projected_stats
    from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
    if not torch.cuda.is_available():
        raise RuntimeError('Local scale smoke requires the CUDA-enabled 3dgs environment')
    with tempfile.TemporaryDirectory(prefix='local_scale_smoke_') as tmp:
        folder = Path(tmp)
        xyz = np.array([[(x-1.5)*.02, (y-1.5)*.02, .3] for x in range(4) for y in range(4)], np.float32)
        values = np.zeros(len(xyz), dtype=[(k,'f4') for k in ('x','y','z')])
        for i, key in enumerate(('x','y','z')):
            values[key] = xyz[:,i]
        PlyData([PlyElement.describe(values, 'vertex')]).write(folder/'input.ply')
        prepare(folder/'input.ply', folder/'base_cache', samples=16, workers=1)
        prepare(folder/'input.ply', folder/'cache', samples=16, workers=1,
                ceiling_multiplier=args.ceiling_multiplier,target_multiplier=args.target_multiplier,reuse_cache=folder/'base_cache')
        opt_parser = argparse.ArgumentParser()
        params = OptimizationParams(opt_parser)
        opt = params.extract(opt_parser.parse_args(['--fixed_lidar', '--scale_control', 'hybrid',
            '--max_scale_ratio', '1', '--max_scale_world', '.015', '--local_scale_cache', str(folder/'cache'),
            '--local_scale_ceiling_multiplier',str(args.ceiling_multiplier),
            '--local_scale_target_multiplier',str(args.target_multiplier),
            '--position_lr_init', '0', '--position_lr_final', '0', '--densify_until_iter', '0']))
        if args.training_schedule is not None:
            # Exercise the user's endpoints through all schedule phases in four
            # native CUDA updates, without a full-scene training warmup.
            from utils.local_scale_schedule import validate, values
            policy=json.loads(json.dumps(args.training_schedule))
            if policy['cap'] is not None:
                policy['cap'].update(start=1,end=3)
            if policy['learning_rates'] is not None:
                policy['learning_rates'].update(start=1,end=4)
            validate(policy,args.ceiling_multiplier,4)
            opt.iterations=4
            opt.local_scale_schedule=json.dumps(policy,sort_keys=True)
        model = GaussianModel(1)
        model.create_from_pcd(BasicPointCloud(points=xyz, colors=np.tile([.7,.3,.2], (len(xyz),1)),
            normals=np.zeros_like(xyz)), [SimpleNamespace(image_name='smoke')], 1000.)
        model.oneupSHdegree()
        model.training_setup(opt)
        original = model._xyz.detach().clone()
        original_color = model._features_dc.detach().clone()
        state = model.fixed_lidar_state
        base_cap = torch.tensor(np.load(folder/'base_cache/ceiling.npy'),device='cuda')[:,None]
        assert (model.get_scaling <= base_cap*(1+1e-6)).all(), 'Relaxing the ceiling must not enlarge initialization'
        with torch.no_grad():
            model._scaling[0,0] = torch.maximum(state['log_scale_cap'][0,0],state['log_scale_target'][0,0])+.1
        grad = torch.autograd.grad(scale_regularization(model), model._scaling)[0]
        assert torch.isfinite(grad).all() and (grad > 0).any() and (grad >= 0).all()
        # A value between the target and ceiling is penalized but not projected.
        with torch.no_grad():
            middle = (torch.minimum(state['log_scale_target'],state['log_scale_cap'])+state['log_scale_cap'])/2
            model._scaling.copy_(middle.expand_as(model._scaling))
        project_scales(model)
        assert torch.equal(model._scaling, middle.expand_as(model._scaling))
        camera = SimpleNamespace(image_width=48, image_height=48, FoVx=math.pi/2, FoVy=math.pi/2,
            world_view_transform=torch.eye(4, device='cuda'),
            full_proj_transform=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T,
            camera_center=torch.zeros(3, device='cuda'))
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=True, antialiasing=True, occlusion_mode='off')
        for step in range(1,5):
            model.update_learning_rate(step)
            if args.training_schedule is not None:
                expected=values(policy,step)
                if 'ceiling_multiplier' in expected:
                    torch.testing.assert_close(state['log_scale_cap'].exp(),base_cap*expected['ceiling_multiplier'],rtol=1e-6,atol=0)
                for group in model.optimizer.param_groups:
                    if group['name'] in expected.get('lrs',{}):
                        assert group['lr']==expected['lrs'][group['name']]
            result = render(camera, model, pipe, torch.zeros(3, device='cuda'))
            assert torch.isfinite(result['render']).all()
            ((result['render']-.6).square().mean()+scale_regularization(model)).backward()
            assert model._xyz.grad is None and model._features_dc.grad.abs().sum() > 0
            assert model._features_rest.grad.abs().sum() > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
            model.optimizer.step()
            project_scales(model)
            model.optimizer.zero_grad(set_to_none=True)
            audit(model, step)
        # Explicit overflow checks projection + selective Adam-moment clearing.
        with torch.no_grad():
            model._scaling[0,0] = state['log_scale_cap'][0,0]+.1
        project_scales(model)
        assert model.optimizer.state[model._scaling]['exp_avg'][0,0] == 0
        assert model.optimizer.state[model._scaling]['exp_avg_sq'][0,0] == 0
        assert torch.equal(original, model._xyz) and model._xyz not in model.optimizer.state
        assert not torch.equal(original_color, model._features_dc)
        # Compare cached broad-phase diagnostics to all pairs on the GPU.
        ids = model.local_scale_audit['sample_ids']
        delta = model._xyz[ids,None]-model._xyz[None]
        radius = 3*(model.get_scaling[ids].amax(1)[:,None]+model.get_scaling.amax(1)[None])
        brute = (delta.square().sum(-1) <= radius.square())
        brute[torch.arange(len(ids), device='cuda'),ids] = False
        assert torch.equal(support_counts(model)[0], brute.sum(1))
        restored = GaussianModel(1)
        restored._exposure = torch.nn.Parameter(model._exposure.detach().clone())
        restored.restore(model.capture(), opt)
        torch.testing.assert_close(restored._scaling, model._scaling, atol=0, rtol=0)
        assert torch.equal(restored._xyz, original)
        model.save_ply(str(folder/'point_cloud.ply'))
        vertex = PlyData.read(folder/'point_cloud.ply')['vertex']
        logs = np.column_stack([vertex['scale_'+str(i)] for i in range(3)])
        np.testing.assert_array_equal(logs, model._scaling.detach().cpu().numpy())
        assert np.all(logs <= state['log_scale_cap'].cpu().numpy())
        report = dict(passed=True, cuda_device=torch.cuda.get_device_name(), torch=torch.__version__,
            native_aa_forward_backward=True, local_penalty_gradient=True, hard_projection_and_moments=True,
            sampled_counts_match_all_pairs=True, fixed_xyz_no_adam_state=True,
            checkpoint_round_trip=True, exported_scales_match=True,
            relaxed_cap_keeps_initialization=True,
            scheduled_cap_and_lrs=args.training_schedule is not None,
            **projected_stats(result['radii']), **audit(model, 4))
        if args.output:
            from utils.checkpoint_snapshots import atomic_json
            atomic_json(args.output, report)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
