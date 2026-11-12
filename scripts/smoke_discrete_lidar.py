#!/usr/bin/env python3
"""Native CUDA AA backward, discrete growth/pruning and fixed-center resume."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud, getProjectionMatrix
    from utils.discrete_lidar_cache import prepare
    from utils.discrete_lidar import adapt, audit
    from scene.dataset_readers import storePly
    if not torch.cuda.is_available():
        raise RuntimeError('Discrete LiDAR smoke requires the CUDA-enabled 3dgs environment')
    with tempfile.TemporaryDirectory(prefix='discrete-lidar-smoke-') as temp:
        root = Path(temp)
        xyz = np.array([[(x-1.5)*.04, (y-1.5)*.04, 2.] for x in range(4) for y in range(4)], np.float32)
        rgb = np.tile([180, 90, 40], (len(xyz), 1)).astype(np.uint8)
        storePly(str(root/'input.ply'), xyz, rgb)
        prepare(root/'input.ply', root/'cache', fraction=.5, voxel=.15, neighbors=8, workers=1)
        opt_parser = argparse.ArgumentParser(); params = OptimizationParams(opt_parser)
        opt = params.extract(opt_parser.parse_args(['--discrete_lidar', '--discrete_lidar_cache', str(root/'cache'),
            '--position_lr_init', '0', '--position_lr_final', '0', '--discrete_lidar_max_new', '4']))
        model = GaussianModel(0)
        cloud = BasicPointCloud(points=xyz, colors=rgb/255., normals=np.zeros_like(xyz))
        model.create_from_pcd(cloud, [SimpleNamespace(image_name='smoke')], 100.)
        model.training_setup(opt)
        camera = SimpleNamespace(image_width=128, image_height=128, FoVx=.8, FoVy=.8,
            world_view_transform=torch.eye(4, device='cuda'),
            full_proj_transform=getProjectionMatrix(.01,100.,.8,.8).cuda().T,
            camera_center=torch.zeros(3, device='cuda'))
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=True, antialiasing=True, occlusion_mode='off')
        event = None
        for step in range(1, 7):
            assert model.update_learning_rate(step) == 0
            output = render(camera, model, pipe, torch.zeros(3, device='cuda'))
            image = output['render']
            assert torch.isfinite(image).all()
            (image-.6).square().mean().backward()
            assert model._xyz.grad is None
            assert model._features_dc.grad is not None and model._features_dc.grad.abs().sum() > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
            model.add_densification_stats(output['viewspace_points'], output['visibility_filter'])
            model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
            if step == 3:
                assert model.xyz_gradient_accum.sum() > 0, 'Frozen XYZ must still produce screen-space growth gradients'
                with torch.no_grad():
                    model._scaling[0, 0] = math.log(.06)
                event = adapt(model, step, 1e-12)
                assert event['added'] > 0 and event['scale_pruned'] >= 1
            audit(model, step)
        checkpoint = root/'checkpoint.pth'
        torch.save(model.capture(), checkpoint)
        restored = GaussianModel(0)
        restored._exposure = torch.nn.Parameter(model._exposure.detach().clone())
        restored.restore(torch.load(checkpoint, weights_only=True), opt)
        assert torch.equal(restored._xyz, model._xyz)
        assert torch.equal(restored.discrete_lidar_state['active_ids'], model.discrete_lidar_state['active_ids'])
        torch.testing.assert_close(restored.optimizer.state[restored._features_dc]['exp_avg'],
                                   model.optimizer.state[model._features_dc]['exp_avg'], atol=0, rtol=0)
        result = dict(passed=True, cuda_device=torch.cuda.get_device_name(), torch=torch.__version__,
            native_aa_forward_backward=True, frozen_xyz_screen_gradients=True, growth_event=event,
            render_backward_after_population_edit=True, checkpoint_round_trip=True, **audit(restored, 6))
        if args.output:
            from utils.checkpoint_snapshots import atomic_json
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
