#!/usr/bin/env python3
"""Native CUDA AA render/backward, distance-weighted Adam and checkpoint smoke."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='0'); parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import numpy as np
    import torch
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud,getProjectionMatrix
    from utils.fixed_lidar import audit,project_scales
    from utils.distance_updates import weights,adam_step,policy
    if not torch.cuda.is_available():
        raise RuntimeError('Distance smoke requires the CUDA-enabled 3dgs environment')
    # Same gradient on near/far rows: plain Adam would normalize positive
    # constant gradient factors away, but our actual displacements must differ.
    p=torch.nn.Parameter(torch.zeros(3,2,device='cuda'))
    optimizer=torch.optim.Adam([p],lr=.01,eps=1e-15)
    p.grad=torch.ones_like(p)
    w=torch.tensor([1.,.25,.05],device='cuda')
    adam_step(optimizer,w,chunk_size=2)
    torch.testing.assert_close(p,-.01*w[:,None].expand_as(p),atol=1e-7,rtol=1e-5)
    # Unit weights agree with native Adam over several changing gradients.
    a=torch.nn.Parameter(torch.ones(4,3,device='cuda')); b=torch.nn.Parameter(a.detach().clone())
    oa=torch.optim.Adam([a],lr=.003,eps=1e-15); ob=torch.optim.Adam([b],lr=.003,eps=1e-15,foreach=False)
    for iteration in range(5):
        gradient=torch.arange(12,device='cuda').reshape(4,3).float().sin()*(iteration+1)
        a.grad=gradient.clone(); b.grad=gradient.clone()
        adam_step(oa,torch.ones(4,device='cuda'),chunk_size=2); ob.step()
    torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-6)
    opt_parser=argparse.ArgumentParser(); params=OptimizationParams(opt_parser)
    opt=params.extract(opt_parser.parse_args(['--fixed_lidar','--distance_weighting',
        '--position_lr_init','0','--position_lr_final','0','--densify_until_iter','0']))
    xyz=np.array([[-.15,-.1,1],[.15,-.1,2],[-.15,.1,4],[.15,.1,8]],np.float32)
    model=GaussianModel(0)
    model.create_from_pcd(BasicPointCloud(points=xyz,colors=np.tile([.7,.3,.2],(4,1)),normals=np.zeros_like(xyz)),
        [SimpleNamespace(image_name='smoke')],1000.)
    model.training_setup(opt)
    original=model._xyz.detach().clone(); original_color=model._features_dc.detach().clone()
    camera=SimpleNamespace(image_width=48,image_height=48,FoVx=math.pi/2,FoVy=math.pi/2,
        world_view_transform=torch.eye(4,device='cuda'),
        full_proj_transform=getProjectionMatrix(.01,100,math.pi/2,math.pi/2).cuda().T,
        camera_center=torch.zeros(3,device='cuda'))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=True,
                         antialiasing=True,occlusion_mode='off')
    for step in range(1,5):
        result=render(camera,model,pipe,torch.zeros(3,device='cuda'))
        assert torch.isfinite(result['render']).all()
        (result['render']-.6).square().mean().backward()
        assert model._xyz.grad is None and model._features_dc.grad.abs().sum()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
        w=weights(model._xyz,camera.camera_center,policy(opt),chunk_size=2)
        adam_step(model.optimizer,w,chunk_size=2)
        project_scales(model); model.optimizer.zero_grad(set_to_none=True)
        audit(model,step)
    assert torch.equal(model._xyz,original) and model._xyz not in model.optimizer.state
    assert not torch.equal(model._features_dc,original_color)
    restored=GaussianModel(0); restored._exposure=torch.nn.Parameter(model._exposure.detach().clone())
    restored.restore(model.capture(),opt)
    torch.testing.assert_close(restored._scaling,model._scaling,atol=0,rtol=0)
    assert restored.fixed_lidar_state['distance_update_policy']==policy(opt)
    report=dict(passed=True,torch=torch.__version__,cuda_device=torch.cuda.get_device_name(),
        native_aa_forward_backward=True,unit_weight_adam_parity=True,actual_step_attenuation=True,
        fixed_xyz=True,checkpoint_round_trip=True,**audit(model,4))
    if args.output:
        from utils.checkpoint_snapshots import atomic_json
        atomic_json(args.output,report)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
