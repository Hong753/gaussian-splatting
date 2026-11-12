#!/usr/bin/env python3
"""Native CUDA mesh depth, corrected-pose gradients and blocked-background checks."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def write_wall(root, depth, rotation=None, center=None):
    import numpy as np
    from utils.lidar_occlusion import file_digest
    from utils.scene_occlusion import FORMAT
    vertices=np.array([[-10,-10,depth],[10,-10,depth],[10,10,depth],[-10,10,depth]],np.float32)
    if rotation is not None: vertices=vertices@rotation+center
    np.savez_compressed(root/'mesh.npz',vertices=vertices,faces=np.array([[0,1,2],[0,2,3]],np.int32))
    report=dict(format=FORMAT,complete=True,geometry_source='full_initialization_lidar',relative_margin=0.,
                mesh='mesh.npz',mesh_sha256=file_digest(root/'mesh.npz'),source=str(root))
    (root/'scene.json').write_text(json.dumps(report))
    return root/'scene.json'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='3');p.add_argument('--output',type=Path)
    args=p.parse_args(argv);os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import getProjectionMatrix
    from utils.pose_refinement import rotation_matrix,rotation_quaternion,camera_with_transform,corrected_transform,render_pose
    from utils.pose_culling import DEFAULTS,provider
    from utils.occlusion_culling import masked_adam_step
    from scripts.pose_gradient_check import probe_model,cuda_pose_gradient_check
    if not torch.cuda.is_available(): raise RuntimeError('Pose culling smoke requires the actual CUDA extensions')
    bg=torch.zeros(3,device='cuda')
    R=rotation_matrix(rotation_quaternion(bg.new_tensor([.12,-.08,.05])))
    center=bg.new_tensor([.1,-.1,0.]);w2c=torch.eye(4,device='cuda')
    w2c[:3,:3]=R;w2c[:3,3]=-R@center
    projection=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T.contiguous()
    cam=SimpleNamespace(image_name='smoke.jpg',image_width=48,image_height=48,FoVx=math.pi/2,FoVy=math.pi/2,
        znear=.01,zfar=100.,world_view_transform=w2c.T.contiguous(),projection_matrix=projection,
        full_proj_transform=w2c.T@projection,camera_center=center)
    with tempfile.TemporaryDirectory(prefix='pose_culling_smoke_') as tmp:
        root=Path(tmp);mesh=write_wall(root,2.8,R.cpu().numpy(),center.cpu().numpy())
        pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=True,antialiasing=True,
            occlusion_mode='mesh',occlusion_backend='ray',occlusion_geometry=str(mesh),occlusion_cache=str(root/'cache'),
            occlusion_margin=0.,pose_culling=dict(DEFAULTS))
        depth=provider(pipe).frame(cam,bg.device)['depth']
        torch.testing.assert_close(depth[8:-8,8:-8],torch.full_like(depth[8:-8,8:-8],2.8),atol=1e-4,rtol=1e-4)
        moved=camera_with_transform(cam,corrected_transform(cam,bg.new_tensor([0.,0.,.02]),bg)[0])
        moved.occlusion_transient=True
        corrected=provider(pipe).frame(moved,bg.device)['depth']
        torch.testing.assert_close(corrected[8:-8,8:-8],torch.full_like(corrected[8:-8,8:-8],2.78),atol=1e-4,rtol=1e-4)
        checks=cuda_pose_gradient_check(cam,probe_model(bg.device),pipe,bg)
        probe=probe_model(bg.device);m=GaussianModel(1);m.active_sh_degree=1
        rear=bg.new_tensor([[0.,0.,5.]])@R+center
        m._xyz=torch.nn.Parameter(torch.cat([probe.get_xyz,rear]))
        m._scaling=torch.nn.Parameter(probe.get_scaling.repeat(2,1).log())
        m._rotation=torch.nn.Parameter(probe.get_rotation.repeat(2,1))
        m._opacity=torch.nn.Parameter(torch.logit(probe.get_opacity.repeat(2,1)))
        m._features_dc=torch.nn.Parameter(probe.get_features[:,:1].repeat(2,1,1))
        m._features_rest=torch.nn.Parameter(probe.get_features[:,1:].repeat(2,1,1))
        params=[m._xyz,m._scaling,m._rotation,m._opacity,m._features_dc,m._features_rest]
        optimizer=torch.optim.Adam(params,lr=1e-4)
        # Seed nonzero historical Adam momentum, then verify hidden rows freeze.
        for param in params: param.grad=torch.ones_like(param)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        before=[p.detach().clone() for p in params]
        moments=[{k:optimizer.state[p][k].clone() for k in ('exp_avg','exp_avg_sq')} for p in params]
        package=render(cam,m,pipe,bg)
        assert package['occlusion_fragment_flags'].tolist()==[1,2]
        package['render'].sum().backward()
        for param in params:
            assert torch.isfinite(param.grad).all() and torch.count_nonzero(param.grad[1])==0
        masked_adam_step(optimizer,package['occlusion_keep'])
        for param,old,state in zip(params,before,moments):
            assert torch.equal(param[1],old[1])
            for key in state: assert torch.equal(optimizer.state[param][key][1],state[key][1])
        optimizer.zero_grad(set_to_none=True)
        t=bg.new_tensor([.005,-.003,.002],requires_grad=True)
        r=bg.new_tensor([.001,-.002,.0015],requires_grad=True)
        image=render_pose(cam,m,pipe,bg,t,r)
        gradient=torch.autograd.grad(image.sum(),(t,r))
        assert all(torch.isfinite(g).all() for g in gradient) and all(p.grad is None for p in params)
    report=dict(passed=True,cuda_device=torch.cuda.get_device_name(),torch=torch.__version__,
        corrected_pose_mesh_depth=True,pose_gradient_checks=checks,blocked_background_rgb_gradients_zero=True,
        hidden_parameters_and_adam_moments_frozen=True,gaussians_detached_in_pose_step=True)
    if args.output:
        from utils.checkpoint_snapshots import atomic_json
        atomic_json(args.output,report)
    print(json.dumps(report,indent=2));return 0


if __name__=='__main__': raise SystemExit(main())
