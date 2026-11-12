#!/usr/bin/env python3
"""Native AA alternation with bounded LiDAR XYZ, checkpoint resume and PLY checks."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',default='2');p.add_argument('--output',type=Path)
    p.add_argument('--ceiling-multiplier',type=float,default=5.)
    args=p.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import numpy as np
    import torch
    from plyfile import PlyData,PlyElement
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud,getProjectionMatrix
    from utils.local_scale import prepare,support_counts
    from utils.fixed_lidar import audit,project_scales,scale_regularization
    from utils.lidar_offsets import DEFAULTS,regularization,state
    from utils.pose_refinement import PoseRefiner,DEFAULTS as POSES,source_fingerprints,render_pose
    from utils.local_scale_schedule import GROUP_LRS
    if not torch.cuda.is_available():raise RuntimeError('LiDAR offset smoke requires CUDA in the 3dgs environment')
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix='lidar_offsets_smoke_') as tmp:
        root=Path(tmp);source=root/'source/sparse/0';source.mkdir(parents=True)
        for name in ('images','cameras'): (source/(name+'.txt')).write_text('# immutable smoke calibration\n')
        fingerprints=source_fingerprints(root/'source')
        xyz=np.array([[(x-2)*.02,(y-2)*.02,.3+.002*x] for y in range(5) for x in range(5)],np.float32)
        vertex=np.zeros(len(xyz),dtype=[(k,'f4') for k in ('x','y','z')])
        for i,k in enumerate(('x','y','z')):vertex[k]=xyz[:,i]
        PlyData([PlyElement.describe(vertex,'vertex')]).write(root/'input.ply')
        prepare(root/'input.ply',root/'cache',samples=25,workers=1,
                ceiling_multiplier=args.ceiling_multiplier,position_padding=.005)
        parser=argparse.ArgumentParser();params=OptimizationParams(parser)
        opt=params.extract(parser.parse_args(['--fixed_lidar','--scale_control','hybrid',
            '--max_scale_ratio','1','--max_scale_world','.015','--local_scale_cache',str(root/'cache'),
            '--local_scale_ceiling_multiplier',str(args.ceiling_multiplier),'--position_lr_init','0','--position_lr_final','0',
            '--densify_until_iter','0','--iterations','6']))
        policy={**DEFAULTS,'warmup':1,'ramp':1,'end':6}
        pose_policy={**POSES,'start':1,'end':4,'max_width':64}
        opt.lidar_offsets=json.dumps(policy);opt.pose_refinement=json.dumps(pose_policy)
        opt.local_scale_schedule=json.dumps(dict(schema=1,
            cap=dict(initial_multiplier=1.,final_multiplier=args.ceiling_multiplier,start=1,end=4),
            learning_rates=dict(start=1,end=6,groups={k:[v,v*.1] for k,v in GROUP_LRS.items()})))
        m=GaussianModel(1)
        m.create_from_pcd(BasicPointCloud(points=xyz,colors=np.tile([.7,.3,.2],(25,1)),normals=np.zeros_like(xyz)),
                          [SimpleNamespace(image_name='smoke.jpg')],1000.)
        initial_sigma=m._scaling[:,:1].detach().exp().clone()
        m.oneupSHdegree();m.training_setup(opt)
        torch.testing.assert_close(state(m)['radius'],(initial_sigma*.25).clamp_max(.005),rtol=0,atol=0)
        projection=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T.contiguous()
        cam=SimpleNamespace(image_name='smoke.jpg',image_height=64,image_width=64,FoVx=math.pi/2,FoVy=math.pi/2,
            world_view_transform=torch.eye(4,device='cuda'),projection_matrix=projection,full_proj_transform=projection,
            camera_center=torch.zeros(3,device='cuda'),original_image=torch.zeros(3,64,64),alpha_mask=torch.ones(1,64,64))
        pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=True,antialiasing=True,occlusion_mode='off')
        bg=torch.zeros(3,device='cuda')
        with torch.no_grad():
            cam.original_image=render_pose(cam,m,pipe,bg,bg.new_tensor([.003,-.002,.001]),bg.new_tensor([.001,0.,-.001])).cpu()
        m.pose_refiner=PoseRefiner([cam],pose_policy,root/'source',6)
        attrs=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')

        def step(model,i):
            lr=model.update_learning_rate(i)
            before=[getattr(model,k).detach().clone() for k in attrs]
            row=model.pose_refiner.update(cam,model,pipe,bg,i)
            for k,value in zip(attrs,before):
                assert torch.equal(getattr(model,k),value) and getattr(model,k).grad is None
            poses=[p.detach().clone() for p in model.pose_refiner.params[cam.image_name]]
            image=render(model.pose_refiner.camera(cam),model,pipe,bg)['render']
            loss=(image-cam.original_image.cuda()).square().mean()+scale_regularization(model)+regularization(model)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
            if lr:
                assert model._xyz.grad is not None and model._xyz.grad.abs().sum()>0
            else:
                assert model._xyz.grad is None and model._xyz not in model.optimizer.state
            model.optimizer.step();project_scales(model);model.optimizer.zero_grad(set_to_none=True)
            for a,b in zip(poses,model.pose_refiner.params[cam.image_name]):assert torch.equal(a,b)
            audit(model,i)
            return row

        rows=[]
        for i in range(1,4):rows.append(step(m,i))
        assert not torch.equal(m._xyz,state(m)['anchors'])
        assert m.optimizer.param_groups[0]['lr']<=1e-5 # scene extent=1000 must not amplify LR
        with torch.no_grad():m._xyz[0].add_(1.)
        moments=m.optimizer.state[m._xyz]
        others={k:moments[k][1:].clone() for k in ('exp_avg','exp_avg_sq')}
        project_scales(m)
        for k,value in others.items():
            assert (moments[k][0]==0).all() and torch.equal(moments[k][1:],value)
        audit(m,3)
        buffer=io.BytesIO();torch.save(m.capture(),buffer);buffer.seek(0)
        restored=GaussianModel(1);restored._exposure=torch.nn.Parameter(m._exposure.detach().clone())
        restored.restore(torch.load(buffer,weights_only=True),opt)
        restored.pose_refiner=PoseRefiner([cam],pose_policy,root/'source',6,saved=restored.pose_refinement_state)
        for k in ('exp_avg','exp_avg_sq'):assert torch.equal(restored.optimizer.state[restored._xyz][k],moments[k])
        for i in range(4,7):rows.append(step(restored,i))
        ids=restored.local_scale_audit['sample_ids']
        distance=torch.cdist(restored._xyz[ids],restored._xyz,compute_mode='donot_use_mm_for_euclid_dist')
        radius=3*(restored.get_scaling[ids].amax(1)[:,None]+restored.get_scaling.amax(1)[None])
        hit=distance<=radius;hit[torch.arange(len(ids),device='cuda'),ids]=False
        assert torch.equal(support_counts(restored)[0],hit.sum(1))
        restored.save_ply(str(root/'point_cloud.ply'))
        vertex=PlyData.read(root/'point_cloud.ply')['vertex']
        np.testing.assert_array_equal(np.column_stack([vertex[k] for k in ('x','y','z')]),restored._xyz.detach().cpu().numpy())
        assert source_fingerprints(root/'source')==fingerprints
        report=dict(passed=True,cuda_device=torch.cuda.get_device_name(),torch=torch.__version__,
            native_aa_alternating_updates=True,gaussians_frozen_during_pose_step=True,poses_frozen_during_gaussian_step=True,
            warmup_has_no_xyz_moments=True,absolute_xyz_lr=True,selective_xyz_moment_projection=True,
            original_anchors_and_xyz_optimizer_resume=True,overlap_counts_match_all_pairs=True,
            exported_xyz_match=True,source_calibration_unchanged=True,
            **audit(restored,6))
        if args.output:
            from utils.checkpoint_snapshots import atomic_json
            atomic_json(args.output,report)
        print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
