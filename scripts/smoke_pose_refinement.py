#!/usr/bin/env python3
"""Native CUDA pose-gradient parity, finite differences, bounded updates and cache."""
import argparse
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
    p.add_argument('--gpu',default='0');p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import numpy as np
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud,getProjectionMatrix
    from utils.pose_refinement import (render_pose,rotation_quaternion,rotation_matrix,corrected_transform,
        camera_with_transform,PoseRefiner,DEFAULTS,source_fingerprints)
    if not torch.cuda.is_available(): raise RuntimeError('Pose smoke requires the CUDA-enabled 3dgs environment')
    torch.manual_seed(42)
    xyz=np.array([[(x-2)*.18,(y-2)*.16,2.+.03*(x+5*y)] for y in range(5) for x in range(5)],np.float32)
    model=GaussianModel(1)
    model.create_from_pcd(BasicPointCloud(points=xyz,colors=np.tile([.7,.3,.2],(len(xyz),1)),normals=np.zeros_like(xyz)),
                          [SimpleNamespace(image_name='smoke.jpg')],1.)
    model.oneupSHdegree()
    with torch.no_grad():
        model._scaling.copy_(torch.tensor([.10,.06,.04],device='cuda').log().expand(len(xyz),3))
        model._features_rest.normal_(std=.05)
        model._opacity.fill_(2.)
        model._rotation.copy_(rotation_quaternion(torch.randn(len(xyz),3,device='cuda')*.3))
    R=rotation_matrix(rotation_quaternion(torch.tensor([.12,-.08,.05],device='cuda')))
    center=torch.tensor([.1,-.1,0.],device='cuda')
    w2c=torch.eye(4,device='cuda');w2c[:3,:3]=R;w2c[:3,3]=-R@center
    projection=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T.contiguous()
    camera=SimpleNamespace(image_name='smoke.jpg',image_width=48,image_height=48,FoVx=math.pi/2,FoVy=math.pi/2,
        world_view_transform=w2c.T.contiguous(),projection_matrix=projection,full_proj_transform=w2c.T@projection,
        camera_center=center,original_image=torch.zeros(3,48,48),alpha_mask=torch.ones(1,48,48))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=True,antialiasing=True,occlusion_mode='off')
    bg=torch.zeros(3,device='cuda')
    translation=torch.tensor([.005,-.003,.002],device='cuda',requires_grad=True)
    rotation=torch.tensor([.001,-.002,.0015],device='cuda',requires_grad=True)
    with torch.no_grad():
        native=render(camera_with_transform(camera,corrected_transform(camera,translation,rotation)[0]),model,pipe,bg)['render']
    transformed=render_pose(camera,model,pipe,bg,translation,rotation)
    torch.testing.assert_close(transformed,native,atol=4e-5,rtol=3e-4)
    # The dense image still checks full forward parity. Its hard 1/255 alpha
    # cutoffs make an arbitrary full-image finite difference nondifferentiable.
    # Check derivatives on a separate anisotropic, nonidentity, nonconstant-SH
    # probe with guarded stable branches, in AA-off AND AA-on modes.
    del transformed,native
    from scripts.pose_gradient_check import probe_model,cuda_pose_gradient_check
    gradient_checks=cuda_pose_gradient_check(camera,probe_model(bg.device),pipe,bg)
    assert all(getattr(model,name).grad is None for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest'))
    with tempfile.TemporaryDirectory(prefix='pose_smoke_') as tmp:
        root=Path(tmp);sparse=root/'source/sparse/0';sparse.mkdir(parents=True)
        for name in ('images','cameras','points3D'): (sparse/(name+'.txt')).write_text('# smoke calibration\n')
        original=source_fingerprints(root/'source')
        with torch.no_grad():
            target=render_pose(camera,model,pipe,bg,translation,rotation).cpu()
        camera.original_image=target
        config={**DEFAULTS,'start':0,'end':12,'max_width':48,'prior_weight':1e-7}
        refiner=PoseRefiner([camera],config,root/'source',12)
        rows=[]
        before=[getattr(model,name).detach().clone() for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')]
        for step in range(1,13): rows.append(refiner.update(camera,model,pipe,bg,step))
        assert any(row['accepted'] and row['objective_candidate']<row['objective_before'] for row in rows)
        for name,value in zip(('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest'),before):
            assert torch.equal(getattr(model,name),value) and getattr(model,name).grad is None
        restored=PoseRefiner([camera],config,root/'source',12,saved=refiner.capture())
        for a,b in zip(restored.params['smoke.jpg'],refiner.params['smoke.jpg']): assert torch.equal(a,b)
        with torch.no_grad():
            restored.params['smoke.jpg'][0].fill_(1);restored.params['smoke.jpg'][1].fill_(1)
        assert restored.project('smoke.jpg')
        assert float(restored.params['smoke.jpg'][0].norm())<=config['max_translation']+1e-7
        assert float(restored.params['smoke.jpg'][1].norm())<=math.radians(config['max_rotation_degrees'])+1e-7
        path=refiner.export(root/'snapshot',12)
        assert path.is_file() and source_fingerprints(root/'source')==original
    report=dict(passed=True,cuda_device=torch.cuda.get_device_name(),torch=torch.__version__,
        native_vs_pose_render_parity=True,six_dof_finite_difference=True,
        gradient_checks=gradient_checks,exact_pose_aa_compensation=True,
        gaussians_frozen_during_pose_step=True,accepted_pose_steps=sum(r['accepted'] for r in rows),
        bounds_and_resume=True,source_calibration_unchanged=True)
    if args.output:
        from utils.checkpoint_snapshots import atomic_json
        atomic_json(args.output,report)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
