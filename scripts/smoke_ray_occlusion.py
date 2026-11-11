#!/usr/bin/env python3
"""Run actual CUDA AA parity, wall/hole clipping, backward and Adam checks."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def require_finite(named_tensors, stage):
    import torch
    for name, tensor in named_tensors:
        bad = int((~torch.isfinite(tensor)).sum().item())
        if bad:
            raise AssertionError(f'{stage}: {name} has {bad}/{tensor.numel()} non-finite values')


def check_raw_depth_gradient_layouts(camera, pc, bg, names, parameters, aa):
    """Exercise the CUDA binding directly, without the shared render hook."""
    import torch
    from diff_gaussian_rasterization_occlusion import GaussianRasterizationSettings, GaussianRasterizer
    h, w = camera.image_height, camera.image_width
    settings = GaussianRasterizationSettings(image_height=h, image_width=w,
        tanfovx=math.tan(camera.FoVx/2), tanfovy=math.tan(camera.FoVy/2), bg=bg,
        scale_modifier=1., viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform, sh_degree=pc.active_sh_degree,
        campos=camera.camera_center, prefiltered=False, debug=True, antialiasing=aa,
        occlusion_depth=torch.full((h,w),float('inf'),device=bg.device))
    rasterizer = GaussianRasterizer(settings)
    def gradients(seed):
        _, _, depth, _ = rasterizer(means3D=pc.get_xyz, means2D=torch.zeros_like(pc.get_xyz),
            opacities=pc.get_opacity, shs=pc.get_features, scales=pc.get_scaling, rotations=pc.get_rotation)
        require_finite([('depth',depth)], f'AA={aa} raw ray output')
        return torch.autograd.grad(depth, parameters, grad_outputs=seed)
    seeds = dict(expanded=torch.tensor(.01,device=bg.device).expand(1,h,w),
        transposed=torch.linspace(.01,.02,h*w,device=bg.device).reshape(1,w,h).transpose(1,2))
    for layout, seed in seeds.items():
        assert not seed.is_contiguous()
        actual, expected = gradients(seed), gradients(seed.contiguous())
        require_finite(zip(names,actual), f'AA={aa} raw ray {layout} gradient')
        require_finite(zip(names,expected), f'AA={aa} raw ray contiguous gradient')
        for name, a, b in zip(names,actual,expected):
            torch.testing.assert_close(a,b,atol=3e-5,rtol=2e-4,
                msg=f'AA={aa} raw ray {layout} gradient differs for {name}')
    return list(seeds)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='5')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from scripts.occlusion_options import configure_pipeline
    from utils.graphics_utils import getProjectionMatrix
    from utils.sh_utils import RGB2SH
    from utils.occlusion_culling import masked_adam_step
    from utils.ray_occlusion import classify_points_depth
    if not torch.cuda.is_available():
        raise RuntimeError('This check requires the actual CUDA renderer; no CPU substitute')
    view = torch.eye(4, device='cuda')
    camera = SimpleNamespace(image_width=64, image_height=64, FoVx=math.pi/2, FoVy=math.pi/2,
        world_view_transform=view, full_proj_transform=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).cuda().T,
        camera_center=torch.zeros(3,device='cuda'))
    pc = GaussianModel(0)
    param = lambda x: torch.nn.Parameter(torch.tensor(x,dtype=torch.float32,device='cuda'))
    pc._xyz = param([[.02,-.03,1.],[.06,-.09,3.]])
    pc._scaling = param([[math.log(.12)]*3,[math.log(.36)]*3])
    pc._rotation = param([[1,0,0,0]]*2)
    pc._opacity = param([[-2.2],[0.]])  # Transparent foreground still blocks by fixed geometry.
    pc._features_dc = torch.nn.Parameter(RGB2SH(torch.tensor([[.8,.2,.1],[.1,.8,.2]],device='cuda'))[:,None,:])
    pc._features_rest = torch.nn.Parameter(torch.empty(2,0,3,device='cuda'))
    names = ('_xyz','_scaling','_rotation','_opacity','_features_dc')
    parameters = [getattr(pc,n) for n in names]
    bg = torch.tensor([.03,.02,.01],device='cuda')
    pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=True,antialiasing=True)
    configure_pipeline(pipe,dict(occlusion_mode='off',occlusion_backend='ray',occlusion_margin=0.))
    wall = torch.full((64,64),2.,device='cuda')
    intrinsic=torch.tensor([[32.,0,31.5],[0,32.,31.5],[0,0,1]],device='cuda')
    point_labels=classify_points_depth(pc.get_xyz.detach(),intrinsic,view,wall,margin=0.,relative_margin=0.)
    assert point_labels['culled_ids'].tolist()==[1]
    assert point_labels['in_front_ids'].tolist()==[0]
    tests=[]
    for aa in (False,True):
        print(f'CUDA smoke: AA={aa}, checking native/ray forward and backward parity',flush=True)
        pipe.antialiasing=aa
        baseline=render(camera,pc,pipe,bg)
        unbounded=render(camera,pc,pipe,bg,occlusion_depth=torch.full_like(wall,float('inf')))
        for label, result in (('native',baseline),('ray',unbounded)):
            require_finite([(key,result[key]) for key in ('render','depth')],f'AA={aa} {label} output')
        torch.testing.assert_close(unbounded['render'],baseline['render'],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(unbounded['depth'],baseline['depth'],atol=2e-6,rtol=2e-5)
        g0=torch.autograd.grad(baseline['render'].square().sum()+baseline['depth'].sum()*.01,parameters)
        require_finite(zip(names,g0),f'AA={aa} native gradient')
        g1=torch.autograd.grad(unbounded['render'].square().sum()+unbounded['depth'].sum()*.01,parameters)
        require_finite(zip(names,g1),f'AA={aa} ray gradient')
        for name,a,b in zip(names,g0,g1):
            torch.testing.assert_close(a,b,atol=3e-5,rtol=2e-4,msg=f'AA={aa} native/ray gradient mismatch: {name}')
        layouts=check_raw_depth_gradient_layouts(camera,pc,bg,names,parameters,aa)
        blocked=render(camera,pc,pipe,bg,occlusion_depth=wall)
        require_finite([(key,blocked[key]) for key in ('render','depth')],f'AA={aa} wall output')
        assert blocked['occlusion_fragment_flags'].tolist()==[1,2]
        grads=torch.autograd.grad(blocked['render'].square().sum()+blocked['depth'].sum(),parameters)
        require_finite(zip(names,grads),f'AA={aa} wall gradient')
        for name,gradient in zip(names,grads):
            assert gradient[1].count_nonzero().item()==0,name
        assert grads[-1][0].count_nonzero().item()>0
        hole=wall.clone();hole[31,32]=float('inf')
        partial=render(camera,pc,pipe,bg,occlusion_depth=hole)
        require_finite([(key,partial[key]) for key in ('render','depth')],f'AA={aa} hole output')
        assert partial['occlusion_fragment_flags'].tolist()==[1,3]
        difference=(partial['render']-blocked['render']).abs().sum(0)
        assert difference[31,32]>0
        difference[31,32]=0
        assert difference.max().item()==0,'A hole changed other rays'
        loss_pixels=torch.ones_like(wall);loss_pixels[31,32]=0
        gradients=torch.autograd.grad((partial['render']*loss_pixels).sum(),parameters)
        require_finite(zip(names,gradients),f'AA={aa} hole gradient')
        assert all(g[1].count_nonzero().item()==0 for g in gradients),'Blocked-pixel loss leaked to background'
        tests.append(dict(antialiasing=aa,baseline_forward_backward_parity=True,
            raw_depth_gradient_layouts=layouts,all_outputs_and_gradients_finite=True,
            transparent_foreground_wall=True,hole_changes_only_its_pixel=True,blocked_pixel_gradients_zero=True))
    optimizer=torch.optim.Adam([pc._features_dc],lr=.01)
    pc._features_dc.grad=torch.ones_like(pc._features_dc);optimizer.step()
    before=pc._features_dc.detach().clone()
    momentum=optimizer.state[pc._features_dc]['exp_avg'].clone()
    masked_adam_step(optimizer,blocked['occlusion_keep'])
    assert torch.equal(before[1],pc._features_dc[1])
    assert torch.equal(momentum[1],optimizer.state[pc._features_dc]['exp_avg'][1])
    report=dict(cuda_device=torch.cuda.get_device_name(),torch=torch.__version__,tests=tests,
        hidden_adam_momentum_frozen=True,cuda_point_depth_classification=True,passed=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
