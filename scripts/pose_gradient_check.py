"""Tiny independent float64 reference for the CUDA pose preflight, never training."""
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def probe_model(device):
    from utils.pose_refinement import rotation_quaternion
    from utils.sh_utils import RGB2SH
    features=torch.linspace(-.04,.04,12,device=device).reshape(1,4,3)
    features[:,0]=RGB2SH(torch.tensor([.7,.3,.2],device=device))
    return SimpleNamespace(get_xyz=torch.tensor([[.12,.08,2.1]],device=device),
        get_scaling=torch.tensor([[.30,.18,.10]],device=device),
        get_rotation=rotation_quaternion(torch.tensor([[.25,-.1,.3]],device=device)),
        get_opacity=torch.tensor([[.6]],device=device),get_features=features,
        active_sh_degree=1,max_sh_degree=1)


def single_splat_patch(camera,gaussians,background,translation,rotation,antialiasing):
    """Analytic one-splat image on a fixed 6x6 patch, away from native cutoffs.

    Unlike the production transformed-quaternion path, this reference rotates
    world covariance matrices with a matrix exponential. All arithmetic is
    float64; finite differences still use the installed CUDA renderer.
    """
    from utils.pose_refinement import rotation_matrix
    from utils.sh_utils import eval_sh
    if len(gaussians.get_xyz)!=1: raise ValueError('Reference requires exactly one splat')
    t,r=translation.double(),rotation.double()
    zero=r.new_zeros(())
    skew=torch.stack((zero,-r[2],r[1],r[2],zero,-r[0],-r[1],r[0],zero)).reshape(3,3)
    base=camera.world_view_transform.T.detach().double()
    R=torch.matrix_exp(skew)@base[:3,:3]
    center=camera.camera_center.detach().double()+base[:3,:3].T@t
    xyz=gaussians.get_xyz.detach().double()[0]
    mean=R@(xyz-center)
    q=gaussians.get_rotation.detach().double()[0]
    Q=rotation_matrix(q)
    covariance=R@(Q@torch.diag(gaussians.get_scaling.detach().double()[0].square())@Q.T)@R.T
    fx=camera.image_width/(2*math.tan(camera.FoVx/2))
    fy=camera.image_height/(2*math.tan(camera.FoVy/2))
    x,y,z=mean.unbind()
    J=torch.stack((fx/z,zero,-fx*x/z.square(),zero,fy/z,-fy*y/z.square())).reshape(2,3)
    screen=J@covariance@J.T
    filtered=screen+.3*torch.eye(2,dtype=screen.dtype,device=screen.device)
    factor=(torch.linalg.det(screen)/torch.linalg.det(filtered)).clamp_min(.000025).sqrt() if antialiasing else 1.
    clip=torch.cat((mean,mean.new_ones(1)))@camera.projection_matrix.double()
    ndc=clip[:2]/(clip[3]+1e-7)
    pixel=((ndc+1)*mean.new_tensor([camera.image_width,camera.image_height])-1)*.5
    yy,xx=torch.meshgrid(torch.arange(21,27,device=mean.device),torch.arange(19,25,device=mean.device),indexing='ij')
    delta=pixel-torch.stack((xx,yy),-1).to(mean.dtype)
    power=-.5*torch.einsum('...i,ij,...j->...',delta,torch.linalg.inv(filtered),delta)
    alpha=gaussians.get_opacity.detach().double()[0,0]*factor*power.exp()
    # Explicitly guard the probe's branch: every sampled fragment contributes,
    # no opacity cap/termination, no FOV clamping, and tiles contain the patch.
    mid=.5*filtered.trace();radius=torch.ceil(3*torch.sqrt(mid+torch.sqrt((mid.square()-torch.linalg.det(filtered)).clamp_min(.1))))
    rect_min=torch.trunc((pixel-radius)/16).clamp_min(0)
    rect_max=torch.trunc((pixel+radius+15)/16)
    if not (float(z)>.2 and abs(float(x/z))<1.3*math.tan(camera.FoVx/2) and
            abs(float(y/z))<1.3*math.tan(camera.FoVy/2) and float(alpha.min())>.02 and float(alpha.max())<.95 and
            bool((xx//16>=rect_min[0]).all() and (xx//16<rect_max[0]).all()) and
            bool((yy//16>=rect_min[1]).all() and (yy//16<rect_max[1]).all())):
        raise AssertionError('Pose gradient probe crossed a hard rasterization branch')
    colors=(eval_sh(gaussians.active_sh_degree,gaussians.get_features.detach().double().transpose(1,2),
                   F.normalize((xyz-center)[None],dim=-1))+.5)[0]
    if not bool(((colors>0)&(colors<1)).all()): raise AssertionError('Probe SH color reached clamp')
    return colors[:,None,None]*alpha+background.double()[:,None,None]*(1-alpha)


def cuda_pose_gradient_check(camera,model,pipe,background):
    from gaussian_renderer import render
    from utils.pose_refinement import render_pose,corrected_transform,camera_with_transform
    rows=[]
    for antialiasing in (False,True):
        pipe.antialiasing=antialiasing
        translation=torch.tensor([.005,-.003,.002],device=background.device,requires_grad=True)
        rotation=torch.tensor([.001,-.002,.0015],device=background.device,requires_grad=True)
        transformed=render_pose(camera,model,pipe,background,translation,rotation)[:,21:27,19:25]
        reference=single_splat_patch(camera,model,background,translation,rotation,antialiasing)
        weights=torch.linspace(.2,1.5,reference.numel(),device=background.device,dtype=torch.float64).reshape_as(reference)
        analytic=torch.cat(torch.autograd.grad((transformed.double()*weights).mean(),(translation,rotation)))
        expected=torch.cat(torch.autograd.grad((reference*weights).mean(),(translation,rotation)))
        print(f'Pose gradient AA={antialiasing}: analytic={analytic.tolist()}, float64 reference={expected.tolist()}',flush=True)
        torch.testing.assert_close(transformed.double(),reference,atol=4e-5,rtol=3e-4)
        torch.testing.assert_close(analytic,expected,atol=2e-5,rtol=.05)
        assert torch.isfinite(analytic).all() and analytic[:3].norm()>1e-7 and analytic[3:].norm()>1e-7
        row=dict(antialiasing=antialiasing,analytic=analytic.tolist(),reference=expected.tolist(),finite_differences=[])
        for eps in (.001,.0005):
            numeric=[]
            with torch.no_grad():
                for axis in range(6):
                    values=[]
                    for sign in (-1,1):
                        t,r=translation.detach().clone(),rotation.detach().clone()
                        (t if axis<3 else r)[axis%3]+=sign*eps
                        single_splat_patch(camera,model,background,t,r,antialiasing) # assert stable branches
                        cam=camera_with_transform(camera,corrected_transform(camera,t,r)[0])
                        patch=render(cam,model,pipe,background)['render'][:,21:27,19:25]
                        values.append(float((patch.double()*weights).mean()))
                    numeric.append((values[1]-values[0])/(2*eps))
            numeric=torch.tensor(numeric,device=background.device,dtype=analytic.dtype)
            row['finite_differences'].append(dict(epsilon=eps,gradient=numeric.tolist()))
            # Print vectors before asserting so a failure remains diagnostic.
            print(f'Pose gradient AA={antialiasing}, eps={eps}: analytic={analytic.tolist()}, numeric={numeric.tolist()}',flush=True)
            torch.testing.assert_close(analytic,numeric,atol=2e-5,rtol=.05)
        rows.append(row)
    return rows
