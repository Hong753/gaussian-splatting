"""Bounded, train-view-only pose corrections with alternating frozen-scene steps.

Source camera objects and COLMAP files are never modified. Native rasterization
is differentiated through transformed Gaussian means/orientations and world-SH
colors, because camera settings themselves do not have a native backward path.
"""
from copy import copy, deepcopy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULTS = dict(schema=1, start=7000, end=20000, every=1, max_translation=.03,
                max_rotation_degrees=.5, translation_lr=.001, rotation_lr=.0002,
                final_lr_factor=.1, prior_weight=.001, max_width=960)


def validate(policy, iterations):
    if not isinstance(policy, dict) or set(policy) != set(DEFAULTS) or policy['schema'] != 1:
        raise ValueError('Invalid pose-refinement policy')
    if (not all(type(policy[k]) is int for k in ('start','end','every','max_width')) or
            not 0 <= policy['start'] < policy['end'] <= iterations or
            policy['every'] < 1 or policy['max_width'] < 32):
        raise ValueError('Invalid pose update interval or image resolution')
    for k in ('max_translation','max_rotation_degrees','translation_lr','rotation_lr','final_lr_factor','prior_weight'):
        if not isinstance(policy[k], (float,int)) or not math.isfinite(policy[k]) or policy[k] <= 0:
            raise ValueError(f'Pose {k} must be positive and finite')
    if policy['max_rotation_degrees'] > 10 or policy['final_lr_factor'] > 1:
        raise ValueError('Pose corrections must be small (<=10 degrees), with non-increasing LR')
    return policy


def source_fingerprints(source):
    from utils.colmap_paths import colmap_model_dir
    from utils.checkpoint_snapshots import digest
    root=Path(colmap_model_dir(source))
    suffix='bin' if (root/'images.bin').is_file() and (root/'cameras.bin').is_file() else 'txt'
    return {str((root/(name+'.'+suffix)).resolve()):digest(root/(name+'.'+suffix)) for name in ('images','cameras')}


def quaternion_product(a,b):
    aw,ax,ay,az=a.unbind(-1); bw,bx,by,bz=b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw),-1)


def rotation_quaternion(vector):
    theta=vector.square().sum(-1,keepdim=True).clamp_min(1e-12).sqrt()
    return torch.cat((torch.cos(theta/2),.5*torch.sinc(theta/(2*math.pi))*vector),-1)


def rotation_matrix(q):
    w,x,y,z=F.normalize(q,dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def base_quaternion(camera):
    from scipy.spatial.transform import Rotation
    xyzw=Rotation.from_matrix(camera.world_view_transform[:3,:3].T.detach().cpu().numpy()).as_quat()
    return torch.as_tensor(xyzw[[3,0,1,2]],device=camera.world_view_transform.device,dtype=camera.world_view_transform.dtype)


def corrected_transform(camera, translation, rotation):
    base=camera.world_view_transform.T.detach()
    R=rotation_matrix(rotation_quaternion(rotation)) @ base[:3,:3]
    center=camera.camera_center.detach()+base[:3,:3].T @ translation
    upper=torch.cat((R,(-R @ center)[:,None]),1)
    w2c=torch.cat((upper,upper.new_tensor([[0,0,0,1]])),0)
    return w2c,center


def camera_with_transform(camera,w2c):
    out=copy(camera)
    out.world_view_transform=w2c.T.contiguous()
    out.full_proj_transform=out.world_view_transform @ camera.projection_matrix
    out.camera_center=-w2c[:3,:3].T @ w2c[:3,3]
    # R/T are only metadata for the ordinary renderer; keep them consistent too.
    out.R=w2c[:3,:3].T.detach().cpu().numpy()
    out.T=w2c[:3,3].detach().cpu().numpy()
    return out


def render_pose(camera, gaussians, pipe, background, translation, rotation, quaternion=None):
    """Native AA rasterization; only the six pose coordinates receive gradients."""
    from utils.sh_utils import eval_sh
    culling = getattr(pipe, 'occlusion_mode', 'off') != 'off'
    if getattr(pipe,'absgs',False) or (culling and not getattr(pipe, 'pose_culling', None)):
        raise ValueError('Pose refinement requires native 3DGS or explicit pose-aware ray culling')
    extra = {}
    w2c,center=corrected_transform(camera,translation,rotation)
    if culling:
        from diff_gaussian_rasterization_occlusion import GaussianRasterizationSettings, GaussianRasterizer
        from utils.ray_occlusion import pixel_cutoff
        with torch.no_grad():
            posed = camera_with_transform(camera, w2c.detach())
            posed.occlusion_transient = True
            extra['occlusion_depth'] = pixel_cutoff(posed, gaussians, pipe)
    else:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    with torch.no_grad():
        xyz=gaussians.get_xyz.detach()
        scales=gaussians.get_scaling.detach()
        rotations=gaussians.get_rotation.detach()
        opacity=gaussians.get_opacity.detach()
        sh=gaussians.get_features.detach().transpose(1,2)
    means=(xyz-center) @ w2c[:3,:3].T
    q=quaternion_product(rotation_quaternion(rotation),quaternion if quaternion is not None else base_quaternion(camera))
    rotations=quaternion_product(q,rotations)
    if pipe.antialiasing:
        from utils.pose_antialiasing import pose_compensation
        opacity=opacity*pose_compensation(means,scales,rotations,camera)
    directions=F.normalize(xyz-center,dim=-1,eps=1e-8)
    colors=(eval_sh(gaussians.active_sh_degree,sh,directions)+.5).clamp_min(0).contiguous()
    identity=torch.eye(4,device=xyz.device,dtype=xyz.dtype)
    settings=GaussianRasterizationSettings(image_height=camera.image_height,image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2),tanfovy=math.tan(camera.FoVy/2),bg=background,
        scale_modifier=1.,viewmatrix=identity,projmatrix=camera.projection_matrix,
        sh_degree=gaussians.active_sh_degree,campos=xyz.new_zeros(3),prefiltered=False,
        # Explicit compensation above retains the AA forward image and avoids
        # the pinned native AA compensation derivative's double-added variance.
        debug=getattr(pipe,'debug',False),antialiasing=False, **extra)
    result=GaussianRasterizer(raster_settings=settings)(means3D=means,means2D=torch.zeros_like(means),
        shs=None,colors_precomp=colors,opacities=opacity,scales=scales,rotations=rotations,cov3D_precomp=None)
    if len(result)>2 and result[2].requires_grad:
        result[2].register_hook(lambda g:g.contiguous() if g is not None else None)
    return result[0].clamp(0,1)


def image_loss(prediction,target,mask):
    # Smooth L1-like photometric objective; normalize over valid RGB samples.
    residual=((prediction-target).square()+1e-6).sqrt()-.001
    return (residual*mask).sum()/(3*mask.sum()).clamp_min(1)


class PoseRefiner:
    def __init__(self,cameras,policy,source,iterations, saved=None):
        self.policy=validate(policy,iterations)
        self.total_iterations=iterations
        self.source=str(Path(source).resolve())
        self.fingerprints=source_fingerprints(source)
        self.cameras={c.image_name:c for c in cameras}
        if len(self.cameras)!=len(cameras) or not cameras:
            raise ValueError('Pose refinement requires unique, nonempty training cameras')
        self.names=sorted(self.cameras)
        self.base={n:self.cameras[n].world_view_transform.T.detach().cpu().tolist() for n in self.names}
        self.params={n:[torch.nn.Parameter(self.cameras[n].camera_center.new_zeros(3)) for _ in range(2)] for n in self.names}
        self.quaternions={n:base_quaternion(self.cameras[n]) for n in self.names}
        self.optimizers={}
        self.counts={n:dict(attempts=0,accepted=0) for n in self.names}
        self.iteration=0
        if saved is not None:
            self.restore(saved)

    def optimizer(self,name):
        if name not in self.optimizers:
            self.optimizers[name]=torch.optim.Adam([
                dict(params=[self.params[name][0]],lr=self.policy['translation_lr']),
                dict(params=[self.params[name][1]],lr=self.policy['rotation_lr'])],eps=1e-8)
        return self.optimizers[name]

    def prior(self,name):
        t,r=self.params[name]
        return self.policy['prior_weight']*((t/self.policy['max_translation']).square().sum()
            +(r/math.radians(self.policy['max_rotation_degrees'])).square().sum())

    @torch.no_grad()
    def project(self,name):
        clipped=False
        for param,limit in zip(self.params[name],(self.policy['max_translation'],math.radians(self.policy['max_rotation_degrees']))):
            norm=float(param.norm())
            if norm>limit:
                param.mul_(limit/norm)
                for key in ('exp_avg','exp_avg_sq'):
                    value=self.optimizer(name).state.get(param,{}).get(key)
                    if value is not None: value.zero_()
                clipped=True
        return clipped

    def update(self,camera,gaussians,pipe,background,iteration,renderer=render_pose):
        if iteration<self.iteration:
            raise ValueError('Pose-refinement iteration cannot rewind')
        self.iteration=iteration
        cfg=self.policy
        if not cfg['start']<iteration<=cfg['end'] or (iteration-cfg['start']-1)%cfg['every']:
            return None
        name=camera.image_name
        if name not in self.params: raise ValueError('Only training cameras can be optimized')
        cam=copy(camera)
        factor=min(1.,cfg['max_width']/camera.image_width)
        h,w=max(1,round(camera.image_height*factor)),max(1,round(camera.image_width*factor))
        cam.image_height,cam.image_width=h,w
        target=F.interpolate(camera.original_image[:3].to(background.device)[None],size=(h,w),mode='bilinear',align_corners=False,antialias=True)[0]
        mask=camera.alpha_mask
        mask=target.new_ones((1,h,w)) if mask is None else F.interpolate(mask.to(background.device)[None],size=(h,w),mode='nearest')[0]
        if getattr(pipe, 'pose_culling', None) and getattr(pipe, 'occlusion_mode', 'off') != 'off':
            from utils.pose_culling import provider
            with torch.no_grad():
                posed=self.camera(cam)
                posed.occlusion_transient=True
                mask=mask*provider(pipe).frame(posed, background.device)['pose_mask'][None]
            if int((mask > 0).sum()) < pipe.pose_culling['pose_min_pixels']:
                return dict(iteration=iteration, image=name, accepted=False, skipped='insufficient_mesh_interior',
                            pose_pixels=int((mask > 0).sum()))
        if not bool(mask.any()): return None
        optimizer=self.optimizer(name)
        phase=(iteration-cfg['start'])/(cfg['end']-cfg['start'])
        factor=cfg['final_lr_factor']**phase
        for group,lr in zip(optimizer.param_groups,(cfg['translation_lr'],cfg['rotation_lr'])): group['lr']=lr*factor
        optimizer.zero_grad(set_to_none=True)
        old_params=[p.detach().clone() for p in self.params[name]]
        old_optimizer=deepcopy(optimizer.state_dict())
        pred=renderer(cam,gaussians,pipe,background,*self.params[name],quaternion=self.quaternions[name])
        photo=image_loss(pred,target,mask)
        objective=photo+self.prior(name)
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f'Nonfinite initial pose objective for {name}')
        objective.backward()
        grad_ok=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.params[name])
        gradients=[float(p.grad.norm()) if p.grad is not None else None for p in self.params[name]]
        before,before_photo=float(objective.detach()),float(photo.detach())
        del pred,photo,objective
        clipped=False
        after=after_photo=None
        if grad_ok:
            optimizer.step()
            if all(bool(torch.isfinite(p).all()) for p in self.params[name]):
                clipped=self.project(name)
                with torch.no_grad():
                    pred=renderer(cam,gaussians,pipe,background,*self.params[name],quaternion=self.quaternions[name])
                    photo=image_loss(pred,target,mask)
                    after=float(photo+self.prior(name));after_photo=float(photo)
                del pred,photo
        accepted=after is not None and math.isfinite(after) and after<=before+1e-8
        if not accepted:
            with torch.no_grad():
                for p,old in zip(self.params[name],old_params): p.copy_(old)
            optimizer.load_state_dict(old_optimizer)
        optimizer.zero_grad(set_to_none=True)
        self.counts[name]['attempts']+=1;self.counts[name]['accepted']+=int(accepted)
        return dict(iteration=iteration,image=name,accepted=accepted,projected=clipped,pose_pixels=int((mask > 0).sum()),
            objective_before=before,objective_candidate=after if after is not None and math.isfinite(after) else None,
            photo_before=before_photo,photo_candidate=after_photo if after_photo is not None and math.isfinite(after_photo) else None,
            translation_norm=float(self.params[name][0].norm()),rotation_degrees=math.degrees(float(self.params[name][1].norm())),
            translation_grad_norm=gradients[0] if grad_ok else None,rotation_grad_norm=gradients[1] if grad_ok else None,
            translation_lr=optimizer.param_groups[0]['lr'],rotation_lr=optimizer.param_groups[1]['lr'])

    @torch.no_grad()
    def camera(self,camera):
        if camera.image_name not in self.params: return camera
        w2c,_=corrected_transform(camera,*self.params[camera.image_name])
        result=camera_with_transform(camera,w2c)
        result.occlusion_transient=self.policy['start'] < self.iteration <= self.policy['end']
        return result

    def capture(self):
        return dict(schema=1,policy=self.policy,source=self.source,fingerprints=self.fingerprints,base=self.base,
            iteration=self.iteration,counts=deepcopy(self.counts),
            offsets={n:[p.detach().cpu().clone() for p in self.params[n]] for n in self.names},
            optimizers={n:deepcopy(o.state_dict()) for n,o in self.optimizers.items()})

    def restore(self,saved):
        if (saved.get('schema')!=1 or saved['policy']!=self.policy or saved['source']!=self.source or
                saved['fingerprints']!=self.fingerprints or saved['base']!=self.base or
                set(saved['offsets'])!=set(self.names) or set(saved['counts'])!=set(self.names)):
            raise ValueError('Pose checkpoint differs from the policy or original training cameras')
        if (type(saved.get('iteration')) is not int or not 0<=saved['iteration']<=self.total_iterations or
                not set(saved['optimizers'])<=set(self.names)):
            raise ValueError('Invalid pose checkpoint iteration or optimizer cameras')
        with torch.no_grad():
            for name in self.names:
                if len(saved['offsets'][name])!=2:
                    raise ValueError('Pose checkpoint needs translation and rotation offsets')
                for p,value,limit in zip(self.params[name],saved['offsets'][name],
                        (self.policy['max_translation'],math.radians(self.policy['max_rotation_degrees']))):
                    if value.shape!=(3,) or not torch.isfinite(value).all() or float(value.norm())>limit+1e-7:
                        raise ValueError('Invalid bounded pose checkpoint offset')
                    p.copy_(value)
        for name,value in saved['optimizers'].items(): self.optimizer(name).load_state_dict(value)
        self.counts=deepcopy(saved['counts']);self.iteration=saved['iteration']

    @torch.no_grad()
    def export(self,folder,iteration):
        from utils.checkpoint_snapshots import atomic_json
        if source_fingerprints(self.source)!=self.fingerprints:
            raise ValueError('Original camera data changed during pose refinement')
        frames={}
        for name in self.names:
            w2c,_=corrected_transform(self.cameras[name],*self.params[name])
            frames[name]=dict(original_world_to_camera=self.base[name],world_to_camera=w2c.cpu().tolist(),
                translation_offset=self.params[name][0].cpu().tolist(),rotation_axis_angle=self.params[name][1].cpu().tolist(),
                **self.counts[name])
        result=dict(schema=1,iteration=iteration,source=self.source,source_fingerprints=self.fingerprints,
                    policy=self.policy,coordinate_convention='COLMAP world-to-camera; offsets in original camera axes',
                    training_images=self.names,frames=frames,heldout_policy='original poses; never optimized')
        path=Path(folder)/'poses.json'
        atomic_json(path,result)
        return path
