"""Small, persistent displacement balls around immutable input LiDAR rows."""
import json
import math

import torch


DEFAULTS=dict(schema=1,max_world=.005,sigma_ratio=.25,warmup=7000,ramp=2000,
              end=30000,lr_init=1e-5,lr_final=1e-6,prior_weight=.001)


def validate(policy,iterations):
    if not isinstance(policy,dict) or set(policy)!=set(DEFAULTS) or policy['schema']!=1:
        raise ValueError('Invalid LiDAR offset policy')
    if (not all(type(policy[k]) is int for k in ('warmup','ramp','end')) or
            not 0<=policy['warmup']<policy['warmup']+policy['ramp']<policy['end']<=iterations):
        raise ValueError('LiDAR offsets need warmup < ramp end < training end')
    for key in ('max_world','sigma_ratio','lr_init','lr_final'):
        if not isinstance(policy[key],(int,float)) or not math.isfinite(policy[key]) or policy[key]<=0:
            raise ValueError(f'LiDAR offset {key} must be positive and finite')
    if (policy['sigma_ratio']>.5 or policy['lr_final']>policy['lr_init'] or
            not math.isfinite(policy['prior_weight']) or policy['prior_weight']<0):
        raise ValueError('Use local offsets <= half initial spacing, decreasing LR and nonnegative prior')
    return policy


def requested(opt):
    value=getattr(opt,'lidar_offsets','')
    policy=validate(json.loads(value),opt.iterations) if value else None
    if policy is not None and (not opt.fixed_lidar or opt.scale_control!='hybrid' or not getattr(opt,'pose_refinement','')):
        raise ValueError('LiDAR offsets require the constant-population local-scale pose-refinement run')
    return policy


@torch.no_grad()
def initialize(model,policy):
    from utils.fixed_lidar import xyz_digest
    anchors=model._xyz.detach().clone()
    radius=(model._scaling[:,:1].detach().exp()*policy['sigma_ratio']).clamp_max(policy['max_world'])
    return dict(policy=policy,anchors=anchors,radius=radius,radius_sha256=xyz_digest(radius),iteration=0)


def state(model):
    return (model.fixed_lidar_state or {}).get('position_offsets')


def bind(model,opt):
    from utils.fixed_lidar import xyz_digest
    policy=requested(opt);saved=state(model)
    if (saved['policy'] if saved else None)!=policy:
        raise ValueError('Checkpoint LiDAR offset policy differs from requested policy')
    if saved is None: return
    if (saved['anchors'].shape!=model._xyz.shape or saved['radius'].shape!=(len(model._xyz),1) or
            xyz_digest(saved['anchors'])!=model.fixed_lidar_state['xyz_sha256'] or
            xyz_digest(saved['radius'])!=saved['radius_sha256'] or
            not torch.isfinite(saved['radius']).all() or not (saved['radius']>0).all() or
            bool((saved['radius']>policy['max_world']).any()) or
            type(saved['iteration']) is not int or not 0<=saved['iteration']<=opt.iterations):
        raise ValueError('Invalid original LiDAR anchors, displacement radii or saved iteration')
    padding=model.local_scale_audit['metadata']['config'].get('position_padding',0.)
    if padding<policy['max_world']:
        raise ValueError('Local-scale overlap cache must include LiDAR offset padding')
    with torch.no_grad():
        distance=(model._xyz-saved['anchors']).norm(dim=1)
        if (not torch.isfinite(distance).all() or bool((distance>saved['radius'][:,0]).any()) or
                (saved['iteration']<=policy['warmup'] and bool((distance!=0).any()))):
            raise ValueError('Saved LiDAR positions violate displacement or warmup bounds')


def learning_rate(policy,iteration):
    if iteration<=policy['warmup']: return 0.
    phase=min(1.,(iteration-policy['warmup'])/policy['ramp'])
    ramp=.5-.5*math.cos(math.pi*phase)
    decay=max(0.,min(1.,(iteration-policy['warmup']-policy['ramp'])/
                            (policy['end']-policy['warmup']-policy['ramp'])))
    return ramp*policy['lr_init']*(policy['lr_final']/policy['lr_init'])**decay


def update(model,iteration):
    saved=state(model)
    if saved is None: return 0.
    if iteration<saved['iteration']: raise ValueError('LiDAR offset schedule cannot rewind')
    saved['iteration']=iteration
    lr=learning_rate(saved['policy'],iteration)
    model._xyz.requires_grad_(lr>0)
    if not lr: model._xyz.grad=None
    for group in model.optimizer.param_groups:
        if group['name']=='xyz': group['lr']=lr
    return lr


@torch.no_grad()
def project(model,chunk=1048576):
    saved=state(model)
    if saved is None: return
    if len(model._xyz)!=len(saved['anchors']): raise RuntimeError('LiDAR anchor population changed')
    momentum=model.optimizer.state.get(model._xyz,{}) if model.optimizer else {}
    for start in range(0,len(model._xyz),chunk):
        part=slice(start,start+chunk)
        xyz,anchor,radius=model._xyz[part],saved['anchors'][part],saved['radius'][part]
        if not torch.isfinite(xyz).all(): raise RuntimeError('Nonfinite LiDAR offset proposal')
        delta=xyz-anchor;distance=delta.norm(dim=1,keepdim=True);outside=distance>radius
        # Reserve a representable margin before adding a tiny displacement to
        # large absolute coordinates. Never relax the physical bound for FP32.
        margin=4*torch.finfo(xyz.dtype).eps*(anchor.abs().amax(1,keepdim=True)+radius+1)
        safe=(radius-margin).clamp_min(0)
        candidate=anchor+delta*(safe/distance.clamp_min(torch.finfo(xyz.dtype).tiny)).clamp_max(1)
        xyz.copy_(torch.where(outside,candidate,xyz))
        still_outside=(xyz-anchor).norm(dim=1,keepdim=True)>radius
        xyz.copy_(torch.where(still_outside,anchor,xyz))
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if key in momentum: momentum[key][part].masked_fill_(outside.expand_as(xyz),0.)


def regularization(model):
    saved=state(model)
    if saved is None or not model._xyz.requires_grad: return model._xyz.new_zeros(())
    relative=(model._xyz-saved['anchors'])/saved['radius']
    return saved['policy']['prior_weight']*relative.square().sum(1).mean()


@torch.no_grad()
def audit(model):
    from utils.fixed_lidar import xyz_digest
    saved=state(model)
    anchors_ok=xyz_digest(saved['anchors'])==model.fixed_lidar_state['xyz_sha256']
    d=(model._xyz-saved['anchors']).norm(dim=1)
    radius=saved['radius'][:,0]
    within=bool(torch.isfinite(d).all() and (d<=radius).all())
    unchanged=bool((d==0).all())
    if not anchors_ok or not within or (saved['iteration']<=saved['policy']['warmup'] and not unchanged):
        raise RuntimeError('LiDAR offset audit failed: original anchor, displacement bound or warmup changed')
    return dict(positions_unchanged=unchanged,positions_within_bounds=within,anchors_unchanged=anchors_ok,
        lidar_offset_policy=saved['policy'],offset_iteration=saved['iteration'],
        offset_max=float(d.max()),offset_mean=float(d.mean()),offset_radius_max=float(radius.max()),
        offset_moved_fraction=float((d>0).float().mean()),
        offset_near_bound_fraction=float((d>=.99*radius).float().mean()),
        offset_xyz_lr=next(g['lr'] for g in model.optimizer.param_groups if g['name']=='xyz'))
