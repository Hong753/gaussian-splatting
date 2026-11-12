"""Adaptive activation of reserved, unique original LiDAR anchors.

Training keeps fixed-size parameter/Adam arrays. Inactive rows have exactly zero
rendering opacity and never update; activating a row exposes its untouched LiDAR
initialization. Each active anchor may learn the existing bounded XYZ offset.
"""
import json
import math
from pathlib import Path

import torch

DEFAULTS = dict(schema=1, fraction=.5, voxel=.05, neighbors=16, seed=0,
                start=7000, end=22000, interval=100, gradient_threshold=.0002,
                min_observations=2, max_new=20000, radius=.08)


def validate(policy, iterations):
    if not isinstance(policy, dict) or set(policy) != set(DEFAULTS) or policy['schema'] != 1:
        raise ValueError('Invalid LiDAR activation policy')
    ints=('neighbors','seed','start','end','interval','min_observations','max_new')
    if (not all(type(policy[k]) is int for k in ints) or
            not 0 <= policy['start'] < policy['end'] <= iterations or policy['seed'] < 0 or
            not 3 <= policy['neighbors'] <= 128 or
            min(policy[k] for k in ('interval','min_observations','max_new')) < 1 or
            not all(math.isfinite(policy[k]) and policy[k] > 0 for k in ('fraction','voxel','radius','gradient_threshold')) or
            policy['fraction'] >= 1):
        raise ValueError('Invalid LiDAR activation bounds, reserve fraction or schedule')
    return dict(policy)


def requested(opt):
    value=getattr(opt,'lidar_activation','')
    cfg=validate(json.loads(value),opt.iterations) if value else None
    if cfg and (not getattr(opt,'pose_culling','') or not getattr(opt,'lidar_offsets','') or
                not getattr(opt,'fixed_lidar',False) or not getattr(opt,'lidar_activation_cache','')):
        raise ValueError('LiDAR activation requires bounded offsets, pose-aware culling and a candidate cache')
    return cfg


def state(model):
    return (getattr(model,'fixed_lidar_state',None) or {}).get('lidar_activation')


def active_mask(model):
    saved=state(model)
    return saved['active'] if saved is not None else None


@torch.no_grad()
def bind(model,opt,*,fresh):
    import numpy as np
    from utils.discrete_lidar_cache import read_metadata
    cfg=requested(opt);saved=state(model)
    if not fresh and (saved['policy'] if saved else None) != cfg:
        raise ValueError('Checkpoint LiDAR activation policy differs from requested policy')
    if cfg is None: return
    directory=Path(opt.lidar_activation_cache).resolve()
    meta=read_metadata(directory)
    if any(meta['config'][k] != cfg[k] for k in ('fraction','voxel','neighbors','seed')):
        raise ValueError('LiDAR activation cache settings differ from policy')
    anchors=model.fixed_lidar_state['position_offsets']['anchors']
    if len(anchors) != meta['original_count']:
        raise ValueError('LiDAR activation cache and original anchor counts differ')
    device=anchors.device
    source=torch.tensor(np.load(directory/'source_ids.npy',mmap_mode='r'),device=device,dtype=torch.long)
    xyz=np.load(directory/'xyz.npy',mmap_mode='r')
    for start in range(0,len(source),1048576):
        end=min(start+1048576,len(source))
        if not torch.equal(anchors[source[start:end]],torch.tensor(xyz[start:end],device=device)):
            raise ValueError('LiDAR activation cache differs from immutable original anchors')
    reverse=torch.full((len(anchors),),-1,device=device,dtype=torch.int32)
    reverse[source]=torch.arange(len(source),device=device,dtype=torch.int32)
    model.lidar_activation_pool=dict(source_ids=source,reverse=reverse,metadata=meta,
        neighbors=torch.tensor(np.load(directory/'neighbors.npy',mmap_mode='r'),device=device))
    if fresh:
        seeds=torch.tensor(np.load(directory/'initial_ids.npy'),device=device,dtype=torch.long)
        active=torch.zeros(len(anchors),device=device,dtype=torch.bool)
        active[source[seeds]]=True
        if len(seeds) >= len(source):
            raise ValueError('Coverage seeds consumed the entire pool; increase activation voxel size for a reserve')
        saved=dict(schema=1,policy=cfg,cache_id=meta['cache_id'],active=active,
                   initial_active=int(active.sum()),last_event=0,total_added=0)
        model.fixed_lidar_state['lidar_activation']=saved
    elif saved['cache_id'] != meta['cache_id']:
        raise ValueError('Checkpoint LiDAR activation cache changed')
    audit(model,0)


@torch.no_grad()
def audit(model,iteration):
    saved=state(model)
    if saved is None: return {}
    active=saved['active'];pool=model.lidar_activation_pool
    if active.dtype != torch.bool or active.shape != (len(model._xyz),) or not bool(active.any()):
        raise RuntimeError('Invalid LiDAR activation mask')
    if bool((pool['reverse'][active] < 0).any()):
        raise RuntimeError('Duplicate/noncanonical LiDAR anchor activated')
    if int(active.sum()) != saved['initial_active']+saved['total_added']:
        raise RuntimeError('LiDAR active population changed outside adaptive activation')
    anchors=model.fixed_lidar_state['position_offsets']['anchors']
    for start in range(0,len(active),1048576):
        part=slice(start,start+1048576);inactive=~active[part]
        if not torch.equal(model._xyz[part][inactive],anchors[part][inactive]):
            raise RuntimeError('Inactive LiDAR anchor moved')
    return dict(iteration=iteration,allocated_gaussians=len(active),active_gaussians=int(active.sum()),
        unique_anchor_pool=len(pool['source_ids']),unused_unique_anchors=len(pool['source_ids'])-int(active.sum()),
        initial_active=saved['initial_active'],total_activated=saved['total_added'],
        unique_anchors=True,inactive_positions_unchanged=True,activation_policy=saved['policy'])


@torch.no_grad()
def visible_candidates(model,ids,camera,pipe):
    from utils.ray_occlusion import pixel_cutoff
    from utils.view_tangent import camera_intrinsics
    xyz=model.fixed_lidar_state['position_offsets']['anchors'][ids]
    view=camera.world_view_transform
    points=xyz@view[:3,:3]+view[3,:3];z=points[:,2]
    k=camera_intrinsics(camera,xyz.device)
    uv=points[:,:2]/z.clamp_min(1e-6)[:,None]*k.diag()[:2]+k[:2,2]
    inside=(z>.2)&torch.isfinite(points).all(1)&(uv[:,0]>=-.5)&(uv[:,0]<camera.image_width-.5)&(uv[:,1]>=-.5)&(uv[:,1]<camera.image_height-.5)
    pix=torch.floor(uv+.5).long()
    cutoff=pixel_cutoff(camera,model,pipe)[pix[:,1].clamp(0,camera.image_height-1),pix[:,0].clamp(0,camera.image_width-1)]
    # Growth requires a supported ray; unknown pixels still render normally for
    # existing active Gaussians but cannot validate a new anchor's visibility.
    return inside&torch.isfinite(cutoff)&(z<=cutoff)


@torch.no_grad()
def adapt(model,camera,pipe,iteration,contributing):
    saved=state(model)
    if saved is None: return None
    cfg=saved['policy']
    if not cfg['start'] < iteration <= cfg['end'] or iteration % cfg['interval']: return None
    if iteration <= saved['last_event']: raise ValueError('LiDAR activation events cannot rewind')
    active=saved['active'];pool=model.lidar_activation_pool
    gradient=(model.xyz_gradient_accum[:,0]/model.denom[:,0].clamp_min(1)).nan_to_num()
    eligible=active&contributing&(model.denom[:,0]>=cfg['min_observations'])&(gradient>=cfg['gradient_threshold'])
    parents=eligible.nonzero().flatten()
    if len(parents)>cfg['max_new']:
        parents=parents[torch.topk(gradient[parents],cfg['max_new'],sorted=True).indices]
    elif len(parents):
        parents=parents[torch.argsort(gradient[parents],descending=True,stable=True)]
    selected=parents.new_empty(0)
    proposals=0
    if len(parents):
        neighbor=pool['neighbors'][pool['reverse'][parents].long()].long()
        ids=pool['source_ids'][neighbor.clamp_min(0)]
        anchors=model.fixed_lidar_state['position_offsets']['anchors']
        valid=(neighbor>=0)&~active[ids]
        valid&=(anchors[ids]-anchors[parents,None]).square().sum(-1)<=cfg['radius']**2
        proposals=int(valid.sum())
        candidates=valid.nonzero()
        if len(candidates):
            valid[candidates[:,0],candidates[:,1]]&=visible_candidates(model,ids[valid],camera,pipe)
        ranks=torch.arange(neighbor.shape[1],device=ids.device).expand_as(ids)
        ranks=torch.where(valid,ranks,neighbor.shape[1])
        best=torch.topk(ranks,min(2,neighbor.shape[1]),largest=False,sorted=True).indices
        proposed=ids.gather(1,best)[valid.gather(1,best)]
        if len(proposed):
            unique,inverse=torch.unique(proposed,sorted=True,return_inverse=True)
            first=torch.full_like(unique,len(proposed))
            first.scatter_reduce_(0,inverse,torch.arange(len(proposed),device=ids.device),reduce='amin')
            selected=proposed[first.sort().values[:cfg['max_new']]]
            # No point duplication, copying of parent attributes, interpolation,
            # pruning or mutation of existing active rows is performed here.
            for group in model.optimizer.param_groups:
                for param in group['params']:
                    momentum=model.optimizer.state.get(param,{})
                    for key in ('exp_avg','exp_avg_sq'):
                        if key in momentum and bool((momentum[key][selected]!=0).any()):
                            raise RuntimeError('Reserved LiDAR rows acquired optimizer momentum')
            active[selected]=True
    saved['last_event']=iteration;saved['total_added']+=len(selected)
    model.xyz_gradient_accum.zero_();model.denom.zero_();model.max_radii2D.zero_()
    return dict(iteration=iteration,image=camera.image_name,eligible_parents=len(parents),
        nearby_unused_proposals=proposals,added=len(selected),active_gaussians=int(active.sum()),
        unused_unique_anchors=len(pool['source_ids'])-int(active.sum()),last_added_original_rows=selected[:32].tolist())
