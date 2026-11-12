"""Range/depth-attenuated gradients AND Adam displacements for LiDAR anchors.

Gradient scaling alone can cancel in Adam's first/second-moment normalization.
We therefore feed w*g to its moments and multiply its resulting displacement by
w as well. This is an explicit optimizer policy, not a change to forward alpha
compositing, ray visibility, or the photometric loss value.
"""
import math

import torch

OPTION_FIELDS=('reference','power','min_weight','metric','decay','half_life',
               'normalization','near_quantile','far_quantile','rank_samples')

def policy(opt):
    if not getattr(opt, 'distance_weighting', False):
        return None
    reference = opt.distance_reference
    power = opt.distance_power
    floor = opt.distance_min_weight
    if (not math.isfinite(reference) or reference <= 0 or not math.isfinite(power) or power <= 0 or
            not math.isfinite(floor) or not 0 < floor <= 1 or opt.distance_chunk_size < 1):
        raise ValueError('Distance reference/power/chunk must be positive; min weight must be in (0, 1]')
    metric=getattr(opt,'distance_metric','range')
    decay=getattr(opt,'distance_decay','inverse_power')
    half_life=getattr(opt,'distance_half_life',2.)
    normalization=getattr(opt,'distance_normalization','absolute')
    near=getattr(opt,'distance_near_quantile',.05)
    far=getattr(opt,'distance_far_quantile',.95)
    samples=getattr(opt,'distance_rank_samples',8192)
    if (metric not in ('range','depth') or decay not in ('inverse_power','exponential') or
            not math.isfinite(half_life) or half_life <= 0):
        raise ValueError('Distance metric/decay must be supported and half-life positive and finite')
    if (normalization not in ('absolute','relative','rank') or
            not math.isfinite(near) or not math.isfinite(far) or not 0<=near<far<=1 or
            type(samples) is not int or samples<16 or
            (normalization!='absolute' and (metric!='depth' or decay!='exponential'))):
        raise ValueError('Relative/rank weighting needs exponential camera depth, valid quantiles and at least 16 samples')
    if (not opt.fixed_lidar or opt.scale_control not in ('clamp','hybrid') or opt.optimizer_type != 'default'):
        raise ValueError('Distance ablation requires LiDAR anchors, hard/hybrid scale control and default Adam')
    config=dict(reference=reference, power=power, min_weight=floor,
                semantics='weighted_gradient_and_adam_displacement_v1')
    # Preserve exact policy identity for already-published range/power runs.
    if metric!='range' or decay!='inverse_power':
        config.update(metric=metric,decay=decay,half_life=half_life,
                      normalization=normalization,near_quantile=near,far_quantile=far,rank_samples=samples,
                      semantics='weighted_gradient_and_adam_displacement_v2')
    return config


def bind_policy(model, opt, *, fresh):
    requested = policy(opt)
    state = model.fixed_lidar_state
    if fresh:
        if requested is not None:
            state['distance_update_policy'] = requested
    elif state.get('distance_update_policy') != requested:
        raise ValueError('Checkpoint distance update policy differs from requested policy')


def _distance(xyz, center, config, view):
    if config.get('metric','range')=='depth':
        # Graphdeco stores the transposed world-to-camera matrix.
        return xyz@view[:3,2]+view[3,2]
    return (xyz-center).norm(dim=1)


@torch.no_grad()
def weights(xyz, camera_center, config, chunk_size=131072, *, world_view_transform=None, keep=None, diagnostics=None):
    if xyz.ndim != 2 or xyz.shape[1] != 3 or camera_center.shape != (3,):
        raise ValueError('Distance weighting needs XYZ rows and a world-space camera center')
    if camera_center.device != xyz.device or not torch.isfinite(camera_center).all():
        raise ValueError('Camera center must be finite and on the Gaussian device')
    if chunk_size<1:
        raise ValueError('Distance chunk size must be positive')
    if config.get('metric','range')=='depth':
        view=world_view_transform
        if (view is None or view.shape!=(4,4) or view.device!=xyz.device or
                view.dtype!=xyz.dtype or not torch.isfinite(view).all()):
            raise ValueError('Depth weighting needs the current corrected world-view transform on the Gaussian device')
    normalization=config.get('normalization','absolute')
    if keep is not None and (keep.shape!=(len(xyz),) or keep.device!=xyz.device or keep.dtype!=torch.bool):
        raise ValueError('Depth reference needs a boolean contribution mask on the Gaussian device')
    if normalization!='absolute' and keep is None:
        raise ValueError('Relative/rank depth needs the current rendering contribution mask')
    result = xyz.new_zeros(len(xyz))
    reference = config['reference']
    if normalization!='absolute':
        indices=keep.nonzero().flatten()
        if len(indices)>config['rank_samples']:
            # Private, deterministic sampling does not perturb training RNGs.
            generator=torch.Generator(device=xyz.device).manual_seed(1729)
            indices=indices[torch.randint(len(indices),(config['rank_samples'],),device=xyz.device,generator=generator)]
        samples=_distance(xyz[indices],camera_center,config,world_view_transform)
        samples=samples[(samples>0)&torch.isfinite(samples)]
        if not len(samples):
            if diagnostics is not None:diagnostics['reference_sample_count']=0
            return result
        samples=samples.sort().values.contiguous()
        reference=torch.quantile(samples,config['near_quantile'])
        if diagnostics is not None:
            diagnostics.update(reference_sample_count=len(samples),reference_depth=float(reference),
                               reference_far_depth=float(torch.quantile(samples,config['far_quantile'])))
    for start in range(0, len(xyz), chunk_size):
        rows=slice(start,start+chunk_size) if keep is None else keep[start:start+chunk_size].nonzero().flatten()+start
        if keep is not None and not len(rows):continue
        distance = _distance(xyz[rows],camera_center,config,world_view_transform)
        if normalization=='rank':
            # Equal-depth rows receive equal rank. In an all-equal-depth view,
            # every row remains full weight instead of arbitrary order decay.
            rank=torch.searchsorted(samples,distance.contiguous(),right=False)/len(samples)
            rank=((rank-config['near_quantile'])/(config['far_quantile']-config['near_quantile'])).clamp(0,1)
            factor=torch.exp(rank*math.log(config['min_weight']))
        elif config.get('decay','inverse_power')=='exponential':
            factor=torch.exp2(-(distance-reference).clamp_min(0)/config['half_life'])
        else:
            factor=(reference/distance.clamp_min(reference)).pow(config['power'])
        factor=factor.clamp_min(config['min_weight'])
        if config.get('metric','range')=='depth':
            factor=torch.where((distance>0)&torch.isfinite(distance),factor,0.)
        result[rows]=factor
    return result


@torch.no_grad()
def diagnostic_sample(xyz, center, weight, visible, limit=2048, *, config=None, world_view_transform=None):
    indices = visible.flatten()
    if indices.dtype == torch.bool:
        indices = indices.nonzero().flatten()
    if len(indices) > limit:
        indices = indices[torch.linspace(0, len(indices)-1, limit, device=xyz.device).long()]
    result = dict(sampled_visible_gaussians=len(indices))
    if len(indices):
        distance = _distance(xyz[indices],center,config or {},world_view_transform)
        for name, value in (('distance',distance), ('weight',weight[indices])):
            q = torch.quantile(value.float(), value.new_tensor([0,.5,.9,1.]))
            result.update({name+'_'+k:float(v) for k,v in zip(('min','median','p90','max'),q)})
        result['weight_at_floor_fraction']=float((weight[indices]<=(config or {}).get('min_weight',0.)+1e-6).float().mean())
    return indices, result


@torch.no_grad()
def adam_step(optimizer, weight, chunk_size=131072, sample_indices=None, *, keep=None):
    """Chunked default Adam; checkpoint state remains torch.optim.Adam compatible.

With keep supplied, excluded parameters AND both moments stay unchanged. This
preserves culling and inactive-reserve freezing. Without keep, retain the legacy
all-row update semantics. The global per-parameter Adam clock still advances.
"""
    if not isinstance(optimizer, torch.optim.Adam) or weight.ndim != 1 or chunk_size < 1:
        raise ValueError('Distance updates require Adam, one weight per row and a positive chunk size')
    if not bool((torch.isfinite(weight)&(weight>=0)&(weight<=1)).all()):
        raise ValueError('Distance weights must be finite and within [0,1]')
    if keep is not None and (keep.shape!=weight.shape or keep.dtype!=torch.bool or keep.device!=weight.device):
        raise ValueError('Contribution mask must match the distance weight rows/device')
    # Validate every group before modifying any parameters/state.
    for group in optimizer.param_groups:
        if any(group.get(k, False) for k in ('amsgrad','maximize','capturable','differentiable','fused')) or group.get('weight_decay',0):
            raise ValueError('Distance updates require default Adam without weight decay or optional variants')
        for parameter in group['params']:
            if parameter.grad is not None and (parameter.ndim < 1 or len(parameter) != len(weight) or
                    parameter.device != weight.device or parameter.grad.is_sparse):
                raise ValueError('Distance weights do not match dense optimizer rows/device')
    result = {}
    for group_id, group in enumerate(optimizer.param_groups):
        beta1, beta2 = group['betas']
        for parameter_id, parameter in enumerate(group['params']):
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            if not state:
                state['step'] = torch.tensor(0.)
                state['exp_avg'] = torch.zeros_like(parameter)
                state['exp_avg_sq'] = torch.zeros_like(parameter)
            state['step'] += 1
            step = state['step'].item()
            correction1, correction2 = 1-beta1**step, math.sqrt(1-beta2**step)
            if sample_indices is not None:
                before = parameter[sample_indices].clone()
            for start in range(0,len(weight),chunk_size):
                if keep is None:
                    rows = slice(start,start+chunk_size)
                else:
                    rows = keep[start:start+chunk_size].nonzero().flatten()+start
                    if not len(rows):
                        continue
                w = weight[rows].reshape(-1,*([1]*(parameter.ndim-1)))
                gradient = parameter.grad[rows]*w
                first, second = state['exp_avg'][rows], state['exp_avg_sq'][rows]
                first.lerp_(gradient,1-beta1)
                second.mul_(beta2).addcmul_(gradient,gradient,value=1-beta2)
                denominator = second.sqrt().div_(correction2).add_(group['eps'])
                if keep is None:
                    parameter[rows].addcdiv_(first*w,denominator,value=-group['lr']/correction1)
                else:
                    # Advanced indexing returns copies; explicitly scatter all
                    # three updated arrays back to their original rows.
                    parameter[rows]=parameter[rows]-(group['lr']/correction1)*first*w/denominator
                    state['exp_avg'][rows],state['exp_avg_sq'][rows]=first,second
            if sample_indices is not None:
                # Compare measured displacement to the SAME weighted-moment
                # Adam direction without its final distance multiplier.
                raw = parameter.grad[sample_indices]
                w = weight[sample_indices].reshape(-1,*([1]*(parameter.ndim-1)))
                proposal = (group['lr']/correction1)*state['exp_avg'][sample_indices]/(
                    state['exp_avg_sq'][sample_indices].sqrt()/correction2+group['eps'])
                if keep is not None:
                    allowed=keep[sample_indices].reshape(-1,*([1]*(parameter.ndim-1)))
                    proposal=proposal*allowed
                    raw=raw*allowed
                moved = parameter[sample_indices]-before
                raw_norm, proposed_norm = float(raw.norm()), float(proposal.norm())
                prefix = group.get('name',f'group{group_id}')+f'_{parameter_id}_'
                result.update({prefix+'raw_gradient_norm':raw_norm,
                    prefix+'weighted_gradient_norm':float((raw*w).norm()),
                    prefix+'adam_proposal_norm':proposed_norm,
                    prefix+'applied_step_norm':float(moved.norm()),
                    prefix+'applied_to_proposal_ratio':float(moved.norm())/proposed_norm if proposed_norm else None})
    return result
