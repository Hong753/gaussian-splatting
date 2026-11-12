"""Distance-attenuated gradients AND Adam displacements for fixed LiDAR.

Gradient scaling alone can cancel in Adam's first/second-moment normalization.
We therefore feed w*g to its moments and multiply its resulting displacement by
w as well. This is an explicit optimizer policy, not a change to forward alpha
compositing, ray visibility, or the photometric loss value.
"""
import math

import torch


def policy(opt):
    if not getattr(opt, 'distance_weighting', False):
        return None
    reference = opt.distance_reference
    power = opt.distance_power
    floor = opt.distance_min_weight
    if (not math.isfinite(reference) or reference <= 0 or not math.isfinite(power) or power <= 0 or
            not math.isfinite(floor) or not 0 < floor <= 1 or opt.distance_chunk_size < 1):
        raise ValueError('Distance reference/power/chunk must be positive; min weight must be in (0, 1]')
    if (not opt.fixed_lidar or opt.scale_control != 'clamp' or opt.optimizer_type != 'default'):
        raise ValueError('Distance ablation requires fixed LiDAR, hard scale clamping and default Adam')
    return dict(reference=reference, power=power, min_weight=floor,
                semantics='weighted_gradient_and_adam_displacement_v1')


def bind_policy(model, opt, *, fresh):
    requested = policy(opt)
    state = model.fixed_lidar_state
    if fresh:
        if requested is not None:
            state['distance_update_policy'] = requested
    elif state.get('distance_update_policy') != requested:
        raise ValueError('Checkpoint distance update policy differs from requested policy')


@torch.no_grad()
def weights(xyz, camera_center, config, chunk_size=131072):
    if xyz.ndim != 2 or xyz.shape[1] != 3 or camera_center.shape != (3,):
        raise ValueError('Distance weighting needs XYZ rows and a world-space camera center')
    if camera_center.device != xyz.device or not torch.isfinite(camera_center).all():
        raise ValueError('Camera center must be finite and on the Gaussian device')
    result = xyz.new_empty(len(xyz))
    reference = config['reference']
    for start in range(0, len(xyz), chunk_size):
        distance = (xyz[start:start+chunk_size]-camera_center).norm(dim=1)
        result[start:start+chunk_size] = (reference/distance.clamp_min(reference)).pow(config['power']).clamp_min(config['min_weight'])
    return result


@torch.no_grad()
def diagnostic_sample(xyz, center, weight, visible, limit=2048):
    indices = visible.flatten()
    if indices.dtype == torch.bool:
        indices = indices.nonzero().flatten()
    if len(indices) > limit:
        indices = indices[torch.linspace(0, len(indices)-1, limit, device=xyz.device).long()]
    result = dict(sampled_visible_gaussians=len(indices))
    if len(indices):
        distance = (xyz[indices]-center).norm(dim=1)
        for name, value in (('distance',distance), ('weight',weight[indices])):
            q = torch.quantile(value.float(), value.new_tensor([0,.5,.9,1.]))
            result.update({name+'_'+k:float(v) for k,v in zip(('min','median','p90','max'),q)})
    return indices, result


@torch.no_grad()
def adam_step(optimizer, weight, chunk_size=131072, sample_indices=None):
    """Chunked default Adam; checkpoint state remains torch.optim.Adam compatible.

All rows receive the distance factor, including residual momentum on rows with
zero current RGB gradient. No extra frustum mask or population edit is applied.
The production weights() function guarantees finite weights in (0,1].
"""
    if not isinstance(optimizer, torch.optim.Adam) or weight.ndim != 1 or chunk_size < 1:
        raise ValueError('Distance updates require Adam, one weight per row and a positive chunk size')
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
                rows = slice(start,start+chunk_size)
                w = weight[rows].reshape(-1,*([1]*(parameter.ndim-1)))
                gradient = parameter.grad[rows]*w
                first, second = state['exp_avg'][rows], state['exp_avg_sq'][rows]
                first.lerp_(gradient,1-beta1)
                second.mul_(beta2).addcmul_(gradient,gradient,value=1-beta2)
                denominator = second.sqrt().div_(correction2).add_(group['eps'])
                parameter[rows].addcdiv_(first*w,denominator,value=-group['lr']/correction1)
            if sample_indices is not None:
                # Compare measured displacement to the SAME weighted-moment
                # Adam direction without its final distance multiplier.
                raw = parameter.grad[sample_indices]
                w = weight[sample_indices].reshape(-1,*([1]*(parameter.ndim-1)))
                proposal = (group['lr']/correction1)*state['exp_avg'][sample_indices]/(
                    state['exp_avg_sq'][sample_indices].sqrt()/correction2+group['eps'])
                moved = parameter[sample_indices]-before
                raw_norm, proposed_norm = float(raw.norm()), float(proposal.norm())
                prefix = group.get('name',f'group{group_id}')+f'_{parameter_id}_'
                result.update({prefix+'raw_gradient_norm':raw_norm,
                    prefix+'weighted_gradient_norm':float((raw*w).norm()),
                    prefix+'adam_proposal_norm':proposed_norm,
                    prefix+'applied_step_norm':float(moved.norm()),
                    prefix+'applied_to_proposal_ratio':float(moved.norm())/proposed_norm if proposed_norm else None})
    return result
