"""Fixed measured centers with hard projection or a soft log-scale penalty.

The radius reference is the native initialization scale (RMS distance to the
three nearest points, with the native 1e-7 squared-distance floor). It is not
scene extent or sensor range. References survive optimizer-checkpoint round trips;
exported PLYs contain the learned scales and need no special inference wrapper.
"""
import hashlib
import math

import torch


def validate_limits(ratio, world):
    if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(world) or world < 0:
        raise ValueError('max_scale_ratio must be positive; max_scale_world must be nonnegative; both finite')


def validate_control(mode, weight):
    if mode not in ('clamp', 'regularize', 'hybrid') or not math.isfinite(weight) or weight <= 0:
        raise ValueError('scale_control must be clamp, regularize or hybrid; regularization weight must be positive and finite')


def configure_training(opt, dataset):
    """Canonicalize the fixed-population ablation before recording cfg_args."""
    if not getattr(opt, 'fixed_lidar', False):
        return
    validate_limits(opt.max_scale_ratio, opt.max_scale_world)
    validate_control(opt.scale_control, opt.scale_regularization_weight)
    if opt.scale_control == 'hybrid' and not getattr(opt, 'local_scale_cache', ''):
        raise ValueError('Hybrid scale control requires the original LiDAR local_scale_cache')
    if (dataset.model_type != '3dgs' or opt.optimizer_type != 'default' or
            opt.surface_mode != 'off' or opt.soft_surface_geometry or opt.absgs or
            dataset.occlusion_mode != 'off' or dataset.train_depth_occlusion_dir or
            opt.lidar_depth_weight or dataset.depths or dataset.train_test_exp):
        raise ValueError('Fixed LiDAR ablation uses native 3DGS, default Adam, no exposure, depth or surface losses/culling')
    opt.position_lr_init = opt.position_lr_final = 0.0
    opt.position_warmup_steps = opt.position_ramp_steps = opt.densification_phase_offset = 0
    opt.densify_until_iter = 0  # Also bypasses pruning and opacity resets in train.py.
    for name in ('fixed_lidar', 'max_scale_ratio', 'max_scale_world', 'scale_control', 'scale_regularization_weight', 'local_scale_cache'):
        setattr(dataset, name, getattr(opt, name))


def xyz_digest(xyz, chunk=1048576):
    digest = hashlib.sha256()
    for part in xyz.detach().split(chunk):
        digest.update(part.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def initialize(model, ratio, world, mode='clamp', weight=.01, cache=''):
    validate_limits(ratio, world)
    validate_control(mode, weight)
    if (not len(model._xyz) or not torch.isfinite(model._xyz).all() or
            not torch.isfinite(model._scaling).all()):
        raise ValueError('Fixed LiDAR needs nonempty, finite initialization positions and scales')
    # Native point-cloud initialization is isotropic. Do not silently treat a
    # learned PLY's already inflated scales as the LiDAR spacing reference.
    if not torch.equal(model._scaling[:, :1].expand_as(model._scaling), model._scaling):
        raise ValueError('Initialize fixed LiDAR from the original point cloud, not learned anisotropic scales')
    cap = model._scaling[:, :1].detach().clone() + math.log(ratio)
    if world > 0:
        cap.clamp_(max=math.log(world))
    state = dict(schema=1, ratio=ratio, world=world, count=len(model._xyz),
                 xyz_sha256=xyz_digest(model._xyz), log_scale_cap=cap)
    if mode in ('regularize', 'hybrid'):
        state.update(scale_control=mode, scale_regularization_weight=weight)
    if mode == 'hybrid':
        if not cache:
            raise ValueError('Hybrid scale control requires a local-scale cache')
        from utils.local_scale import load_bounds
        data, cap, target = load_bounds(cache, model._xyz.device, state['xyz_sha256'], state['count'], ratio, world)
        state.update(log_scale_cap=cap, log_scale_target=target, local_cache_id=data['cache_id'])
    return state


@torch.no_grad()
def project_scales(model, chunk=1048576, *, initializing=False):
    state = model.fixed_lidar_state
    if state is None:
        return
    if len(model._xyz) != state['count']:
        raise RuntimeError('Fixed LiDAR population changed')
    if state.get('scale_control', 'clamp') == 'regularize' and not initializing:
        return  # No training-time projection or Adam moment reset in this mode.
    momentum = model.optimizer.state.get(model._scaling, {}) if model.optimizer else {}
    for start in range(0, state['count'], chunk):
        part = slice(start, start+chunk)
        scale, cap = model._scaling[part], state['log_scale_cap'][part]
        exceeded = scale > cap
        torch.minimum(scale, cap, out=scale)
        # Clear both moments on projected axes so stale Adam momentum cannot
        # repeatedly push outward or delay later contraction. Other axes retain
        # their optimizer state. No XYZ optimizer state is ever accumulated.
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in momentum:
                momentum[key][part].masked_fill_(exceeded, 0.)


def scale_regularization(model):
    """lambda * mean_{all points, all 3 axes} relu(log(sigma / target))^2.

    Targets are immutable and independent of view, opacity and scene extent.
    Means/rotation/opacity do not receive gradients from this term. Calling it
    after RGB backward releases the large rasterizer graph first.
    """
    state = model.fixed_lidar_state
    if state is None or state.get('scale_control', 'clamp') not in ('regularize', 'hybrid'):
        return model._scaling.new_zeros(())
    excess = torch.relu(model._scaling - state.get('log_scale_target', state['log_scale_cap']))
    return state['scale_regularization_weight'] * excess.square().mean()


@torch.no_grad()
def sample_scale_gradients(model, visible, limit=2048):
    indices = visible.flatten()
    if indices.dtype == torch.bool:
        indices = indices.nonzero().flatten()
    if len(indices) > limit:
        indices = indices[torch.linspace(0, len(indices)-1, limit, device=indices.device).long()]
    grad = model._scaling.grad
    before = grad[indices].clone() if grad is not None else model._scaling.new_zeros((len(indices), 3))
    return indices, before


@torch.no_grad()
def added_scale_gradient_stats(model, sample):
    indices, before = sample
    after = model._scaling.grad[indices] if model._scaling.grad is not None else torch.zeros_like(before)
    rgb, reg = float(before.norm()), float((after-before).norm())
    return dict(sampled_visible_gaussians=len(indices), rgb_scale_grad_norm=rgb,
                regularizer_scale_grad_norm=reg, regularizer_to_rgb_scale_grad_ratio=reg/rgb if rgb else None)


@torch.no_grad()
def audit(model, iteration):
    state = model.fixed_lidar_state
    same = len(model._xyz) == state['count'] and xyz_digest(model._xyz) == state['xyz_sha256']
    finite = bool(torch.isfinite(model._scaling).all() and torch.isfinite(model.get_scaling).all())
    within = bool((model._scaling <= state['log_scale_cap']).all())
    mode = state.get('scale_control', 'clamp')
    if not same or not finite or (mode != 'regularize' and not within):
        raise RuntimeError('Fixed LiDAR audit failed: changed centers/count or invalid/oversized scales')
    result = dict(iteration=iteration, gaussians=len(model._xyz), positions_unchanged=True,
                scale_control=mode, scales_finite=finite, scales_within_bounds=within,
                max_scale_ratio=state['ratio'], max_scale_world=state['world'],
                above_target_fraction=float((model._scaling > state.get('log_scale_target', state['log_scale_cap'])).any(1).float().mean()),
                at_cap_fraction=float((model._scaling >= state['log_scale_cap']-1e-6).any(1).float().mean()),
                largest_sigma=float(model.get_scaling.max()))
    if mode == 'hybrid':
        from utils.local_scale import audit as local_audit
        result.update(local_audit(model))
    if state.get('distance_update_policy') is not None:
        result['distance_update_policy'] = state['distance_update_policy']
    return result
