"""Fixed measured centers and projected Adam updates for bounded log-scales.

The radius reference is the native initialization scale (RMS distance to the
three nearest points, with the native 1e-7 squared-distance floor). It is not
scene extent or sensor range. Bounds survive optimizer-checkpoint round trips;
ordinary exported PLYs contain the bounded physical scales and need no wrapper.
"""
import hashlib
import math

import torch


def validate_limits(ratio, world):
    if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(world) or world < 0:
        raise ValueError('max_scale_ratio must be positive; max_scale_world must be nonnegative; both finite')


def configure_training(opt, dataset):
    """Canonicalize the fixed-population ablation before recording cfg_args."""
    if not getattr(opt, 'fixed_lidar', False):
        return
    validate_limits(opt.max_scale_ratio, opt.max_scale_world)
    if (dataset.model_type != '3dgs' or opt.optimizer_type != 'default' or
            opt.surface_mode != 'off' or opt.soft_surface_geometry or opt.absgs or
            dataset.occlusion_mode != 'off' or dataset.train_depth_occlusion_dir or
            opt.lidar_depth_weight or dataset.depths or dataset.train_test_exp):
        raise ValueError('Fixed LiDAR ablation uses native 3DGS, default Adam, no exposure, depth or surface losses/culling')
    opt.position_lr_init = opt.position_lr_final = 0.0
    opt.position_warmup_steps = opt.position_ramp_steps = opt.densification_phase_offset = 0
    opt.densify_until_iter = 0  # Also bypasses pruning and opacity resets in train.py.
    for name in ('fixed_lidar', 'max_scale_ratio', 'max_scale_world'):
        setattr(dataset, name, getattr(opt, name))


def xyz_digest(xyz, chunk=1048576):
    digest = hashlib.sha256()
    for part in xyz.detach().split(chunk):
        digest.update(part.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def initialize(model, ratio, world):
    validate_limits(ratio, world)
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
    return dict(schema=1, ratio=ratio, world=world, count=len(model._xyz),
                xyz_sha256=xyz_digest(model._xyz), log_scale_cap=cap)


@torch.no_grad()
def project_scales(model, chunk=1048576):
    state = model.fixed_lidar_state
    if state is None:
        return
    if len(model._xyz) != state['count']:
        raise RuntimeError('Fixed LiDAR population changed')
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


@torch.no_grad()
def audit(model, iteration):
    state = model.fixed_lidar_state
    same = len(model._xyz) == state['count'] and xyz_digest(model._xyz) == state['xyz_sha256']
    valid = torch.isfinite(model._scaling).all() and (model._scaling <= state['log_scale_cap']).all()
    if not same or not valid:
        raise RuntimeError('Fixed LiDAR audit failed: changed centers/count or invalid/oversized scales')
    return dict(iteration=iteration, gaussians=len(model._xyz), positions_unchanged=True,
                scales_within_bounds=True, max_scale_ratio=state['ratio'], max_scale_world=state['world'],
                at_cap_fraction=float((model._scaling >= state['log_scale_cap']-1e-6).any(1).float().mean()),
                largest_sigma=float(model.get_scaling.max()))
