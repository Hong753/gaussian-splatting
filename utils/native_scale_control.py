"""Optional native 3DGS spatial-resolution control; no per-point persistent state.

Scale units are the source scene's coordinates. Raster visibility and opacity
are eligibility proxies, not measured per-pixel compositing contribution.
"""
import math


DEFAULTS = dict(native_forced_split_sigma=0.0, native_forced_split_fraction=.01,
                native_forced_split_max_parents=50000, native_forced_split_population=12000000,
                native_forced_split_min_observations=1, native_forced_split_min_opacity=.05,
                native_longest_axis_split=False, native_split_offset_fraction=.8,
                native_scale_soft_sigma=.05, native_scale_loss_weight=0.0)


def options(opt):
    p = {k:getattr(opt,k,v) for k,v in DEFAULTS.items()}
    if not all(math.isfinite(v) for v in p.values()):
        raise ValueError('Nonfinite native scale control option')
    if (p['native_forced_split_sigma'] < 0 or not 0 < p['native_forced_split_fraction'] <= 1
            or p['native_forced_split_max_parents'] < 1 or p['native_forced_split_population'] < 1
            or p['native_forced_split_min_observations'] < 1
            or not 0 <= p['native_forced_split_min_opacity'] < 1
            or not 0 < p['native_split_offset_fraction'] < 1
            or p['native_scale_soft_sigma'] <= 0 or p['native_scale_loss_weight'] < 0):
        raise ValueError('Invalid native scale control option')
    enabled = p['native_forced_split_sigma'] > 0 or p['native_longest_axis_split'] or p['native_scale_loss_weight'] > 0
    if enabled and (opt.fixed_lidar or opt.discrete_lidar or opt.freeze_lidar_positions
                    or opt.surface_mode != 'off' or opt.optimizer_type != 'default'):
        raise ValueError('Native scale control requires adaptive native 3DGS, default Adam and surface_mode=off')
    return p


def scale_loss(log_scales, sigma):
    """Squared log hinge on the longest axis. No penalty on thin axes."""
    return (log_scales.amax(dim=1)-math.log(sigma)).clamp_min(0).square().mean()


def forced_mask(log_scales, opacity, observations, clone_grad, split_grad,
                clone_threshold, split_threshold, clone_split_sigma, policy):
    """Select additional parents before native growth clears observation stats."""
    import torch
    n = len(log_scales)
    sigma = log_scales.amax(dim=1)
    clone = (clone_grad.reshape(-1) >= clone_threshold) & (sigma <= math.log(clone_split_sigma))
    split = (split_grad.reshape(-1) >= split_threshold) & (sigma > math.log(clone_split_sigma))
    # Require larger than the clone/split boundary so forced parents cannot
    # also be cloned. Existing gradient-eligible split parents are excluded.
    eligible = ((sigma > math.log(max(policy['native_forced_split_sigma'],clone_split_sigma)))
                & (observations.reshape(-1) >= policy['native_forced_split_min_observations'])
                & (opacity.reshape(-1) >= policy['native_forced_split_min_opacity'])
                & ~split & ~clone & torch.isfinite(sigma))
    ids = eligible.nonzero(as_tuple=True)[0]
    native_growth = int(clone.sum()+split.sum())
    budget = min(max(1,int(n*policy['native_forced_split_fraction'])),
                 policy['native_forced_split_max_parents'],
                 max(0,policy['native_forced_split_population']-n-native_growth))
    k = min(len(ids),budget)
    result = torch.zeros(n,dtype=torch.bool,device=log_scales.device)
    if k:
        chosen = ids[torch.topk(sigma[ids],k,sorted=False).indices]
        result[chosen] = True
    stats = dict(forced_split_candidates=len(ids), forced_split_budget=budget,
                 forced_split_parents=k, native_gradient_clone_parents=int(clone.sum()),
                 native_gradient_split_parents=int(split.sum()),
                 forced_split_sigma=policy['native_forced_split_sigma'],
                 longest_axis_split=bool(policy['native_longest_axis_split']))
    return result,stats


def longest_axis_children(log_scales, rotations, opacity_logits, fraction):
    """Equal-weight mixture matches the parent's spatial mean and covariance.

    Optical thickness heuristic for child opacity: preserve approximate
    integrated optical thickness for a tangential split; not an exact
    alpha-compositing invariant across views. SH/rotation are inherited.
    Returns rows ordered as all minus children, then all plus children.
    """
    import torch
    import torch.nn.functional as F
    if not len(log_scales):
        return log_scales.new_empty((0,3)), log_scales.new_empty((0,3)), opacity_logits.new_empty((0,1))
    axis = log_scales.argmax(dim=1,keepdim=True)
    maximum = log_scales.gather(1,axis).exp()
    unit = rotations.gather(2,axis[:,None,:].expand(-1,3,1)).squeeze(2)
    offset = unit * maximum * fraction
    shrink = math.sqrt(1-fraction*fraction)
    new_scales = log_scales.clone()
    new_scales.scatter_add_(1,axis,torch.full_like(maximum,math.log(shrink)))
    # tau=softplus(logit), logit(alpha_child)=log(expm1(tau_child)).
    # Stable for both large and small optical thickness.
    tau = F.softplus(opacity_logits)/(2*shrink)
    child_logits = tau + torch.log((-torch.expm1(-tau)).clamp_min(torch.finfo(tau.dtype).tiny))
    return torch.cat((-offset,offset),0), new_scales.repeat(2,1), child_logits.repeat(2,1)


def self_test():
    import torch
    s = torch.tensor([[.02,.12,.004],[.2,.03,.01]],dtype=torch.float64).log()
    r = torch.eye(3,dtype=s.dtype).repeat(2,1,1)
    logits = torch.tensor([[0.],[4.]],dtype=s.dtype)
    offsets, children, alpha = longest_axis_children(s,r,logits,.8)
    assert torch.allclose(offsets[:2]+offsets[2:],torch.zeros_like(offsets[:2]))
    assert torch.allclose(children[:2].exp().square()+offsets[:2].square(),s.exp().square())
    assert torch.isfinite(alpha).all() and (alpha.sigmoid()>0).all()
    x=s.clone().requires_grad_()
    scale_loss(x,.05).backward()
    assert (x.grad>=0).all() and x.grad.count_nonzero()==2
    p=dict(DEFAULTS,native_forced_split_sigma=.1,native_forced_split_fraction=1.,
           native_forced_split_population=10)
    mask,stats=forced_mask(s,logits.sigmoid(),torch.ones(2,1),torch.zeros(2,1),
                          torch.tensor([[0.],[1.]]),.1,.1,.03,p)
    assert mask.tolist()==[True,False] and stats['forced_split_parents']==1
    p['native_forced_split_population']=2
    mask,_=forced_mask(s,logits.sigmoid(),torch.ones(2,1),torch.zeros(2,1),
                       torch.zeros(2,1),.1,.1,.03,p)
    assert not mask.any()
    print('Native scale-control CPU self-test passed')


if __name__=='__main__':
    self_test()
