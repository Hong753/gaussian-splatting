"""Adaptive activation/pruning on immutable, unique LiDAR coordinates.

This is neither continuous splitting nor scale projection: size pruning deletes
rows; growth selects unused measured neighbors. Surviving rows keep their Adam
moments and centers. Population edits occur after the RGB optimizer step.
"""
import math

import torch


def policy(opt):
    cfg = {key: getattr(opt, 'discrete_lidar_'+key) for key in
           ('prune_sigma', 'seed_sigma_ratio', 'radius', 'max_new', 'min_observations')}
    if (not all(math.isfinite(cfg[k]) and cfg[k] > 0 for k in ('prune_sigma', 'radius')) or
            not 0 < cfg['seed_sigma_ratio'] < 1 or cfg['max_new'] < 1 or cfg['min_observations'] < 1):
        raise ValueError('Invalid discrete LiDAR scale, radius or growth budget')
    return cfg


def configure_training(opt, dataset):
    if not getattr(opt, 'discrete_lidar', False):
        return
    policy(opt)
    if (opt.fixed_lidar or dataset.model_type != '3dgs' or opt.optimizer_type != 'default' or
            opt.surface_mode != 'off' or opt.soft_surface_geometry or opt.absgs or
            dataset.occlusion_mode != 'off' or dataset.train_depth_occlusion_dir or
            opt.lidar_depth_weight or dataset.depths or dataset.train_test_exp):
        raise ValueError('Discrete LiDAR uses native 3DGS and Adam without other surface, culling or fixed-population modes')
    if not opt.discrete_lidar_cache or opt.densification_interval < 1 or opt.densify_from_iter < 0:
        raise ValueError('Discrete LiDAR needs a prepared cache and a valid growth schedule')
    opt.position_lr_init = opt.position_lr_final = 0.
    opt.position_warmup_steps = opt.position_ramp_steps = opt.densification_phase_offset = 0
    for key, value in vars(opt).items():
        if key.startswith('discrete_lidar'):
            setattr(dataset, key, value)


def freeze_xyz(model):
    model._xyz.requires_grad_(False)
    model._xyz.grad = None
    if model.optimizer is not None:
        model.optimizer.state.pop(model._xyz, None)
        for group in model.optimizer.param_groups:
            if group['name'] == 'xyz':
                group['lr'] = 0.


@torch.no_grad()
def setup(model, opt):
    from utils.discrete_lidar_cache import load
    from pathlib import Path
    cfg = policy(opt)
    cache = str(Path(opt.discrete_lidar_cache).expanduser().resolve())
    pool = getattr(model, 'discrete_lidar_pool', None)
    if pool is None or pool['directory'] != cache:
        pool = load(cache, model._xyz.device)
        model.discrete_lidar_pool = pool
    state = model.discrete_lidar_state
    if state is None:
        if len(model._xyz) != pool['metadata']['original_count']:
            raise ValueError('Initialize discrete LiDAR from the complete original input cloud')
        for start in range(0, len(pool['xyz']), 1048576):
            part = slice(start, start+1048576)
            if not torch.equal(model._xyz[pool['source_ids'][part]], pool['xyz'][part]):
                raise ValueError('Discrete LiDAR cache coordinates differ from initialization')
        ids = pool['initial_ids'].long().clone()
        original_rows = pool['source_ids'][ids].long()
        for key in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation'):
            setattr(model, key, torch.nn.Parameter(getattr(model, key)[original_rows].clone(), requires_grad=key != '_xyz'))
        sigma = pool['sigma'][ids].clamp_max(cfg['seed_sigma_ratio']*cfg['prune_sigma'])
        model._scaling.copy_(sigma.log()[:, None].expand(-1, 3))
        model.max_radii2D = model._xyz.new_zeros(len(ids))
        state = dict(schema=1, cache_id=pool['metadata']['cache_id'], cache=cache,
                     policy=cfg, initial_count=len(ids), active_ids=ids)
        model.discrete_lidar_state = state
    elif state['cache_id'] != pool['metadata']['cache_id'] or state['policy'] != cfg:
        raise ValueError('Checkpoint discrete LiDAR cache or policy differs from requested settings')
    freeze_xyz(model)
    audit(model, 0)


@torch.no_grad()
def audit(model, iteration):
    pool, state = model.discrete_lidar_pool, model.discrete_lidar_state
    ids = state['active_ids']
    if (len(ids) != len(model._xyz) or not len(ids) or ids.min() < 0 or ids.max() >= len(pool['xyz']) or
            model._xyz.requires_grad or model._xyz.grad is not None or
            (model.optimizer is not None and model._xyz in model.optimizer.state)):
        raise RuntimeError('Discrete LiDAR identity/frozen-XYZ invariant failed')
    if torch.bincount(ids, minlength=len(pool['xyz'])).max() > 1:
        raise RuntimeError('More than one active Gaussian at a LiDAR coordinate')
    for start in range(0, len(ids), 1048576):
        part = slice(start, start+1048576)
        if not torch.equal(model._xyz[part], pool['xyz'][ids[part]]):
            raise RuntimeError('Gaussian moved off its discrete LiDAR coordinate')
    largest = model._scaling.amax(dim=1)
    if not torch.isfinite(largest).all() or not torch.isfinite(largest.exp()).all():
        raise RuntimeError('Nonfinite discrete LiDAR scales')
    return dict(iteration=iteration, gaussians=len(ids), pool_size=len(pool['xyz']),
                unused_coordinates=len(pool['xyz'])-len(ids), positions_in_pool=True,
                unique_coordinates=True, xyz_frozen=True, largest_sigma=float(largest.max().exp()),
                above_prune_fraction=float((largest > math.log(state['policy']['prune_sigma'])).float().mean()))


@torch.no_grad()
def candidate_children(model, prune, max_grad):
    pool, state = model.discrete_lidar_pool, model.discrete_lidar_state
    cfg, active_ids = state['policy'], state['active_ids']
    gradient = (model.xyz_gradient_accum[:, 0]/model.denom[:, 0].clamp_min(1)).nan_to_num()
    eligible = (model.denom[:, 0] >= cfg['min_observations']) & ((gradient >= max_grad) | prune)
    parents = eligible.nonzero().flatten()
    if not len(parents):
        return parents, parents
    # Replace broad observed primitives preferentially; remaining growth follows
    # image gradients. A bounded proposal set avoids N*k feature replication.
    score = gradient[parents] + prune[parents]*(gradient.max()+1)
    order = torch.topk(score, min(len(parents), cfg['max_new']), sorted=True).indices
    parents = parents[order]
    neighbor = pool['neighbors'][active_ids[parents]].long()
    occupied = torch.zeros(len(pool['xyz']), dtype=torch.bool, device=active_ids.device)
    occupied[active_ids] = True  # Pruned centers cannot be immediately re-seeded in the same event.
    safe = neighbor.clamp_min(0)
    valid = (neighbor >= 0) & ~occupied[safe]
    delta = pool['xyz'][safe] - model._xyz[parents, None]
    valid &= delta.square().sum(-1) <= cfg['radius']**2
    ranks = torch.arange(neighbor.shape[1], device=neighbor.device).expand_as(neighbor)
    ranks = torch.where(valid, ranks, neighbor.shape[1])
    best = torch.topk(ranks, min(2, neighbor.shape[1]), largest=False, sorted=True).indices
    child = neighbor.gather(1, best)
    keep = valid.gather(1, best)
    parent = parents[:, None].expand_as(child)[keep]
    child = child[keep]
    if not len(child):
        return parent, child
    # Conflicting proposals keep the highest-ranked parent, with unique centers.
    unique, inverse = torch.unique(child, sorted=True, return_inverse=True)
    first = torch.full_like(unique, len(child))
    first.scatter_reduce_(0, inverse, torch.arange(len(child), device=child.device), reduce='amin')
    take = first.sort().values[:cfg['max_new']]
    return parent[take], child[take]


@torch.no_grad()
def adapt(model, iteration, max_grad):
    """Call after Adam and zero_grad, never before the current RGB update."""
    state, pool = model.discrete_lidar_state, model.discrete_lidar_pool
    cfg = state['policy']
    before = len(model._xyz)
    if not torch.isfinite(model._scaling).all():
        raise RuntimeError('Nonfinite scales before discrete LiDAR adaptation')
    prune = model._scaling.amax(dim=1) > math.log(cfg['prune_sigma'])
    parent, ids = candidate_children(model, prune, max_grad)
    removed = int(prune.sum())
    if before-removed+len(ids) == 0:
        raise RuntimeError('Scale pruning would remove the entire population; inspect scale statistics')
    new = None
    if len(ids):
        sigma = pool['sigma'][ids].clamp_max(cfg['seed_sigma_ratio']*cfg['prune_sigma'])
        new = (pool['xyz'][ids], model._features_dc[parent], model._features_rest[parent],
               model._opacity[parent].clamp_max(math.log(.1/.9)),
               sigma.log()[:, None].expand(-1, 3).contiguous(), model._rotation[parent],
               model._xyz.new_zeros(len(ids)))
    model.tmp_radii = model._xyz.new_zeros(before)
    model._discrete_population_edit = True
    try:
        if removed:
            model.prune_points(prune)
            state['active_ids'] = state['active_ids'][~prune]
        if new is not None:
            model.densification_postfix(*new)
            state['active_ids'] = torch.cat((state['active_ids'], ids))
    finally:
        model._discrete_population_edit = False
        freeze_xyz(model)
        model.tmp_radii = None
    model.xyz_gradient_accum.zero_()
    model.denom.zero_()
    model.max_radii2D.zero_()
    return dict(iteration=iteration, before=before, added=len(ids), scale_pruned=removed,
                after=len(model._xyz), unused_coordinates=len(pool['xyz'])-len(model._xyz),
                prune_sigma=cfg['prune_sigma'], largest_sigma=float(model.get_scaling.max()))
