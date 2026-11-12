"""Local LiDAR scale targets, hard ceilings, and sampled support diagnostics.

Distances and source-row identity are cached once on CPU. Training uses fixed
GPU tensors; only the current scales/rotations change. Diagnostics count 3-sigma
bounding-sphere intersections (an upper bound for ellipsoid intersections), not
visible contributors per pixel. No scene extent enters the size rule.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from utils.checkpoint_snapshots import atomic_json, digest


def validate(neighbors, support_sigma, overlap_factor, ratio, world, samples, pair_budget, workers):
    if (not 3 <= neighbors <= 128 or not math.isfinite(support_sigma) or support_sigma <= 0 or
            not math.isfinite(overlap_factor) or overlap_factor < 1 or
            not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(world) or world < 0 or
            samples < 1 or pair_budget < 1 or workers < 1):
        raise ValueError('Invalid local-scale KNN, support, bounds or audit settings')


def bounds(distances, *, support_sigma=3., overlap_factor=2., ratio=1., world=.015,
           ceiling_multiplier=1., target_multiplier=1.):
    """distances excludes self; last column is the requested K-th neighbor."""
    reference = np.sqrt(np.maximum(np.mean(distances[:, :3]**2, axis=1), 1e-7))
    kth = distances[:, -1]
    if not np.isfinite(distances).all() or np.any(kth <= 0):
        raise ValueError('Nonfinite/zero K-th neighbor distance: inspect coincident LiDAR rows or invalid input')
    ceiling = np.minimum(ratio*reference, kth/support_sigma)
    if world > 0:
        ceiling = np.minimum(ceiling, world)
    target = np.minimum(ceiling, kth/(overlap_factor*support_sigma))
    ceiling = ceiling*ceiling_multiplier
    target = np.minimum(ceiling, target*target_multiplier)
    return ceiling.astype(np.float32), target.astype(np.float32)


def distribution(values):
    return dict(zip(('min', 'median', 'p90', 'p99', 'max'),
                    map(float, np.quantile(values, [0, .5, .9, .99, 1]))))


def identity(data):
    return hashlib.sha256(json.dumps({k: data[k] for k in
        ('schema', 'config', 'points', 'xyz_sha256', 'files')}, sort_keys=True).encode()).hexdigest()


def metadata(directory, verify=True):
    directory = Path(directory)
    data = json.loads((directory/'metadata.json').read_text())
    if data.get('schema') != 1 or data.get('cache_id') != identity(data):
        raise ValueError('Invalid local-scale cache identity/schema')
    if verify:
        for name, sha in data['files'].items():
            if digest(directory/name) != sha:
                raise ValueError(f'Local-scale cache changed: {name}')
    return data


def prepare(point_cloud, directory, *, neighbors=10, support_sigma=3., overlap_factor=2.,
            ratio=1., world=.015, samples=2048, pair_budget=2000000, workers=8, chunk=65536,
            ceiling_multiplier=1., target_multiplier=1., reuse_cache=None):
    import fcntl
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    validate(neighbors, support_sigma, overlap_factor, ratio, world, samples, pair_budget, workers)
    if not all(math.isfinite(v) and v > 0 for v in (ceiling_multiplier,target_multiplier)):
        raise ValueError('Local scale multipliers must be positive and finite')
    if chunk < 1:
        raise ValueError('Local-scale query chunk must be positive')
    point_cloud, directory = Path(point_cloud).expanduser().resolve(), Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = dict(source_sha256=digest(point_cloud), neighbors=neighbors, support_sigma=support_sigma,
                      overlap_factor=overlap_factor, ratio=ratio, world=world, samples=samples, pair_budget=pair_budget)
        # Omit identity defaults to keep existing 1x caches/plans compatible.
        if ceiling_multiplier != 1. or target_multiplier != 1.:
            config.update(ceiling_multiplier=ceiling_multiplier,target_multiplier=target_multiplier)
        if (directory/'metadata.json').is_file():
            old = metadata(directory)
            if old['config'] != config:
                raise ValueError('Local-scale cache inputs/settings differ; choose a new cache directory')
            print(f'Reusing local-scale cache: {directory}', flush=True)
            return old
        vertex = PlyData.read(str(point_cloud), mmap='r')['vertex']
        xyz = np.column_stack([vertex[k] for k in ('x', 'y', 'z')]).astype(np.float32)
        del vertex
        if len(xyz) <= neighbors or not np.isfinite(xyz).all():
            raise ValueError('Need more finite input points than the requested neighbor count')
        xyz_hash = hashlib.sha256(memoryview(xyz).cast('B')).hexdigest()
        tree = cKDTree(xyz)
        count = len(xyz)
        cap = np.lib.format.open_memmap(directory/'ceiling.npy', mode='w+', dtype=np.float32, shape=(count,))
        target = np.lib.format.open_memmap(directory/'target.npy', mode='w+', dtype=np.float32, shape=(count,))
        kth = np.lib.format.open_memmap(directory/'neighbor_distance.npy', mode='w+', dtype=np.float32, shape=(count,))
        reused = None
        if reuse_cache is not None:
            reuse_cache = Path(reuse_cache).expanduser().resolve()
            base = metadata(reuse_cache)
            required = ('source_sha256','neighbors','support_sigma','overlap_factor','ratio','world')
            if (base['xyz_sha256'] != xyz_hash or base['points'] != count or
                    any(base['config'][k] != config[k] for k in required) or
                    base['config'].get('ceiling_multiplier',1.) != 1. or
                    base['config'].get('target_multiplier',1.) != 1.):
                raise ValueError('Reuse the matching original 1x local-scale cache')
            reused = [np.load(reuse_cache/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                      for name in ('ceiling','target','neighbor_distance')]
            print(f'Reusing original KNN distances/bounds: {reuse_cache}; rebuilding complete overlap audit for new ceilings.',flush=True)
        for start in range(0, count, chunk):
            stop = min(count, start+chunk)
            # There is always one self-distance of zero. With duplicate rows the
            # zero entries are interchangeable; discarding one still excludes self.
            if reused is None:
                distance = tree.query(xyz[start:stop], k=neighbors+1, workers=workers)[0][:, 1:]
                cap[start:stop], target[start:stop] = bounds(distance, support_sigma=support_sigma,
                    overlap_factor=overlap_factor, ratio=ratio, world=world,
                    ceiling_multiplier=ceiling_multiplier,target_multiplier=target_multiplier)
                kth[start:stop] = distance[:, -1]
            else:
                cap[start:stop] = reused[0][start:stop]*ceiling_multiplier
                target[start:stop] = np.minimum(cap[start:stop],reused[1][start:stop]*target_multiplier)
                kth[start:stop] = reused[2][start:stop]
            if start == 0 or stop == count or (start//chunk) % 40 == 0:
                print(f'Local scale bounds: {stop:,}/{count:,}', flush=True)
        for value in (cap, target, kth):
            value.flush()
        if np.any(target <= 0) or not np.isfinite(cap).all():
            raise ValueError('Local scale bounds cannot be represented as positive float32 values')
        # Uniform deterministic sampling. Query all possible support neighbors
        # under the immutable hard ceilings, then retain only feasible pairs.
        # Whole samples are omitted on budget overflow; never report truncated
        # neighbor lists as complete overlap counts.
        proposed = np.random.default_rng(0).choice(count, min(samples, count), replace=False)
        sample_ids, slots, neighbor_ids = [], [], []
        omitted, pairs = 0, 0
        max_radius = support_sigma*float(cap.max())
        for query_id in proposed:
            radius = support_sigma*float(cap[query_id])+max_radius
            radius = radius*(1+1e-5)
            candidate_count = tree.query_ball_point(xyz[query_id], radius, return_length=True)
            if candidate_count > max(250000, pair_budget):
                omitted += 1
                continue
            candidates = np.asarray(tree.query_ball_point(xyz[query_id], radius), dtype=np.int64)
            candidates = candidates[candidates != query_id]
            delta = xyz[candidates].astype(np.float64)-xyz[query_id].astype(np.float64)
            limit = support_sigma*(cap[query_id].astype(np.float64)+cap[candidates].astype(np.float64))
            candidates = candidates[np.einsum('ij,ij->i', delta, delta) <= (limit*(1+1e-5))**2]
            if pairs+len(candidates) > pair_budget:
                omitted += 1
                continue
            slot = len(sample_ids)
            sample_ids.append(query_id)
            slots.append(np.full(len(candidates), slot, dtype=np.int32))
            neighbor_ids.append(candidates)
            pairs += len(candidates)
        if not sample_ids:
            raise ValueError('No complete overlap audit samples fit the pair budget; increase it')
        np.save(directory/'sample_ids.npy', np.asarray(sample_ids, dtype=np.int64), allow_pickle=False)
        np.save(directory/'pair_slots.npy', np.concatenate(slots), allow_pickle=False)
        np.save(directory/'pair_neighbors.npy', np.concatenate(neighbor_ids), allow_pickle=False)
        names = ('ceiling.npy', 'target.npy', 'neighbor_distance.npy',
                 'sample_ids.npy', 'pair_slots.npy', 'pair_neighbors.npy')
        data = dict(schema=1, config=config, points=count, xyz_sha256=xyz_hash,
                    files={name: digest(directory/name) for name in names},
                    ceiling_sigma=distribution(cap), target_sigma=distribution(target),
                    neighbor_distance=distribution(kth), audit_samples=len(sample_ids),
                    audit_samples_omitted=omitted, audit_candidate_pairs=pairs)
        data['cache_id'] = identity(data)
        atomic_json(directory/'metadata.json', data)
        print(json.dumps({k: data[k] for k in ('points','ceiling_sigma','target_sigma','audit_samples',
                                              'audit_samples_omitted','audit_candidate_pairs')}, indent=2), flush=True)
        return data


def load_bounds(directory, device, xyz_sha256, count, ratio, world, ceiling_multiplier=1., target_multiplier=1.):
    import torch
    directory = Path(directory).expanduser().resolve()
    data = metadata(directory)
    if (data['xyz_sha256'] != xyz_sha256 or data['points'] != count or
            data['config']['ratio'] != ratio or data['config']['world'] != world or
            data['config'].get('ceiling_multiplier',1.) != ceiling_multiplier or
            data['config'].get('target_multiplier',1.) != target_multiplier):
        raise ValueError('Local-scale cache does not match ordered initialization points or scale settings')
    tensors = [torch.tensor(np.load(directory/(name+'.npy'), mmap_mode='r', allow_pickle=False), device=device)[:, None]
               for name in ('ceiling', 'target')]
    cap, target = tensors
    if (cap.shape != (count, 1) or target.shape != cap.shape or
            not torch.isfinite(cap).all() or not torch.isfinite(target).all() or
            not (target > 0).all() or not (target <= cap).all()):
        raise ValueError('Invalid local-scale ceiling/target arrays')
    return data, cap.log(), target.log()


def attach_audit(model, directory):
    import torch
    directory = Path(directory).expanduser().resolve()
    data = metadata(directory)
    state = model.fixed_lidar_state
    if data['cache_id'] != state['local_cache_id'] or data['xyz_sha256'] != state['xyz_sha256']:
        raise ValueError('Checkpoint local-scale cache differs from requested cache')
    runtime = dict(metadata=data)
    for key in ('sample_ids', 'pair_slots', 'pair_neighbors'):
        runtime[key] = torch.tensor(np.load(directory/(key+'.npy'), mmap_mode='r', allow_pickle=False),
                                    device=model._xyz.device, dtype=torch.long)
    model.local_scale_audit = runtime


def support_counts(model, chunk=131072):
    """Complete sampled sphere intersections and centers inside own ellipsoid."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        rt = model.local_scale_audit
        samples = rt['sample_ids']
        q = rt['metadata']['config']['support_sigma']
        sphere = torch.zeros(len(samples), device=samples.device, dtype=torch.long)
        centers = torch.zeros_like(sphere)
        for start in range(0, len(rt['pair_slots']), chunk):
            slot = rt['pair_slots'][start:start+chunk]
            other = rt['pair_neighbors'][start:start+chunk]
            query = samples[slot]
            delta = model._xyz[other]-model._xyz[query]
            scale = model._scaling[query].exp()
            radius = q*(scale.amax(1)+model._scaling[other].exp().amax(1))
            hit = delta.square().sum(1) <= radius.square()
            sphere.scatter_add_(0, slot, hit.long())
            # Inverse unit-quaternion rotation, without constructing a matrix for every pair.
            quat = F.normalize(model._rotation[query], dim=1)
            cross = torch.cross(quat[:, 1:], delta, dim=1)
            local = delta - 2*quat[:, :1]*cross + 2*torch.cross(quat[:, 1:], cross, dim=1)
            mahalanobis = (local/scale.clamp_min(torch.finfo(scale.dtype).tiny)).square().sum(1)
            centers.scatter_add_(0, slot, (mahalanobis <= q*q).long())
        return sphere, centers


def audit(model):
    import torch
    with torch.no_grad():
        rt = model.local_scale_audit
        spheres, centers = support_counts(model)
        result = dict(local_scale_audit_samples=len(spheres),
                      local_scale_audit_samples_omitted=rt['metadata']['audit_samples_omitted'])
        for name, value in (('support_sphere_overlaps', spheres), ('support_centers_inside', centers),
                            ('sample_largest_sigma', model._scaling[rt['sample_ids']].exp().amax(1))):
            quantiles = torch.quantile(value.float(), value.new_tensor([.5, .9, .99, 1.], dtype=torch.float32))
            result.update({name+'_'+label: float(v) for label, v in zip(('median', 'p90', 'p99', 'max'), quantiles)})
        result['support_sphere_fraction_over_k'] = float((spheres > rt['metadata']['config']['neighbors']).float().mean())
        return result


def projected_stats(radii, limit=20000):
    import torch
    with torch.no_grad():
        visible = radii[radii > 0]
        if len(visible) > limit:
            visible = visible[torch.linspace(0, len(visible)-1, limit, device=radii.device).long()]
        if not len(visible):
            return dict(sampled_visible_radius_px_count=0)
        q = torch.quantile(visible.float(), visible.new_tensor([.5,.9,.99,1.], dtype=torch.float32))
        return dict(sampled_visible_radius_px_count=len(visible),
                    **{'visible_radius_px_'+k: float(v) for k, v in zip(('median','p90','p99','max'), q)})


def pressure_stats(model, sample):
    """Is the cap resisting RGB-requested growth on actually supervised rows?"""
    import torch
    with torch.no_grad():
        indices, rgb_gradient = sample
        active = rgb_gradient.ne(0).any(1)
        indices, rgb_gradient = indices[active], rgb_gradient[active]
        result = dict(sampled_rgb_active_gaussians=len(indices))
        if not len(indices):
            return result
        state = model.fixed_lidar_state
        scale = model._scaling[indices]
        at_cap = scale >= state['log_scale_cap'][indices]-1e-5
        result.update(active_at_cap_fraction=float(at_cap.any(1).float().mean()),
            active_growth_at_cap_fraction=float((at_cap & (rgb_gradient < 0)).any(1).float().mean()),
            active_above_target_fraction=float((scale > state['log_scale_target'][indices]).any(1).float().mean()))
        if model._scaling.grad is not None:
            combined=model._scaling.grad[indices]
            rgb_norm=float(rgb_gradient.norm())
            result.update(active_net_growth_at_cap_fraction=float((at_cap & (combined < 0)).any(1).float().mean()),
                active_regularizer_to_rgb_scale_grad_ratio=float((combined-rgb_gradient).norm())/rgb_norm if rgb_norm else None)
        return result
