"""Full-resolution first-hit depth cutoffs and point labels on the selected device.

Point labels sample the CUDA-rendered pixel depth. They describe point centers;
Gaussian fragment flags come from the CUDA splat compositor, not these labels.
"""
import torch


def pixel_cutoff(camera, pc, pipe):
    if not hasattr(pipe, '_occlusion_provider'):
        if pipe.occlusion_mode == 'mesh':
            from utils.scene_occlusion import MeshDepthCache
            provider = MeshDepthCache(pipe.occlusion_geometry, pipe.occlusion_cache, pipe.occlusion_edge_pixels)
        else:
            from utils.occlusion_depth import LidarDepthCache
            provider = LidarDepthCache(pipe.occlusion_geometry, pipe.occlusion_cache, pipe.occlusion_edge_pixels)
        pipe._occlusion_provider = provider
    return (pipe._occlusion_provider.pixel_depth(camera, pc.get_xyz.device).detach() + pipe.occlusion_margin).contiguous()


@torch.no_grad()
def classify_points_depth(points, intrinsic, extrinsic, depth, margin=.05, relative_margin=.10,
                          chunk_size=262144):
    """Original-row IDs, full pixel first-hit z; CUDA projection/lookup/comparison.

    Unknown pixel depths stay unknown. Out-of-frustum/invalid rows are separate.
    The mask partitions all input rows; on_surface uses the same margin band as
    the depth cutoff. This discretizes each point ray to a nearest image pixel.
    """
    if min(margin, relative_margin) < 0 or chunk_size < 1:
        raise ValueError('Invalid point visibility parameters')
    device = depth.device
    k = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
    e = torch.as_tensor(extrinsic, dtype=torch.float32, device=device)
    h, w = depth.shape
    parts = {key: [] for key in ('frustum', 'outside', 'on_surface', 'culled', 'unknown', 'in_front')}
    for start in range(0, len(points), chunk_size):
        xyz = torch.as_tensor(points[start:start+chunk_size], dtype=torch.float32, device=device)
        cam = xyz @ e[:3, :3].T + e[:3, 3]
        z = cam[:, 2]
        u = k[0, 0]*cam[:, 0]/z+k[0, 2]
        v = k[1, 1]*cam[:, 1]/z+k[1, 2]
        inside = torch.isfinite(cam).all(1) & (z > .2) & (u >= -.5) & (u < w-.5) & (v >= -.5) & (v < h-.5)
        ids = inside.nonzero().flatten()
        parts['frustum'].append((ids+start).cpu())
        parts['outside'].append(((~inside).nonzero().flatten()+start).cpu())
        if not len(ids):
            continue
        ui, vi = torch.floor(u[ids]+.5).long().clamp(0,w-1), torch.floor(v[ids]+.5).long().clamp(0,h-1)
        hit = depth[vi, ui]
        known = torch.isfinite(hit) & (hit > .2)
        delta = z[ids]-hit
        tolerance = margin + relative_margin*hit
        masks = dict(on_surface=known & (delta.abs() <= tolerance), culled=known & (delta > tolerance),
                     in_front=known & (delta < -tolerance), unknown=~known)
        for key, mask in masks.items():
            parts[key].append((ids[mask]+start).cpu())
    return {key+'_ids': torch.cat(values).numpy() if values else torch.empty(0,dtype=torch.long).numpy()
            for key, values in parts.items()}


def reference_composite(alpha, colors, z, cutoff, background):
    """Small differentiable CPU reference for the hard fragment visibility rule.

    Inputs are depth-sorted [N,H,W] alpha and [N,3] colors. This is a test oracle,
    never a replacement for the native CUDA training renderer.
    """
    blocked = torch.isfinite(cutoff)[None] & (cutoff > 0)[None] & (z[:,None,None] > cutoff)
    effective = torch.where(blocked, torch.zeros_like(alpha), alpha)
    trans = torch.cumprod(torch.cat((torch.ones_like(effective[:1]), 1-effective), dim=0), dim=0)
    weights = effective*trans[:-1]
    return torch.einsum('nhw,nc->chw', weights, colors)+trans[-1][None]*background[:,None,None]
