"""Full-resolution first-hit depth cutoffs and point labels on the selected device.

Point labels sample the CUDA-rendered pixel depth. They describe point centers;
Gaussian fragment flags come from the CUDA splat compositor, not these labels.
"""
import torch


def pixel_cutoff(camera, pc, pipe):
    if getattr(pipe, 'pose_culling', None):
        from utils.pose_culling import provider
        return provider(pipe).pixel_depth(camera, pc.get_xyz.device).detach()
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
                          chunk_size=262144, surface_band=None):
    """Original-row IDs, full pixel first-hit z; CUDA projection/lookup/comparison.

    Unknown pixel depths stay unknown. Out-of-frustum/invalid rows are separate.
    With surface_band=None, on_surface retains the legacy cutoff tolerance band.
    An explicit metric surface_band separates surface proximity from the culling
    safety margin; points in the rear margin are 'uncertain', not surface points.
    This discretizes each point ray to a nearest image pixel.
    """
    import math
    if (not all(math.isfinite(x) for x in (margin, relative_margin)) or
            min(margin, relative_margin) < 0 or chunk_size < 1 or
            (surface_band is not None and (not math.isfinite(surface_band) or surface_band < 0))):
        raise ValueError('Invalid point visibility parameters')
    device = depth.device
    k = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
    e = torch.as_tensor(extrinsic, dtype=torch.float32, device=device)
    h, w = depth.shape
    parts = {key: [] for key in ('frustum', 'outside', 'on_surface', 'culled', 'unknown', 'in_front')}
    if surface_band is not None:
        parts['uncertain'] = []
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
        band = tolerance if surface_band is None else surface_band
        culled = known & (delta > tolerance)
        masks = dict(on_surface=known & (delta.abs() <= band) & ~culled, culled=culled,
                     in_front=known & (delta < -band), unknown=~known)
        if surface_band is not None:
            masks['uncertain'] = known & (delta > band) & ~culled
        for key, mask in masks.items():
            parts[key].append((ids[mask]+start).cpu())
    return {key+'_ids': torch.cat(values).numpy() if values else torch.empty(0,dtype=torch.long).numpy()
            for key, values in parts.items()}


@torch.no_grad()
def projected_lidar_depth(points, intrinsic, extrinsic, height, width, device, chunk_size=262144):
    """Nearest measured point per native camera pixel, for auditing the mesh.

    This sparse z-buffer is diagnostic only. It is never substituted for the
    connected occlusion surface and never dilated across an opening.
    """
    k = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)
    e = torch.as_tensor(extrinsic, dtype=torch.float32, device=device)
    depth = torch.full((height*width,), float('inf'), device=device)
    for start in range(0, len(points), chunk_size):
        xyz = torch.as_tensor(points[start:start+chunk_size], dtype=torch.float32, device=device)
        cam = xyz @ e[:3, :3].T + e[:3, 3]
        z = cam[:, 2]
        u, v = k[0, 0]*cam[:, 0]/z+k[0, 2], k[1, 1]*cam[:, 1]/z+k[1, 2]
        inside = torch.isfinite(cam).all(1) & (z > .2) & (u >= -.5) & (u < width-.5) & (v >= -.5) & (v < height-.5)
        ui = torch.floor(u[inside]+.5).long().clamp(0, width-1)
        vi = torch.floor(v[inside]+.5).long().clamp(0, height-1)
        depth.scatter_reduce_(0, vi*width+ui, z[inside], reduce='amin', include_self=True)
    return depth.reshape(height, width)


def first_surface_audit(lidar_depth, mesh_depth, rgb, tolerance=.05):
    """Image-space measured-foreground disagreement; does not certify alignment."""
    import numpy as np
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid first-surface audit tolerance')
    measured, hit = np.isfinite(lidar_depth), np.isfinite(mesh_depth)
    both = measured & hit
    delta = np.zeros_like(mesh_depth)
    delta[both] = mesh_depth[both]-lidar_depth[both]
    masks = dict(agreement=both & (np.abs(delta) <= tolerance),
        missing_mesh=measured & ~hit, mesh_behind_lidar=both & (delta > tolerance),
        mesh_in_front_of_lidar=both & (delta < -tolerance))
    overlay = (np.asarray(rgb, dtype=np.float32)*.35).astype(np.uint8)
    colors = dict(agreement=(35,210,90), missing_mesh=(175,80,225),
                  mesh_behind_lidar=(245,55,55), mesh_in_front_of_lidar=(30,195,235))
    for key, mask in masks.items():
        overlay[mask] = colors[key]
    count = int(measured.sum())
    report = dict(measured_pixels=count, measured_pixel_fraction=float(measured.mean()),
        tolerance=tolerance, counts={k: int(v.sum()) for k, v in masks.items()},
        agreement_fraction_of_measured=float(masks['agreement'].sum()/max(count, 1)),
        foreground_miss_fraction_of_measured=float((masks['missing_mesh'] | masks['mesh_behind_lidar']).sum()/max(count, 1)),
        median_absolute_error=float(np.median(np.abs(delta[both]))) if both.any() else None,
        note='Nearest projected LiDAR vs mesh at the same pixel; sparse samples, mixed pixels, outliers and calibration errors can disagree')
    return report, overlay


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
