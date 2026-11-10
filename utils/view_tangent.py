"""Visibility-gated local tangent neighborhoods for native 3DGS densification.

Surface depths are camera-z, not range, and must be *uninflated*. Occlusion tile
maxima include safety margins and are deliberately never used to fit planes.
Nothing here creates geometry in unknown pixels or changes camera poses.
"""
import math

import torch
import torch.nn.functional as F


DEFAULTS = dict(view_tangent_radius_pixels=3, view_tangent_max_radius=.1,
                view_tangent_depth_abs=.03, view_tangent_depth_relative=.01,
                view_tangent_min_observations=2, view_tangent_max_parents=50000,
                view_tangent_max_points=12000000)


def options(args):
    result = {key: getattr(args, key, value) for key, value in DEFAULTS.items()}
    if (any(not math.isfinite(float(v)) for v in result.values()) or
            result['view_tangent_radius_pixels'] < 2 or
            result['view_tangent_max_radius'] <= 0 or
            min(result['view_tangent_depth_abs'], result['view_tangent_depth_relative']) < 0 or
            result['view_tangent_depth_abs'] + result['view_tangent_depth_relative'] <= 0 or
            min(result['view_tangent_min_observations'], result['view_tangent_max_parents'],
                result['view_tangent_max_points']) < 1):
        raise ValueError('Invalid view-tangent neighborhood, depth tolerance, or growth budget')
    return result


def camera_intrinsics(camera, device):
    w, h = camera.image_width, camera.image_height
    return torch.tensor([[w/(2*math.tan(camera.FoVx/2)), 0, (w-1)/2],
                         [0, h/(2*math.tan(camera.FoVy/2)), (h-1)/2],
                         [0, 0, 1]], device=device, dtype=torch.float32)


@torch.no_grad()
def surface_frame(depth, valid, intrinsic, radius_pixels=3, max_radius=.1):
    """Fit local camera-space planes; holes, jumps, and nonplanar patches fail.

Full square neighborhoods must be trusted. Normals use opposite backprojected
neighbors, with diagonal/cardinal plane residual checks. The supported tangent
disk occupies at most half the neighborhood in image space (linearized), with
an additional scene-unit radius cap. No spatial interpolation across holes.
"""
    if radius_pixels < 2 or max_radius <= 0:
        raise ValueError('Require radius_pixels >= 2 and max_radius > 0')
    if depth.ndim != 2 or valid.shape != depth.shape or intrinsic.shape != (3, 3):
        raise ValueError('Invalid surface depth/mask/intrinsic shape')
    if not torch.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError('Invalid surface intrinsics')
    h, w = depth.shape
    r = radius_pixels
    valid = valid.bool() & torch.isfinite(depth) & (depth > 0)
    z = torch.where(valid, depth, 0)
    yy, xx = torch.meshgrid(torch.arange(h, device=z.device, dtype=z.dtype),
                           torch.arange(w, device=z.device, dtype=z.dtype), indexing='ij')
    x, y = (xx-intrinsic[0, 2])/intrinsic[0, 0], (yy-intrinsic[1, 2])/intrinsic[1, 1]
    points = torch.stack((x*z, y*z, z), -1)
    normal = torch.zeros_like(points)
    support = torch.zeros_like(z)
    if min(h, w) <= 2*r:
        return dict(depth=z, normal=normal, support=support, intrinsic=intrinsic)
    center = points[r:-r, r:-r]
    horizontal = points[r:-r, 2*r:]-points[r:-r, :-2*r]
    vertical = points[2*r:, r:-r]-points[:-2*r, r:-r]
    cross = torch.cross(horizontal, vertical, dim=-1)
    normals = F.normalize(cross, dim=-1)
    full = F.avg_pool2d(valid.float()[None, None], 2*r+1, stride=1)[0, 0] > 1-1e-6
    max_z = F.max_pool2d(z[None, None], 2*r+1, stride=1)[0, 0]
    min_z = -F.max_pool2d(-z[None, None], 2*r+1, stride=1)[0, 0]
    cz = z[r:-r, r:-r]
    good = full & ((max_z-min_z) <= .05*cz) & (cross.norm(dim=-1) > 1e-12)
    # A tilted plane is valid; corners and curved/noisy depth should not become
    # an unbounded tangent plane. Check all 8 perimeter samples around it.
    for dy, dx in ((-r,-r), (-r,0), (-r,r), (0,-r), (0,r), (r,-r), (r,0), (r,r)):
        neighbor = points[r+dy:h-r+dy, r+dx:w-r+dx]
        residual = ((neighbor-center)*normals).sum(-1).abs()
        good &= residual <= .005 + .002*cz
    ray = F.normalize(center, dim=-1)
    good &= (ray*normals).sum(-1).abs() >= .05
    # Projection Jacobian restricted to the tangent plane bounds a disk's
    # image footprint, including oblique surfaces and asymmetric intrinsics.
    jx = torch.stack((torch.ones_like(cz), torch.zeros_like(cz), -x[r:-r, r:-r]), -1)
    jy = torch.stack((torch.zeros_like(cz), torch.ones_like(cz), -y[r:-r, r:-r]), -1)
    jx *= (intrinsic[0, 0]/cz.clamp_min(1e-8))[..., None]
    jy *= (intrinsic[1, 1]/cz.clamp_min(1e-8))[..., None]
    jx -= (jx*normals).sum(-1, keepdim=True)*normals
    jy -= (jy*normals).sum(-1, keepdim=True)*normals
    radius = (.5*r/torch.maximum(jx.norm(dim=-1), jy.norm(dim=-1)).clamp_min(1e-8)).clamp_max(max_radius)
    normal[r:-r, r:-r] = torch.where(good[..., None], normals, 0)
    support[r:-r, r:-r] = torch.where(good, radius, 0)
    return dict(depth=z, normal=normal, support=support, intrinsic=intrinsic)


@torch.no_grad()
def match_visible_surface(camera, xyz, visible, frame, depth_abs=.03, depth_relative=.01, chunk_size=131072):
    """Associate only rendered, near-first-surface Gaussian centers with patches."""
    visible = visible.reshape(-1)
    if visible.dtype == torch.bool:
        visible = visible.nonzero().flatten()
    view = camera.world_view_transform.to(device=xyz.device, dtype=xyz.dtype)
    k = frame['intrinsic']
    h, w = frame['depth'].shape
    gathered = {key: [] for key in ('indices', 'anchor', 'normal', 'support', 'tolerance')}
    for start in range(0, len(visible), chunk_size):
        idx = visible[start:start+chunk_size]
        cam = xyz[idx] @ view[:3, :3] + view[3, :3]
        z = cam[:, 2]
        u = torch.floor(k[0, 0]*cam[:, 0]/z.clamp_min(1e-8)+k[0, 2]+.5).nan_to_num().long()
        v = torch.floor(k[1, 1]*cam[:, 1]/z.clamp_min(1e-8)+k[1, 2]+.5).nan_to_num().long()
        inside = (u >= 0) & (v >= 0) & (u < w) & (v < h) & (z > .01)
        u, v = u.clamp(0, w-1), v.clamp(0, h-1)
        depth, radius = frame['depth'][v, u], frame['support'][v, u]
        n = frame['normal'][v, u] @ view[:3, :3].T
        point = torch.stack(((u-k[0, 2])*depth/k[0, 0], (v-k[1, 2])*depth/k[1, 1], depth), 1)
        anchor = (point-view[3, :3]) @ view[:3, :3].T
        tolerance = depth_abs + depth_relative*depth
        delta = xyz[idx]-anchor
        distance = (delta*n).sum(-1)
        tangent = delta-distance[:, None]*n
        good = (inside & (radius > 0) & ((z-depth).abs() <= tolerance) &
                (distance.abs() <= tolerance) & (tangent.norm(dim=1) < .9*radius))
        for key, value in (('indices', idx), ('anchor', anchor), ('normal', n),
                           ('support', radius), ('tolerance', tolerance)):
            gathered[key].append(value[good])
    if not gathered['indices']:
        return visible[:0], dict(anchor=xyz[:0].detach(), normal=xyz[:0].detach(),
            support=xyz.new_empty(0), tolerance=xyz.new_empty(0), confidence=xyz.new_empty(0))
    result = {key: torch.cat(value) for key, value in gathered.items()}
    indices = result.pop('indices')
    result['confidence'] = torch.ones_like(result['support'])
    return indices, result


@torch.no_grad()
def eligible_parents(xyz, refs, observations, min_observations=2):
    delta = xyz-refs['anchor']
    distance = (delta*refs['normal']).sum(-1)
    tangent = delta-distance[:, None]*refs['normal']
    return ((refs['confidence'] > 0) & (observations.reshape(-1) >= min_observations) &
            (distance.abs() <= refs['tolerance']) & (tangent.norm(dim=1) < .9*refs['support']))


@torch.no_grad()
def tangent_children(parent, offsets, refs):
    """Preserve parent plane distance; keep children inside its measured disk."""
    n = refs['normal']
    offset = offsets-(offsets*n).sum(-1, keepdim=True)*n
    step = .5*refs['support']
    offset *= (step/offset.norm(dim=1).clamp_min(1e-12)).clamp_max(1)[:, None]
    delta = parent-refs['anchor']
    normal = (delta*n).sum(-1, keepdim=True)*n
    tangent = delta-normal+offset
    tangent *= (.9*refs['support']/tangent.norm(dim=1).clamp_min(1e-12)).clamp_max(1)[:, None]
    return refs['anchor']+normal+tangent
