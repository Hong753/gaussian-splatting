"""Conservative, opacity-independent whole-Gaussian culling on the GPU."""
import math
import torch
import torch.nn.functional as F

from utils.occlusion_depth import TILE, LidarDepthCache


def depth_pyramid(tiles):
    levels = [tiles]
    while max(levels[-1].shape) > 1:
        levels.append(F.max_pool2d(levels[-1][None, None], 2, stride=2, ceil_mode=True)[0, 0])
    offsets, offset = [], 0
    for level in levels:
        offsets.append(offset)
        offset += level.numel()
    metadata = torch.tensor([(o, *a.shape) for o, a in zip(offsets, levels)], device=tiles.device, dtype=torch.long)
    return torch.cat([a.flatten() for a in levels]), metadata


def rectangle_maximum(atlas, metadata, lo, hi):
    """Max over an enclosing set of <=4 mip cells; never underestimate a rectangle."""
    size = (hi-lo+1).amax(dim=1).clamp_min(1)
    level = torch.ceil(torch.log2(size.float())).long().clamp_max(len(metadata)-1)
    cell = torch.bitwise_left_shift(torch.ones_like(level), level)
    a, b = lo // cell[:, None], hi // cell[:, None]
    offsets, height, width = metadata[level].unbind(1)
    result = torch.full_like(size, -float("inf"), dtype=atlas.dtype)
    for x, y in ((a[:, 0], a[:, 1]), (a[:, 0], b[:, 1]), (b[:, 0], a[:, 1]), (b[:, 0], b[:, 1])):
        idx = offsets + y.clamp_min(0).minimum(height-1)*width + x.clamp_min(0).minimum(width-1)
        result = torch.maximum(result, atlas[idx])
    return result


def rotation_matrix(q):
    q = F.normalize(q, dim=1)
    w, x, y, z = q.unbind(1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=1).reshape(-1, 3, 3)


@torch.no_grad()
def conservative_keep(camera, xyz, scales, rotations, tiles, margin=.05, sigma=3., chunk_size=131072,
                      scaling_modifier=1.):
    """Keep partial/uncertain splats; reject only fully covered, definitely hidden support.

    Footprints follow dr_aa's covariance +0.3 pixel filter and 3-sigma radius,
    expanded to whole 16x16 rasterizer tiles. Depth uses the nearest sigma extent.
    """
    if sigma < 3 or margin < 0 or chunk_size < 1:
        raise ValueError("Invalid conservative occlusion parameters")
    device = xyz.device
    view = camera.world_view_transform.to(device=device, dtype=xyz.dtype)
    projection = camera.full_proj_transform.to(device=device, dtype=xyz.dtype)
    atlas, metadata = depth_pyramid(tiles.to(device))
    w, h = camera.image_width, camera.image_height
    tx, ty = math.tan(camera.FoVx/2), math.tan(camera.FoVy/2)
    fx, fy = w/(2*tx), h/(2*ty)
    tile_bounds = torch.tensor([(w-1)//TILE, (h-1)//TILE], device=device)
    keep = torch.ones(len(xyz), dtype=torch.bool, device=device)
    for start in range(0, len(xyz), chunk_size):
        end = start + chunk_size
        points = xyz[start:end]
        cam = points @ view[:3, :3] + view[3, :3]
        z = cam[:, 2]
        safe_z = z.clamp_min(1e-6)
        a = view[:3, :3].T @ (rotation_matrix(rotations[start:end]) * (scales[start:end]*scaling_modifier)[:, None, :])
        cov = a @ a.transpose(1, 2)
        j = torch.zeros((len(points), 2, 3), dtype=xyz.dtype, device=device)
        j[:, 0, 0], j[:, 1, 1] = fx/safe_z, fy/safe_z
        j[:, 0, 2] = -fx*(cam[:, 0]/safe_z).clamp(-1.3*tx, 1.3*tx)/safe_z
        j[:, 1, 2] = -fy*(cam[:, 1]/safe_z).clamp(-1.3*ty, 1.3*ty)/safe_z
        screen_cov = j @ cov @ j.transpose(1, 2)
        xx, yy, xy = screen_cov[:, 0, 0]+.3, screen_cov[:, 1, 1]+.3, screen_cov[:, 0, 1]
        midpoint = .5*(xx+yy)
        radius = torch.ceil(3*torch.sqrt((midpoint + torch.sqrt((midpoint.square() - (xx*yy-xy.square())).clamp_min(.1))).clamp_min(0)))
        clip = points @ projection[:3] + projection[3]
        ndc = clip[:, :2] / (clip[:, 3:4] + 1e-7)
        center = ((ndc+1)*torch.tensor([w, h], device=device)-1)*.5
        # Extra two pixels cover arithmetic rounding and extend to the complete
        # rasterizer tile bounds (the CUDA loop can contribute beyond 3 sigma).
        lo_pixel, hi_pixel = center-radius[:, None]-2, center+radius[:, None]+2
        intersects = (hi_pixel[:, 0] >= 0) & (hi_pixel[:, 1] >= 0) & (lo_pixel[:, 0] < w) & (lo_pixel[:, 1] < h)
        lo = torch.floor(lo_pixel/TILE).nan_to_num().long().clamp_min(0).minimum(tile_bounds)
        hi = torch.floor(hi_pixel/TILE).nan_to_num().long().clamp_min(0).minimum(tile_bounds)
        farthest_surface = rectangle_maximum(atlas, metadata, lo, hi)
        nearest = z - sigma*torch.sqrt(cov[:, 2, 2].clamp_min(0))
        reliable = intersects & (nearest > .2) & torch.isfinite(radius) & torch.isfinite(center).all(dim=1)
        keep[start:end] = ~(reliable & torch.isfinite(farthest_surface) & (nearest > farthest_surface + margin))
    return keep


def visibility_mask(camera, pc, pipe, scaling_modifier=1.):
    if not hasattr(pipe, "_occlusion_provider"):
        pipe._occlusion_provider = LidarDepthCache(pipe.occlusion_geometry, pipe.occlusion_cache, pipe.occlusion_edge_pixels)
    tiles = pipe._occlusion_provider.tiles(camera, pc.get_xyz.device)
    return conservative_keep(camera, pc.get_xyz.detach(), pc.get_scaling.detach(), pc.get_rotation.detach(), tiles,
        pipe.occlusion_margin, pipe.occlusion_sigma, pipe.occlusion_chunk_size, scaling_modifier)


@torch.no_grad()
def masked_adam_step(optimizer, keep, chunk_size=131072):
    """Default Adam with frozen culled rows, including both momentum buffers.

    Uses Adam's existing global per-parameter clock, state format and LR schedule.
    Densification-replaced tensors have no gradient and are skipped as upstream.
    """
    for group in optimizer.param_groups:
        if group.get("amsgrad") or group.get("weight_decay", 0) or group.get("maximize"):
            raise ValueError("Occlusion updates currently require default Adam without weight decay/AMSGrad/maximize")
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if len(parameter) != len(keep):
                raise ValueError("Occlusion mask no longer matches optimizer rows")
            state = optimizer.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            step = state["step"].item()
            correction1, correction2 = 1-beta1**step, math.sqrt(1-beta2**step)
            for start in range(0, len(keep), chunk_size):
                indices = keep[start:start+chunk_size].nonzero().flatten() + start
                if not len(indices):
                    continue
                gradient = parameter.grad[indices]
                first, second = state["exp_avg"][indices], state["exp_avg_sq"][indices]
                first.lerp_(gradient, 1-beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
                denominator = second.sqrt().div_(correction2).add_(group["eps"])
                parameter[indices] = parameter[indices] - (group["lr"]/correction1)*first/denominator
                state["exp_avg"][indices], state["exp_avg_sq"][indices] = first, second
