"""DA3 camera-z loss and a local coplanarity loss on triangle vertices."""

from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import smooth_l1_loss


def subsample(index, maximum):
    if len(index) <= maximum:
        return index
    return index[torch.arange(maximum, device=index.device) * len(index) // maximum]


def gather_depth(render, prior, intrinsic, camera):
    z, v, u = prior
    estimate = render["surf_depth"][0]
    fx = camera.image_width / (2 * math.tan(camera.FoVx / 2))
    fy = camera.image_height / (2 * math.tan(camera.FoVy / 2))
    x = torch.round((u - intrinsic[0, 2]) / intrinsic[0, 0] * fx + (camera.image_width - 1) / 2).long()
    y = torch.round((v - intrinsic[1, 2]) / intrinsic[1, 1] * fy + (camera.image_height - 1) / 2).long()
    inside = (x >= 0) & (x < camera.image_width) & (y >= 0) & (y < camera.image_height)
    x, y, z = x[inside], y[inside], z[inside]
    predicted = estimate[y, x]
    with torch.no_grad():
        supported = torch.isfinite(predicted) & (predicted > 0) & (render["rend_alpha"][0, y, x] >= .5)
    return predicted[supported], z[supported]


def plane_residuals(vertices, render, plane_ids, planes, k, world_view):
    """Use actively contributing faces wholly supported by compatible patches."""
    with torch.no_grad():
        visible = render["visibility_filter"].reshape(-1) & (render["max_blending"].reshape(-1) > .05)
        indices = subsample(torch.where(visible)[0], 8192)
    cam = vertices[indices] @ world_view[:3, :3] + world_view[3, :3]
    with torch.no_grad():
        samples = torch.cat((cam, cam.mean(dim=1, keepdim=True)), dim=1)
        xy = samples[..., :2] / samples[..., 2:].clamp_min(1e-6)
        u, v = torch.round(xy * k.diag()[:2] + k[:2, 2]).long().unbind(-1)
        h, w = plane_ids.shape
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (samples[..., 2] > 0)
        ids = plane_ids[v.clamp(0, h - 1), u.clamp(0, w - 1)].long()
        valid = inside.all(dim=1) & (ids >= 0).all(dim=1)
        samples, ids = samples[valid], ids[valid]
        candidates = planes[ids]
        reference = candidates[:, -1]
        normals_agree = ((candidates[..., :3] * reference[:, None, :3]).sum(-1) >= .995).all(1)
        offset_delta = (candidates[..., 3] - reference[:, None, 3]).abs()
        offsets_agree = (offset_delta <= .03 * samples[:, -1, 2:3]).all(1)
        compatible = normals_agree & offsets_agree
        reference = reference[compatible]
        scale = samples[compatible, -1, 2:3].clamp_min(1e-6)
    selected = cam[valid][compatible]
    return ((selected * reference[:, None, :3]).sum(-1) - reference[:, 3:4]) / scale


class GeometryPriors:
    def __init__(self, directory, depth_weight, plane_weight):
        self.directory = Path(directory)
        self.depth_weight = depth_weight
        self.plane_weight = plane_weight
        self.counts = np.zeros(2, dtype=np.int64)

    @lru_cache(maxsize=12)
    def frame(self, name):
        with np.load(self.directory / "frames" / f"{name}.npz", allow_pickle=False) as data:
            v, u = np.where(data["valid"])
            index = np.arange(min(len(u), 4096)) * len(u) // max(1, min(len(u), 4096))
            u, v = u[index], v[index]
            depth = np.stack((data["depth"][v, u], v, u)).astype(np.float32)
            plane_id, planes = (data["plane_id"], data["planes"]) if self.plane_weight else (None, None)
            return depth, data["intrinsic"], plane_id, planes

    def __call__(self, iteration, camera, triangles, render):
        # Ramp both terms after the current triangles have acquired RGB support.
        depth_scale = self.depth_weight * min(1., max(0., (iteration - 1000) / 1000))
        plane_scale = self.plane_weight * min(1., max(0., (iteration - 3000) / 1000))
        surface = render["surf_depth"]
        loss = surface.new_zeros(())
        if depth_scale or plane_scale:
            depth, k, plane_id, planes = self.frame(camera.image_name)
            intrinsic = torch.as_tensor(k, device=surface.device)
            if depth_scale and depth.shape[1]:
                predicted, target = gather_depth(render, torch.as_tensor(depth, device=surface.device), intrinsic, camera)
                if predicted.numel():
                    loss = loss + depth_scale * smooth_l1_loss(predicted.log(), target.log(), beta=.02)
                self.counts[0] += predicted.numel()
            if plane_scale and len(planes):
                residual = plane_residuals(triangles.get_triangles_points, render,
                                           torch.as_tensor(plane_id, device=surface.device),
                                           torch.as_tensor(planes, device=surface.device), intrinsic,
                                           camera.world_view_transform)
                if residual.numel():
                    loss = loss + plane_scale * smooth_l1_loss(residual, torch.zeros_like(residual), beta=.02)
                self.counts[1] += residual.shape[0]
        if iteration % 500 == 0:
            print(f"[priors {iteration}] depth pixels={self.counts[0]} plane faces={self.counts[1]} "
                  f"depth weight={depth_scale:.3g} plane weight={plane_scale:.3g}", flush=True)
            self.counts.fill(0)
        return loss
