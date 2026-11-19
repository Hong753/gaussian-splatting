"""Inverse-square attenuation of RGB gradients according to rendered ray range."""

import math

import torch


def active_policy(options):
    if not getattr(options, "ray_depth_gradients", False):
        return None
    reference = options.ray_depth_reference
    start = options.ray_depth_start
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Ray depth gradient reference distance must be positive and finite")
    if not 1 <= start <= options.iterations:
        raise ValueError("Ray depth gradient start must lie within the training schedule")
    return dict(reference_m=reference, start=start, min_weight=0.05, min_opacity=0.10)


@torch.no_grad()
def ray_weights(camera, accumulated_invdepth, opacity, policy):
    """Normalize composited inverse z and convert it to inverse Euclidean range."""
    width, height = camera.image_width, camera.image_height
    if accumulated_invdepth.shape != (1, height, width) or opacity.shape != accumulated_invdepth.shape:
        raise ValueError("Ray gradient weights require 1×H×W inverse depth and opacity")
    kwargs = dict(device=accumulated_invdepth.device, dtype=accumulated_invdepth.dtype)
    x = ((torch.arange(width, **kwargs) + 0.5) * (2 / width) - 1) * math.tan(camera.FoVx / 2)
    y = ((torch.arange(height, **kwargs) + 0.5) * (2 / height) - 1) * math.tan(camera.FoVy / 2)
    ray_norm_squared = 1 + x.square()[None, :] + y.square()[:, None]
    valid = torch.isfinite(opacity) & (opacity >= policy["min_opacity"])
    valid &= torch.isfinite(accumulated_invdepth) & (accumulated_invdepth > 0)
    inverse_z = accumulated_invdepth / opacity.clamp_min(policy["min_opacity"])
    weights = (policy["reference_m"] * inverse_z).square() / ray_norm_squared
    return torch.where(valid, weights.clamp(policy["min_weight"], 1), 1), valid


@torch.no_grad()
def render_weights(render, camera, model, pipe, policy, **visibility):
    """A white foreground over black exposes the native rasterizer's accumulated alpha."""
    xyz = model.get_xyz
    buffers = render(
        camera, model, pipe, xyz.new_zeros(3), override_color=xyz.new_ones(xyz.shape), **visibility
    )
    return ray_weights(camera, buffers["depth"], buffers["render"][:1], policy)


def weight_rgb_backward(image, weights):
    """Scale the complete image-loss gradient at each ray, including SSIM contributions."""
    weights = weights.detach()
    image.register_hook(lambda grad: grad * weights)
