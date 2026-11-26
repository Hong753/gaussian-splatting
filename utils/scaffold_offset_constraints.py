"""Optional projection of Scaffold child-center displacements in scene units.

The decoder uses ``anchor + exp(log_scale[:3]) * offset``.  Bounding the raw
offset would therefore not bound the physical displacement when scales learn.
Only the offset parameters are changed here; the decoder and Adam state remain
unchanged.  A radius of zero is an exact no-op.
"""
import math

import torch


def validate_offset_radius(value):
    radius = float(value)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('offset_max_radius must be finite and nonnegative; 0 disables the constraint')
    return radius


@torch.no_grad()
def project_offsets_(offsets, log_scales, max_radius, *, chunk_size=65536, record_stats=False):
    """Project each child radially into a world-space ball around its anchor.

    ``offsets`` has shape [anchors, children, 3], and ``log_scales`` has at
    least three scale channels per anchor.  Work is chunked by anchor to avoid
    a full-scene displacement temporary.  Device-to-host synchronization only
    occurs when ``record_stats`` is true; counts refer to this projection call.
    """
    radius = validate_offset_radius(max_radius)
    if radius == 0 and not record_stats:
        return None
    if offsets.ndim != 3 or offsets.shape[-1] != 3:
        raise ValueError('Expected Scaffold offsets with shape [anchors, children, 3]')
    if log_scales.ndim != 2 or log_scales.shape[0] != offsets.shape[0] or log_scales.shape[1] < 3:
        raise ValueError('Scaffold scales must have at least three channels per anchor')
    if not offsets.is_floating_point() or offsets.dtype != log_scales.dtype or offsets.device != log_scales.device:
        raise ValueError('Scaffold offsets and scales must share floating dtype and device')
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError('Offset projection chunk_size must be a positive integer')
    if record_stats:
        count = torch.zeros((), dtype=torch.int64, device=offsets.device)
        maximum_before = offsets.new_zeros(())
        maximum_after = offsets.new_zeros(())
    # A small inward rounding margin keeps projected float32 children within
    # the radius on the following step, without changing any interior child.
    inward = 1. - 4. * torch.finfo(offsets.dtype).eps
    tiny = torch.finfo(offsets.dtype).tiny
    for begin in range(0, len(offsets), int(chunk_size)):
        part = offsets[begin:begin + int(chunk_size)]
        scale = log_scales[begin:begin + int(chunk_size), :3].exp()[:, None, :]
        displacement = part * scale
        # Normalize before squaring, avoiding norm overflow for large finite
        # displacements while preserving the radial projection direction.
        largest = displacement.abs().amax(dim=-1, keepdim=True)
        normalized_norm = torch.linalg.vector_norm(displacement / largest.clamp_min(tiny), dim=-1, keepdim=True)
        norm = largest * normalized_norm
        if radius:
            outside = norm > radius
            factor = torch.where(outside, ((radius * inward) / largest.clamp_min(tiny)) /
                                 normalized_norm.clamp_min(tiny), torch.ones_like(norm))
            part.mul_(factor)
        if record_stats and part.numel():
            if radius:
                count += outside.sum()
            maximum_before = torch.maximum(maximum_before, norm.amax())
            after = torch.linalg.vector_norm(part * scale, dim=-1).amax() if radius else norm.amax()
            maximum_after = torch.maximum(maximum_after, after)
    if not record_stats:
        return None
    values = torch.stack((count.to(torch.float64), maximum_before.to(torch.float64),
                          maximum_after.to(torch.float64))).cpu().tolist()
    children = offsets.shape[0] * offsets.shape[1]
    return dict(offset_max_radius=radius, offset_children=children, offset_projected_children=int(values[0]),
                offset_projection_fraction=values[0] / max(1, children),
                offset_radius_before_projection_max=values[1], offset_radius_after_projection_max=values[2])
