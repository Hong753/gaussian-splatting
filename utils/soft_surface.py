"""Training-only surface opacity and local normal attraction for native 3DGS.

RGB uses ordinary alpha compositing, without a hard visibility gate. Surface
membership is recomputed from current centers every view, including new children.
"""
import math
from pathlib import Path


DEFAULTS = dict(
    soft_surface_geometry='', soft_surface_cache='',
    soft_surface_opacity_weight=.05, soft_surface_position_weight=.01,
    soft_surface_opacity_target=.98, soft_surface_opacity_band=.05,
    soft_surface_capture_distance=.10, soft_surface_ramp_steps=1000,
    soft_surface_radius_pixels=3, soft_surface_max_radius=.10,
    soft_surface_position_samples=100000, soft_surface_memory_views=4,
)


def options(args):
    values = args if isinstance(args, dict) else vars(args)
    result = {key: values.get(key, value) for key, value in DEFAULTS.items()}
    numeric = [v for k, v in result.items() if k not in ('soft_surface_geometry', 'soft_surface_cache')]
    if not all(math.isfinite(v) for v in numeric):
        raise ValueError('Soft surface options must be finite')
    if (min(result['soft_surface_opacity_weight'], result['soft_surface_position_weight']) < 0 or
            result['soft_surface_opacity_weight'] + result['soft_surface_position_weight'] <= 0 or
            not 0 < result['soft_surface_opacity_target'] < 1 or
            not 0 < result['soft_surface_opacity_band'] <= result['soft_surface_capture_distance'] or
            result['soft_surface_max_radius'] <= 0 or result['soft_surface_radius_pixels'] < 2 or
            result['soft_surface_ramp_steps'] < 0 or result['soft_surface_position_samples'] < 1 or
            result['soft_surface_memory_views'] < 1):
        raise ValueError('Invalid soft surface weights, bands, neighborhood or cache limits')
    return result


def ramp(iteration, start, steps):
    return min(1., max(0., (iteration-start)/max(1, steps)))


def sample_rgb_gradients(gaussians, visible, limit=2048):
    """Small deterministic sample for auditing the added regularizer gradient.

    Do not draw random samples: diagnostics must not change the training RNG.
    """
    import torch
    indices = visible.reshape(-1)
    if indices.dtype == torch.bool:
        indices = indices.nonzero().reshape(-1)
    indices = indices[::max(1, math.ceil(len(indices)/limit))]
    saved = {}
    for name in ('opacity', 'xyz'):
        parameter = getattr(gaussians, '_'+name)
        saved[name] = (parameter.grad[indices].detach().clone() if parameter.grad is not None
                       else torch.zeros_like(parameter[indices]))
    return indices, saved


def added_gradient_stats(gaussians, sample):
    import torch
    indices, before = sample
    stats = dict(gradient_sample_count=len(indices))
    for name, rgb in before.items():
        parameter = getattr(gaussians, '_'+name)
        after = parameter.grad[indices].detach() if parameter.grad is not None else torch.zeros_like(rgb)
        rgb_norm, soft_norm = rgb.norm().item(), (after-rgb).norm().item()
        stats.update({f'rgb_{name}_grad_norm': rgb_norm, f'soft_{name}_grad_norm': soft_norm,
                      f'soft_to_rgb_{name}_grad_ratio': soft_norm/rgb_norm if rgb_norm > 0 else None})
    return stats


def surface_opacity_loss(opacity, valid, target):
    """Reward surface compositing weight only at supported reference pixels."""
    import torch
    if not valid.any():
        return opacity.new_zeros(())
    return torch.relu(target-opacity[valid]).square().mean()


def normal_attraction(xyz, indices, refs, capture_distance, max_samples):
    """Robust normal-only pull; correspondence and capture decisions are fixed."""
    import torch
    if not len(indices):
        return xyz.new_zeros(())
    selected = torch.arange(len(indices), device=indices.device)
    if len(selected) > max_samples:
        selected = torch.randperm(len(selected), device=indices.device)[:max_samples]
    anchor, normal = refs['anchor'][selected].detach(), refs['normal'][selected].detach()
    signed = ((xyz[indices[selected]]-anchor)*normal).sum(-1)/capture_distance
    # Smooth L1 with a 20%-of-capture quadratic core and bounded slope.
    absolute = signed.abs()
    robust = torch.where(absolute < .2, .5*signed.square()/.2, absolute-.1)
    return robust.mean()


def render_surface_opacity(camera, gaussians, pipe, indices):
    """Native CUDA pass: white surface tags, black other points and background.

    Keep every Gaussian in transmittance, including unassociated foreground.
    Only opacities receive gradients from this pass. It cannot inflate scales,
    move centers or alter RGB/SH to satisfy the coverage target.
    """
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    xyz = gaussians.get_xyz.detach()
    colors = torch.zeros_like(xyz)
    colors[indices] = 1.
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx*.5), tanfovy=math.tan(camera.FoVy*.5),
        bg=xyz.new_zeros(3), scale_modifier=1.,
        viewmatrix=camera.world_view_transform.detach(),
        projmatrix=camera.full_proj_transform.detach(),
        sh_degree=0, campos=camera.camera_center.detach(), prefiltered=False,
        debug=pipe.debug, antialiasing=pipe.antialiasing)
    rendered, _, _ = GaussianRasterizer(raster_settings=settings)(
        means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
        colors_precomp=colors, opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling.detach(), rotations=gaussians.get_rotation.detach(),
        cov3D_precomp=None)
    if rendered.requires_grad:
        rendered.register_hook(lambda g: g.contiguous() if g is not None else g)
    return rendered[0]


class SoftSurfaceRegularizer:
    def __init__(self, opt, dataset, cameras):
        from utils.scene_occlusion import MeshDepthCache
        self.options = options(opt)
        self.warmup = opt.position_warmup_steps
        self.provider = MeshDepthCache(
            opt.soft_surface_geometry, opt.soft_surface_cache,
            memory_views=opt.soft_surface_memory_views,
            point_cloud=Path(dataset.model_path)/'input.ply', source=dataset.source_path)
        if self.provider.report.get('diagnostic_only'):
            raise ValueError('Soft surface training requires the full scene mesh, not a diagnostic preview')
        if {c.image_name for c in cameras} != set(self.provider.report['training_images']):
            raise ValueError('Soft surface reference and actual training cameras differ')

    def loss(self, camera, gaussians, pipe, visible, iteration):
        import torch
        from utils.view_tangent import match_visible_surface
        cfg, xyz = self.options, gaussians.get_xyz
        frame = self.provider.surface(camera, xyz.device,
            cfg['soft_surface_radius_pixels'], cfg['soft_surface_max_radius'])
        with torch.no_grad():
            valid = (frame['support'] > 0) & torch.isfinite(frame['depth']) & (frame['depth'] > 0)
            if getattr(camera, 'alpha_mask', None) is not None:
                valid = valid & (camera.alpha_mask.to(xyz.device).reshape_as(valid) > .5)
            # Exclude masked pixels from both opacity membership and attraction.
            matching_frame = dict(frame, support=torch.where(valid, frame['support'], 0.))
            indices, refs = match_visible_surface(camera, xyz, visible, matching_frame,
                depth_abs=cfg['soft_surface_capture_distance'], depth_relative=0.)
            delta = xyz[indices]-refs['anchor']
            normal_distance = (delta*refs['normal']).sum(-1).abs()
            z_distance = (delta @ camera.world_view_transform[:3, 2]).abs()
            band = cfg['soft_surface_opacity_band']
            surface_indices = indices[(normal_distance <= band) & (z_distance <= band)]
        opacity_loss, position_loss = xyz.new_zeros(()), xyz.new_zeros(())
        surface_opacity = None
        if cfg['soft_surface_opacity_weight'] and len(surface_indices) and valid.any():
            surface_opacity = render_surface_opacity(camera, gaussians, pipe, surface_indices)
            opacity_loss = surface_opacity_loss(surface_opacity, valid, cfg['soft_surface_opacity_target'])
        if cfg['soft_surface_position_weight'] and iteration > self.warmup:
            position_loss = normal_attraction(xyz, indices, refs,
                cfg['soft_surface_capture_distance'], cfg['soft_surface_position_samples'])
        opacity_weight = cfg['soft_surface_opacity_weight'] * ramp(iteration, 0, cfg['soft_surface_ramp_steps'])
        position_weight = cfg['soft_surface_position_weight'] * ramp(iteration, self.warmup, cfg['soft_surface_ramp_steps'])
        loss = opacity_weight*opacity_loss + position_weight*position_loss
        # Keep detached scalars until the caller's logging interval: avoid a
        # device sync for every statistic on every training iteration.
        stats = dict(surface_gaussians=len(surface_indices), captured_gaussians=len(indices),
            reference_fraction=valid.float().mean().detach(),
            surface_opacity_mean=(surface_opacity[valid].mean().detach() if surface_opacity is not None else xyz.new_zeros(())),
            normal_distance_mean=(normal_distance.mean() if len(indices) else xyz.new_zeros(())),
            opacity_loss=opacity_loss.detach(), position_loss=position_loss.detach(),
            opacity_weight=opacity_weight, position_weight=position_weight,
            weighted_loss=loss.detach())
        return loss, stats
