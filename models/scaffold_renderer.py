#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from models.scaffold_gaussian import ScaffoldGaussianModel as GaussianModel

from models.scaffold_decoder import generate_neural_gaussians

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, update_anchors=None, contribution_anchors=None, return_alpha=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if update_anchors is not None and not is_training:
        raise ValueError('Sensed updates require training mode')
    if contribution_anchors is not None and (is_training or visible_mask is None):
        raise ValueError('Parent contribution rendering requires eval mode and a parent frustum mask')
    if is_training or contribution_anchors is not None:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=True)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    

    if update_anchors is not None or contribution_anchors is not None:
        if visible_mask is None:
            raise ValueError('Per-parent rendering requires a parent frustum mask')
        parent = visible_mask.nonzero(as_tuple=True)[0][mask.nonzero(as_tuple=True)[0] // pc.n_offsets]

    if contribution_anchors is not None:
        # RGB stores the actual composited weights: unindexed, indexed, background.
        indexed = contribution_anchors[parent]
        color = torch.stack((~indexed, indexed, torch.zeros_like(indexed)), dim=1).to(color.dtype)
        bg_color = bg_color.new_tensor([0., 0., 1.])

    if update_anchors is not None:
        from utils.scaffold_observations import gate_children
        eligible = update_anchors[parent]
        xyz, color, opacity, scaling, rot = gate_children((xyz, color, opacity, scaling, rot), eligible)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, requires_grad=True) + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    settings = dict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    def rasterizer(background):
        return GaussianRasterizer(raster_settings=GaussianRasterizationSettings(bg=background, **settings))
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if len(xyz) == 0:
        # A valid camera can have no currently positive-opacity offsets. Avoid
        # the extension's ambiguous empty SH gradient shape, keeping a zero graph.
        zero = sum(value.sum()*0 for value in (xyz, screenspace_points, color, opacity, scaling, rot))
        rendered_image = bg_color[:, None, None].expand(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)) + zero
        radii = torch.zeros(0, dtype=torch.int32, device=xyz.device)
        if return_alpha:
            alpha = zero.expand(rendered_image.shape[1:])
    else:
        inputs = dict(means3D=xyz, means2D=screenspace_points, scales=scaling, rotations=rot)
        rendered_image, radii, _ = rasterizer(bg_color)(
            colors_precomp=color, opacities=opacity, shs=None, cov3D_precomp=None, **inputs)
        if return_alpha:
            # White on black gives accumulated opacity. Hold geometry fixed for this loss
            # so expanding footprints cannot directly satisfy the hit target.
            coverage, _, _ = rasterizer(torch.zeros_like(bg_color))(
                colors_precomp=torch.ones_like(color), opacities=opacity, shs=None, cov3D_precomp=None,
                **{name: tensor.detach() for name, tensor in inputs.items()})
            alpha = coverage[0]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    result = dict(render=rendered_image, viewspace_points=screenspace_points,
                  visibility_filter=radii > 0, radii=radii)
    if return_alpha:
        result['alpha'] = alpha
    if is_training:
        result.update(selection_mask=mask, neural_opacity=neural_opacity, scaling=scaling)
        if update_anchors is not None:
            active = eligible & (radii > 0)
            result['active_children'] = active
            result['active_parents'] = parent[active].unique()
    return result


def prefilter_voxel(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
    """Conservative sphere/frustum filter for the standard Graphdeco rasterizer.

    Include offset extents and a three-sigma Gaussian support; anchors that
    intersect the camera frustum are retained even if their centers are outside.
    """
    with torch.no_grad():
        anchor = pc.get_anchor
        scale = pc.get_scaling
        offset_extent = (pc._offset.abs() * scale[:, None, :3]).norm(dim=-1).amax(dim=1)
        radius = offset_extent + 3.0 * scale[:, 3:].amax(dim=1) * scaling_modifier
        ones = torch.ones_like(anchor[:, :1])
        view = torch.cat((anchor, ones), dim=1) @ viewpoint_camera.world_view_transform
        x, y, z = view[:, 0], view[:, 1], view[:, 2]
        tx, ty = math.tan(viewpoint_camera.FoVx * 0.5), math.tan(viewpoint_camera.FoVy * 0.5)
        return ((z + radius > 0.2) & (x.abs() <= z * tx + radius * math.sqrt(1 + tx * tx)) &
                (y.abs() <= z * ty + radius * math.sqrt(1 + ty * ty)))
