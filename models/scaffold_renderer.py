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

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist.clamp_min(1e-8)

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    parent_ids = mask.nonzero(as_tuple=True)[0] // pc.n_offsets
    scaling_repeat, repeat_anchor = grid_scaling[parent_ids], anchor[parent_ids]
    color, scale_rot, offsets = color[mask], scale_rot[mask], offsets[mask]
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, update_anchors=None, contribution_anchors=None):
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

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if len(xyz) == 0:
        # A valid camera can have no currently positive-opacity offsets. Avoid
        # the extension's ambiguous empty SH gradient shape, keeping a zero graph.
        zero = sum(value.sum()*0 for value in (xyz, screenspace_points, color, opacity, scaling, rot))
        rendered_image = bg_color[:, None, None].expand(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)) + zero
        radii = torch.zeros(0, dtype=torch.int32, device=xyz.device)
    else:
        rendered_image, radii, _depth = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        result = {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
        if update_anchors is not None:
            active = eligible & (radii > 0)
            result['active_children'] = active
            result['active_parents'] = parent[active].unique()
        return result
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


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
