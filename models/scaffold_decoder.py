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
from utils.sh_utils import eval_sh

def generate_neural_gaussians(viewpoint_camera, pc, visible_mask=None, is_training=False):
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

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    # combine for parallel masking
    emitted = mask.nonzero(as_tuple=True)[0]
    parent_ids = emitted // pc.n_offsets
    scaling_repeat, repeat_anchor = grid_scaling[parent_ids], anchor[parent_ids]
    scale_rot, offsets = scale_rot[mask], offsets[mask]

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    color_model = pc.get_color_mlp
    if hasattr(color_model, "coefficients"):
        # parent_ids index the filtered rows. Resolve their original anchor IDs
        # before looking up a child, including for the viewer's integer row selection.
        anchor_ids = visible_mask.nonzero(as_tuple=True)[0] if visible_mask.dtype == torch.bool else visible_mask
        coefficients = color_model(anchor_ids[parent_ids], emitted % pc.n_offsets)
        direction = torch.nn.functional.normalize(xyz - viewpoint_camera.camera_center, dim=-1, eps=1e-8)
        color = (eval_sh(color_model.degree, coefficients, direction) + 0.5).clamp_min(0.)
    else:
        color_input = cat_local_view if pc.add_color_dist else cat_local_view_wodist
        if pc.appearance_dim > 0:
            color_input = torch.cat([color_input, appearance], dim=1)
        color = color_model(color_input).reshape(-1, 3)[mask]

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot
