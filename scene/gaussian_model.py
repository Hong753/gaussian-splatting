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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.use_absgs = False
        self.surface_mode = "off"
        self.surface_refs = None
        self.fixed_lidar_state = None
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.surface_refs,
            self.surface_mode,
            self.fixed_lidar_state,
        )
    
    def restore(self, model_args, training_args):
        saved_surface_refs = model_args[12] if len(model_args) > 12 else None
        saved_surface_mode = model_args[13] if len(model_args) > 13 else "off"
        saved_fixed = model_args[14] if len(model_args) > 14 else None
        if (saved_fixed is not None) != bool(getattr(training_args, 'fixed_lidar', False)):
            raise ValueError('Checkpoint fixed LiDAR mode differs from requested mode')
        self.fixed_lidar_state = saved_fixed
        model_args = model_args[:12]
        if saved_surface_refs is not None and saved_surface_mode != training_args.surface_mode:
            raise ValueError("Checkpoint surface mode differs from requested mode")
        if (saved_surface_refs is None) != (training_args.surface_mode == "off"):
            raise ValueError("Checkpoint surface settings do not match requested surface mode")
        self.surface_refs = saved_surface_refs
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        if xyz_gradient_accum.shape[1] != self.xyz_gradient_accum.shape[1]:
            raise ValueError("Checkpoint densification statistics do not match the requested AbsGS setting")
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if self.fixed_lidar_state is not None:
            from utils.fixed_lidar import audit, project_scales
            self.optimizer.state.pop(self._xyz, None)
            project_scales(self)
            audit(self, 0)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def initialize_surface(self, directory, mode, confidence_threshold=0.6, thickness=0.2):
        """Attach immutable input-point patches and initialize trusted Gaussians thin."""
        from pathlib import Path
        from utils.surface_geometry import points_digest
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        xyz = self._xyz.detach().cpu().numpy()
        if metadata.get("points") != len(xyz) or metadata.get("points_sha256_float32") != points_digest(xyz):
            raise ValueError("Surface patches do not match the ordered input point cloud")
        normal = torch.from_numpy(np.load(directory / "normals.npy")).to(self._xyz.device)
        confidence = torch.from_numpy(np.load(directory / "confidence.npy")).to(self._xyz.device)
        support = torch.from_numpy(np.load(directory / "support.npy")).to(self._xyz.device)
        if normal.shape != self._xyz.shape or confidence.shape != self._xyz.shape[:1] or support.shape != confidence.shape:
            raise ValueError("Invalid surface patch array shapes")
        if not (torch.isfinite(normal).all() and torch.isfinite(confidence).all() and torch.isfinite(support).all()):
            raise ValueError("Surface patches contain non-finite values")
        confidence = torch.where((confidence >= confidence_threshold) & (support > 0), confidence, 0)
        self.surface_refs = {"anchor": self._xyz.detach().clone(), "normal": normal,
                             "confidence": confidence, "support": support}
        self.surface_mode = mode
        # Quaternion rotating the local +z axis onto the PCA normal (sign arbitrary).
        # Flip normals with negative z to avoid the anti-parallel singularity.
        normal = torch.where((normal[:, 2] < 0)[:, None], -normal, normal)
        w = torch.sqrt(torch.clamp((1 + normal[:, 2]) * 0.5, min=1e-8))
        quat = torch.stack((w, -normal[:, 1] / (2*w), normal[:, 0] / (2*w), torch.zeros_like(w)), dim=1)
        trusted = confidence > 0
        with torch.no_grad():
            self._rotation[trusted] = quat[trusted]
            self._scaling[trusted, 2] += np.log(thickness)
        print(f"Surface patches: {int(trusted.sum())}/{len(trusted)} trusted; mode={mode}")

    def surface_loss(self, visible, position_weight, normal_weight, thickness_weight, max_samples):
        refs = self.surface_refs
        idx = visible.reshape(-1)
        if idx.numel() > max_samples:
            idx = idx[torch.randint(idx.numel(), (max_samples,), device=idx.device)]
        if idx.numel() == 0:
            return self._xyz.sum() * 0
        confidence = refs["confidence"][idx]
        delta = self._xyz[idx] - refs["anchor"][idx]
        normal = refs["normal"][idx]
        support = refs["support"][idx].clamp_min(1e-8)
        distance = (delta * normal).sum(-1)
        tangential = delta - distance[:, None] * normal
        confidence = confidence * (tangential.detach().norm(dim=1) <= support).to(confidence.dtype)
        denominator = confidence.sum().clamp_min(1)
        position = (confidence * (distance / support).square()).sum() / denominator
        zaxis = build_rotation(self._rotation[idx])[:, :, 2]
        alignment = (confidence * (1 - (zaxis * normal).sum(-1).square()).clamp_min(0)).sum() / denominator
        scales = self.get_scaling[idx]
        ratio = scales[:, 2] / scales[:, :2].amax(dim=1).clamp_min(1e-8)
        thinness = (confidence * (ratio - 0.25).clamp_min(0).square()).sum() / denominator
        return position_weight * position + normal_weight * alignment + thickness_weight * thinness

    def initialize_view_surface(self):
        """Allocate empty references; only trusted rendered observations fill them."""
        xyz = self._xyz.detach()
        self.surface_mode = 'view_tangent'
        self.surface_refs = dict(anchor=torch.zeros_like(xyz), normal=torch.zeros_like(xyz),
            confidence=xyz.new_zeros(len(xyz)), support=xyz.new_zeros(len(xyz)),
            tolerance=xyz.new_zeros(len(xyz)))

    @torch.no_grad()
    def observe_view_surface(self, camera, visible, frame, chunk_size=131072):
        from utils.view_tangent import match_visible_surface
        cfg = self.view_tangent_options
        indices, refs = match_visible_surface(camera, self._xyz, visible, frame,
            cfg['view_tangent_depth_abs'], cfg['view_tangent_depth_relative'], chunk_size)
        previous = self.surface_refs
        agreement = (previous['normal'][indices]*refs['normal']).sum(-1).abs() >= .95
        separation = ((previous['anchor'][indices]-refs['anchor'])*refs['normal']).sum(-1).abs()
        changed = (previous['confidence'][indices] > 0) & (~agreement | (separation > refs['tolerance']))
        self.xyz_gradient_accum[indices[changed]] = 0
        self.denom[indices[changed]] = 0
        for key, values in refs.items():
            self.surface_refs[key][indices] = values
        return indices

    def training_setup(self, training_args):
        fixed = bool(getattr(training_args, 'fixed_lidar', False))
        if fixed:
            from utils.fixed_lidar import initialize, validate_limits
            validate_limits(training_args.max_scale_ratio, training_args.max_scale_world)
            if self.optimizer_type != 'default' or training_args.surface_mode != 'off':
                raise ValueError('Fixed LiDAR requires default Adam and no other surface mode')
            if self.fixed_lidar_state is None:
                self.fixed_lidar_state = initialize(self, training_args.max_scale_ratio, training_args.max_scale_world)
            elif (self.fixed_lidar_state['ratio'], self.fixed_lidar_state['world']) != (
                    training_args.max_scale_ratio, training_args.max_scale_world):
                raise ValueError('Checkpoint scale limits differ from requested limits')
            self._xyz.requires_grad_(False)
            self._xyz.grad = None
        self.surface_mode = training_args.surface_mode
        if self.surface_mode == 'view_tangent':
            from utils.view_tangent import options
            self.view_tangent_options = options(training_args)
            if len(self._xyz) > self.view_tangent_options['view_tangent_max_points']:
                raise ValueError('Initialization exceeds view_tangent_max_points; increase the explicit point budget')
        self.percent_dense = training_args.percent_dense
        self.position_warmup_steps = training_args.position_warmup_steps
        self.position_ramp_steps = training_args.position_ramp_steps
        self.use_absgs = getattr(training_args, "absgs", False)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2 if self.use_absgs else 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': 0. if fixed else training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        if fixed:
            from utils.fixed_lidar import project_scales
            project_scales(self)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                if self.fixed_lidar_state is not None:
                    param_group['lr'] = 0.
                    return 0.
                from utils.warmup_schedule import position_lr_step
                multiplier, schedule_step = position_lr_step(iteration, self.position_warmup_steps, self.position_ramp_steps)
                lr = multiplier * self.xyz_scheduler_args(schedule_step)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        if self.fixed_lidar_state is not None:
            raise RuntimeError('Pruning is disabled for fixed LiDAR')
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.surface_refs is not None:
            self.surface_refs = {k: v[valid_points_mask] for k, v in self.surface_refs.items()}
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_surface_refs=None):
        if self.fixed_lidar_state is not None:
            raise RuntimeError('Cloning/splitting is disabled for fixed LiDAR')
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.surface_refs is not None:
            if new_surface_refs is None:
                raise ValueError("Surface references are required for every new Gaussian")
            self.surface_refs = {k: torch.cat((v, new_surface_refs[k]), dim=0) for k, v in self.surface_refs.items()}
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2 if self.use_absgs else 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_surface_refs = None
        if self.surface_refs is not None:
            new_surface_refs = {k: v[selected_pts_mask].repeat((N,) + (1,) * (v.ndim - 1)) for k, v in self.surface_refs.items()}
            normals = new_surface_refs["normal"]
            parent_delta = self.get_xyz[selected_pts_mask].repeat(N, 1) - new_surface_refs["anchor"]
            parent_normal = (parent_delta * normals).sum(dim=1, keepdim=True) * normals
            inside = (parent_delta - parent_normal).norm(dim=1) <= new_surface_refs["support"]
            new_surface_refs["confidence"] = new_surface_refs["confidence"] * inside
            trusted = new_surface_refs["confidence"] > 0
            tangent = offsets - (offsets * normals).sum(dim=1, keepdim=True) * normals
            offsets = torch.where(trusted[:, None], tangent, offsets)
        new_xyz = offsets + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if new_surface_refs is not None:
            # An inherited planar patch is only valid locally, particularly at edges/curves.
            delta = new_xyz - new_surface_refs["anchor"]
            normal_component = (delta * new_surface_refs["normal"]).sum(dim=1, keepdim=True) * new_surface_refs["normal"]
            inside = (delta - normal_component).norm(dim=1) <= new_surface_refs["support"]
            new_surface_refs["confidence"] = new_surface_refs["confidence"] * inside
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        if new_surface_refs is not None:
            # Keep normal thickness on split; shrink only tangential scales.
            trusted = new_surface_refs["confidence"] > 0
            new_scaling[trusted, 2] = self._scaling[selected_pts_mask].repeat(N, 1)[trusted, 2]
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_surface_refs)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # The upstream clone is colocated, so its displacement is already tangent.
        new_surface_refs = ({k: v[selected_pts_mask] for k, v in self.surface_refs.items()}
                            if self.surface_refs is not None else None)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_surface_refs)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, max_grad_abs=None,
                         record_stats=False):
        if self.surface_mode == 'view_tangent':
            return self.densify_view_tangent(max_grad, min_opacity, extent, max_screen_size, radii)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        clone_grads = grads[:, :1]
        if self.use_absgs:
            if max_grad_abs is None:
                raise ValueError("AbsGS requires a split gradient threshold")
            split_grads, split_threshold = grads[:, 1:2], max_grad_abs
        else:
            split_grads, split_threshold = clone_grads, max_grad

        if record_stats:
            from utils.densification_diagnostics import growth_window_stats
            stats = growth_window_stats(self.get_scaling, clone_grads, split_grads, self.max_radii2D,
                                        extent, self.percent_dense, max_grad, split_threshold)
            stats['points_before'] = len(self._xyz)
        self.tmp_radii = radii
        self.densify_and_clone(clone_grads, max_grad, extent)
        if record_stats: after_clone = len(self._xyz)
        self.densify_and_split(split_grads, split_threshold, extent)
        if record_stats: after_split = len(self._xyz)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        if record_stats:
            self.last_densification_stats = dict(stats, cloned_parents=after_clone-stats['points_before'],
                split_parents=after_split-after_clone, split_children_per_parent=2,
                pruned=after_split-len(self._xyz), points_after=len(self._xyz))
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    @torch.no_grad()
    def densify_view_tangent(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """Gradient-ranked growth from measured visible surface neighborhoods.

        Every selected parent adds exactly one net Gaussian: one displaced clone
        or two split children replacing the parent. No fallback volumetric growth
        for unsupported/hidden parents. Opacity and world-size pruning remain
        enabled. Projected size alone never deletes points: a camera approaching
        a valid wall can make every nearby Gaussian exceed a pixel threshold.
        """
        from utils.view_tangent import eligible_parents, tangent_children
        from utils.occlusion_culling import rotation_matrix
        cfg, refs = self.view_tangent_options, self.surface_refs
        count = len(self._xyz)
        previous_max_radii = self.max_radii2D
        gradient = (self.xyz_gradient_accum[:, 0]/self.denom[:, 0].clamp_min(1)).nan_to_num()
        good = eligible_parents(self._xyz, refs, self.denom, cfg['view_tangent_min_observations'])
        supported_parents = int(good.sum().item())
        good &= (gradient >= max_grad) & (self.get_opacity[:, 0] >= min_opacity)
        # Parents that are about to be pruned should not consume the growth budget.
        # A large image footprint is not a geometric outlier and must neither
        # reject a growth parent nor remove its existing surface coverage.
        scales = self.get_scaling
        if max_screen_size:
            good &= scales.max(dim=1).values <= .1*extent
        candidates = good.nonzero().flatten()
        budget = min(cfg['view_tangent_max_parents'], max(0, cfg['view_tangent_max_points']-count))
        selected = candidates
        if len(selected) > budget:
            selected = selected[torch.topk(gradient[selected], budget, sorted=False).indices]
        split = scales[selected].amax(dim=1) > self.percent_dense*extent
        split_parents = selected[split]
        # First child for all selected parents; second only for split parents.
        parents = torch.cat((selected, split_parents))
        child_split = torch.cat((split, torch.ones_like(split_parents, dtype=torch.bool)))
        self.tmp_radii = radii
        normal_displacement, tangent_fraction = 0., 0.
        if len(parents):
            # Draw once per parent, then use opposite offsets for its split pair.
            samples = torch.randn_like(self._xyz[selected])*scales[selected]
            offsets = torch.bmm(rotation_matrix(self._rotation[selected]), samples[..., None]).squeeze(-1)
            offsets = torch.cat((offsets, -offsets[split]))
            child_refs = {key: value[parents] for key, value in refs.items()}
            new_xyz = tangent_children(self._xyz[parents], offsets, child_refs)
            # Audit the actual new centers, not just whether the mode was set.
            normal_displacement = float(((new_xyz-self._xyz[parents])*child_refs['normal']).sum(-1).abs().max().item())
            delta = new_xyz-child_refs['anchor']
            tangent = delta-(delta*child_refs['normal']).sum(-1, keepdim=True)*child_refs['normal']
            tangent_fraction = float((tangent.norm(dim=1)/child_refs['support'].clamp_min(1e-12)).max().item())
            new_scaling = self._scaling[parents].clone()
            new_scaling[child_split] -= np.log(1.6)
            self.densification_postfix(new_xyz, self._features_dc[parents], self._features_rest[parents],
                self._opacity[parents], new_scaling, self._rotation[parents], radii[parents], child_refs)
        # Always reset the accumulation window, even when no surface is trusted
        # or the budget is full. Unsupported historical gradients must not build up.
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        low_opacity = self.get_opacity[:, 0] < min_opacity
        split_replaced = torch.zeros_like(low_opacity)
        split_replaced[split_parents] = True
        world_oversize = torch.zeros_like(low_opacity)
        if max_screen_size:
            world_oversize = self.get_scaling.amax(dim=1) > .1*extent
        prune_mask = low_opacity | split_replaced | world_oversize
        pruned = int(prune_mask.sum().item())
        # Disjoint causes sum to `pruned`; split replacement is not loss of
        # coverage. Record the previously destructive screen-size candidates.
        prune_stats = dict(prune_stats_version=2,
            replaced_split_parents=len(split_parents),
            pruned_low_opacity=int((low_opacity & ~split_replaced).sum().item()),
            pruned_world_size=int((world_oversize & ~low_opacity & ~split_replaced).sum().item()),
            pruned_screen_size=0,
            large_screen_retained=int(((previous_max_radii > max_screen_size) & ~prune_mask[:count]).sum().item())
                if max_screen_size else 0)
        if pruned:
            self.prune_points(prune_mask)
        self.max_radii2D.zero_()
        self.tmp_radii = None
        self.last_view_tangent_stats = dict(points_before=count, eligible_parents=len(candidates),
            surface_supported_parents=supported_parents,
            selected_parents=len(selected), cloned_parents=int((~split).sum().item()),
            split_parents=len(split_parents), added_children=len(parents), pruned=pruned,
            **prune_stats,
            max_child_normal_displacement=normal_displacement, max_child_tangent_fraction=tangent_fraction,
            points_after=len(self._xyz), point_budget=cfg['view_tangent_max_points'])
        torch.cuda.empty_cache()
        return self.last_view_tangent_stats

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        grad = viewspace_point_tensor.grad
        if self.use_absgs and (grad is None or grad.shape[-1] != 4):
            raise RuntimeError("AbsGS requires the four-component absolute-gradient rasterizer")
        self.xyz_gradient_accum[update_filter, :1] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        if self.use_absgs:
            self.xyz_gradient_accum[update_filter, 1:2] += torch.norm(grad[update_filter,2:4], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

