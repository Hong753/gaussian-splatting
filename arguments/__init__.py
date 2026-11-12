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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.model_type = "3dgs"
        self.scaffold_voxel_size = 0.001
        self.scaffold_n_offsets = 10
        self.scaffold_feat_dim = 32
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self.keyframe_manifest = ""
        self.initial_point_cloud = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.occlusion_mode = "off"
        self.occlusion_backend = "whole"
        self.occlusion_inference_mode = "saved"
        self.train_depth_occlusion_dir = ""
        self.occlusion_geometry = ""
        self.occlusion_points = ""
        self.occlusion_voxel_size = 0.05
        self.occlusion_max_radius = 0.1
        self.occlusion_neighbors = 24
        self.occlusion_confidence = 0.6
        self.occlusion_cache = ""
        self.occlusion_margin = 0.05
        self.occlusion_sigma = 3.0
        self.occlusion_edge_pixels = 1
        self.occlusion_chunk_size = 131072
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.position_warmup_steps = 0
        self.position_ramp_steps = 0
        self.densification_phase_offset = 0
        self.fixed_lidar = False
        self.max_scale_ratio = 2.0
        self.max_scale_world = 0.0
        self.scale_control = "clamp"
        self.scale_regularization_weight = 0.01
        self.local_scale_cache = ""
        self.local_scale_ceiling_multiplier = 1.0
        self.local_scale_target_multiplier = 1.0
        self.local_scale_schedule = ""
        self.pose_refinement = ""
        self.lidar_offsets = ""
        self.pose_culling = ""
        self.distance_weighting = False
        self.distance_reference = 2.0
        self.distance_power = 2.0
        self.distance_min_weight = 0.05
        self.distance_chunk_size = 131072
        self.discrete_lidar = False
        self.discrete_lidar_cache = ""
        self.discrete_lidar_prune_sigma = 0.05
        self.discrete_lidar_seed_sigma_ratio = 0.5
        self.discrete_lidar_radius = 0.10
        self.discrete_lidar_max_new = 50000
        self.discrete_lidar_min_observations = 1
        self.lidar_depth_dir = ""
        self.lidar_depth_weight = 0.0
        self.lidar_depth_start = 0
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.absgs = False
        self.absgs_split_threshold = 0.0004
        self.surface_mode = "off"
        self.surface_patches = ""
        self.surface_confidence = 0.6
        self.surface_thickness = 0.2
        self.surface_position_weight = 0.01
        self.surface_normal_weight = 0.01
        self.surface_thickness_weight = 0.01
        self.surface_loss_samples = 100000
        self.view_tangent_radius_pixels = 3
        self.view_tangent_max_radius = 0.1
        self.view_tangent_depth_abs = 0.03
        self.view_tangent_depth_relative = 0.01
        self.view_tangent_min_observations = 2
        self.view_tangent_max_parents = 50000
        self.view_tangent_max_points = 12000000
        from utils.soft_surface import DEFAULTS as soft_surface_defaults
        for name, value in soft_surface_defaults.items():
            setattr(self, name, value)
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


