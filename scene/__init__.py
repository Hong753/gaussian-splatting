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

import os
import random
import json
import shutil
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        # camera_utils imports scene.cameras, which initializes this package.
        # Defer the reverse dependency until construction so either module can
        # be imported first (including the standalone rendering workers).
        from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.depths, args.eval, args.train_test_exp,
                keyframe_manifest=getattr(args, "keyframe_manifest", ""),
                initial_point_cloud=getattr(args,'initial_point_cloud',''))
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if getattr(args, 'initial_point_cloud', ''):
                from scene.dataset_readers import fetchPly
                initial_path = os.path.abspath(os.path.expanduser(args.initial_point_cloud))
                if os.path.realpath(scene_info.ply_path)!=os.path.realpath(initial_path):
                    scene_info = scene_info._replace(point_cloud=fetchPly(initial_path), ply_path=initial_path)
            destination = os.path.join(self.model_path, 'input.ply')
            if os.path.realpath(scene_info.ply_path) != os.path.realpath(destination):
                shutil.copyfile(scene_info.ply_path, destination)
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        from utils.checkpoint_snapshots import atomic_write, atomic_json
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        atomic_write(os.path.join(point_cloud_path, "point_cloud.ply"),
                     lambda temp: self.gaussians.save_ply(str(temp)))
        if getattr(self.gaussians, "is_scaffold", False):
            self.gaussians.save_mlp_checkpoints(point_cloud_path)
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        atomic_json(os.path.join(self.model_path, "exposure.json"), exposure_dict)
        if getattr(self.gaussians,'pose_refiner',None) is not None:
            self.gaussians.pose_refiner.export(point_cloud_path,iteration)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

