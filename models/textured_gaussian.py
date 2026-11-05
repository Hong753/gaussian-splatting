"""Textured Gaussians adapter for the upstream gsplat-based implementation."""
import math

import numpy as np
import torch
from torch import nn


def _backend():
    try:
        from textured_gaussians.rendering import rasterization_2dgs, rasterization_textured_gaussians
        from textured_gaussians.strategy import MCMCStrategy
    except ImportError as exc:
        raise ImportError("Install the pinned Textured Gaussians backend; see scripts/README.md") from exc
    return rasterization_2dgs, rasterization_textured_gaussians, MCMCStrategy


def camera_matrices(camera, device="cuda"):
    """Graphdeco Camera.R is camera-to-world, T is world-to-camera translation."""
    view = torch.eye(4, dtype=torch.float32, device=device)
    view[:3, :3] = torch.as_tensor(np.asarray(camera.R).T.copy(), device=device)
    view[:3, 3] = torch.as_tensor(np.asarray(camera.T).copy(), device=device)
    width, height = camera.image_width, camera.image_height
    intrinsic = torch.tensor([[width / (2 * math.tan(camera.FoVx / 2)), 0, width / 2],
                              [0, height / (2 * math.tan(camera.FoVy / 2)), height / 2],
                              [0, 0, 1]], device=device, dtype=torch.float32)
    return view[None], intrinsic[None]


class TexturedGaussianModel:
    """Maintains upstream parameter conventions and optimizer layout."""

    def __init__(self, sh_degree=3, max_gaussians=10000, texture_resolution=50,
                 textured_rgb=True, textured_alpha=True):
        self.sh_degree = sh_degree
        self.max_gaussians = max_gaussians
        self.texture_resolution = texture_resolution
        self.textured_rgb = textured_rgb
        self.textured_alpha = textured_alpha
        self.stage = "2dgs"
        self.splats = None
        self.spatial_lr_scale = 1.0

    def create_from_pcd(self, pcd, cameras, spatial_lr_scale):
        from simple_knn._C import distCUDA2
        from utils.sh_utils import RGB2SH
        points = torch.as_tensor(np.asarray(pcd.points), dtype=torch.float32, device="cuda")
        colors = torch.as_tensor(np.asarray(pcd.colors), dtype=torch.float32, device="cuda")
        if len(points) < 4:
            raise ValueError("Textured Gaussians require at least four initial points")
        if len(points) > self.max_gaussians:
            ids = torch.randperm(len(points), device="cuda")[:self.max_gaussians]
            points, colors = points[ids], colors[ids]
        n = len(points)
        dist = torch.clamp_min(distCUDA2(points.contiguous()), 1e-7).sqrt()
        sh = torch.zeros((n, (self.sh_degree + 1) ** 2, 3), device="cuda")
        sh[:, 0, :] = RGB2SH(colors)
        quat = torch.rand((n, 4), device="cuda")
        self.splats = nn.ParameterDict({
            "means": nn.Parameter(points),
            "scales": nn.Parameter(dist.log()[:, None].repeat(1, 3)),
            "quats": nn.Parameter(quat),
            "opacities": nn.Parameter(torch.full((n,), math.log(0.1 / 0.9), device="cuda")),
            "sh0": nn.Parameter(sh[:, :1]),
            "shN": nn.Parameter(sh[:, 1:]),
        })
        self.spatial_lr_scale = float(spatial_lr_scale)
        print(f"Textured Gaussians: initialized {n} 2D Gaussians from input point cloud")

    def add_textures(self):
        if "textures" in self.splats:
            raise ValueError("Textures already initialized")
        n, size = len(self.splats["means"]), self.texture_resolution
        texture = torch.ones((n, size, size, 4), device=self.splats["means"].device)
        texture[..., :3] = 0.1
        self.splats["textures"] = nn.Parameter(texture)
        self.stage = "textured"

    def optimizer_setup(self):
        rates = {"means": 1.6e-4 * self.spatial_lr_scale, "scales": 5e-3,
                 "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
                 "shN": 2.5e-3 / 20, "textures": 2.5e-3}
        return {key: torch.optim.Adam([{"params": param, "lr": rates[key]}],
                                     eps=1e-15, betas=(0.9, 0.999))
                for key, param in self.splats.items()}

    def render(self, camera, background, degree=None):
        raster_2d, raster_texture, _ = _backend()
        view, intrinsic = camera_matrices(camera, self.splats["means"].device)
        p = self.splats
        options = dict(means=p["means"], quats=p["quats"], scales=p["scales"].exp(),
                       opacities=p["opacities"].sigmoid(), colors=torch.cat([p["sh0"], p["shN"]], 1),
                       viewmats=view, Ks=intrinsic, width=camera.image_width, height=camera.image_height,
                       sh_degree=self.sh_degree if degree is None else degree, packed=False,
                       render_mode="RGB", backgrounds=background[None])
        if self.stage == "textured":
            textures = p["textures"]
            rgb = textures[..., :3] if self.textured_rgb else torch.zeros_like(textures[..., :3])
            if self.textured_alpha:
                alpha = textures[..., 3:4]
                alpha = alpha / (alpha.amax(dim=[1, 2], keepdim=True) + 1e-6)
            else:
                alpha = torch.ones_like(textures[..., 3:4])
            options["textures"] = torch.cat([rgb, alpha], -1).clamp(0, 1)
            result = raster_texture(**options)
        else:
            result = raster_2d(**options)
        return result[0][0].permute(2, 0, 1), result[-1]

    def save(self, path):
        torch.save({"model_type": "textured", "stage": self.stage,
                    "sh_degree": self.sh_degree, "texture_resolution": self.texture_resolution,
                    "textured_rgb": self.textured_rgb, "textured_alpha": self.textured_alpha,
                    "splats": {k: v.detach().cpu() for k, v in self.splats.items()}}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("model_type") != "textured" or checkpoint.get("stage") != "textured":
            raise ValueError("Not a finished Textured Gaussians checkpoint")
        model = cls(sh_degree=checkpoint["sh_degree"], texture_resolution=checkpoint["texture_resolution"],
                    textured_rgb=checkpoint["textured_rgb"], textured_alpha=checkpoint["textured_alpha"])
        model.stage = checkpoint["stage"]
        model.splats = nn.ParameterDict({k: nn.Parameter(v) for k, v in checkpoint["splats"].items()})
        return model
