#!/usr/bin/env python3
"""Smoke test AbsGS rasterizer against the standard AA-enabled rasterizer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as StandardSettings
from diff_gaussian_rasterization import GaussianRasterizer as StandardRasterizer
from diff_gaussian_rasterization_abs import GaussianRasterizationSettings as AbsSettings
from diff_gaussian_rasterization_abs import GaussianRasterizer as AbsRasterizer
from utils.graphics_utils import getProjectionMatrix


def run(settings_class, rasterizer_class, components):
    device = "cuda"
    view = torch.eye(4, device=device)
    projection = getProjectionMatrix(znear=0.01, zfar=100, fovX=1.0, fovY=1.0).transpose(0, 1).cuda()
    settings = settings_class(image_height=32, image_width=32,
                             tanfovx=0.5463024898, tanfovy=0.5463024898,
                             bg=torch.zeros(3, device=device), scale_modifier=1.0,
                             viewmatrix=view, projmatrix=projection, sh_degree=0,
                             campos=torch.zeros(3, device=device), prefiltered=False,
                             debug=False, antialiasing=True)
    means = torch.tensor([[0.0, 0.0, 2.0], [0.12, 0.08, 2.3]], device=device, requires_grad=True)
    means2d = torch.zeros((2, components), device=device, requires_grad=True)
    colors = torch.tensor([[0.9, 0.3, 0.2], [0.2, 0.7, 0.8]], device=device, requires_grad=True)
    opacities = torch.full((2, 1), 0.7, device=device, requires_grad=True)
    scales = torch.full((2, 3), 0.2, device=device, requires_grad=True)
    rotations = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], device=device, requires_grad=True)
    rasterizer = rasterizer_class(raster_settings=settings)
    image, radii, _ = rasterizer(means3D=means, means2D=means2d, colors_precomp=colors,
                                 opacities=opacities, scales=scales, rotations=rotations)
    yy, xx = torch.meshgrid(torch.arange(32, device=device), torch.arange(32, device=device), indexing="ij")
    target = ((xx + yy) % 2).float()[None].expand_as(image)
    ((image - target) ** 2).mean().backward()
    return image.detach(), radii, means.grad, means2d.grad


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    standard = run(StandardSettings, StandardRasterizer, 3)
    absolute = run(AbsSettings, AbsRasterizer, 4)
    if not torch.equal(standard[1], absolute[1]) or not bool((absolute[1] > 0).any()):
        raise AssertionError("Gaussian visibility differs or the smoke-test scene is empty")
    torch.testing.assert_close(standard[0], absolute[0], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(standard[2], absolute[2], rtol=2e-3, atol=2e-6)
    torch.testing.assert_close(standard[3][:, :2], absolute[3][:, :2], rtol=2e-3, atol=2e-6)
    signed = absolute[3][:, :2].abs()
    absolute_grad = absolute[3][:, 2:4]
    if not bool(torch.isfinite(absolute_grad).all()) or not bool((absolute_grad > 0).any()):
        raise AssertionError("No valid per-pixel absolute gradient was produced")
    if not bool((absolute_grad + 2e-6 >= signed).all()):
        raise AssertionError("Absolute gradients are smaller than the magnitude of signed sums")
    print("AbsGS AA smoke test passed: forward, signed gradients, and absolute gradients.")


if __name__ == "__main__":
    main()
