from math import pi
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from utils.ray_depth_gradients import active_policy, ray_weights, render_weights, weight_rgb_backward


def test_opacity_normalization_ray_range_and_unknown_depth():
    camera = SimpleNamespace(image_width=5, image_height=1, FoVx=pi / 2, FoVy=pi / 2)
    alpha = torch.tensor([[[0.8, 0.0, 0.2, 0.05, 0.8]]])
    invdepth = torch.tensor([[[0.2, 0.0, 0.05, 0.01, float("nan")]]], requires_grad=True)
    policy = dict(reference_m=2.0, min_weight=0.05, min_opacity=0.10)
    weights, valid = ray_weights(camera, invdepth, alpha, policy)
    expected = torch.tensor([[[0.25 / (1.0 + 0.8**2), 1.0, 0.25, 1.0, 1.0]]])
    torch.testing.assert_close(weights, expected)
    assert valid.tolist() == [[[True, False, True, False, False]]]
    assert not weights.requires_grad


def test_pixel_hook_weights_both_direct_and_window_gradients_exactly():
    reference = torch.linspace(0, 1, 16).reshape(1, 1, 4, 4)
    image = reference.clone().requires_grad_()
    changed = reference.clone().requires_grad_()
    weights = torch.linspace(0.05, 1.0, 16).reshape(1, 4, 4).requires_grad_()
    weight_rgb_backward(changed, weights)

    def loss(pixels):
        return pixels.abs().mean() + F.avg_pool2d(pixels, 3, stride=1).square().mean()

    assert loss(changed).item() == loss(image).item()
    loss(image).backward()
    loss(changed).backward()
    torch.testing.assert_close(changed.grad, image.grad * weights)
    assert weights.grad is None


def test_native_buffers_are_requested_without_gradients():
    camera = SimpleNamespace(image_width=3, image_height=1, FoVx=pi / 2, FoVy=pi / 2)
    model = SimpleNamespace(get_xyz=torch.ones(2, 3, requires_grad=True))
    pipe = object()
    depth_cutoff = object()

    def render(cam, gaussians, renderer, background, **kwargs):
        assert (cam, gaussians, renderer) == (camera, model, pipe)
        assert not torch.is_grad_enabled()
        assert torch.count_nonzero(background) == 0
        assert torch.all(kwargs["override_color"] == 1)
        assert kwargs["occlusion_depth"] is depth_cutoff
        opacity = torch.tensor([[[0.1, 0.7, 0.7]]])
        invdepth = opacity / torch.tensor([[[40.0, 1.0, 4.0]]])
        return {"render": opacity.expand(3, -1, -1), "depth": invdepth}

    policy = dict(reference_m=2.0, min_weight=0.05, min_opacity=0.10)
    weights, valid = render_weights(render, camera, model, pipe, policy, occlusion_depth=depth_cutoff)
    torch.testing.assert_close(weights, torch.tensor([[[0.05, 1.0, 0.25 / (1.0 + (2 / 3) ** 2)]]]))
    assert valid.all()


def test_policy_rejects_start_after_end_and_defaults_off():
    assert active_policy(SimpleNamespace()) is None
    options = SimpleNamespace(
        ray_depth_gradients=True, ray_depth_reference=2.0, ray_depth_start=1001, iterations=1000
    )
    with pytest.raises(ValueError, match="within the training schedule"):
        active_policy(options)
