"""Physical child-offset bounds, real Adam resume, and saved-decoder parity."""
import argparse
import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.scaffold_arguments import ScaffoldOptimizationParams
from models.scaffold_decoder import generate_neural_gaussians
from test_orange4f_scaffold import cpu_backend, build
from utils.scaffold_offset_constraints import project_offsets_, validate_offset_radius


def options(radius, growth=False):
    parser = argparse.ArgumentParser()
    group = ScaffoldOptimizationParams(parser)
    args = ['--freeze_anchors', '--position_lr_init', '0', '--position_lr_final', '0',
            '--offset_max_radius', str(radius)]
    if not growth:
        args.append('--disable_anchor_growth')
    return group.extract(parser.parse_args(args))


def world_offsets(model):
    return model._offset * model.get_scaling[:, None, :3]


def step(model, iteration):
    model.update_learning_rate(iteration)
    camera = SimpleNamespace(camera_center=torch.zeros(3))
    decoded = generate_neural_gaussians(camera, model)
    xyz, color, opacity, scale, rotation = decoded
    # Drive actual decoded centers outward while learning the other networks.
    target = model.get_anchor.detach().repeat_interleave(model.n_offsets, 0) + .5
    loss = ((xyz - target).square().mean() + (color - .7).square().mean()
            + (opacity - .8).square().mean() + scale.square().mean()
            - .01 * model._scaling[:, :3].mean())
    loss.backward()
    assert torch.isfinite(loss)
    model.optimizer.step()
    states = {key: value.clone() for key, value in model.optimizer.state[model._offset].items()
              if torch.is_tensor(value)}
    scales = model._scaling.detach().clone()
    projected = model.project_offsets(record_stats=True, chunk_size=2)
    torch.testing.assert_close(model._scaling, scales, rtol=0, atol=0)
    for key, value in states.items():
        torch.testing.assert_close(model.optimizer.state[model._offset][key], value, rtol=0, atol=0)
    model.optimizer.zero_grad(set_to_none=True)
    return projected


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf'), -float('inf')])
def test_reject_invalid_radius(value):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        validate_offset_radius(value)


def test_disabled_constraint_is_bitwise_noop_including_zero_sign():
    offset = torch.tensor([[[1., -0., 1e20], [-2., 4., 0.]]], requires_grad=True)
    scales = torch.tensor([[0., 1., 2., 3., 4., 5.]], requires_grad=True)
    saved, saved_scales = offset.detach().clone(), scales.detach().clone()
    assert project_offsets_(offset, scales, 0) is None
    assert torch.equal(offset.view(torch.int32), saved.view(torch.int32))
    assert torch.equal(scales, saved_scales)
    stats = project_offsets_(offset, scales, 0, record_stats=True, chunk_size=1)
    assert stats['offset_projected_children'] == 0 and stats['offset_children'] == 2
    assert stats['offset_radius_before_projection_max'] == stats['offset_radius_after_projection_max'] > 1e20
    assert torch.equal(offset.view(torch.int32), saved.view(torch.int32))
    assert torch.equal(scales, saved_scales)


@pytest.mark.parametrize('chunk_size', [1, 2, 65536])
def test_world_radius_uses_anisotropic_scales_and_preserves_interiors(chunk_size):
    offsets = torch.tensor([[[.2, .4, .3], [0., 0., 0.]],
                            [[.001, -.002, .003], [2., -3., 1.]],
                            [[-.8, 1.2, .6], [1., 0., 0.]]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[2., .1, .3, .4, .5, .6],
                           [.8, 1.7, 3., .6, .5, .4],
                           [.01, .02, .03, 2., 3., 4.]], dtype=torch.float64).log()
    before, scale_before = offsets.detach().clone(), scales.clone()
    world_before = before * scales[:, None, :3].exp()
    norm_before = torch.linalg.vector_norm(world_before, dim=-1)
    inside = norm_before <= .05
    stats = project_offsets_(offsets, scales, .05, chunk_size=chunk_size, record_stats=True)
    world_after = offsets * scales[:, None, :3].exp()
    norm_after = torch.linalg.vector_norm(world_after, dim=-1)
    assert torch.all(norm_after <= .05)
    torch.testing.assert_close(offsets[inside], before[inside], rtol=0, atol=0)
    torch.testing.assert_close(scales, scale_before, rtol=0, atol=0)
    torch.testing.assert_close(world_after[~inside] / norm_after[~inside, None],
                               world_before[~inside] / norm_before[~inside, None])
    assert stats['offset_projected_children'] == int((~inside).sum())
    assert stats['offset_radius_after_projection_max'] <= .05
    assert stats['offset_radius_before_projection_max'] == pytest.approx(norm_before.max().item())
    # Scale changes alone can violate the constraint; it must be reapplied.
    scales[:, :3] += 2.
    project_offsets_(offsets, scales, .05, chunk_size=chunk_size)
    assert torch.all(torch.linalg.vector_norm(offsets * scales[:, None, :3].exp(), dim=-1) <= .05)


def test_large_finite_displacements_keep_direction_without_norm_overflow():
    offsets = torch.tensor([[[3e20, -4e20, 0.]]])
    scales = torch.zeros(1, 6)
    project_offsets_(offsets, scales, .05)
    torch.testing.assert_close(offsets[0, 0], torch.tensor([.03, -.04, 0.]), rtol=1e-6, atol=1e-8)
    assert offsets.norm() <= .05


def test_zero_and_empty_offsets_are_supported():
    offsets = torch.zeros(4, 3, 3)
    stats = project_offsets_(offsets, torch.zeros(4, 6), .05, record_stats=True)
    assert not offsets.any() and stats['offset_projected_children'] == 0
    empty = project_offsets_(torch.empty(0, 2, 3), torch.empty(0, 6), .05, record_stats=True)
    assert empty['offset_radius_after_projection_max'] == 0.


def test_model_adam_resume_policy_and_saved_decoder_parity(tmp_path):
    with cpu_backend() as Model:
        torch.manual_seed(7)
        model = build(Model, True)
        opt = options(.05)
        model.training_setup(opt)
        with torch.no_grad():
            model.mlp_opacity[2].weight.zero_()
            model.mlp_opacity[2].bias.fill_(.5)
            model._offset.fill_(2.)
        anchors_before = model.get_anchor.detach().clone()
        model.project_offsets()
        for iteration in range(1, 4):
            step(model, iteration)
            assert world_offsets(model).norm(dim=-1).max() <= .05
        assert torch.equal(model.get_anchor, anchors_before)
        assert not model.get_anchor.requires_grad and model.get_anchor not in model.optimizer.state
        saved = io.BytesIO()
        torch.save(model.capture(), saved)
        saved.seek(0)
        state = torch.load(saved, weights_only=False)
        assert state['offset_max_radius'] == .05
        restored = build(Model, True)
        restored.restore(state, opt)
        step(model, 4)
        step(restored, 4)
        for name in ('_anchor', '_offset', '_scaling', '_anchor_feat'):
            torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
        for bad in (0., .1):
            with pytest.raises(ValueError, match='offset_max_radius'):
                restored.restore(state, options(bad))
        # PLY stores the projected raw offsets; the unchanged decoder sees the
        # same physical centers and appearance after loading the saved networks.
        model.save_ply(str(tmp_path/'point_cloud.ply'))
        model.save_mlp_checkpoints(str(tmp_path))
        loaded = build(Model, True)
        loaded.load_ply(str(tmp_path/'point_cloud.ply'))
        model.eval()
        loaded.eval()
        camera = SimpleNamespace(camera_center=torch.zeros(3))
        for expected, actual in zip(generate_neural_gaussians(camera, model), generate_neural_gaussians(camera, loaded)):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        assert world_offsets(loaded).norm(dim=-1).max() <= .05


def test_legacy_checkpoint_defaults_to_disabled_and_rejects_new_cap():
    with cpu_backend() as Model:
        model = build(Model, True)
        state = model.capture()
        del state['offset_max_radius']
        restored = build(Model, True)
        restored.restore(state, options(0.))
        assert restored.offset_max_radius == 0.
        with pytest.raises(ValueError, match='offset_max_radius'):
            restored.restore(state, options(.05))


def test_bounded_children_growth_pruning_and_real_update_counts():
    with cpu_backend() as Model:
        model = build(Model, True)
        model.training_setup(options(.05, growth=True))
        model.update_depth = 1
        model.update_init_factor = 1
        before = model.get_anchor.detach().clone()
        with torch.no_grad():
            model._offset[:, :, 1] = 20.
            model.project_offsets()
            model.offset_denom.fill_(100)
            model.offset_gradient_accum.fill_(100)
            model.opacity_accum.fill_(100)
            model.anchor_demon.fill_(100)
            with patch.object(torch, 'rand_like', side_effect=lambda x: torch.full_like(x, .9)):
                model.adjust_anchor()
        added = len(model.get_anchor) - len(before)
        assert added > 0
        assert model.last_anchor_update == dict(anchors_before=len(before), anchors_added=added,
            anchors_pruned=0, anchors_after=len(model.get_anchor), anchor_growth_enabled=True)
        torch.testing.assert_close(model.get_anchor[:len(before)], before, rtol=0, atol=0)
        assert not model.get_anchor.requires_grad and world_offsets(model).norm(dim=-1).max() <= .05
        grown = model.get_anchor.detach().clone()
        model.disable_anchor_growth = True
        with torch.no_grad():
            model.anchor_demon.fill_(100)
            model.opacity_accum.fill_(100)
            model.opacity_accum[1] = 0
            model.adjust_anchor()
        assert model.last_anchor_update['anchors_added'] == 0
        assert model.last_anchor_update['anchors_pruned'] == 1
        assert model.last_anchor_update['anchors_after'] == len(grown) - 1
        assert torch.equal(model.get_anchor, grown[torch.arange(len(grown)) != 1])
        assert not model.get_anchor.requires_grad and world_offsets(model).norm(dim=-1).max() <= .05
