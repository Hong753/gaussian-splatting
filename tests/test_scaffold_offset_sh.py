import io
from types import SimpleNamespace

import pytest
import torch

from models.scaffold_decoder import generate_neural_gaussians
from models.scaffold_offset_sh import scaffold_model_class
from test_orange4f_scaffold import build, cpu_backend, options
from utils.scaffold_observations import sensed_step
from utils.sh_utils import C0, C1


def emit(model, rows, position=(0., 0., 0.)):
    camera = SimpleNamespace(camera_center=torch.tensor(position))
    return generate_neural_gaussians(camera, model, rows)


def all_children(model):
    with torch.no_grad():
        model.mlp_opacity[2].weight.zero_()
        model.mlp_opacity[2].bias.fill_(0.5)


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_independent_offset_colors_and_saved_free_camera(tmp_path, degree):
    with cpu_backend() as Base:
        Model = scaffold_model_class(Base, degree)
        m = build(Model, True)
        all_children(m)
        rows = torch.tensor([3, 1])
        table = m.mlp_color.coefficients
        with torch.no_grad():
            table[3, 1, 0, 0] = .4
            if degree:
                table[3, 1, 2, 3] = .6
            m._offset[3, 1, 0] = 8.
        xyz, color, *_ = emit(m, rows)
        expected = torch.full_like(color, .5)
        expected[1, 0] += C0 * .4
        if degree:
            expected[1, 2] -= .6 * C1 * xyz[1, 0] / xyz[1].norm()
        torch.testing.assert_close(color, expected)
        color[1].sum().backward()
        assert torch.nonzero(table.grad.abs().sum(dim=(-1, -2))).tolist() == [[3, 1]]
        m.optimizer.step()
        m.optimizer.zero_grad(set_to_none=True)
        # Non-emitting offsets and masked-out parents must not shift the color lookup.
        with torch.no_grad():
            m.mlp_opacity[2].bias[0] = -.5
        mask = torch.arange(6) % 2 == 1
        colors = emit(m, mask)[1]
        torch.testing.assert_close(colors[[1, 0]], emit(m, rows)[1])
        assert emit(m, mask & ~mask)[1].shape == (0, 3)
        view = (2., -1., .7)
        with torch.no_grad():
            if degree:
                assert not torch.allclose(colors, emit(m, mask, view)[1])
            m.save_ply(str(tmp_path/'point_cloud.ply'))
            m.save_mlp_checkpoints(tmp_path)
            # Ordinary native viewer and evaluation loaders recognize the saved color module.
            for cls in (Base, Model):
                saved = cls(feat_dim=8, n_offsets=2, appearance_dim=0)
                saved.load_ply(str(tmp_path/'point_cloud.ply'))
                saved.eval()
                torch.testing.assert_close(emit(saved, rows, view)[1], emit(m, rows, view)[1])


def test_sh_adam_row_gating_growth_pruning_and_resume():
    with cpu_backend() as Base, torch.no_grad():
        Model = scaffold_model_class(Base, 1)
        m = build(Model, False)
        table = m.mlp_color.coefficients
        initial = torch.arange(len(table))
        table.grad = torch.ones_like(table)
        sensed_step(m.optimizer, initial % 2 == 0)
        assert torch.equal(table[1::2], torch.zeros_like(table[1::2]))
        first = table.clone()
        moments = m.optimizer.state[table]['exp_avg'].clone()
        table.grad = torch.ones_like(table)
        sensed_step(m.optimizer, initial % 2 == 1)
        torch.testing.assert_close(table[::2], first[::2], rtol=0, atol=0)
        torch.testing.assert_close(m.optimizer.state[table]['exp_avg'][::2], moments[::2], rtol=0, atol=0)
        m._offset.fill_(100.)
        first = table.clone()
        torch.manual_seed(15)
        m.anchor_growing(torch.ones(len(table)*2), .1, torch.ones(len(table)*2, dtype=torch.bool))
        grown = m.mlp_color.coefficients
        assert len(grown) == len(m.get_anchor) > len(first)
        torch.testing.assert_close(grown[:len(first)], first)
        assert not grown[len(first):].any() and not m.optimizer.state[grown]['exp_avg'][len(first):].any()
        remove = torch.arange(len(grown)) % 2 == 0
        m.prune_anchor(remove)
        torch.testing.assert_close(m.mlp_color.coefficients, grown[~remove])
        buf = io.BytesIO()
        torch.save(m.capture(), buf)
        buf.seek(0)
        restored = build(Model, False)
        restored.restore(torch.load(buf, weights_only=False), options(False))
        for instance in (m, restored):
            p = instance.mlp_color.coefficients
            p.grad = torch.full_like(p, .3)
            instance.optimizer.step()
        torch.testing.assert_close(m.mlp_color.coefficients, restored.mlp_color.coefficients)
