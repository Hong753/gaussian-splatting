import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.pgsr_lidar import PlaneGuidance, position_schedule
from scripts.run_orange4f_pgsr import protocol
from test_region_ablation import region_source


def toy(guided):
    xyz = torch.nn.Parameter(torch.zeros(3, 3))
    scale = torch.nn.Parameter(torch.zeros_like(xyz))
    optimizer = torch.optim.Adam([dict(params=[xyz], name='xyz'), dict(params=[scale], name='scaling')], lr=.1)
    model = SimpleNamespace(_xyz=xyz, get_xyz=xyz, _scaling=scale, optimizer=optimizer,
                            update_learning_rate=lambda _: .1)
    n = torch.tensor([1., 2., -2.])/3
    table = torch.zeros((3, 8))
    table[:, 6] = 1.
    table[:2, 3:6] = n
    table[1, 0] = 5.  # Plane is no longer supported at this Gaussian.
    return model, PlaneGuidance(table, torch.arange(3), guided), n


def test_project_actual_adam_step_and_clamp_moments():
    updated = []
    for guided in (False, True):
        model, policy, n = toy(guided)
        assert position_schedule(model, 0) == 0 and not model._xyz.requires_grad
        model._scaling.sum().neg().backward()
        policy.step(model)
        assert (model._xyz == 0).all() and model._xyz not in model.optimizer.state
        assert not model._scaling.any()
        assert not model.optimizer.state[model._scaling]['exp_avg'].any()
        model.optimizer.zero_grad(set_to_none=True)
        assert position_schedule(model, 1000) == .05 and model._xyz.requires_grad
        for grad in ([1., -3., 4.], [-1., 1., -2.]):
            model._xyz.grad = torch.tensor(grad).expand_as(model._xyz).clone()
            policy.step(model)
        updated.append(model._xyz.detach().clone())
    plain, biased = updated
    torch.testing.assert_close(biased[0] @ n, .1 * (plain[0] @ n))
    torch.testing.assert_close(biased[0] - (biased[0] @ n)*n, plain[0] - (plain[0] @ n)*n)
    torch.testing.assert_close(biased[1:], plain[1:])


@pytest.mark.parametrize('copies', [1, 2])
def test_split_and_clone_use_original_plane_and_inherit_ids(copies):
    m, policy, n = toy(True)
    mask = torch.tensor([True, False, True])
    offsets = torch.tensor([[.2, -.3, .4], [.5, -.6, .7]]).repeat(copies, 1)
    children = policy.children(m.get_xyz, mask, offsets, copies)
    assert (children[0::2] @ n).abs().max() < 1e-6
    torch.testing.assert_close(children[1::2], offsets[1::2])
    torch.testing.assert_close(policy.ids, torch.tensor([0, 1, 2] + [0, 2]*copies))


def test_matching_pca_warmup_and_initial_scale_caps():
    x, y = torch.meshgrid(torch.arange(6.), torch.arange(6.), indexing='xy')
    xyz = torch.stack((x, y, 2 + .5*x + y), -1).reshape(-1, 3)
    models = []
    for guided in (False, True):
        m = SimpleNamespace(get_xyz=xyz, _scaling=torch.nn.Parameter(torch.zeros_like(xyz)),
                            _rotation=torch.nn.Parameter(torch.zeros(len(xyz), 4)))
        policy = PlaneGuidance.fit(m, guided)
        normal = torch.tensor([-.5, -1., 1.]) / 1.5
        assert torch.allclose((policy.table[:, 3:6] @ normal).abs(), torch.ones(len(xyz)))
        assert torch.allclose(policy.table[:, 7].exp(), torch.full((len(xyz),), 2.))
        assert torch.allclose(m._scaling[:, 2].exp(), torch.full((len(xyz),), .2))
        models.append(m)
    torch.testing.assert_close(models[0]._rotation, models[1]._rotation)


def test_variants_use_same_saved_inputs_without_regional_model(tmp_path, region_source):
    source, cloud, manifest = region_source
    suite = tmp_path/'stride3_noaa_v1'
    fixed = suite/'3dgs_fixed_lidar'
    fixed.mkdir(parents=True)
    (fixed/'cfg_args').write_text(f'Namespace(source_path={str(source)!r}, images="images")')
    (fixed/'input.ply').symlink_to(cloud)
    (fixed/'keyframes.json').write_bytes(manifest.read_bytes())
    configs = []
    for variant in ('guided','covariance'):
        args = SimpleNamespace(suite=suite, variant=variant, source=None, output=None)
        cfg, links, saved, split = protocol(args)
        assert Path(cfg['source'])/'sparse/points3D.ply' in links
        assert split['train'] == sorted(json.loads(manifest.read_text())['selected_images'])
        assert set(split['test']).isdisjoint(split['train'])
        assert cfg['all_frames'] == 6 and cfg['train_frames'] == 3
        assert cfg['checkpoints'] == [7000, 30000, 50000]
        configs.append(cfg)
    assert configs[0]['fingerprints'] == configs[1]['fingerprints']
    assert configs[0]['model'] != configs[1]['model']
