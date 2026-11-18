import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch._dynamo  # Initialize lazy Torch registries before scoped module stubs.

from scripts import prepare_orange4f, run_orange4f_scaffold as runner, run_soft_surface_long as periodic
from test_region_ablation import region_source
from utils.checkpoint_snapshots import digest, publish_snapshot, read_snapshot
from utils.scaffold_growth import new_voxel_mask, candidate_offsets
from models.scaffold_arguments import ScaffoldOptimizationParams

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def cpu_backend():
    """Use real Torch Adam/MLPs/model code; replace only CUDA allocation and tiny KNN."""
    def knn(points):
        squared = torch.cdist(points, points).square()
        squared.fill_diagonal_(float('inf'))
        return squared.topk(min(3, len(points)-1), largest=False).values.mean(1)
    module = ModuleType('simple_knn._C'); module.distCUDA2 = knn
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {'simple_knn': ModuleType('simple_knn'), 'simple_knn._C': module}))
        stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self))
        stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda self, *a, **kw: self))
        for name in ('zeros', 'ones', 'eye', 'tensor'):
            original = getattr(torch, name)
            def allocate(*a, _original=original, **kw):
                if kw.get('device') == 'cuda': kw['device'] = 'cpu'
                return _original(*a, **kw)
            stack.enter_context(patch.object(torch, name, allocate))
        spec = importlib.util.spec_from_file_location('scaffold_cpu_test', ROOT/'models/scaffold_gaussian.py')
        loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
        yield loaded.ScaffoldGaussianModel


def options(fixed):
    p = argparse.ArgumentParser(); op = ScaffoldOptimizationParams(p)
    return op.extract(p.parse_args(['--position_lr_init', '.01', '--position_lr_final', '.001',
        *(['--freeze_anchors', '--disable_anchor_growth'] if fixed else [])]))


def build(Model, fixed):
    m = Model(feat_dim=8, n_offsets=2, voxel_size=.02, appearance_dim=0, update_init_factor=16)
    xyz = np.array([[i*.2, i*.01, 2.] for i in range(6)], np.float32)
    m.create_from_pcd(SimpleNamespace(points=xyz), [SimpleNamespace(image_name='a')], 1.)
    m.training_setup(options(fixed))
    return m


def step(m, iteration):
    m.update_learning_rate(iteration)
    # Simultaneous nonzero geometry/appearance gradients, using real Adam state.
    value = m.get_anchor + m._offset[:, 0]*m.get_scaling[:, :3]
    loss = (value-1).square().sum() + (m._anchor_feat-.3).square().sum()
    loss.backward(); m.optimizer.step(); m.optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize('fixed', [False, True])
def test_real_adam_freeze_survives_pruning_and_optimizer_resume(fixed):
    with cpu_backend() as Model:
        m = build(Model, fixed); before = m.get_anchor.detach().clone()
        for i in range(1, 4): step(m, i)
        assert torch.equal(m.get_anchor, before) == fixed
        assert m._offset.abs().max() > 0
        if fixed:
            assert m.get_anchor.grad is None and m.get_anchor not in m.optimizer.state
        survivor = torch.tensor([False, True, False, False, True, False])
        with torch.no_grad(): m.prune_anchor(survivor)
        xyz = m.get_anchor.detach().clone()
        step(m, 4)
        assert torch.equal(xyz, m.get_anchor) == fixed
        # Serialize rather than alias live optimizer moments.
        buffer = io.BytesIO(); torch.save(m.capture(), buffer); buffer.seek(0)
        restored = build(Model, fixed)
        restored.restore(torch.load(buffer, weights_only=False), options(fixed))
        step(m, 5); step(restored, 5)
        for name in ('_anchor', '_offset', '_anchor_feat', '_scaling'):
            torch.testing.assert_close(getattr(m, name), getattr(restored, name), rtol=0, atol=0)


def test_fixed_anchors_disable_growth_but_still_prune():
    with cpu_backend() as Model:
        m = build(Model, True); original = m.get_anchor.detach().clone()
        m.anchor_demon.fill_(100); m.opacity_accum.fill_(100)
        m.opacity_accum[1] = 0
        with patch.object(m, 'anchor_growing', side_effect=AssertionError('growth must be disabled')):
            with torch.no_grad(): m.adjust_anchor()
        assert len(m.get_anchor) == 5 and not m.get_anchor.requires_grad
        torch.testing.assert_close(m.get_anchor, original[[0, 2, 3, 4, 5]], rtol=0, atol=0)
        assert m.offset_denom.shape == (10, 1)


def test_voxelization_is_identical_and_does_not_mutate_lidar():
    with cpu_backend() as Model:
        m = build(Model, True)
        xyz = np.array([[.011, -.011, .02], [.012, -.012, .021], [.05, .02, .04]], np.float32)
        before = xyz.copy()
        one = m.voxelize_sample(xyz, .02); two = m.voxelize_sample(xyz, .02)
        np.testing.assert_array_equal(xyz, before)
        np.testing.assert_array_equal(one, two)
        assert len(one) == 2


def test_learnable_anchor_growth_adds_only_new_voxels():
    with cpu_backend() as Model:
        m = build(Model, False)
        m.update_depth = 1
        original = m.get_anchor.detach().clone()
        with torch.no_grad():
            m._offset.fill_(20.)
            m.offset_denom.fill_(100); m.offset_gradient_accum.fill_(100)
            m.opacity_accum.fill_(100); m.anchor_demon.fill_(100)
            with patch.object(torch, 'rand_like', side_effect=lambda x: torch.full_like(x, .9)):
                m.adjust_anchor()
        assert len(m.get_anchor) > len(original) and m.get_anchor.requires_grad
        torch.testing.assert_close(m.get_anchor[:len(original)], original, rtol=0, atol=0)
        assert torch.isfinite(m._anchor_feat).all()
        assert m.offset_denom.shape == (len(m.get_anchor)*m.n_offsets, 1)


def test_growth_indexing_matches_dense_reference_including_gradients():
    torch.manual_seed(3)
    anchors = torch.randn(9, 3, requires_grad=True)
    offsets = torch.randn(9, 4, 3, requires_grad=True)
    scales = torch.rand(9, 6, requires_grad=True)
    mask = torch.rand(36) > .4
    actual, parents = candidate_offsets(anchors, offsets, scales, mask)
    expected = (anchors[:, None] + offsets*scales[:, None, :3]).reshape(-1, 3)[mask]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(torch.autograd.grad(actual.square().sum(), (anchors, offsets, scales), retain_graph=True),
                    torch.autograd.grad(expected.square().sum(), (anchors, offsets, scales))):
        torch.testing.assert_close(a, b)
    feat = torch.randn(9, 8)
    torch.testing.assert_close(feat[parents], feat[:, None].repeat(1, 4, 1).reshape(-1, 8)[mask])
    existing = torch.tensor([[0, 0, 0], [-1, 2, 3], [0, 0, 0]])
    candidates = torch.tensor([[-1, 2, 3], [2, 3, 4], [0, 0, 0], [2, 3, 4]])
    expected_mask = ~(candidates[:, None] == existing).all(-1).any(-1)
    assert torch.equal(new_voxel_mask(existing, candidates), expected_mask)


def test_two_plans_reuse_all_inputs_noaa_without_mutating_old_runs(tmp_path, region_source):
    source, cloud, _ = region_source
    suite = tmp_path/'suite'
    prepare_orange4f.main(['--source', str(source), '--output', str(suite), '--point-cloud', str(cloud)])
    original = {p: digest(p) for p in suite.rglob('*') if p.is_file() and p.name != 'index.html'}
    for variant, gpu in [('learnable', '4'), ('fixed', '5')]:
        argv = ['--suite', str(suite), '--variant', variant, '--gpu', gpu]
        runner.main(argv); runner.main(argv)
        model = suite/f'scaffold_{variant}_anchors_v1'
        data = periodic.load_plan(model)
        args = SimpleNamespace(**data['training_args'])
        assert runner.build_plan(args) == data['plan']
        assert data['iterations'] == [10000, 20000, 30000] and data['frames'] == sorted(periodic.read_colmap_records(source))
        assert (model/'input.ply').resolve() == cloud
        assert digest(model/'keyframes.json') == digest(suite/'shared/keyframes.json')
        command = data['plan']['stages'][0]['argv']
        assert '--antialiasing' not in command and '--resolution' in command
        assert data['plan']['antialiasing'] is False and data['plan']['model_type'] == 'scaffold'
        assert ('--freeze_anchors' in command) == (variant == 'fixed')
        assert ('--disable_anchor_growth' in command) == (variant == 'fixed')
        assert command[command.index('--debug_preview_every')+1] == '500'
        with pytest.raises(ValueError, match='settings changed'): runner.main(argv+['--voxel-size', '.03'])
        with patch.object(runner, 'execute_plan') as execute:
            periodic.train(model)
            assert execute.call_args.args[1] == data['plan']
    assert all(digest(p) == value for p, value in original.items())


def test_snapshot_requires_matching_mlps_and_rejects_modified_weights(tmp_path):
    class Scene:
        model_path = str(tmp_path)
        gaussians = SimpleNamespace(is_scaffold=True, use_feat_bank=False, appearance_dim=0,
            n_offsets=10, get_anchor=torch.zeros(7, 3), capture=lambda: {'schema': 1})
        def save(self, iteration):
            folder = tmp_path/f'point_cloud/iteration_{iteration}'; folder.mkdir(parents=True)
            for name in ('point_cloud.ply', 'opacity_mlp.pt', 'cov_mlp.pt', 'color_mlp.pt'):
                (folder/name).write_text(name)
    (tmp_path/'cfg_args').write_text('Namespace(model_type="scaffold")')
    (tmp_path/'keyframes.json').write_text('{}')
    snapshot = publish_snapshot(Scene(), 10)
    assert snapshot['anchors'] == 7 and snapshot['gaussians'] == 70
    assert read_snapshot(tmp_path, 10) == snapshot
    (tmp_path/'point_cloud/iteration_10/color_mlp.pt').write_text('wrong model!')
    with pytest.raises(ValueError): read_snapshot(tmp_path, 10)


@pytest.mark.parametrize('sensed', [False, True])
def test_training_loop_previews_and_complete_checkpoint_publication(tmp_path, sensed):
    """Execute the actual loop, optimizer and image reports with a CPU RGB probe."""
    with cpu_backend() as Model, ExitStack() as stack:
        cameras = [SimpleNamespace(image_name=name, original_image=torch.ones(3, 16, 16)*.4)
                   for name in ('train.png', 'test.png')]
        class Scene:
            def __init__(self, dataset, gaussians):
                self.model_path = dataset.model_path; self.gaussians = gaussians
                xyz = np.array([[i*.2, i*.01, 2.] for i in range(6)], np.float32)
                gaussians.create_from_pcd(SimpleNamespace(points=xyz), cameras[:1], 1.)
            def getTrainCameras(self): return cameras[:1]
            def getTestCameras(self): return cameras[1:]
            def save(self, iteration):
                folder = tmp_path/f'point_cloud/iteration_{iteration}'; folder.mkdir(parents=True)
                self.gaussians.save_ply(str(folder/'point_cloud.ply'))
                self.gaussians.save_mlp_checkpoints(str(folder))
        def prefilter(camera, model, pipe, bg):
            return torch.ones(len(model.get_anchor), dtype=torch.bool)
        def render(camera, model, pipe, bg, visible_mask=None, retain_grad=False, update_anchors=None):
            points = model.get_anchor.repeat_interleave(model.n_offsets, 0) + model._offset.reshape(-1, 3)
            if retain_grad: points.retain_grad()
            color = (points.mean(0)+model._anchor_feat.mean()).sigmoid()
            result = dict(render=color[:, None, None].expand(3, 16, 16), viewspace_points=points,
                visibility_filter=torch.ones(len(points), dtype=torch.bool),
                selection_mask=torch.ones(len(points), dtype=torch.bool),
                neural_opacity=torch.ones(len(points), 1)*.4, scaling=model.get_scaling[:, 3:].repeat_interleave(model.n_offsets,0))
            if update_anchors is not None:
                result.update(active_parents=update_anchors.nonzero().flatten(),
                              active_children=update_anchors.repeat_interleave(model.n_offsets))
            return result
        scene_mod = ModuleType('scene'); scene_mod.Scene = Scene
        model_mod = ModuleType('models.scaffold_gaussian'); model_mod.ScaffoldGaussianModel = Model
        renderer_mod = ModuleType('models.scaffold_renderer'); renderer_mod.render = render; renderer_mod.prefilter_voxel = prefilter
        stack.enter_context(patch.dict(sys.modules, {'scene': scene_mod, 'models.scaffold_gaussian': model_mod,
                                                     'models.scaffold_renderer': renderer_mod}))
        stack.enter_context(patch.object(torch.cuda, 'get_rng_state_all', return_value=[]))
        stack.enter_context(patch.object(torch.cuda, 'current_device', return_value=0))
        stack.enter_context(patch.object(torch.random, 'fork_rng', side_effect=lambda **kw: nullcontext()))
        spec = importlib.util.spec_from_file_location('scaffold_train_probe', ROOT/'models/train_scaffold.py')
        trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
        ds = SimpleNamespace(model_path=str(tmp_path), source_path=str(tmp_path), scaffold_feat_dim=8,
            scaffold_n_offsets=2, scaffold_voxel_size=.02, white_background=False, train_test_exp=False,
            keyframe_manifest='', images='images', resolution=1)
        (tmp_path/'keyframes.json').write_text('{"selected_images":["train.png"]}')
        launch = SimpleNamespace(variant='sensed' if sensed else 'fixed', source=tmp_path, model=tmp_path, point_cloud=tmp_path/'input.ply',
            keyframes=tmp_path/'keyframes.json', voxel_size=.02, iterations=30000, checkpoint_every=10000,
            anchor_lr_init=.00016, anchor_lr_final=.0000016)
        argv = runner.build_plan(launch)['stages'][0]['argv']
        with patch.object(sys, 'argv', argv), patch.object(trainer, 'safe_state'), patch.object(trainer, 'train') as call:
            trainer.main()
            assert call.call_args.args[1].freeze_anchors
            assert call.call_args.args[3] == {10000, 20000, 30000}
            assert call.call_args.args[6] is True  # complete atomic snapshot publication
        opt = options(True); opt.iterations = 3; opt.start_stat = 0
        if sensed:
            from utils.scaffold_observations import anchor_digest
            folder = tmp_path/'observations'; folder.mkdir()
            xyz = np.array([[i*.2, i*.01, 2.] for i in range(6)], np.float32)
            centers = np.unique(np.round(xyz/.02), axis=0)*.02
            np.save(folder/'parents.npy', np.array([[1]],dtype=np.uint8))
            (folder/'metadata.json').write_text(json.dumps(dict(names=['train.png'],format='lidar_rows_v1',
                                                                anchor_sha256=anchor_digest(centers))))
            opt.sensed_membership = str(folder); opt.disable_anchor_pruning = True
        trainer.train(ds, opt, SimpleNamespace(debug=False, antialiasing=False), {3}, set(),
            dict(every=2, train=1, test=1, seed=0), publish=True)
        history = json.loads((tmp_path/'debug_previews/history.json').read_text())
        assert [s['iteration'] for s in history['snapshots']] == [1, 2, 3]
        assert all(not s['antialiasing'] and len(s['frames']) == 2 for s in history['snapshots'])
        saved = read_snapshot(tmp_path, 3)
        assert saved['anchors'] == 6 and len(saved['files']) == 7
        model = Model(feat_dim=8, n_offsets=2, voxel_size=.02, appearance_dim=0, update_init_factor=16)
        state, iteration = torch.load(tmp_path/'chkpnt3.pth', weights_only=False)
        assert iteration == 3
        if sensed:
            assert state['tensors']['_offset'][0].any() and not state['tensors']['_offset'][1:].any()
            last = json.loads((tmp_path/'scaffold_training.jsonl').read_text().splitlines()[-1])
            assert last['updated_parents'] == 1 and last['eligible_rendered_children'] == 2
        model.restore(state, opt)
        expected = model.get_anchor.detach().clone()
        model.load_ply(str(tmp_path/'point_cloud/iteration_3/point_cloud.ply'))
        torch.testing.assert_close(model.get_anchor, expected, rtol=0, atol=0)
