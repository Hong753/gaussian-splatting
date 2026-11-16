"""Live preview isolation, real split membership, image fidelity and publication."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from utils import training_previews as preview
from scripts.serve_training_previews import preview_directory
from test_view_comparisons import dataset as colmap_dataset


def camera(name, value=.4):
    return SimpleNamespace(image_name=name, original_image=torch.full((3, 8, 16), value))


def make_previewer(tmp_path, every=500):
    return preview.TrainingPreviewer(tmp_path/'debug_previews',
        [('train', camera('a/train.jpg')), ('held-out', camera('b/test.jpg'))],
        dict(preview.DEFAULTS, every=every), dict(source='source'))


def test_selection_is_stable_disjoint_and_does_not_consume_training_rng():
    train = [f'{i:03}.jpg' for i in range(12)]
    candidates = [f'{i:03}.jpg' for i in range(20)]
    state = random.getstate()
    chosen = preview.select_names(train, candidates, preview.DEFAULTS)
    assert chosen == preview.select_names(train[::-1], candidates[::-1], preview.DEFAULTS)
    assert len(chosen['train']) == len(chosen['held-out']) == 2
    assert set(chosen['train']) <= set(train)
    assert not set(chosen['held-out']) & set(train)
    assert state == random.getstate()


def test_keyframe_run_loads_only_selected_untrained_rgb_frames(tmp_path):
    source, manifest, names, selected = colmap_dataset(tmp_path)
    trains = [camera(name) for name in selected]
    scene = SimpleNamespace(model_path=str(tmp_path/'model'), getTrainCameras=lambda: trains,
                            getTestCameras=lambda: [])  # Actual launcher has eval=False.
    args = SimpleNamespace(source_path=str(source), images='images', resolution=-1, keyframe_manifest=str(manifest))
    loaded = []
    def load(record, dataset):
        loaded.append(record['name'])
        return camera(record['name'])
    with patch.object(preview, 'load_heldout_camera', side_effect=load):
        observer = preview.create_previewer(scene, args, preview.DEFAULTS)
    assert len(loaded) == 2
    assert set(loaded) <= set(names) - set(selected)
    assert [c for split, c in observer.frames if split == 'train'] == [c for c in trains if c.image_name in
        preview.select_names(selected, set(names)-set(selected), preview.DEFAULTS)['train']]
    assert observer.manifest['eligible_heldout'] == 10
    # Do not trigger metadata/image work when the observer is disabled.
    with patch.object(preview, 'select_names', side_effect=AssertionError('disabled')):
        assert preview.create_previewer(None, None, dict(preview.DEFAULTS, every=0)) is None


def test_missing_test_split_is_not_fabricated_and_never_changes_training(tmp_path):
    cams = [camera('only.jpg')]
    scene = SimpleNamespace(model_path=str(tmp_path), getTrainCameras=lambda: cams, getTestCameras=lambda: cams)
    observer = preview.create_previewer(scene, SimpleNamespace(source_path=str(tmp_path)), preview.DEFAULTS)
    assert [(split, c.image_name) for split, c in observer.frames] == [('train', 'only.jpg')]
    assert len(cams) == 1


def test_interval_overrides_reach_cli_and_invalid_options_fail():
    parser = argparse.ArgumentParser()
    preview.add_arguments(parser)
    args = parser.parse_args(['--debug-preview-every', '250', '--debug_preview_train', '1'])
    assert preview.options(args) == dict(preview.DEFAULTS, every=250, train=1)
    assert preview.command_arguments(args)[:2] == ['--debug_preview_every', '250']
    assert preview.command_arguments(SimpleNamespace()) == []  # Legacy saved plan remains valid.
    with pytest.raises(ValueError, match='nonnegative'):
        preview.options(parser.parse_args(['--debug-preview-every', '-1']))
    with pytest.raises(ValueError, match='at least one'):
        preview.options(parser.parse_args(['--debug-preview-train', '0', '--debug-preview-test', '0']))


def test_capture_keeps_model_optimizer_rng_and_camera_data_unchanged(tmp_path):
    observer = make_previewer(tmp_path)
    assert [i for i in (1, 2, 499, 500, 501, 999, 1000, 1001) if observer.due(i, 1001)] == [1, 500, 1000, 1001]
    assert not make_previewer(tmp_path/'off', every=0).due(1, 1)
    parameter = torch.nn.Parameter(torch.tensor(.5))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    parameter.square().backward(); optimizer.step(); optimizer.zero_grad()
    before, opt_before = parameter.detach().clone(), deepcopy(optimizer.state_dict())
    original_images = [c.original_image.clone() for _, c in observer.frames]
    called = []
    def refine(c):
        return SimpleNamespace(**vars(c), corrected=True)
    refiner = SimpleNamespace(names={'a/train.jpg', 'b/test.jpg'}, camera=refine)
    pipe = SimpleNamespace(antialiasing=True, occlusion_mode='off')
    def render(c, model, settings, background, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs == dict(separate_sh=False, use_trained_exp=False)
        called.append((c.image_name, getattr(c, 'corrected', False)))
        random.random(); np.random.rand(); torch.rand(1)
        settings.debug = True  # The caller's settings remain untouched.
        return {'render': model.expand(3, 8, 16)}
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    snapshot = observer.capture(500, parameter, render, pipe, torch.zeros(3), pose_refiner=refiner)
    assert called == [('a/train.jpg', True), ('b/test.jpg', False)]
    assert random.getstate() == py_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(parameter, before) and parameter.grad is None
    assert not hasattr(pipe, 'debug')
    for key, value in opt_before['state'][0].items():
        assert torch.equal(value, optimizer.state_dict()['state'][0][key])
    for (_, c), before_image in zip(observer.frames, original_images):
        assert torch.equal(c.original_image, before_image)
    expected = -10*math_log10((float(parameter)-.4)**2)
    for row in snapshot['frames']:
        assert row['psnr_db'] == pytest.approx(expected, abs=1e-4)
        for key in ('gt', 'render', 'error'):
            with Image.open(observer.root/row[key]) as image:
                assert image.size == (16, 8)
    history = json.loads((observer.root/'history.json').read_text())
    assert history['snapshots'] == [snapshot]
    assert json.loads((observer.root/'latest.json').read_text()) == snapshot
    assert 'refined' in snapshot['frames'][0]['pose']
    assert snapshot['frames'][1]['pose'] == 'original pose'
    assert snapshot['occlusion_mode'] == 'off'
    assert preview_directory(model=tmp_path) == observer.root


def math_log10(value):
    import math
    return math.log10(value)


def test_failed_frame_never_publishes_partial_history_or_leaves_work_files(tmp_path):
    observer = make_previewer(tmp_path)
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    calls = []
    def render(camera, *args, **kwargs):
        calls.append(camera.image_name)
        random.random(); torch.rand(1)
        return {'render': camera.original_image + (float('nan') if len(calls) == 2 else .1)}
    with pytest.raises(ValueError, match='Invalid debug preview'):
        observer.capture(1, None, render, SimpleNamespace(), torch.zeros(3))
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)
    assert json.loads((observer.root/'history.json').read_text())['snapshots'] == []
    assert not (observer.root/'latest.json').exists()
    assert not (observer.root/'iteration_000001').exists()
    assert not list(observer.root.glob('.preview-*'))
    assert not list(observer.root.rglob('*.tmp'))


def test_full_precision_metrics_and_immutable_history(tmp_path):
    observer = make_previewer(tmp_path)
    def render(camera, *args, **kwargs):
        return {'render': camera.original_image + .001}  # Rounding to PNG would lose this error.
    first = observer.capture(1, None, render, SimpleNamespace(), torch.zeros(3))
    assert first['frames'][0]['psnr_db'] == pytest.approx(60., abs=.001)
    observer.capture(500, None, lambda c,*a,**k: {'render': c.original_image}, SimpleNamespace(), torch.zeros(3))
    history = json.loads((observer.root/'history.json').read_text())
    assert [s['iteration'] for s in history['snapshots']] == [1, 500]
    assert history['snapshots'][1]['frames'][0]['psnr_db'] == 'inf'
    with pytest.raises(FileExistsError):
        observer.capture(500, None, render, SimpleNamespace(), torch.zeros(3))
    with pytest.raises(ValueError, match='selection changed'):
        preview.TrainingPreviewer(observer.root, [('train', camera('different.jpg'))], preview.DEFAULTS, dict(source='source'))


def test_server_selects_latest_run_without_loading_a_model(tmp_path):
    a = make_previewer(tmp_path/'a')
    b = make_previewer(tmp_path/'b')
    import os
    os.utime(a.root/'history.json', ns=(100, 100))
    os.utime(b.root/'history.json', ns=(200, 200))
    assert preview_directory(root=tmp_path) == b.root
    assert preview_directory(model=tmp_path/'a') == a.root
    with pytest.raises(FileNotFoundError):
        preview_directory(model=tmp_path/'missing')


def test_checkpoint_rewind_archives_old_images_and_starts_unmixed_history(tmp_path):
    cams = [camera('a.jpg')]
    scene = SimpleNamespace(model_path=str(tmp_path), getTrainCameras=lambda: cams, getTestCameras=lambda: [])
    args = SimpleNamespace(source_path=str(tmp_path))
    config = dict(preview.DEFAULTS, test=0)
    observer = preview.create_previewer(scene, args, config)
    render = lambda c, *a, **kw: {'render': c.original_image + .1}
    old = observer.capture(500, None, render, SimpleNamespace(), torch.zeros(3))
    # Matching/latest checkpoints retain history; a rewind archives the old
    # branch without deleting data, so duplicate steps can be rendered again.
    resumed = preview.create_previewer(scene, args, config, start_iteration=500)
    assert json.loads((resumed.root/'history.json').read_text())['snapshots'] == [old]
    restarted = preview.create_previewer(scene, args, config, start_iteration=100)
    archives = list(tmp_path.glob('debug_previews_before_restart_*'))
    assert len(archives) == 1
    assert (archives[0]/old['frames'][0]['render']).is_file()
    assert json.loads((restarted.root/'history.json').read_text())['snapshots'] == []
    restarted.capture(500, None, render, SimpleNamespace(), torch.zeros(3))
