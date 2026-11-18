"""Check the original Scaffold-GS comparison split, optimizer settings and report routing."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import render_views_worker, upstream_scaffold
from utils.checkpoint_snapshots import digest


def test_prepare_pair_uses_identical_source_poses_and_only_suite_train_names(tmp_path):
    from scripts.view_comparisons import read_colmap_records
    from test_sparse_observation_viewer import fixture
    source = tmp_path/'capture'
    sparse = fixture(source)
    cloud = sparse/'points3D.ply'
    suite = tmp_path/'suite'
    suite.mkdir()
    selection = suite/'keyframes.json'
    selection.write_text(json.dumps(dict(selected_images=['09882.jpg'])))
    (suite/'suite.json').write_text(json.dumps(dict(source=str(source), cloud=dict(original=str(cloud), cached=str(cloud)),
        selection=str(selection), input_fingerprints={str(selection):digest(selection), str(cloud):digest(cloud)})))
    plans = []
    for variant in ('fixed','adaptive'):
        model = suite/variant
        args = SimpleNamespace(model=model, suite=suite, variant=variant, source_code=tmp_path/'code',
                               dependencies=tmp_path/'deps', voxel_size=.02, point_cloud=None)
        upstream_scaffold.prepare(args)
        upstream_scaffold.prepare(args)
        plans.append(json.loads((model/'upstream.json').read_text()))
        assert plans[-1].pop('variant') == variant
        assert (model/'keyframes.json').read_bytes() == selection.read_bytes()
        assert (model/'input.ply').resolve() == cloud
        assert not (model/'observations').exists()
        assert plans[-1]['train_views'] == plans[-1]['heldout_views'] == 1
        args.voxel_size = .03
        with pytest.raises(ValueError, match='differs'):
            upstream_scaffold.prepare(args)
    assert plans[0] == plans[1]
    assert set(read_colmap_records(source)) == {'09882.jpg','empty.jpg'}


def test_release_defaults_differ_only_in_anchor_adjustment():
    checkout = os.environ.get('SCAFFOLD_UPSTREAM_SOURCE')
    if not checkout:
        pytest.skip('Set SCAFFOLD_UPSTREAM_SOURCE to the pinned authors checkout')
    spec = importlib.util.spec_from_file_location('author_args', Path(checkout)/'arguments/__init__.py')
    arguments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arguments)
    api = SimpleNamespace(arguments=arguments)
    plan = dict(source='/capture', voxel_size=.02)
    fixed = upstream_scaffold.parameters(api, '/model', dict(plan, variant='fixed'))
    adaptive = upstream_scaffold.parameters(api, '/model', dict(plan, variant='adaptive'))
    for i in (0,2):
        assert vars(fixed[i]) == vars(adaptive[i])
    native = arguments.OptimizationParams(argparse.ArgumentParser())
    assert vars(fixed[1]) == dict(vars(adaptive[1]), update_until=0)
    assert vars(adaptive[1]) == vars(native)
    assert adaptive[1].update_until == 15000 and adaptive[1].iterations == 30000
    assert fixed[1].position_lr_init == adaptive[1].position_lr_final == 0.


def test_report_routes_saved_model_to_its_own_renderer(tmp_path):
    spec = dict(path=tmp_path, cfg=dict(scaffold_upstream=True))
    def run(command, check):
        assert check and command[2] == 'render'
        assert command[command.index('--iteration')+1] == '10000'
        assert json.loads(Path(command[command.index('--selection')+1]).read_text()) == dict(names=['a.jpg'])
    with patch.object(render_views_worker.subprocess, 'run', side_effect=run) as called:
        render_views_worker.render_views(spec, ['a.jpg'], 10000, tmp_path/'out', False)
        assert called.call_count == 1
