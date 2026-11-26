"""CPU integration checks for the corrected-image Scaffold experiment suite.

The real launcher validates tiny COLMAP/images fixtures. CUDA training and
rendering are not simulated: parser/config checks stop before model creation,
and execution-order checks replace only subprocess.run.
"""
from argparse import ArgumentParser, Namespace
import ast
from contextlib import contextmanager, redirect_stdout
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from arguments import ModelParams, PipelineParams
from models.scaffold_arguments import ScaffoldOptimizationParams
from scripts import run_orange4f_scaffold_exposure as launcher
from scripts.view_comparisons import model_info, read_colmap_records
from scripts.view_scaffold import discover_checkpoints
from tests.test_exposure_training import fixture, write_json
from utils import training_previews


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('baseline', 'fixed_anchors', 'bounded_offsets', 'fixed_anchors_bounded_offsets')
SAVES = [5000, 6000, 10000, 15000, 20000, 30000, 60000, 100000, 120000]
LR_HORIZONS = ('position', 'offset', 'mlp_opacity', 'mlp_cov', 'mlp_color',
               'mlp_featurebank', 'appearance')


def tree_identity(root):
    """Detect both content changes and empty-directory creation during dry runs."""
    result = {}
    for path in root.rglob('*'):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[name] = ('symlink', os.readlink(path))
        elif path.is_file():
            result[name] = ('file', hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            result[name] = ('directory',)
    return result


def trainer_function(name, namespace):
    """Execute a real function body without importing CUDA-only trainer modules."""
    source = ROOT / 'models/train_scaffold.py'
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


class ScaffoldExposureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='scaffold_exposure_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.export, self.frames = fixture(self.root)
        self.output = self.root / 'scaffold_exposure_v1'
        self.old = self.root / 'scale_control_exposure_v1' / 'baseline'
        self.old.mkdir(parents=True)
        (self.old / 'scale_control_plan.json').write_text('{"existing_run": true}\n')

    @contextmanager
    def locations(self):
        with patch.object(launcher, 'SOURCE', self.source), \
             patch.object(launcher, 'EXPOSURE_ROOT', self.export), \
             patch.object(launcher, 'OUTPUT_ROOT', self.output):
            yield

    def plan(self, variant='baseline', gpu=0, **overrides):
        args = dict(variant=variant, gpu=gpu, exposure_dir=self.export, output_root=self.output,
                    initial_point_cloud=None, render_max_views=0, offset_radius=.05, dry_run=True)
        args.update(overrides)
        with self.locations(), redirect_stdout(io.StringIO()):
            return launcher.build_plan(Namespace(**args))

    def launch(self, variant='baseline', gpu=0, extra=(), dry_run=True):
        output = io.StringIO()
        argv = ['--variant', variant, '--gpu', str(gpu), *extra]
        if dry_run:
            argv.append('--dry-run')
        with self.locations(), redirect_stdout(output):
            result = launcher.main(argv)
        return result, output.getvalue()

    def parse_trainer(self, plan):
        model = Path(plan['model'])
        model.mkdir(parents=True)
        write_json(model / 'keyframes.json', dict(selected_images=plan['registered_names']))
        calls = []
        namespace = dict(
            ArgumentParser=ArgumentParser, ModelParams=ModelParams,
            ScaffoldOptimizationParams=ScaffoldOptimizationParams, PipelineParams=PipelineParams,
            Path=Path, training_previews=training_previews, __doc__='Actual Scaffold CLI test',
            safe_state=lambda quiet: None, train=lambda *args: calls.append(args))
        main = trainer_function('main', namespace)
        command = plan['commands'][1]
        entry = next(index for index, value in enumerate(command)
                     if Path(value).name == 'train_scaffold.py')
        with patch.object(sys, 'argv', command[entry:]):
            main()
        self.assertEqual(len(calls), 1)
        return calls[0]

    def test_four_plans_use_same_data_and_change_only_growth_and_offset_bound(self):
        expected = {
            'baseline': (False, 0.),
            'fixed_anchors': (True, 0.),
            'bounded_offsets': (False, .05),
            'fixed_anchors_bounded_offsets': (True, .05),
        }
        shared = None
        for gpu, variant in enumerate(VARIANTS):
            with self.subTest(variant=variant):
                plan = self.plan(variant, gpu)
                self.assertEqual(plan['source'], str(self.source))
                self.assertEqual(plan['images'], str(self.export / 'corrected'))
                self.assertEqual(plan['model'], str(self.output / variant))
                self.assertEqual(plan['registered_names'], sorted(frame['name'] for frame in self.frames))
                self.assertEqual(plan['training_views'], 2)
                self.assertEqual(plan['initial_point_cloud'], str(self.source / 'sparse/0/points3D.ply'))
                self.assertIsNone(plan['converted_from'])
                self.assertFalse(plan['learned_exposure'])
                self.assertFalse(plan['antialiasing'])
                self.assertEqual(plan['appearance_dim'], 0)
                opt = plan['options']
                self.assertEqual((opt['disable_anchor_growth'], opt['offset_max_radius']), expected[variant])
                self.assertTrue(opt['freeze_anchors'])
                self.assertFalse(opt['disable_anchor_pruning'])
                self.assertEqual((opt['position_lr_init'], opt['position_lr_final']), (0., 0.))
                self.assertEqual(opt['iterations'], 120000)
                for name in LR_HORIZONS:
                    self.assertEqual(opt[name + '_lr_max_steps'], 120000)
                self.assertEqual((opt['start_stat'], opt['update_from'], opt['update_until'],
                                  opt['update_interval']), (2000, 6000, 60000, 400))
                self.assertEqual((plan['first_anchor_update'], plan['last_anchor_update']), (6400, 59600))
                self.assertEqual(plan['save_iterations'], SAVES)
                common = {key: value for key, value in opt.items()
                          if key not in ('disable_anchor_growth', 'offset_max_radius')}
                if shared is None:
                    shared = common
                self.assertEqual(common, shared)
                preflight, train, render = plan['commands']
                self.assertEqual(preflight[preflight.index('--offset-max-radius') + 1],
                                 str(expected[variant][1]))
                self.assertEqual(train[train.index('--images') + 1], str(self.export / 'corrected'))
                self.assertEqual(train[train.index('--keyframe_manifest') + 1],
                                 str(self.output / variant / 'keyframes.json'))
                self.assertEqual(render[render.index('--model') + 1], str(self.output / variant))
                self.assertEqual(render[render.index('--split') + 1], 'both')
                self.assertEqual(render[render.index('--iteration') + 1], '120000')
                self.assertEqual(render[render.index('--gpu') + 1], str(gpu))
        self.assertFalse(self.output.exists())

    def test_real_trainer_parser_accepts_all_four_plans_and_complete_schedule(self):
        for gpu, variant in enumerate(VARIANTS):
            with self.subTest(variant=variant):
                plan = self.plan(variant, gpu)
                dataset, opt, pipe, saves, tests, previews, publish, checkpoint = self.parse_trainer(plan)
                self.assertEqual(dataset.model_type, 'scaffold')
                self.assertEqual(dataset.source_path, str(self.source))
                self.assertEqual(dataset.images, str(self.export / 'corrected'))
                self.assertEqual((dataset.scaffold_voxel_size, dataset.scaffold_n_offsets,
                                  dataset.scaffold_feat_dim), (.02, 10, 32))
                self.assertEqual(dataset.scaffold_offset_sh_degree, -1)
                self.assertEqual(dataset.resolution, 1600)
                self.assertEqual(dataset.data_device, 'cpu')
                self.assertTrue(dataset.colmap_principal_point)
                self.assertFalse(dataset.train_test_exp)
                self.assertFalse(dataset.eval)
                self.assertFalse(pipe.antialiasing)
                self.assertTrue(opt.freeze_anchors)
                self.assertFalse(opt.disable_anchor_pruning)
                self.assertEqual(opt.sensed_membership, '')
                self.assertFalse(opt.indexed_forward_only)
                self.assertEqual(opt.offset_max_radius, plan['options']['offset_max_radius'])
                self.assertEqual(opt.disable_anchor_growth, plan['options']['disable_anchor_growth'])
                self.assertEqual((opt.start_stat, opt.update_from, opt.update_until,
                                  opt.update_interval), (2000, 6000, 60000, 400))
                for name in LR_HORIZONS:
                    self.assertEqual(getattr(opt, name + '_lr_max_steps'), 120000)
                self.assertEqual(saves, set(SAVES))
                self.assertEqual(tests, set(SAVES))
                self.assertEqual(previews, dict(every=500, train=2, test=2, seed=0))
                self.assertTrue(publish)
                self.assertIsNone(checkpoint)

    def test_real_trainer_config_routes_saved_scaffold_report_to_corrected_rgb(self):
        plan = self.plan('bounded_offsets', 2)
        dataset, opt, pipe, saves, tests, previews, publish, checkpoint = self.parse_trainer(plan)

        class StopBeforeCUDA(Exception):
            pass

        def stop_before_cuda(dataset):
            raise StopBeforeCUDA

        namespace = dict(os=os, math=math, json=json, Path=Path, Namespace=Namespace,
                         make_model=stop_before_cuda)
        train = trainer_function('train', namespace)
        with self.assertRaises(StopBeforeCUDA):
            train(dataset, opt, pipe, saves, tests, previews, publish, checkpoint)

        spec = model_info(Path(plan['model']))
        self.assertEqual(spec['type'], 'scaffold')
        self.assertEqual(spec['cfg']['offset_max_radius'], .05)
        self.assertEqual((spec['cfg']['scaffold_voxel_size'], spec['cfg']['scaffold_n_offsets'],
                          spec['cfg']['scaffold_feat_dim']), (.02, 10, 32))
        self.assertFalse(spec['cfg']['train_test_exp'])
        self.assertFalse(spec['cfg']['antialiasing'])
        original = read_colmap_records(self.source)
        self.assertEqual(spec['splits']['train'], set(original))
        self.assertEqual(spec['splits']['eval'], set())
        for name, record in spec['records'].items():
            self.assertEqual(Path(record['image_path']), self.export / 'corrected' / name)
            with Image.open(record['image_path']) as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 255, 0))
            for key in ('R', 'T', 'width', 'height', 'fx', 'fy', 'cx', 'cy'):
                self.assertEqual(record[key], original[name][key])

    def test_all_dry_runs_leave_dataset_export_and_older_runs_untouched(self):
        before = tree_identity(self.root)
        with patch.object(launcher.subprocess, 'run') as run:
            for gpu, variant in enumerate(VARIANTS):
                with self.subTest(variant=variant):
                    result, text = self.launch(variant, gpu)
                    self.assertEqual(result, 0)
                    plan, _ = json.JSONDecoder().raw_decode(text[text.index('{'):])
                    self.assertEqual(plan['images'], str(self.export / 'corrected'))
                    self.assertEqual(plan['model'], str(self.output / variant))
                    self.assertNotIn('registered_names', plan)
            run.assert_not_called()
        self.assertEqual(tree_identity(self.root), before)

    def test_execution_validates_cuda_before_publishing_manifest_and_metadata(self):
        expected = self.plan('bounded_offsets', 2)
        model = Path(expected['model'])
        inputs = tree_identity(self.source), tree_identity(self.export), tree_identity(self.old)
        calls = []

        def run(command, *, cwd, env, check):
            index = len(calls)
            self.assertEqual(command, expected['commands'][index])
            self.assertEqual(cwd, launcher.ROOT)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
            self.assertTrue(check)
            if index == 0:
                self.assertFalse(model.exists())
            else:
                manifest = json.loads((model / 'keyframes.json').read_text())
                self.assertEqual(manifest['selected_images'], sorted(frame['name'] for frame in self.frames))
                saved = json.loads((model / 'scaffold_exposure_plan.json').read_text())
                self.assertEqual(saved['images'], str(self.export / 'corrected'))
                self.assertNotIn('registered_names', saved)
                for filename, digest in saved['exposure_preprocessing']['metadata_sha256'].items():
                    copied = model / 'exposure_preprocessing' / filename
                    self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), digest)
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(launcher.subprocess, 'run', side_effect=run):
            result, _ = self.launch('bounded_offsets', 2, dry_run=False)
        self.assertEqual(result, 0)
        self.assertEqual(calls, expected['commands'])
        self.assertEqual((tree_identity(self.source), tree_identity(self.export), tree_identity(self.old)), inputs)

    def test_failed_cuda_preflight_does_not_reserve_run_directory(self):
        before = tree_identity(self.root)
        failure = subprocess.CalledProcessError(1, ['synthetic_preflight'])
        with patch.object(launcher.subprocess, 'run', side_effect=failure) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.launch(dry_run=False)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(tree_identity(self.root), before)

    def test_incomplete_export_fails_even_in_dry_run_before_any_subprocess(self):
        write_json(self.export / 'status.json', dict(status='exporting', registered_images=2))
        before = tree_identity(self.root)
        with patch.object(launcher.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                self.launch()
            run.assert_not_called()
        self.assertEqual(tree_identity(self.root), before)

    def test_mismatched_frame_manifest_or_source_fails_before_publishing_run(self):
        summary_path = self.export / 'summary.json'
        summary = json.loads(summary_path.read_text())
        cases = ('wrong_source', 'wrong_registered_name')
        for case in cases:
            with self.subTest(case=case):
                write_json(summary_path, summary)
                write_json(self.export / 'frames.json', self.frames)
                if case == 'wrong_source':
                    write_json(summary_path, dict(summary, source=str(self.root / 'another_scene')))
                else:
                    frames = copy.deepcopy(self.frames)
                    frames[0].update(name='unregistered.png', corrected_rel='corrected/unregistered.png')
                    write_json(self.export / 'frames.json', frames)
                before = tree_identity(self.root)
                with patch.object(launcher.subprocess, 'run') as run:
                    with self.assertRaises(ValueError):
                        self.launch()
                    run.assert_not_called()
                self.assertEqual(tree_identity(self.root), before)

    def test_output_cannot_be_inside_source_or_exposure_export(self):
        before = tree_identity(self.root)
        for protected in (self.source, self.export):
            with self.subTest(protected=protected), patch.object(launcher.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    self.launch(extra=('--output-root', str(protected / 'runs')))
                run.assert_not_called()
        self.assertEqual(tree_identity(self.root), before)

    def test_existing_run_is_never_overwritten(self):
        model = self.output / 'baseline'
        model.mkdir(parents=True)
        (model / 'scaffold_exposure_plan.json').write_text('{"existing": true}\n')
        before = tree_identity(self.root)
        with patch.object(launcher.subprocess, 'run') as run:
            with self.assertRaises(FileExistsError):
                self.launch()
            run.assert_not_called()
        self.assertEqual(tree_identity(self.root), before)

    def test_explicit_output_root_and_offset_radius_are_used_without_writes(self):
        output = self.root / 'custom_scaffold_runs'
        before = tree_identity(self.root)
        with patch.object(launcher.subprocess, 'run') as run:
            result, text = self.launch('bounded_offsets', 2,
                extra=('--output-root', str(output), '--offset-radius', '.025'))
            run.assert_not_called()
        plan, _ = json.JSONDecoder().raw_decode(text[text.index('{'):])
        self.assertEqual(result, 0)
        self.assertEqual(plan['model'], str(output / 'bounded_offsets'))
        self.assertEqual(plan['options']['offset_max_radius'], .025)
        self.assertEqual(tree_identity(self.root), before)

    def test_missing_source_ply_is_converted_only_inside_new_run(self):
        (self.source / 'sparse/0/points3D.ply').unlink()
        source_points = self.source / 'sparse/0/points3D.txt'
        source_points.write_text('1 0 0 1 255 0 0 0\n')
        before = tree_identity(self.root)
        plan = self.plan()
        self.assertEqual(plan['converted_from'], str(source_points))
        self.assertEqual(plan['initial_point_cloud'], str(self.output / 'baseline' / 'input.ply'))
        self.assertEqual(tree_identity(self.root), before)

    def test_browser_suite_discovers_four_scaffold_runs_without_suite_json(self):
        for gpu, variant in enumerate(VARIANTS):
            plan = self.plan(variant, gpu)
            dataset, *_ = self.parse_trainer(plan)
            model = Path(plan['model'])
            (model / 'cfg_args').write_text(str(Namespace(**vars(dataset))))
            write_json(model / 'cameras.json', [dict(position=[0, 0, 0],
                rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], height=6, fy=11)])
            folder = model / 'point_cloud/iteration_5000'
            folder.mkdir(parents=True)
            (folder / 'point_cloud.ply').write_text('ply\nformat ascii 1.0\nend_header\n')
            for name in ('opacity_mlp.pt', 'cov_mlp.pt', 'color_mlp.pt'):
                with zipfile.ZipFile(folder / name, 'w') as archive:
                    archive.writestr('catalog_fixture', 'Not executable model weights')
        self.assertFalse((self.output / 'suite.json').exists())
        catalog = discover_checkpoints(self.output)
        self.assertEqual(set(catalog), set(VARIANTS))
        for variant in VARIANTS:
            self.assertEqual(catalog[variant], {
                5000: self.output / variant / 'point_cloud/iteration_5000/point_cloud.ply'})

    def test_browser_waits_for_exposure_snapshot_publication_but_keeps_legacy_support(self):
        plan = self.plan()
        model = Path(plan['model'])
        folder = model / 'point_cloud/iteration_5000'
        folder.mkdir(parents=True)
        (model / 'cfg_args').write_text("Namespace(model_type='scaffold')")
        write_json(model / 'cameras.json', [dict(position=[0, 0, 0],
            rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], height=6, fy=11)])
        plan_path = model / 'scaffold_exposure_plan.json'
        write_json(plan_path, plan)
        ply = folder / 'point_cloud.ply'
        ply.write_text('ply\nformat ascii 1.0\nend_header\n')
        for name in ('opacity_mlp.pt', 'cov_mlp.pt', 'color_mlp.pt'):
            with zipfile.ZipFile(folder / name, 'w') as archive:
                archive.writestr('catalog_fixture', 'Not executable model weights')

        # Complete inference files can precede optimizer/metadata publication.
        self.assertEqual(discover_checkpoints(self.output), {'baseline': {}})
        marker = folder / 'ready.json'
        write_json(marker, dict(iteration=5000))
        self.assertEqual(discover_checkpoints(self.output), {'baseline': {5000: ply}})

        # Older Scaffold runs never promised this snapshot publication protocol.
        marker.unlink()
        plan_path.unlink()
        self.assertEqual(discover_checkpoints(self.output), {'baseline': {5000: ply}})


if __name__ == '__main__':
    unittest.main()
