"""CPU checks for corrected-image validation and native training/report paths."""
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from arguments import ModelParams
from scripts import run_orange4f_scale_control as launcher
from scripts.view_comparisons import model_info, read_colmap_records
from utils.exposure_training import validate_exposure_export


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fixture(root):
    """Two registered views with distinct raw/corrected pixels and a nested name."""
    source, export = root / 'scene', root / 'exposure'
    sparse, original = source / 'sparse' / '0', source / 'images'
    corrected = export / 'corrected'
    sparse.mkdir(parents=True)
    corrected.mkdir(parents=True)
    width, height = 8, 6
    names = ('floor/00001.png', '00002.png')
    (sparse / 'cameras.txt').write_text(f'1 PINHOLE {width} {height} 10 11 4 3\n')
    (sparse / 'images.txt').write_text(''.join(
        f'{index + 1} 1 0 0 0 {index} 0 0 1 {name}\n\n'
        for index, name in enumerate(names)))
    # The launcher only needs a file here; no GPU point-cloud loader is invoked.
    (sparse / 'points3D.ply').write_text(
        'ply\nformat ascii 1.0\nelement vertex 1\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n0 0 1 255 0 0\n')
    frames = []
    for index, name in enumerate(names):
        for directory, color in ((original, (255, 0, 0)), (corrected, (0, 255, 0))):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (width, height), color).save(path)
        frames.append(dict(name=name, image_id=index + 1, width=width, height=height,
                           original_rel=f'original/{name}', corrected_rel=f'corrected/{name}'))
    (export / 'original').symlink_to(original, target_is_directory=True)
    write_json(export / 'status.json', dict(status='complete', registered_images=len(names)))
    write_json(export / 'summary.json', dict(
        source=str(source), images=str(original), registered_images=len(names),
        warnings=['Synthetic fixture: unsupported frames keep their original exposure.']))
    write_json(export / 'frames.json', frames)
    return source, export, frames


class ExposureTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='exposure_training_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.export, self.frames = fixture(self.root)
        self.corrected = self.export / 'corrected'

    def test_complete_export_nested_names_and_provenance(self):
        corrected, provenance = validate_exposure_export(self.export, self.source)
        self.assertEqual(corrected, self.corrected.resolve())
        summary = json.loads((self.export / 'summary.json').read_text())
        self.assertEqual(provenance['warnings'], summary['warnings'])
        self.assertEqual(provenance['registered_images'], len(self.frames))
        for name in ('status.json', 'summary.json', 'frames.json'):
            self.assertEqual(provenance['metadata_sha256'][name],
                             hashlib.sha256((self.export / name).read_bytes()).hexdigest())
        json.dumps(provenance, allow_nan=False)

    def test_incomplete_export_is_rejected(self):
        write_json(self.export / 'status.json', dict(status='exporting', registered_images=2))
        with self.assertRaises(ValueError):
            validate_exposure_export(self.export, self.source)

    def test_mismatched_source_is_rejected(self):
        path = self.export / 'summary.json'
        summary = json.loads(path.read_text())
        summary['source'] = str(self.root / 'another_scene')
        write_json(path, summary)
        with self.assertRaises(ValueError):
            validate_exposure_export(self.export, self.source)

    def test_status_and_summary_counts_must_match_colmap(self):
        for filename in ('status.json', 'summary.json'):
            path = self.export / filename
            original = json.loads(path.read_text())
            with self.subTest(filename=filename):
                write_json(path, dict(original, registered_images=99))
                with self.assertRaises(ValueError):
                    validate_exposure_export(self.export, self.source)
            write_json(path, original)

    def test_missing_extra_and_duplicate_manifest_frames_are_rejected(self):
        extra = dict(self.frames[1], name='extra.png', corrected_rel='corrected/extra.png')
        cases = {
            'missing': self.frames[:-1],
            'extra': self.frames + [extra],
            'duplicate': [self.frames[0], self.frames[0]],
            'replacement_with_equal_count': [self.frames[0], extra],
        }
        for label, frames in cases.items():
            with self.subTest(label=label):
                write_json(self.export / 'frames.json', frames)
                with self.assertRaises(ValueError):
                    validate_exposure_export(self.export, self.source)

    def test_unsafe_or_redirected_manifest_paths_are_rejected(self):
        cases = (
            ('name', '../outside.png'),
            ('name', '/tmp/outside.png'),
            ('corrected_rel', '../outside.png'),
            ('corrected_rel', str(self.corrected / self.frames[0]['name'])),
            ('corrected_rel', self.frames[0]['original_rel']),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                frames = copy.deepcopy(self.frames)
                frames[0][field] = value
                write_json(self.export / 'frames.json', frames)
                with self.assertRaises(ValueError):
                    validate_exposure_export(self.export, self.source)

    def test_wrong_manifest_dimensions_are_rejected(self):
        frames = copy.deepcopy(self.frames)
        frames[0]['width'] += 1
        write_json(self.export / 'frames.json', frames)
        with self.assertRaises(ValueError):
            validate_exposure_export(self.export, self.source)

    def test_wrong_original_dimensions_are_rejected(self):
        Image.new('RGB', (9, 6), (255, 0, 0)).save(
            self.source / 'images' / self.frames[0]['name'])
        with self.assertRaises(ValueError):
            validate_exposure_export(self.export, self.source)

    def test_wrong_corrected_dimensions_are_rejected(self):
        Image.new('RGB', (8, 7), (0, 255, 0)).save(self.corrected / self.frames[0]['name'])
        with self.assertRaises(ValueError):
            validate_exposure_export(self.export, self.source)

    def test_missing_corrected_image_is_rejected(self):
        (self.corrected / self.frames[0]['name']).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            validate_exposure_export(self.export, self.source)

    def test_corrected_image_cannot_link_back_to_original(self):
        path = self.corrected / self.frames[0]['name']
        path.unlink()
        path.symlink_to(self.source / 'images' / self.frames[0]['name'])
        with self.assertRaises(ValueError):
            validate_exposure_export(self.export, self.source)

    def test_saved_native_config_uses_corrected_pixels_and_original_cameras(self):
        model = self.root / 'model'
        model.mkdir()
        parser = ArgumentParser()
        params = ModelParams(parser)
        cfg = params.extract(parser.parse_args([
            '-s', str(self.source), '--images', str(self.corrected), '-m', str(model)]))
        self.assertFalse(cfg.train_test_exp)
        (model / 'cfg_args').write_text(str(Namespace(**vars(cfg))))

        spec = model_info(model)
        original = read_colmap_records(self.source)
        self.assertFalse(spec['cfg']['train_test_exp'])
        self.assertEqual(spec['splits']['train'], set(original))
        self.assertEqual(spec['splits']['eval'], set())
        for name, record in spec['records'].items():
            self.assertEqual(Path(record['image_path']), self.corrected / name)
            with Image.open(record['image_path']) as corrected:
                self.assertEqual(corrected.getpixel((0, 0)), (0, 255, 0))
            with Image.open(original[name]['image_path']) as raw:
                self.assertEqual(raw.getpixel((0, 0)), (255, 0, 0))
            for key in ('R', 'T', 'width', 'height', 'fx', 'fy', 'cx', 'cy', 'fov_x', 'fov_y'):
                self.assertEqual(record[key], original[name][key])

    def dry_run(self, variant, gpu, *extra):
        cloud = self.source / 'sparse' / '0' / 'points3D.ply'
        cfg = dict(source_path=str(self.source), images='images', train_test_exp=False)
        output = io.StringIO()
        with patch.object(launcher, 'SOURCE', self.source), \
             patch.object(launcher, 'OUTPUT_ROOT', self.root / 'raw_runs'), \
             patch.object(launcher, 'EXPOSURE_OUTPUT_ROOT', self.root / 'corrected_runs'), \
             patch('utils.native_scale_control.self_test'), \
             patch('scripts.run_orange4f_blur_ablation.alternate_source',
                   return_value=(cfg, 1.1, cloud, None, len(self.frames))), \
             patch.object(launcher.subprocess, 'run') as subprocess_run, \
             redirect_stdout(output):
            result = launcher.main(['--variant', variant, '--gpu', str(gpu), '--dry-run', *extra])
        self.assertEqual(result, 0)
        subprocess_run.assert_not_called()
        return json.loads(output.getvalue())

    def test_all_four_corrected_dry_runs_keep_policy_schedule_and_separate_outputs(self):
        before = set(self.root.rglob('*'))
        for gpu, variant in enumerate(launcher.VARIANTS):
            with self.subTest(gpu=gpu, variant=variant):
                plan = self.dry_run(variant, gpu, '--exposure-dir', str(self.export))
                raw_plan = self.dry_run(variant, gpu)
                expected_output = self.root / 'corrected_runs' / variant
                self.assertEqual(plan['images'], str(self.corrected))
                self.assertEqual(plan['source'], str(self.source))
                self.assertEqual(plan['training_views'], len(self.frames))
                self.assertFalse(plan['learned_exposure'])
                self.assertEqual(plan['exposure_preprocessing']['registered_images'], len(self.frames))
                self.assertEqual(plan['options'], raw_plan['options'])
                self.assertEqual(plan['initial_point_cloud'], raw_plan['initial_point_cloud'])
                train, render, audit = plan['commands']
                self.assertEqual(train[train.index('--images') + 1], str(self.corrected))
                self.assertEqual(train[train.index('-s') + 1], str(self.source))
                self.assertEqual(train[train.index('-m') + 1], str(expected_output))
                self.assertNotIn('--train_test_exp', train)
                saved_iterations = ['5000', '6000', '10000', '15000', '20000',
                                    '30000', '60000', '100000', '120000']
                for flag in ('--test_iterations', '--save_iterations', '--checkpoint_iterations'):
                    start = train.index(flag) + 1
                    self.assertEqual(train[start:start + len(saved_iterations)], saved_iterations)
                self.assertEqual(render[render.index('--split') + 1], 'both')
                self.assertEqual(render[render.index('--gpu') + 1], str(gpu))
                self.assertEqual(render[render.index('--model') + 1], str(expected_output))
                self.assertEqual(audit[audit.index('--model') + 1], str(expected_output))
        self.assertEqual(set(self.root.rglob('*')), before)

    def test_dry_run_allows_explicit_output_root_and_keeps_original_default(self):
        explicit = self.root / 'custom_runs'
        corrected = self.dry_run('baseline', 0, '--exposure-dir', str(self.export),
                                 '--output-root', str(explicit))
        command = corrected['commands'][0]
        self.assertEqual(command[command.index('-m') + 1], str(explicit / 'baseline'))
        raw = self.dry_run('baseline', 0)
        command = raw['commands'][0]
        self.assertEqual(command[command.index('-m') + 1], str(self.root / 'raw_runs' / 'baseline'))
        self.assertEqual(raw['images'], 'images')
        self.assertIsNone(raw['exposure_preprocessing'])
        self.assertFalse(explicit.exists())
        self.assertFalse((self.root / 'raw_runs').exists())


if __name__ == '__main__':
    unittest.main()
