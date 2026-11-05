import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('render_model_comparison', ROOT / 'scripts/render_model_comparison.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def camera(name, width=100, height=50):
    return types.SimpleNamespace(image_name=name, width=width, height=height,
                                 FovX=1.0, FovY=0.55, R=np.eye(3), T=np.zeros(3))


class ComparisonTests(unittest.TestCase):
    def test_natural_order_selection_and_dimensions(self):
        cams = [camera('frame10.png'), camera('frame2.png'), camera('frame1.png'), camera('frame3.png')]
        selected = mod.make_trajectory(cams, 1, 2, None)
        self.assertEqual([c.image_name for c in selected], ['frame2.png', 'frame10.png'])
        self.assertEqual(mod.video_dimensions(selected, 1280), (1280, 640))
        self.assertEqual(len(mod.trajectory_record(selected)), 2)
        with self.assertRaisesRegex(ValueError, 'different aspect'):
            mod.make_trajectory([camera('1', 100, 50), camera('2', 100, 70)], 0, 1, None)
        with self.assertRaises(ValueError):
            mod.video_dimensions(selected, 1279)
        with self.assertRaises(ValueError):
            mod.make_trajectory(cams, 0, 0, None)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg and ffprobe required')
    def test_real_video_encoding_both_models_share_identical_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'scene'
            source.mkdir()
            paths = []
            for kind in ('3dgs', 'scaffold'):
                run = root / kind
                ply = run / 'point_cloud/iteration_30000/point_cloud.ply'
                ply.parent.mkdir(parents=True)
                ply.touch()
                (run / 'cfg_args').write_text(f"Namespace(model_type='{kind}', source_path={str(source)!r}, white_background=False)")
                if kind == 'scaffold':
                    for name in ('opacity_mlp.pt', 'cov_mlp.pt', 'color_mlp.pt'):
                        (ply.parent / name).touch()
                paths.append(run)
            cams = [camera('frame10.png'), camera('frame2.png'), camera('frame1.png')]
            calls = []
            def fake_frames(spec, sequence, width, height, aa):
                calls.append((spec['type'], [c.image_name for c in sequence], aa))
                for idx, _ in enumerate(sequence):
                    yield np.full((height, width, 3), idx * 40, dtype=np.uint8)
            output = root / 'comparison'
            argv = ['--source', str(source), '--model-3dgs', str(paths[0]), '--model-scaffold', str(paths[1]),
                    '--output-dir', str(output), '--3dgs-aa', 'on', '--scaffold-aa', 'off', '--width', '64', '--fps', '10']
            with patch.object(mod, 'read_cameras', return_value=cams), patch.object(mod, 'rendered_frames', side_effect=fake_frames):
                mod.main(argv)
                with self.assertRaises(SystemExit):
                    mod.main(argv)
            self.assertEqual(calls, [('3dgs', ['frame1.png','frame2.png','frame10.png'], True),
                                     ('scaffold', ['frame1.png','frame2.png','frame10.png'], False)])
            details = json.loads((output / 'trajectory.json').read_text())
            self.assertEqual(details['frame_count'], 3)
            self.assertEqual(details['side_by_side'], {'left': '3dgs', 'right': 'scaffold'})
            for name, expected_width in [('3dgs.mp4',64), ('scaffold.mp4',64), ('side_by_side.mp4',128)]:
                result = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames',
                        '-of', 'json', str(output / name)], text=True)
                stream = json.loads(result)['streams'][0]
                self.assertEqual((stream['width'], stream['height'], int(stream['nb_read_frames'])),
                                 (expected_width, 32, 3))

    def test_rejects_wrong_model_and_missing_mlps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'run'
            ply = run / 'point_cloud/iteration_30000/point_cloud.ply'
            ply.parent.mkdir(parents=True)
            ply.touch()
            (run / 'cfg_args').write_text(f"Namespace(model_type='scaffold', source_path={str(root)!r})")
            with self.assertRaisesRegex(FileNotFoundError, 'Missing Scaffold MLP'):
                mod.model_spec(run, 'scaffold', 30000, root)
            with self.assertRaisesRegex(ValueError, 'expected'):
                mod.model_spec(run, '3dgs', 30000, root)

if __name__ == '__main__':
    unittest.main()
