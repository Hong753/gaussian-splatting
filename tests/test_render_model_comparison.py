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
from PIL import Image

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
            image_dir = source / 'images'
            image_dir.mkdir()
            for info in cams:
                Image.new('RGB', (100, 50), (255, 0, 0)).save(image_dir / info.image_name)
            calls = []
            def fake_frames(spec, sequence, width, height, aa):
                calls.append((spec['type'], [c.image_name for c in sequence], aa))
                color = [0, 255, 0] if spec['type'] == '3dgs' else [0, 0, 255]
                for _ in sequence:
                    yield np.broadcast_to(np.array(color, dtype=np.uint8), (height, width, 3)).copy()
            output = root / 'comparison'
            argv = ['--source', str(source), '--model-3dgs', str(paths[0]), '--model-scaffold', str(paths[1]),
                    '--output-dir', str(output), '--3dgs-aa', 'on', '--scaffold-aa', 'off', '--width', '64', '--fps', '10']
            fake_tqdm = types.SimpleNamespace(tqdm=lambda xs, **kwargs: xs)
            with patch.object(mod, 'read_cameras', return_value=cams), patch.object(mod, 'rendered_frames', side_effect=fake_frames), patch.dict(sys.modules, {'tqdm': fake_tqdm}):
                mod.main(argv)
                with self.assertRaises(SystemExit):
                    mod.main(argv)
            self.assertEqual(calls, [('3dgs', ['frame1.png','frame2.png','frame10.png'], True),
                                     ('scaffold', ['frame1.png','frame2.png','frame10.png'], False)])
            details = json.loads((output / 'trajectory.json').read_text())
            self.assertEqual(details['frame_count'], 3)
            self.assertEqual(details['side_by_side'], {'left': 'original', 'middle': '3dgs', 'right': 'scaffold'})
            self.assertEqual(details['original_images'], str(image_dir))
            for name, expected_width in [('original.mp4',64), ('3dgs.mp4',64), ('scaffold.mp4',64), ('side_by_side.mp4',192)]:
                result = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames',
                        '-of', 'json', str(output / name)], text=True)
                stream = json.loads(result)['streams'][0]
                self.assertEqual((stream['width'], stream['height'], int(stream['nb_read_frames'])),
                                 (expected_width, 32, 3))
            raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output / 'side_by_side.mp4'),
                                           '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
            image = np.frombuffer(raw, dtype=np.uint8).reshape(32, 192, 3)
            for x, dominant in [(32, 0), (96, 1), (160, 2)]:
                pixel = image[16, x]
                self.assertGreater(pixel[dominant], 200)
                self.assertLess(max(pixel[(dominant + 1) % 3], pixel[(dominant + 2) % 3]), 30)
            # Missing input frames must be rejected before either model renders.
            (image_dir / 'frame2.png').unlink()
            argv[argv.index('--output-dir') + 1] = str(root / 'missing_test')
            with patch.object(mod, 'read_cameras', return_value=cams), patch.object(mod, 'rendered_frames', side_effect=AssertionError('must not render')):
                with self.assertRaises(SystemExit):
                    mod.main(argv)
            self.assertFalse((root / 'missing_test').exists())

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

