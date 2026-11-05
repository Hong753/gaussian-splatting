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
spec = importlib.util.spec_from_file_location('render_novel_comparison', ROOT / 'scripts/render_novel_comparison.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def cam(i, center, rotation=None):
    r = np.eye(3) if rotation is None else rotation
    center = np.array(center, dtype=float)
    return types.SimpleNamespace(image_name=f'frame{i:04d}.png', R=r, T=-r.T @ center,
                                 width=100, height=50, FovX=1.0, FovY=0.55)


class TestNovelComparison(unittest.TestCase):
    def test_rotation_roundtrip_and_camera_convention(self):
        for angle in [0, 0.5, 2, np.pi]:
            r = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            np.testing.assert_allclose(mod.quat_to_rot(mod.rot_to_quat(r)), r, atol=1e-7)
        c = [cam(i, [i, 0, 0]) for i in range(30)]
        result, info = mod.synthesize_path(c, 60, control_step=5, lateral_fraction=0.1)
        self.assertEqual(len(result), 60)
        self.assertGreater(info['lateral_amplitude'], 0)
        for item in result:
            np.testing.assert_allclose(item.R.T @ item.R, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(np.linalg.det(item.R), 1, places=6)
            center = -item.R @ item.T
            self.assertGreaterEqual(center[0], -1)
            self.assertLessEqual(center[0], 30)
        centers = np.array([-item.R @ item.T for item in result])
        self.assertTrue(any(np.min(np.abs(np.arange(30) - c[0])) > 1e-4 for c in centers))

    def test_skips_discontinuous_capture_jump(self):
        cameras = [cam(i, [i, 0, 0]) for i in range(20)] + [cam(i+20, [i+1000, 0, 0]) for i in range(20)]
        segment, info = mod.choose_source_segment(cameras, window=12)
        self.assertEqual(len(segment), 12)
        self.assertEqual(info['continuous_segment_length'], 20)
        self.assertTrue(all(int(x.image_name[5:9]) < 20 for x in segment))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg needed')
    def test_two_panel_encoding_uses_shared_synthetic_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'scene'
            source.mkdir()
            runs = []
            for kind in ('3dgs', 'scaffold'):
                run = root / kind
                folder = run / 'point_cloud/iteration_30000'
                folder.mkdir(parents=True)
                (folder / 'point_cloud.ply').touch()
                (run / 'cfg_args').write_text(f"Namespace(model_type='{kind}', source_path={str(source)!r}, white_background=False)")
                if kind == 'scaffold':
                    for name in ('opacity_mlp.pt', 'cov_mlp.pt', 'color_mlp.pt'):
                        (folder / name).touch()
                runs.append(run)
            cameras = [cam(i, [i*0.1, 0, 0]) for i in range(24)]
            output = root / 'out'
            seen = []
            def fake_frames(spec, seq, width, height, aa):
                seen.append((spec['type'], [(-c.R @ c.T).tolist() for c in seq], aa))
                color = [255, 0, 0] if spec['type'] == '3dgs' else [0, 0, 255]
                for _ in seq:
                    yield np.broadcast_to(np.array(color, dtype=np.uint8), (height, width, 3)).copy()
            args = ['--source', str(source), '--model-3dgs', str(runs[0]), '--model-scaffold', str(runs[1]),
                    '--output-dir', str(output), '--3dgs-aa', 'on', '--scaffold-aa', 'off',
                    '--width', '64', '--fps', '5', '--duration', '1', '--source-window', '20']
            with patch.object(mod.core, 'read_cameras', return_value=cameras), patch.object(mod.core, 'rendered_frames', side_effect=fake_frames):
                mod.main(args)
            self.assertEqual(seen[0][1], seen[1][1])
            self.assertEqual((seen[0][2], seen[1][2]), (True, False))
            data = json.loads((output / 'trajectory.json').read_text())
            self.assertEqual(data['frame_count'], 5)
            self.assertEqual(data['side_by_side'], {'left':'3dgs', 'right':'scaffold'})
            self.assertEqual(len(data['cameras']), 5)
            self.assertFalse((output / 'original.mp4').exists())
            info = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=width,height,nb_read_frames','-of','json',str(output/'side_by_side.mp4')], text=True)
            stream = json.loads(info)['streams'][0]
            self.assertEqual((stream['width'],stream['height'],int(stream['nb_read_frames'])), (128,32,5))

if __name__ == '__main__': unittest.main()
