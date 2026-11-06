import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.warmup_schedule import position_lr_step, densification_phase
from utils.sparse_lidar import frame_filename, check_dataset, load_frame, lidar_loss

spec = importlib.util.spec_from_file_location('preprocess_lidar_depth', ROOT/'scripts/preprocess_lidar_depth.py')
projector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projector)


class WarmupLidarTests(unittest.TestCase):
    def test_schedule_and_densification_timing(self):
        self.assertEqual(position_lr_step(1, 0, 0), (1., 1))
        self.assertEqual(position_lr_step(7000, 7000, 3000), (0., 0))
        self.assertEqual(position_lr_step(8500, 7000, 3000), (.5, 0))
        self.assertEqual(position_lr_step(10000, 7000, 3000), (1., 0))
        self.assertEqual(position_lr_step(10001, 7000, 3000), (1., 1))
        phase = [i for i in range(1, 60001) if 0 < densification_phase(i,7000) < 15000]
        split = [i for i in phase if densification_phase(i,7000) > 500 and densification_phase(i,7000)%100 == 0]
        reset = [i for i in phase if densification_phase(i,7000)%3000 == 0]
        self.assertEqual((split[0], split[-1], len(split)), (7600, 21900, 144))
        self.assertEqual(reset, [10000, 13000, 16000, 19000])

    def test_projection_frontmost_camera_space_and_resolution(self):
        points = torch.tensor([[0.,0.,2.], [0.,0.,4.], [1.,0.,2.], [0.,0.,-1.]])
        u, v, z = projector.project(points, torch.eye(3), torch.zeros(3), 4, 4, 9, 9, 1)
        self.assertEqual(sorted(zip(u.tolist(),v.tolist(),z.tolist())), [(4,4,2.), (6,4,2.)])
        self.assertEqual(projector.image_resolution(2000, 1000, -1), (1600, 800))
        self.assertEqual(projector.image_resolution(101, 51, 2), (50, 26))

    def test_sparse_roundtrip_and_loss(self):
        c = types.SimpleNamespace(image_name='nested/image.png', image_height=4, image_width=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scene = root/'source'; scene.mkdir(); data = root/'depth'; data.mkdir()
            (data/'metadata.json').write_text(json.dumps({'source':str(scene), 'complete':True, 'images':[c.image_name]}))
            np.savez_compressed(data/frame_filename(c.image_name), image=c.image_name, shape=[4,5], u=[2,1], v=[1,2], z=[2.,4.])
            check_dataset(data, scene, [c]); u, v, z = load_frame(data,c)
            inv = torch.zeros(1,4,5, requires_grad=True)
            with torch.no_grad():
                inv[0,1,2] = .5
                inv[0,2,1] = .25
            self.assertAlmostEqual(lidar_loss(inv,u,v,z).item(), 0.)
            with torch.no_grad(): inv[0,2,1] = .1
            loss = lidar_loss(inv,u,v,z); loss.backward()
            self.assertGreater(loss.item(), 0)
            self.assertLess(inv.grad[0,2,1].item(), 0)
            self.assertEqual(inv.grad[0,1,2].item(), 0)
            with self.assertRaises(ValueError):
                check_dataset(data, scene/'wrong', [c])
            c.image_width = 6
            with self.assertRaises(ValueError): load_frame(data,c)


if __name__ == '__main__': unittest.main()
