import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from scripts.keyframe_pipeline.common import Camera, Image, project
from scripts import render_pose_video as video


class PoseVideoTest(unittest.TestCase):
    def test_calibrated_frustum_reprojects_to_image_edges(self):
        cam = Camera(1, 120, 70, np.array([100, 80, 35, 41]))
        view = Image(1, 1, "frame", Rotation.from_euler("xyz", [20, 15, 70], degrees=True).as_matrix(),
                     np.array([3., -2., 1.]))
        points = video.frustum_vertices(view, cam, 2.5)
        np.testing.assert_allclose(points[0], view.C)
        uv, depth = project(points[1:], view.R, view.C, cam.params)
        np.testing.assert_allclose(uv, [[0,0], [120,0], [120,70], [0,70]], atol=1e-10)
        np.testing.assert_allclose(depth, 2.5)

    def test_full_cloud_overview_depth_buffer(self):
        cam = Camera(1, 100, 100, np.array([100, 100, 50, 50]))
        view = Image(1, 1, "frame", np.eye(3), np.zeros(3))
        points = np.array([(1., 2., 1.), (5., 2., 1.), (0., -3., 2.), (np.nan, 0., 0.)],
                          dtype=[(a, "f4") for a in "xyz"])
        overview = video.Overview(points, [view], {1: cam}, 500, 400, 0, 0, 3)
        occupied = overview.world[np.isfinite(overview.world[:,:,0])]
        np.testing.assert_equal(sorted(map(tuple, occupied)), [(0., -3., 2.), (5., 2., 1.)])

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required to exercise the MP4 encoder")
    def test_mp4_pairs_each_photo_with_its_pose_and_preserves_previous_output_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse = root/"sparse/0"
            sparse.mkdir(parents=True)
            (root/"images").mkdir()
            (sparse/"cameras.txt").write_text("1 PINHOLE 80 60 80 70 35 25\n2 PINHOLE 60 80 65 70 35 42\n")
            # Names are deliberately out of capture order in the COLMAP file.
            (sparse/"images.txt").write_text("1 1 0 0 0 -4 0 0 2 10.png\n\n"
                                             "2 1 0 0 0 0 0 0 1 2.png\n\n"
                                             "3 1 0 0 0 -2 0 0 1 4.png\n\n")
            photos = [("2", (60,80), (0,0,253)), ("4", (60,80), (0,251,0)), ("10", (80,60), (252,0,0))]
            for name, dims, bgr in photos:
                cv2.imwrite(str(root/"images"/(name+".png")), np.full((*dims,3), bgr, np.uint8))
            rng = np.random.default_rng(3)
            points = np.empty(4000, dtype=[(k, "f4") for k in "xyz"])
            for k in "xyz":
                points[k] = rng.uniform(-1, 5, len(points))
            cloud = root/"cloud.ply"
            PlyData([PlyElement.describe(points, "vertex")]).write(cloud)
            suite = root/"suite.json"
            suite.write_text(json.dumps(dict(source=str(root), cloud=dict(cached=str(cloud)))))
            out = root/"pair.mp4"
            args = ["--suite", str(suite), "--output", str(out), "--width", "640", "--fps", "4",
                    "--start-frame", "2.png", "--stride", "2", "--max-frames", "2"]
            video.main(args)
            reader = cv2.VideoCapture(str(out))
            try:
                self.assertEqual(reader.get(cv2.CAP_PROP_FPS), 4.)
                self.assertEqual(reader.get(cv2.CAP_PROP_FRAME_COUNT), 2)
                panels = []
                for expected_channel in (2, 0):
                    ok, frame = reader.read()
                    self.assertTrue(ok)
                    photo = frame[100:130, 460:510]
                    self.assertGreater(photo[:,:,expected_channel].mean(), 240)
                    panels.append(frame[20:-30,:320].copy())
                self.assertGreater(cv2.norm(panels[0], panels[1], cv2.NORM_L1), 10000)
                self.assertFalse(reader.read()[0])
            finally:
                reader.release()
            good = out.read_bytes()
            (root/"images/10.png").unlink()
            with self.assertRaisesRegex(ValueError, "10.png"):
                video.main(args)
            self.assertEqual(out.read_bytes(), good)
