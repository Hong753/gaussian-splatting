import importlib.util
import json
from pathlib import Path
import sys
import socket
import tempfile
import time
import types
import unittest
import urllib.request
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("view_scaffold", ROOT / "scripts/view_scaffold.py")
viewer_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer_mod)


class ScaffoldViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.folder = self.run / "point_cloud/iteration_7000"
        self.folder.mkdir(parents=True)
        self.ply = self.folder / "point_cloud.ply"
        self.ply.touch()
        for name in viewer_mod.MLP_FILES:
            (self.folder / name).touch()
        (self.run / "cfg_args").write_text("Namespace(model_type='scaffold', scaffold_n_offsets=10)")
        (self.run / "cameras.json").write_text(json.dumps([{"position": [2, 3, 4],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "height": 1000, "fy": 500}]))

    def test_starts_in_home_without_data_directory_or_browse_argument(self):
        captured = []
        class Server:
            def __init__(self, **kwargs): captured.append(kwargs)
            def get_port(self): return 8080
            def stop(self): captured.append("stopped")
        with patch.object(Path, "home", return_value=self.root), \
             patch.object(sys, "argv", ["view_scaffold.py", "--gpu", "0", "--port", "8080"]), \
             patch.dict(sys.modules, {"viser": types.SimpleNamespace(ViserServer=Server)}), \
             patch.object(viewer_mod, "Viewer") as viewer, \
             patch.object(viewer_mod.time, "sleep", side_effect=KeyboardInterrupt):
            viewer_mod.main()
        self.assertEqual(viewer.call_args.args[1], self.root)
        self.assertEqual(captured[-1], "stopped")

    def test_model_selection_requires_matching_mlp_and_config(self):
        ply, root, cfg = viewer_mod.model_paths(self.ply)
        self.assertEqual(ply, self.ply)
        self.assertEqual(root, self.run)
        self.assertEqual(cfg["scaffold_n_offsets"], 10)
        (self.folder / "color_mlp.pt").unlink()
        with self.assertRaisesRegex(ValueError, "color_mlp.pt"):
            viewer_mod.model_paths(self.ply)
        (self.folder / "color_mlp.pt").touch()
        (self.run / "cfg_args").write_text("Namespace(model_type='3dgs')")
        with self.assertRaisesRegex(ValueError, "not marked"):
            viewer_mod.model_paths(self.ply)

    @unittest.skipUnless(importlib.util.find_spec("viser"), "optional viser dependency not installed")
    def test_camera_coordinates_follow_opencv(self):
        pos, rot, fovy, distance = viewer_mod.first_camera(self.run / "cameras.json")
        np.testing.assert_allclose(pos, [2, 3, 4])
        np.testing.assert_allclose(rot[:, 2], [0, 0, 1])
        self.assertAlmostEqual(fovy, np.pi / 2)
        w2c, fovx = viewer_mod.camera_matrices([1, 0, 0, 0], pos, fovy, 1600, 1000)
        np.testing.assert_allclose(w2c[:3, 3], [-2, -3, -4])
        self.assertAlmostEqual(fovx, 2 * np.arctan(1.6))
        # World point one unit in front of camera maps to +Z in camera space.
        np.testing.assert_allclose(w2c @ np.array([2, 3, 5, 1]), [0, 0, 1, 1])
        q = [2 ** -0.5, 0, 0, 2 ** -0.5]
        rotated, _ = viewer_mod.camera_matrices(q, [2, 3, 4], fovy, 1600, 1000)
        np.testing.assert_allclose(rotated @ np.array([2, 4, 4, 1]), [1, 0, 0, 1], atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("viser"), "optional viser dependency not installed")
    def test_actual_viser_gui_load_and_render_with_fake_backend(self):
        import viser
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
        self.addCleanup(server.stop)
        renders = []
        class Backend:
            def __init__(self, ply, cfg): self.ply = ply
            def close(self): pass
            def render_image(self, wxyz, pos, fov, width, height, aa):
                renders.append((width, height, aa))
                return np.zeros((height, width, 3), dtype=np.uint8)
        app = viewer_mod.Viewer(server, self.folder, backend_factory=Backend)
        self.assertIn("point_cloud.ply", app.entries.options)
        app.directory.value = str(self.root)
        app.refresh()
        self.assertIn("run/", app.entries.options)
        app.directory.value = str(self.folder)
        app.refresh()
        self.assertIn("point_cloud.ply", app.entries.options)
        app.load(self.ply)
        self.assertIn("Loaded", app.status.content)
        with urllib.request.urlopen(f"http://127.0.0.1:{server.get_port()}/", timeout=5) as response:
            self.assertEqual(response.status, 200)
        frames = []
        camera = types.SimpleNamespace(aspect=2.0, wxyz=np.array([1, 0, 0, 0]),
                                       position=np.array([2, 3, 4]), fov=np.pi/2)
        client = types.SimpleNamespace(client_id=3, camera=camera,
            scene=types.SimpleNamespace(set_background_image=lambda image, **kw: frames.append(image)))
        app.schedule(client)
        deadline = time.monotonic() + 3
        while not frames and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(renders, [(960, 480, True)])
        self.assertEqual(frames[0].shape, (480, 960, 3))


if __name__ == "__main__":
    unittest.main()

