import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class NetworkViewerTests(unittest.TestCase):
    def test_virtual_camera_does_not_require_training_image_exposure(self):
        source = (Path(__file__).resolve().parents[1] / "train.py").read_text()
        tree = ast.parse(source)
        training = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "training")
        render_calls = [node for node in ast.walk(training)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "render" and node.args
                        and isinstance(node.args[0], ast.Name)]
        calls_by_camera = {call.args[0].id: call for call in render_calls}
        self.assertIn("custom_cam", calls_by_camera)
        self.assertIn("viewpoint_cam", calls_by_camera)
        received = []

        def fake_render(camera, *args, **kwargs):
            if kwargs["use_trained_exp"]:
                # This mirrors the real renderer's per-image exposure lookup.
                _ = camera.image_name
            received.append(kwargs["use_trained_exp"])
            return {"render": "image"}

        namespace = {"render": fake_render, "gaussians": object(), "pipe": object(),
                     "background": object(), "scaling_modifer": 1.0,
                     "SPARSE_ADAM_AVAILABLE": False,
                     "dataset": SimpleNamespace(train_test_exp=True),
                     "custom_cam": SimpleNamespace(),
                     "viewpoint_cam": SimpleNamespace(image_name="frame.png"),
                     "bg": object()}
        for camera in ("custom_cam", "viewpoint_cam"):
            expr = ast.Expression(body=calls_by_camera[camera])
            eval(compile(expr, "train.py", "eval"), namespace)
        self.assertEqual(received, [False, True])


if __name__ == "__main__":
    unittest.main()
