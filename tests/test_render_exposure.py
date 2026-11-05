import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class RenderExposureTests(unittest.TestCase):
    def test_exposure_without_eval_renders_full_images(self):
        source = (Path(__file__).resolve().parents[1] / "render.py").read_text()
        tree = ast.parse(source)
        render_set = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == "render_set")
        namespace = {
            "os": __import__("os"),
            "makedirs": lambda *args, **kwargs: None,
            "tqdm": lambda views, **kwargs: views,
            "render": lambda *args, **kwargs: {"render": np.zeros((3, 4, 6))},
        }
        saved = []
        namespace["torchvision"] = SimpleNamespace(
            utils=SimpleNamespace(save_image=lambda image, path: saved.append(image.shape))
        )
        exec(compile(ast.Module(body=[render_set], type_ignores=[]), "render.py", "exec"), namespace)
        view = SimpleNamespace(original_image=np.zeros((3, 4, 6)))
        namespace["args"] = SimpleNamespace(train_test_exp=True, eval=False)
        namespace["render_set"]("output", "train", 1, [view], None, None, None, True, False)
        self.assertEqual(saved, [(3, 4, 6), (3, 4, 6)])

        saved.clear()
        namespace["args"].eval = True
        namespace["render_set"]("output", "test", 1, [view], None, None, None, True, False)
        self.assertEqual(saved, [(3, 4, 3), (3, 4, 3)])


if __name__ == "__main__":
    unittest.main()
