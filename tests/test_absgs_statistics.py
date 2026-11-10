import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np


def method(name):
    source = (Path(__file__).resolve().parents[1] / "scene/gaussian_model.py").read_text()
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": SimpleNamespace(norm=lambda x, dim, keepdim: np.linalg.norm(x, axis=dim, keepdims=keepdim),
                                          cuda=SimpleNamespace(empty_cache=lambda: None))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<GaussianModel>", "exec"), namespace)
    return namespace[name]


class AbsGSStatisticsTests(unittest.TestCase):
    def test_split_uses_absolute_but_clone_uses_signed_signal(self):
        class Array(np.ndarray):
            def isnan(self):
                return np.isnan(self)
        densify = method("densify_and_prune")
        calls = []
        model = SimpleNamespace(
            surface_mode='off',
            use_absgs=True,
            xyz_gradient_accum=np.array([[0.0003, 0.0009], [0.0001, 0.0008]]).view(Array),
            denom=np.ones((2, 1)),
            get_opacity=np.array([[0.5], [0.5]]),
            densify_and_clone=lambda g, t, e: calls.append(("clone", g.copy(), t)),
            densify_and_split=lambda g, t, e: calls.append(("split", g.copy(), t)),
            prune_points=lambda mask: calls.append(("prune", mask.copy())),
        )
        densify(model, 0.0002, 0.005, 1.0, None, np.ones(2), max_grad_abs=0.0004)
        self.assertEqual(calls[0][0], "clone")
        np.testing.assert_allclose(calls[0][1][:, 0], [0.0003, 0.0001])
        self.assertEqual(calls[0][2], 0.0002)
        self.assertEqual(calls[1][0], "split")
        np.testing.assert_allclose(calls[1][1][:, 0], [0.0009, 0.0008])
        self.assertEqual(calls[1][2], 0.0004)

    def test_signed_and_per_pixel_absolute_gradients_are_accumulated_separately(self):
        add = method("add_densification_stats")
        model = SimpleNamespace(use_absgs=True, xyz_gradient_accum=np.zeros((3, 2)), denom=np.zeros((3, 1)))
        grad = SimpleNamespace(grad=np.array([[3, 4, 6, 8], [2, 2, 2, 2], [5, 12, 8, 15]], dtype=float))
        selected = np.array([[0], [2]])
        add(model, grad, selected)
        np.testing.assert_allclose(model.xyz_gradient_accum, [[5, 10], [0, 0], [13, 17]])
        np.testing.assert_allclose(model.denom, [[1], [0], [1]])
        with self.assertRaisesRegex(RuntimeError, "four-component"):
            add(model, SimpleNamespace(grad=np.zeros((3, 3))), selected)

    def test_regular_3dgs_keeps_single_signed_accumulator(self):
        add = method("add_densification_stats")
        model = SimpleNamespace(use_absgs=False, xyz_gradient_accum=np.zeros((2, 1)), denom=np.zeros((2, 1)))
        grad = SimpleNamespace(grad=np.array([[3., 4., 0.], [1., 0., 0.]]))
        add(model, grad, np.array([[0], [1]]))
        np.testing.assert_allclose(model.xyz_gradient_accum[:, 0], [5, 1])
        np.testing.assert_allclose(model.denom[:, 0], [1, 1])


if __name__ == "__main__":
    unittest.main()
