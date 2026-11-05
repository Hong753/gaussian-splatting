import unittest
import numpy as np
from pathlib import Path
import tempfile

from utils.depth_alignment import apply_alignment, fit_alignment
from utils.colmap_paths import depth_params_path


class DepthAlignmentTests(unittest.TestCase):
    def test_depth_specific_parameters_take_precedence_with_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "scene/sparse/0"
            model.mkdir(parents=True)
            legacy = model / "depth_params.json"
            legacy.touch()
            maps = root / "depths"
            maps.mkdir()
            self.assertEqual(depth_params_path(root / "scene", model, str(maps)), legacy)
            own = maps / "depth_params.json"
            own.touch()
            self.assertEqual(depth_params_path(root / "scene", model, str(maps)), own)

    def test_bad_alignment_and_invalid_pixels_do_not_become_depth_targets(self):
        image = np.array([[0.1, 0.4, 0.7]], dtype=np.float32)
        for params in (None, {"scale": 1.0, "offset": 0, "reliable": False},
                       {"scale": 0, "offset": 0, "reliable": True}):
            aligned, mask, reliable = apply_alignment(image, params)
            self.assertFalse(reliable)
            self.assertFalse(mask.any())
        aligned, mask, reliable = apply_alignment(image, {"scale": 2, "offset": -0.5, "reliable": True})
        self.assertTrue(reliable)
        np.testing.assert_array_equal(mask, [[False, True, True]])
        np.testing.assert_allclose(aligned, [[0, 0.3, 0.9]], atol=1e-6)

    def test_recovers_inverse_depth_transform_with_outliers(self):
        rng = np.random.default_rng(12)
        prediction = rng.uniform(0, 1, 2000)
        target = 1.4 * prediction + 0.2 + rng.normal(0, 0.005, len(prediction))
        target[:100] = rng.uniform(2, 5, 100)
        fit = fit_alignment(prediction, target)
        self.assertTrue(fit["reliable"], fit)
        self.assertAlmostEqual(fit["scale"], 1.4, delta=0.03)
        self.assertAlmostEqual(fit["offset"], 0.2, delta=0.02)
        self.assertLess(fit["validation_median_relative_error"], 0.03)

    def test_rejects_inadequate_or_inverted_alignment(self):
        self.assertEqual(fit_alignment(np.arange(3), np.arange(3) + 1)["reason"], "too_few_points")
        self.assertEqual(fit_alignment(np.ones(100), np.arange(100) + 1)["reason"], "insufficient_depth_variation")
        x = np.linspace(0, 1, 1000)
        fit = fit_alignment(x, 2 - x)
        self.assertFalse(fit["reliable"])
        self.assertEqual(fit["reason"], "nonpositive_or_nonfinite_scale")

    def test_checks_holdout_fit_and_determinism(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(size=1000)
        y = rng.uniform(0.1, 1.0, 1000)
        first = fit_alignment(x, y)
        self.assertEqual(first, fit_alignment(x, y))
        self.assertFalse(first["reliable"])


if __name__ == "__main__":
    unittest.main()
