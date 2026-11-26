"""Numerical checks for the CPU exposure solver; no scene data or CUDA needed."""
import json
import unittest

import numpy as np

from utils.exposure_balance import (
    apply_exposure, evaluate_holdout, fit_exposures, linear_to_srgb, srgb_to_linear,
)


class ExposureBalanceTests(unittest.TestCase):
    def test_known_exposure_with_outliers_and_track_holdout(self):
        rng = np.random.default_rng(9)
        truth = np.array([-0.8, 0.1, 0.55, -0.3, 0.45])
        truth -= truth.mean()
        tracks = np.repeat(np.arange(500, dtype=np.int64) * 101 + 7, len(truth))
        frames = np.tile(np.arange(len(truth)), 500)
        radiance = np.repeat(rng.uniform(-6, -2, 500), len(truth))
        values = radiance + truth[frames] + rng.normal(0, 0.015, len(frames))
        outliers = rng.random(len(values)) < 0.12
        values[outliers] += rng.normal(0, 1.8, outliers.sum())
        result = fit_exposures(frames, tracks, values, len(truth), seed=6)
        np.testing.assert_allclose(result["exposure_ev"], truth, atol=0.025)
        self.assertTrue(result["supported"].all())
        self.assertAlmostEqual(result["exposure_ev"].mean(), 0.0, places=12)
        self.assertLess(result["summary"]["fit_observations"], len(frames))
        heldout = result["summary"]["heldout"]
        self.assertEqual(heldout["status"], "ok")
        self.assertGreater(heldout["tracks"], 10)
        self.assertLess(heldout["median_abs_deviation_after_ev"],
                        heldout["median_abs_deviation_before_ev"] * 0.1)
        applied = evaluate_holdout(frames, tracks, values,
            np.clip(result["exposure_ev"], -0.1, 0.1), result["supported"],
            result["component"], result["heldout_mask"])
        self.assertGreater(applied["median_abs_deviation_after_ev"],
                           heldout["median_abs_deviation_after_ev"])
        # Summary must be writable with strict JSON, including missing diagnostics.
        json.dumps(result["summary"], allow_nan=False)
        again = fit_exposures(frames[::-1], tracks[::-1], values[::-1], len(truth), seed=6)
        np.testing.assert_allclose(again["exposure_ev"], result["exposure_ev"], atol=1e-9)

    def test_independent_components_and_unsupported_frame(self):
        f = np.tile([0, 1, 2, 3], 100)
        t = np.repeat(np.arange(100), 4)
        # Tracks in the second component use a different identifier and radiance.
        t[f >= 2] += 1000
        capture = np.array([2.0, 3.0, -2.2, -1.8, 8.0])
        z = -5.0 + capture[f]
        result = fit_exposures(f, t, z, 5, min_observations=20, seed=2)
        np.testing.assert_allclose(result["exposure_ev"], [-0.5, 0.5, -0.2, 0.2, 0], atol=1e-10)
        self.assertEqual(result["summary"]["components"], 2)
        self.assertEqual(result["component"][4], -1)
        self.assertFalse(result["supported"][4])
        self.assertTrue(np.isnan(result["residual_median_ev"][4]))
        self.assertNotEqual(result["component"][0], result["component"][2])

    def test_insufficient_tracks_and_holdout_disabled(self):
        result = fit_exposures(np.array([0, 1]), np.array([8, 8]),
                               np.array([-2.0, -3.0]), 3, min_observations=5)
        self.assertFalse(result["supported"].any())
        self.assertEqual(result["summary"]["status"], "insufficient_observations")
        self.assertEqual(result["summary"]["heldout"]["status"], "insufficient")
        np.testing.assert_array_equal(result["exposure_ev"], 0)
        result = fit_exposures(np.array([0, 1]), np.array([8, 8]),
                               np.array([-2.0, -3.0]), 2, min_observations=1,
                               holdout_fraction=0)
        self.assertEqual(result["summary"]["heldout"]["status"], "disabled")
        np.testing.assert_allclose(result["exposure_ev"], [0.5, -0.5], atol=1e-10)

    def test_reject_invalid_observations(self):
        cases = [
            (np.array([0, 0]), np.array([5, 5]), np.array([-1., -2.])),
            (np.array([0, 2]), np.array([5, 6]), np.array([-1., -2.])),
            (np.array([0, 1]), np.array([5, -1]), np.array([-1., -2.])),
            (np.array([0, 1]), np.array([5, 6]), np.array([-1., np.nan])),
            (np.array([0., 1.]), np.array([5, 6]), np.array([-1., -2.])),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                fit_exposures(*args, n_frames=2)

    def test_holdout_never_compares_independent_component_offsets(self):
        report = evaluate_holdout(np.arange(4), np.ones(4, dtype=np.int64),
            np.array([2., 3., -2.2, -1.8]), np.array([-.5, .5, -.2, .2]),
            np.ones(4, dtype=bool), np.array([0, 0, 9, 9]),
            np.ones(4, dtype=bool))
        self.assertEqual(report["track_component_groups"], 2)
        self.assertAlmostEqual(report["median_abs_deviation_before_ev"], .35)
        self.assertAlmostEqual(report["median_abs_deviation_after_ev"], 0)
        self.assertEqual(report["status"], "insufficient")
        with self.assertRaises(ValueError):
            evaluate_holdout(np.arange(2), np.array([3, 3]), np.array([-2., -3.]),
                np.zeros(2), np.ones(2, dtype=bool), np.zeros(2, dtype=int),
                np.array([True, False]))

    def test_color_conversion_and_exposure(self):
        ramp = np.arange(256, dtype=np.uint8)
        rgb = np.stack((ramp, ramp[::-1], np.roll(ramp, 23)), axis=-1)[None]
        np.testing.assert_array_equal(apply_exposure(rgb, 0), rgb)
        normalized = rgb.astype(np.float64) / 255
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(normalized)), normalized, atol=1e-12)
        color = np.array([[[30, 60, 100], [180, 220, 255]]], dtype=np.uint8)
        bright = apply_exposure(color, 1)
        expected = linear_to_srgb(np.clip(srgb_to_linear(color / 255) * 2, 0, 1))
        np.testing.assert_array_equal(bright, np.rint(expected * 255).astype(np.uint8))
        self.assertFalse(np.array_equal(bright[0, 0], color[0, 0] * 2))
        self.assertEqual(bright[0, 1, 2], 255)
        with self.assertRaises(ValueError):
            apply_exposure(color.astype(float), 0)
        with self.assertRaises(ValueError):
            apply_exposure(color, np.nan)


if __name__ == "__main__":
    unittest.main()
