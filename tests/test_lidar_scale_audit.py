import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_lidar_scale import cap_outliers, compare_area, gallery, initial_sigma, select_frames


class FixedLidarScaleAuditTest(unittest.TestCase):
    def test_fixed_centers_use_entire_original_cloud_including_pruned_neighbors(self):
        xyz = np.zeros((6, 3), np.float32)
        xyz[:, 0] = [0, 0, 1, 2, 5, 11]
        np.testing.assert_allclose(initial_sigma(xyz, xyz[[4, 0]]), np.sqrt([50 / 3, 5 / 3]))
        with self.assertRaisesRegex(ValueError, "absent"):
            initial_sigma(xyz, xyz[[1]] + .001)

    def test_tail_is_ranked_by_growth_and_only_oversized_axes_change(self):
        scales = np.array([[1., 2, .1], [4., .2, .1], [.2, 3., .5], [.5, 1., 1.]])
        spacing = np.array([1., 2., .25, .1])
        saved = np.log(scales)
        ids, capped, stats = cap_outliers(saved, spacing, ratio=4, percent=25)
        np.testing.assert_array_equal(ids, [2])
        np.testing.assert_allclose(np.exp(capped[0]), [.2, 1, .5])
        np.testing.assert_array_equal(saved, np.log(scales))
        self.assertEqual(stats["capped"], 1)
        self.assertEqual(len(cap_outliers(saved, spacing, 20, 100)[0]), 0)

    def test_held_out_reference_includes_nearby_training_camera(self):
        spec = dict(records=dict.fromkeys(["01", "02", "03", "04", "05"]), splits=dict(train={"01", "03", "05"}))
        self.assertEqual(select_frames(spec, "04", 3), ["04", "03", "01", "05"])
        self.assertEqual(select_frames(spec, "03", 0), ["03"])

    def test_coverage_and_gt_agreement_in_report(self):
        gt = np.zeros((12, 20, 3), np.float32)
        gt[:, 10:] = 1
        blurred = gt.copy()
        blurred[:, 9:11] = .5
        alpha = np.ones(gt.shape[:2], np.float32)
        holes = alpha.copy()
        holes[:, :2] = .1
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            row = compare_area(output, "cmp", gt, blurred, gt, alpha, holes)
            self.assertEqual(row['capped']['psnr_db'], 'inf')
            self.assertGreater(row['capped']['edge_strength'], row['baseline']['edge_strength'])
            self.assertAlmostEqual(row['new_hole_percent'], 10)
            self.assertAlmostEqual(row['mean_alpha_loss'], .09)
            self.assertTrue(all((output / f).is_file() for f in row['images'].values()))
            gallery(output, dict(iteration=3, training=dict(camera_extent=15, world_prune_sigma=1.5),
                                 scales=dict(capped=4, total=400, max_percent=1, cap_ratio=4),
                                 views=[dict(name="12569.jpg", split="held-out", areas={"Selected crop": row})]))
            self.assertIn("12569.jpg", (output / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
