"""Scene-spanning context, shared metric calibration and nonrecursive donor checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from utils.da3_trajectory import trajectory_groups, calibrate_group, validate_from_donors
from scripts.audit_da3_occlusion import summarize


def frame(name, anchors=True, depth=1.):
    h, w = 128, 128
    yy, xx = np.indices((h, w))
    k = np.array([[64., 0, 63.5], [0, 64., 63.5], [0, 0, 1]], np.float32)
    u, v = xx.ravel(), yy.ravel()
    z = np.full(len(u), 2., np.float32)
    if not anchors:
        u, v, z = u[:0], v[:0], z[:0]
    return dict(image=name, raw=np.full((h, w), depth, np.float32),
        confident=np.ones((h, w), bool), confidence=np.ones((h, w), np.float32),
        anchors=(u, v, z), intrinsic=k, extrinsic=np.eye(4, dtype=np.float32))


class TrajectoryTests(unittest.TestCase):
    def test_groups_cover_all_targets_and_mix_global_and_local_context(self):
        names = [f'{i:05d}.jpg' for i in range(1670)]
        groups = list(trajectory_groups(names))
        owned = [n for _, targets in groups for n in targets]
        self.assertEqual(len(owned), len(names))
        self.assertEqual(set(owned), set(names))
        shared = set(groups[0][0])
        for batch, targets in groups:
            self.assertLessEqual(len(batch), 40)
            self.assertEqual(len(batch), len(set(batch)))
            self.assertGreater(int(batch[-1][:5])-int(batch[0][:5]), 1600)
            shared &= set(batch)
            for target in targets:
                index = int(target[:5])
                close = [n for n in batch if n != target and abs(int(n[:5])-index) <= 2]
                self.assertGreaterEqual(len(close), 2)
        self.assertGreaterEqual(len(shared), 8)

    def test_small_groups_and_invalid_budget(self):
        names = ['a', 'b', 'c']
        self.assertEqual(list(trajectory_groups(names)), [(names, names)])
        with self.assertRaises(ValueError):
            list(trajectory_groups(names, group_size=8, shared_views=7, local_neighbors=2))

    def test_anchorless_target_can_be_verified_by_metric_donor_views(self):
        inputs = [frame('target', anchors=False), frame('a'), frame('b'), frame('c')]
        # Different known camera translations; plane stays at camera z=2.
        inputs[1]['extrinsic'][0, 3] = .02
        inputs[2]['extrinsic'][0, 3] = -.02
        items = calibrate_group(inputs)
        self.assertTrue(all(item['report']['scale'] == 2 for item in items))
        self.assertFalse(items[0]['local'].any())
        trusted, support, donors = validate_from_donors(items[0], items[1:])
        self.assertGreater(trusted.mean(), .9)
        self.assertTrue((support[trusted] >= 2).all())
        self.assertGreaterEqual(len(donors), 2)

    def test_one_donor_cannot_validate_and_borrowed_predictions_cannot_bootstrap(self):
        items = calibrate_group([frame('target', False), frame('a'), frame('b'), frame('c')])
        trusted, _, _ = validate_from_donors(items[0], items[1:2])
        self.assertFalse(trusted.any())
        for item in items[1:]:
            item['local'][:] = False
        trusted, _, _ = validate_from_donors(items[0], items[1:])
        self.assertFalse(trusted.any())
        unsupported = calibrate_group([frame('a', False), frame('b', False), frame('c', False)])
        self.assertFalse(unsupported[0]['report']['group_scale']['accepted'])
        self.assertFalse(validate_from_donors(unsupported[0], unsupported[1:])[0].any())

    def test_wrong_depth_or_nonoverlapping_room_cannot_validate_a_target(self):
        items = calibrate_group([frame('target', False, depth=3), frame('a'), frame('b'), frame('c')])
        self.assertFalse(validate_from_donors(items[0], items[1:])[0].any())
        items[0]['depth'][:] = 2
        for item in items[1:]:
            item['extrinsic'][0, 3] = 100
        self.assertFalse(validate_from_donors(items[0], items[1:])[0].any())

    def test_shared_scale_uses_equal_view_proposals_and_rejects_disagreement(self):
        inputs = [frame('a'), frame('b'), frame('c'), frame('outlier', depth=.1)]
        items = calibrate_group(inputs)
        self.assertEqual(items[0]['report']['scale'], 2.)
        self.assertFalse(items[-1]['local'].any())
        disagreement = calibrate_group([frame('a', depth=.1), frame('b', depth=1), frame('c', depth=10)])
        self.assertFalse(disagreement[0]['report']['group_scale']['accepted'])

    def test_audit_distinguishes_local_rejection_and_missing_donor_support(self):
        rows = [dict(image='a', reliable_tile_fraction=0, locally_calibrated_pixel_fraction=0,
                     global_fit={'reason': 'insufficient anchors'}, rejected_regions={'insufficient_local_anchors_or_holdout': 30}),
                dict(image='b', reliable_tile_fraction=.1),
                dict(image='c', reliable_tile_fraction=0, trusted_pixel_fraction=0, group_scale={'accepted': True})]
        report = summarize({'images': ['a', 'b', 'c', 'd']}, rows)
        self.assertEqual(report['empty_frames'], 2)
        self.assertEqual(report['stages']['no_local_lidar_validation'], 1)
        self.assertEqual(report['stages']['no_metric_donor_agreement'], 1)
        self.assertEqual(report['rejected_local_regions']['insufficient_local_anchors_or_holdout'], 30)
        self.assertEqual(report['prepared_frames'], 3)


if __name__ == '__main__':
    unittest.main()
