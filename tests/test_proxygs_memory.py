"""Numerical and allocation regressions for large-scene anchor growth."""
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from scripts.proxygs_memory import (occupied_voxel_mask, pooled_candidate_features,
                                    selected_offset_positions)


@unittest.skipIf(torch is None, "PyTorch required")
class ProxyGSMemoryTests(unittest.TestCase):
    def test_feature_pooling_matches_repeat_mask_and_group_max(self):
        for seed in range(5):
            torch.manual_seed(seed)
            features = torch.randn(31, 9) - 2  # Include all-negative feature groups.
            mask = torch.rand(31 * 10) > .35
            old = features[:, None, :].repeat(1, 10, 1).reshape(-1, 9)[mask]
            assignments = torch.randint(0, 23, (old.shape[0],))
            _, inverse = torch.unique(assignments, return_inverse=True)
            keep = torch.rand(int(inverse.max()) + 1) > .3
            expected = torch.stack([old[inverse == i].max(0).values
                                    for i in range(keep.numel()) if keep[i]])
            for chunk_size in (1, 7, 65536):
                with torch.no_grad():
                    actual = pooled_candidate_features(features, 10, mask, inverse, keep, chunk_size)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pooling_does_not_repeat_features_and_bounds_each_gather(self):
        features = torch.randn(123, 32)
        mask = torch.ones(1230, dtype=torch.bool)
        inverse = torch.arange(1230).remainder(17)
        keep = torch.ones(17, dtype=torch.bool)
        original = torch.Tensor.index_select
        sizes = []
        def gather(tensor, dimension, indices):
            sizes.append(indices.numel())
            return original(tensor, dimension, indices)
        with patch.object(torch.Tensor, "repeat", side_effect=AssertionError("Feature replication is forbidden")), \
                patch.object(torch.Tensor, "index_select", gather), torch.no_grad():
            result = pooled_candidate_features(features, 10, mask, inverse, keep, chunk_size=19)
        self.assertEqual(result.shape, (17, 32))
        self.assertGreater(len(sizes), 1)
        self.assertLessEqual(max(sizes), 19)

    def test_empty_candidates_or_retained_groups(self):
        features = torch.randn(3, 4)
        for mask, inverse, keep in (
            (torch.zeros(30, dtype=torch.bool), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)),
            (torch.ones(30, dtype=torch.bool), torch.zeros(30, dtype=torch.long), torch.tensor([False])),
        ):
            self.assertEqual(pooled_candidate_features(features, 10, mask, inverse, keep).shape, (0, 4))

    def test_selected_positions_match_full_offset_expansion(self):
        torch.manual_seed(5)
        anchors, offsets, scales = torch.randn(11, 3), torch.randn(11, 10, 3), torch.rand(11, 6)
        for mask in (torch.rand(110) > .8, torch.ones(110, dtype=torch.bool), torch.zeros(110, dtype=torch.bool)):
            expected = (anchors[:, None, :] + offsets * scales[:, None, :3]).reshape(-1, 3)[mask]
            actual = selected_offset_positions(anchors, offsets, scales, mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_voxel_membership_matches_pairwise_reference(self):
        torch.manual_seed(42)
        for n, m in ((0, 5), (4, 0), (29, 37), (200, 400)):
            old = torch.randint(-3, 4, (n, 3), dtype=torch.int32)
            new = torch.randint(-3, 4, (m, 3), dtype=torch.int32)
            expected = (new[:, None, :] == old[None, :, :]).all(-1).any(-1)
            self.assertTrue(torch.equal(occupied_voxel_mask(old, new), expected))


if __name__ == "__main__":
    unittest.main()
