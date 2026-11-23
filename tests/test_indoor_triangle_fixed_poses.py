from types import SimpleNamespace
import unittest

import numpy as np

from scripts.indoor_triangles.room import colmap_loader, fixed_pose


class IndoorTriangleFixedPoseTests(unittest.TestCase):
    def test_colmap_roundoff_is_removed_before_constructing_rotation(self):
        colmap = colmap_loader()
        angle = .53
        expected_qvec = np.array([np.cos(angle / 2), 0., np.sin(angle / 2), 0.])
        source_qvec = expected_qvec * (1 + 2.5e-7)
        frame = SimpleNamespace(name="12561.jpg", qvec=source_qvec.copy())
        expected_rotation = colmap.qvec2rotmat(expected_qvec)
        self.assertGreater(np.max(abs(colmap.qvec2rotmat(source_qvec) - expected_rotation)), 1e-8)

        qvec, rotation = fixed_pose(colmap, frame)

        np.testing.assert_allclose(qvec, expected_qvec, atol=1e-15, rtol=0)
        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(frame.qvec, source_qvec)

    def test_invalid_colmap_quaternion_identifies_the_frame(self):
        frame = SimpleNamespace(name="12561.jpg", qvec=np.zeros(4))
        with self.assertRaisesRegex(ValueError, r"12561\.jpg"):
            fixed_pose(colmap_loader(), frame)


if __name__ == "__main__":
    unittest.main()
