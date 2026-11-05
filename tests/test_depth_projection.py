import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

from scripts.preprocess_depth import sample_projected_points


class FakeTensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    @property
    def device(self):
        return "cpu"

    def long(self):
        return self.astype(np.int64).view(FakeTensor)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def scatter_reduce_(self, dim, index, source, reduce, include_self):
        assert dim == 0 and reduce == "amin" and include_self
        np.minimum.at(self, np.asarray(index), np.asarray(source))
        return self


class DepthProjectionTests(unittest.TestCase):
    def test_projection_uses_nearest_positive_z_and_matching_image_pixel(self):
        torch = types.ModuleType("torch")
        torch.float32 = np.float32
        torch.tensor = lambda value, **kwargs: FakeTensor(value)
        torch.isfinite = np.isfinite
        torch.full = lambda shape, value, **kwargs: FakeTensor(np.full(shape, value))
        torch.linspace = lambda start, end, count, **kwargs: FakeTensor(np.linspace(start, end, count))
        points = FakeTensor([[0, 0, 2], [0, 0, 4], [1, 0, 2], [0, 0, -2], [100, 0, 2]])
        camera = types.SimpleNamespace(qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3))
        intrinsic = types.SimpleNamespace(model="PINHOLE", params=[2, 2, 2, 2], width=4, height=4)
        image = np.zeros((4, 4), dtype=np.uint16)
        image[2, 2], image[2, 3] = 32768, 16384
        with patch.dict(sys.modules, {"torch": torch}):
            prediction, reference = sample_projected_points(points, camera, intrinsic, image.shape, image, 1, 20)
        np.testing.assert_allclose(prediction, [0.5, 0.25])
        np.testing.assert_allclose(reference, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
