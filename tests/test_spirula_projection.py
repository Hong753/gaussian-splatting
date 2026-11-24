import torch

from utils.graphics_utils import focal2fov, getProjectionMatrix


def test_spirula_projection_matches_colmap_after_resize():
    width, height = 2048, 1536
    fx, fy, cx, cy = 1142., 1270., 1036., 739.
    fov = focal2fov(fx, width), focal2fov(fy, height)
    points = torch.tensor([[.1, -.2, 1., 1.], [-.8, .4, 3., 1.]])
    projection = getProjectionMatrix(.01, 100., *fov, (cx/width, cy/height))
    clip = points @ projection.T
    output_wh = torch.tensor([1600, 1200])
    pixels = (clip[:, :2] / clip[:, 3:] + 1) * output_wh / 2
    reference = points[:, :2] / points[:, 2:3] * torch.tensor([fx, fy]) + torch.tensor([cx, cy])
    torch.testing.assert_close(pixels, reference * output_wh / torch.tensor([width, height]))
    torch.testing.assert_close(getProjectionMatrix(.01, 100., *fov),
                               getProjectionMatrix(.01, 100., *fov, (.5, .5)))
