#!/usr/bin/env python3
"""GPU smoke test for proxy orientation, native AA rendering, and hidden gradients."""
import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires the CUDA-enabled 3dgs environment and a GPU")
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scripts.occlusion_options import configure_pipeline
    from utils.graphics_utils import getProjectionMatrix
    from utils.occlusion_depth import MeshDepthRenderer
    from utils.sh_utils import RGB2SH

    with tempfile.TemporaryDirectory(prefix="3dgs_occlusion_smoke_") as tmp:
        mesh = Path(tmp)/"two_depths.ply"
        mesh.write_text("ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\n"
                        "element face 4\nproperty list uchar int vertex_indices\nend_header\n"
                        "-2 -2 2\n2 -2 2\n2 0 2\n-2 0 2\n-4 0 4\n4 0 4\n4 4 4\n-4 4 4\n"
                        "3 0 1 2\n3 0 2 3\n3 4 5 6\n3 4 6 7\n")
        view = torch.eye(4, device="cuda")
        projection = getProjectionMatrix(.01, 100., math.pi/2, math.pi/2).cuda().T
        cam = SimpleNamespace(image_width=128, image_height=128, FoVx=math.pi/2, FoVy=math.pi/2,
            znear=.01, zfar=100., world_view_transform=view, full_proj_transform=view@projection,
            camera_center=torch.zeros(3, device="cuda"))
        depth = MeshDepthRenderer(mesh, torch.device("cuda:0"))(cam)
        torch.testing.assert_close(depth[32, 64], torch.tensor(2., device="cuda"), atol=.002, rtol=0)
        torch.testing.assert_close(depth[96, 64], torch.tensor(4., device="cuda"), atol=.002, rtol=0)

        pc = GaussianModel(0)
        p = lambda x: torch.nn.Parameter(torch.tensor(x, dtype=torch.float32, device="cuda"))
        pc._xyz = p([[0, -.5, 1], [0, -.75, 3], [0, 1.5, 3]])
        pc._scaling = p([[math.log(.03)]*3]*3)
        pc._rotation = p([[1, 0, 0, 0]]*3)
        pc._opacity = p([[0]]*3)
        pc._features_dc = torch.nn.Parameter(RGB2SH(torch.tensor([[.8,.2,.1],[.1,.8,.2],[.2,.1,.8]], device="cuda"))[:,None,:])
        pc._features_rest = torch.nn.Parameter(torch.empty(3, 0, 3, device="cuda"))
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=True)
        bg = torch.zeros(3, device="cuda")
        plain = render(cam, pc, pipe, bg)["render"].detach()
        configure_pipeline(pipe, dict(occlusion_mode="proxy", occlusion_mesh=str(mesh), occlusion_cache=tmp))
        result = render(cam, pc, pipe, bg)
        assert result["occlusion_keep"].tolist() == [True, False, True]
        assert result["radii"][1].item() == 0
        assert not torch.allclose(plain, result["render"])
        result["render"].square().sum().backward()
        for attribute in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc"):
            gradient = getattr(pc, attribute).grad
            assert gradient is not None and torch.count_nonzero(gradient[1]).item() == 0, attribute
        assert torch.count_nonzero(pc._features_dc.grad[[0,2]]).item() > 0
        assert torch.count_nonzero(result["viewspace_points"].grad[1]).item() == 0
        print("CUDA smoke test passed: proxy depth/orientation, AA rendering, full-length visibility, and zero hidden gradients.")


if __name__ == "__main__":
    main()
