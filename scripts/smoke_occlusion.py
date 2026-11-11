#!/usr/bin/env python3
"""GPU smoke test for LiDAR depth, novel poses, native AA, and hidden gradients."""
import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# At 128px/90deg, the hidden center projects to row 31.5, safely inside the
# near surface. Row 47.5 touched a tile contaminated by the farther surface
# through edge padding, so the legacy whole-Gaussian test correctly kept it.
GAUSSIAN_CENTERS = ((0., -.5, 1.), (0., -1.5, 3.), (0., 1.5, 3.))
TILE_BOUNDARY_GAUSSIAN = (0., -.75, 3.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="2")
    parser.add_argument('--mesh', action='store_true', help='Test the whole-scene mesh renderer and native AA culling')
    parser.add_argument('--lidar-mesh', action='store_true', help='Reconstruct the test mesh from measured samples, with no DA3 input')
    parser.add_argument('--backend', choices=('whole','ray'), default='whole',
                        help='Exercise the same geometry-to-renderer path as the selected workflow')
    args = parser.parse_args(argv)
    args.mesh = args.mesh or args.lidar_mesh
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires the CUDA-enabled 3dgs environment and a GPU")
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scripts.occlusion_options import configure_pipeline
    from utils.graphics_utils import getProjectionMatrix
    from utils.occlusion_depth import LidarDepthRenderer
    from utils.lidar_occlusion import save_geometry
    import numpy as np
    from utils.sh_utils import RGB2SH

    with tempfile.TemporaryDirectory(prefix="3dgs_occlusion_smoke_") as tmp:
        geometry = Path(tmp)/"two_depths.npz"
        centers, radii = [], []
        for z, low, high, step in [(2, -2, 0, .125), (4, 0, 4, .25)]:
            if args.lidar_mesh:
                step = .1
            x, y = np.meshgrid(np.arange(-z, z+step/2, step), np.arange(low, high+step/2, step))
            centers.extend(np.stack((x.ravel(), y.ravel(), np.full(x.size, z)), axis=1))
            radii.extend([step*1.2]*x.size)
        save_geometry(geometry, centers, np.tile([0, 0, 1], (len(centers), 1)), radii)
        if args.mesh:
            import json
            from utils.scene_occlusion import FORMAT, MeshDepthRenderer
            from utils.lidar_occlusion import file_digest
            mesh_path = Path(tmp)/'mesh.npz'
            vertices = np.array([[-3,-3,2],[3,-3,2],[3,0,2],[-3,0,2],
                                 [-5,0,4],[5,0,4],[5,5,4],[-5,5,4]], np.float32)
            faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]], np.int32)
            if args.lidar_mesh:
                from utils.lidar_surface import build_lidar_mesh
                vertices, faces, _, _, stats = build_lidar_mesh(np.asarray(centers), threads=2)
                print(f"LiDAR smoke fixture: {stats['triangles']:,} measured triangles; no predicted depths", flush=True)
            np.savez(mesh_path, vertices=vertices, faces=faces)
            geometry = Path(tmp)/'scene.json'
            geometry.write_text(json.dumps(dict(format=FORMAT, complete=True, mesh='mesh.npz',
                mesh_sha256=file_digest(mesh_path), relative_margin=0. if args.lidar_mesh else .05)))
        view = torch.eye(4, device="cuda")
        projection = getProjectionMatrix(.01, 100., math.pi/2, math.pi/2).cuda().T
        cam = SimpleNamespace(image_width=128, image_height=128, FoVx=math.pi/2, FoVy=math.pi/2,
            znear=.01, zfar=100., world_view_transform=view, full_proj_transform=view@projection,
            camera_center=torch.zeros(3, device="cuda"))
        renderer = MeshDepthRenderer(mesh_path, torch.device('cuda:0')) if args.mesh else LidarDepthRenderer(geometry, torch.device("cuda:0"))
        depth = renderer(cam)
        torch.testing.assert_close(depth[32, 64], torch.tensor(2., device="cuda"), atol=.002, rtol=0)
        torch.testing.assert_close(depth[96, 64], torch.tensor(4., device="cuda"), atol=.002, rtol=0)
        moved_view = view.clone()
        moved_view[3, 2] = -.25
        novel = SimpleNamespace(**{**vars(cam), 'world_view_transform': moved_view, 'full_proj_transform': moved_view@projection})
        novel_depth = renderer(novel)
        torch.testing.assert_close(novel_depth[32, 64], torch.tensor(1.75, device="cuda"), atol=.002, rtol=0)
        torch.testing.assert_close(novel_depth[96, 64], torch.tensor(3.75, device="cuda"), atol=.002, rtol=0)

        pc = GaussianModel(0)
        p = lambda x: torch.nn.Parameter(torch.tensor(x, dtype=torch.float32, device="cuda"))
        centers = list(GAUSSIAN_CENTERS)
        if args.backend == 'ray':
            # Preserve the original boundary case: ray clipping must reject
            # it even when the conservative whole-tile bound cannot.
            centers[1] = TILE_BOUNDARY_GAUSSIAN
        pc._xyz = p(centers)
        pc._scaling = p([[math.log(.03)]*3]*3)
        pc._rotation = p([[1, 0, 0, 0]]*3)
        pc._opacity = p([[0]]*3)
        pc._features_dc = torch.nn.Parameter(RGB2SH(torch.tensor([[.8,.2,.1],[.1,.8,.2],[.2,.1,.8]], device="cuda"))[:,None,:])
        pc._features_rest = torch.nn.Parameter(torch.empty(3, 0, 3, device="cuda"))
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=True)
        bg = torch.zeros(3, device="cuda")
        plain = render(cam, pc, pipe, bg)["render"].detach()
        configure_pipeline(pipe, dict(occlusion_mode='mesh' if args.mesh else 'lidar',
            occlusion_backend=args.backend, occlusion_geometry=str(geometry), occlusion_cache=tmp))
        result = render(cam, pc, pipe, bg)
        actual = result['occlusion_keep'].tolist()
        assert actual == [True, False, True], f'{args.backend} culling: expected [True, False, True], got {actual}'
        if args.backend == 'ray':
            flags = result['occlusion_fragment_flags'].tolist()
            assert flags == [1, 2, 1], f'Expected contributing/blocked/contributing CUDA flags, got {flags}'
        assert result["radii"][1].item() == 0
        assert not torch.allclose(plain, result["render"])
        result["render"].square().sum().backward()
        for attribute in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc"):
            gradient = getattr(pc, attribute).grad
            assert gradient is not None and torch.count_nonzero(gradient[1]).item() == 0, attribute
        assert torch.count_nonzero(pc._features_dc.grad[[0,2]]).item() > 0
        assert torch.count_nonzero(result["viewspace_points"].grad[1]).item() == 0
        print(f"CUDA smoke test passed ({args.backend} backend): {'scene mesh' if args.mesh else 'LiDAR'} depth/orientation, unseen camera pose, AA rendering, and zero hidden gradients.")


if __name__ == "__main__":
    main()
