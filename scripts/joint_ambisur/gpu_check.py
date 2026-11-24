"""Exercise both compiled renderers and the mesh-to-Gaussian gradient before DA3."""

import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import nvdiffrast.torch as dr
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from joint_runtime import View
from joint_report import write_json
from joint_surface import extract, pivots, raster, tetrahedra
from joint_tetmesh import marching_tetrahedra


def main():
    run = Path(os.environ["JOINT_RUN"])
    cam = View(
        dict(id=0, width=80, height=64, K=[55.0, 52.0, 34.0, 36.0], w2c=np.eye(4).tolist(), name="fixture"),
        run,
    )
    theta = 0.35
    xy = np.array([[-0.22, -0.22], [0.22, -0.22], [-0.22, 0.22], [0.22, 0.22], [0.0, 0.0]], np.float32)
    xyz = np.column_stack([xy, 2.0 - math.tan(theta) * xy[:, 0]]).astype(np.float32)
    gs = GaussianModel(3)
    gs.create_from_pcd(BasicPointCloud(xyz, np.full_like(xyz, 0.6), np.zeros_like(xyz)), 1.0)
    gs.disable_trunc, gs.trunc_sigma = False, 2.0
    with torch.no_grad():
        gs._scaling[:] = gs._xyz.new_tensor([0.28, 0.32, 0.04]).log()
        gs._rotation[:] = gs._xyz.new_tensor([math.cos(theta / 2), 0, math.sin(theta / 2), 0])
        gs._opacity.fill_(3.0)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    pkg = render(cam, gs, pipe, torch.zeros(3, device="cuda"), return_depth_normal=False, ray_reg=-1)
    yy, xx = 36, 38
    expected = 2.0 / (1 + math.tan(theta) * (xx - cam.Cx) / cam.Fx)
    z_gs = pkg["plane_depth"][0, yy, xx]
    ctx = dr.RasterizeCudaContext()
    verts = gs._xyz[:4]
    faces = torch.tensor([[0, 1, 2], [2, 1, 3]], device="cuda")
    z, _, _ = raster(ctx, cam, verts, faces)
    torch.testing.assert_close(z[yy, xx], z.new_tensor(expected), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(z_gs, z[yy, xx], rtol=5e-4, atol=5e-4)
    loss = pkg["render"].mean() + (z_gs - 1.8).square()
    loss.backward()
    if not torch.isfinite(gs._xyz.grad).all() or gs._xyz.grad.norm() == 0:
        raise RuntimeError("AmbiSuR backward produced invalid position gradients")
    gs._xyz.grad = gs._scaling.grad = gs._rotation.grad = None
    p, r = pivots(gs.get_xyz, gs.get_scaling, gs.get_rotation_matrix(), gs._xyz.new_full((len(xyz),), 0.2))
    field = torch.nn.Parameter((2.0 - p.detach()[:, 2]) / 0.5)
    tets = tetrahedra(p)
    v, f, _, _ = extract(p, r, field, tets, torch.ones_like(field, dtype=torch.bool), marching_tetrahedra)
    depth, _, coverage = raster(ctx, cam, v, f)
    hit = depth.detach() > 0
    if hit.sum() < 16:
        raise RuntimeError("Synthetic in-loop mesh has too few first hits")
    (depth[hit].mean() + 0.05 * coverage.mean()).backward()
    gradients = {}
    for label, tensor in (
        ("means", gs._xyz),
        ("scales", gs._scaling),
        ("quaternions", gs._rotation),
        ("signed_field", field),
    ):
        grad = tensor.grad
        if grad is None or not grad.isfinite().all() or grad.norm() == 0:
            raise RuntimeError(f"No finite mesh gradient to {label}")
        gradients[label] = float(grad.norm())
    write_json(
        run / "gpu_check.json",
        dict(
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gaussian_z=float(z_gs.detach()),
            mesh_z=float(z[yy, xx].detach()),
            analytic_z=expected,
            mesh_gradients=gradients,
            faces=len(f),
            mesh_pixels=int(hit.sum()),
        ),
    )
    print(f"CUDA forward/backward checks passed. {run / 'gpu_check.json'}", flush=True)


if __name__ == "__main__":
    main()
