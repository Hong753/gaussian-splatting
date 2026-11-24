"""CPU checks of the data boundary, coupled surface, reports and video outputs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from scripts.joint_ambisur import data
from scripts.orange4f_joint_mesh import build_runtime
from utils import read_write_model as colmap


def fixture(tmp_path):
    source, run = tmp_path / "scene", tmp_path / "run"
    run.mkdir()
    model = source / "sparse/0"
    model.mkdir(parents=True)
    (source / "images").mkdir()
    cam = colmap.Camera(1, "PINHOLE", 64, 48, np.array([50.0, 52.0, 29.0, 26.0]))
    frames = {}
    for i in range(12):
        name = f"frame{i:02d}.png"
        x = i * 0.02
        frames[i + 1] = colmap.Image(
            i + 1,
            np.array([1.0, 0, 0, 0]),
            np.array([-x, 0, 0]),
            1,
            name,
            np.empty((0, 2)),
            np.empty(0, dtype=np.int64),
        )
        im = np.full((48, 64, 3), 30 + i, np.uint8)
        im[12:32, 15:22] = 200
        Image.fromarray(im).save(source / "images" / name)
    colmap.write_model({1: cam}, frames, {}, str(model), ".bin")
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"selected_images": [v.name for i, v in frames.items() if i % 2 == 1]}))
    rows = data.inputs(source, run, 64, 2, keys, 9)
    return run, rows


def test_da3_fit_only_cache_and_calibration(tmp_path, monkeypatch):
    run, rows = fixture(tmp_path)
    batch_names = []

    class MockDA3:
        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def inference(self, images, extrinsics, intrinsics, **kw):
            batch_names.extend([int(im[0, 0, 0]) - 30 for im in images])
            assert kw["align_to_input_ext_scale"]
            np.testing.assert_allclose(intrinsics[0, 0, 2], 29.0)
            z = np.full((len(images), 48, 64), 2.0, np.float32)
            return SimpleNamespace(depth=z, conf=z, intrinsics=intrinsics)

    monkeypatch.setitem(sys.modules, "depth_anything_3.api", SimpleNamespace(DepthAnything3=MockDA3))
    data.preprocess(run, rows, "fixture", 64, 80)
    data.preprocess(run, rows, "fixture", 64, 80)
    assert batch_names == [0, 2, 4, 6, 8, 10]
    with np.load(run / "da3" / "seed.npz") as cloud:
        np.testing.assert_allclose(cloud["xyz"][:, 2], 2.0)
        assert len(cloud["xyz"]) == 80
    with np.load(data.prior_path(run, rows[0])) as p:
        assert p["K"][0, 2] == 28.5
    assert not any(data.prior_path(run, r).exists() for r in rows if r["split"] == "test")
    path = json.loads((run / "cameras.json").read_text())["random"]
    assert len(path) == 9
    assert any(abs(r["w2c"][0][2]) > 0.001 for r in path)


def test_projection_and_mesh_photo_derivative():
    torch = pytest.importorskip("torch")
    from scripts.joint_ambisur.surface import reprojection

    class Pinhole:
        Fx = Fy = 21.0
        Cx, Cy = 7.25, 8.2
        image_width = image_height = 18
        znear = 0.01

        def __init__(self, tx):
            self.world_view_transform = torch.eye(4)
            self.world_view_transform[3, 0] = tx

        def get_image(self):
            return torch.arange(18).float()[None, None, :].expand(3, 18, 18) / 18, None

        def prior(self):
            return torch.full((18, 18), 2.0), torch.ones((18, 18), dtype=torch.bool)

    a, b = Pinhole(0), Pinhole(-0.1)
    mask = torch.zeros(18, 18, dtype=torch.bool)
    mask[4:12, 5:12] = True
    target = a.get_image()[0] * 0.7

    def evaluate(z):
        return reprojection(a, b, z.expand(18, 18), mask, target)

    z = torch.tensor(2.0, requires_grad=True)
    loss, n = evaluate(z)
    loss.backward()
    numerical = (evaluate(z.detach() + 0.001)[0] - evaluate(z.detach() - 0.001)[0]) / 0.002
    assert n == 56 and abs(z.grad.item()) > 0.001
    torch.testing.assert_close(z.grad, numerical, atol=2e-5, rtol=0.01)
    assert evaluate(torch.tensor(3.0))[1] == 0


def reference_runtime(tmp_path):
    ambi = os.environ.get("TEST_AMBISUR")
    milo = os.environ.get("TEST_MILO")
    if not ambi or not milo:
        pytest.skip("Set TEST_AMBISUR and TEST_MILO to pinned local checkouts")
    return build_runtime(tmp_path, Path(ambi), Path(milo))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_coupled_surface_wrong_depth(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.joint_ambisur.surface import pivots, tetrahedra, extract

    source = reference_runtime(tmp_path)
    marching = load_module("tetmesh", source / "joint_tetmesh.py").marching_tetrahedra
    torch.manual_seed(3)
    means = torch.randn(8, 3) * 0.22
    means[:, 2] = 2.0 + 0.04 * means[:, 2]
    means.requires_grad_()
    field = None
    scale = torch.full_like(means, 0.16, requires_grad=True)
    rotation = torch.eye(3).repeat(len(means), 1, 1).requires_grad_()
    spacing = torch.full((len(means),), 0.12)
    for step in range(31):
        p, r = pivots(means, scale, rotation, spacing)
        if step == 0:
            field = torch.nn.Parameter((2.0 - p[:, 2]).detach())
            tets = tetrahedra(p)
            valid = torch.ones_like(field, dtype=torch.bool)
            optimizer = torch.optim.Adam([means, scale, field], lr=0.004)
        verts, faces, _, _ = extract(p, r, field, tets, valid, marching)
        assert len(faces) > 0
        z = verts[faces, 2]
        loss = (z - 2.16).square().mean()
        if step == 0:
            initial = loss.item()
        optimizer.zero_grad()
        loss.backward()
        if step == 0:
            assert all(t.grad is not None and t.grad.norm() > 0 for t in (means, scale, rotation, field))
        optimizer.step()
    print(f"CPU mesh z RMSE {initial**0.5:.5f} -> {loss.item() ** 0.5:.5f}")
    assert loss.item() < initial * 0.10


def test_previews_complete_eval_and_videos(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    run, frames = fixture(tmp_path)
    src = reference_runtime(run)
    monkeypatch.syspath_prepend(str(src))
    # Exercise production reporting independently of native CUDA camera/rasterizer imports.
    monkeypatch.setitem(sys.modules, "scene.cameras", SimpleNamespace(Camera=object))
    monkeypatch.setitem(sys.modules, "scene.gaussian_model", SimpleNamespace(GaussianModel=object))
    monkeypatch.setitem(sys.modules, "utils.graphics_utils", SimpleNamespace(BasicPointCloud=object))
    m = load_module("joint_eval_fixture", src / "joint_runtime.py")

    class CpuCamera:
        def __init__(self, r, run):
            self.r = r

        def get_image(self):
            return torch.from_numpy(data.rgb(self.r)).permute(2, 0, 1) / 255, None

    class CpuSurface:
        ids = torch.tensor([0, 1, 2])
        sdf = torch.zeros(27)
        gs = SimpleNamespace(_features_dc=torch.zeros(3, 1, 3))

        def mesh(self):
            return (
                torch.eye(3),
                torch.tensor([[0, 1, 2]]),
                torch.tensor([[0, 0], [9, 9], [18, 18]]),
                torch.zeros(3, 1),
            )

        def render(self, cam, *mesh):
            shape = (cam.r["height"], cam.r["width"])
            return torch.ones(shape), torch.ones(*shape, 3), None

    def renderer(cam, geometry=True):
        shape = (cam.r["height"], cam.r["width"])
        return {
            "render": torch.full((3, *shape), 0.2),
            "plane_depth": torch.ones(1, *shape),
            "rendered_alpha": torch.ones(1, *shape),
        }

    m.View = CpuCamera
    ev = m.Evaluator(run, json.loads((run / "cameras.json").read_text()), CpuSurface(), renderer)
    z = torch.tensor([2.0, 2.0, 2.0, 2.0, float("nan")])
    counts = m.first_hit_counts(
        z, torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0]), torch.tensor([2.0, 3.0, 0.0, 2.0, 2.0])
    )
    assert counts == dict(reference_rays=3, mesh_hits=2, agreeing_hits=1)
    ev.preview(7, density_stage="before")
    rows, metric = ev.preview(7, density_stage="after")
    assert (run / "density/index.html").is_file()
    for stage in ("before", "after"):
        result = json.loads((run / "density" / f"{stage}.json").read_text())
        assert result["faces"] == 1 and all((run / r["panel"]).is_file() for r in result["frames"])
    assert metric["test"]["frames"] == 4
    assert all((run / r["panel"]).exists() for r in rows)
    import socket
    from urllib.request import urlopen
    from urllib.parse import urljoin
    from scripts.orange4f_joint_mesh import serve

    # Occupy the preferred port and verify that the announced fallback actually serves the current images.
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with serve(run, occupied.getsockname()[1]):
            url = (run / "preview_url.txt").read_text().strip()
            with urlopen(url, timeout=5) as response:
                assert rows[0]["panel"] in response.read().decode()
            with urlopen(urljoin(url, rows[0]["panel"]), timeout=5) as response:
                assert response.read(2) == b"\xff\xd8"

    ev.evaluate(run / "final", 7)
    result = json.loads((run / "final/metrics.json").read_text())
    assert [result["metrics"][k]["frames"] for k in ("train", "test", "all")] == [6, 6, 12]
    assert len(list((run / "final/comparisons").glob("*.jpg"))) == 12
    from plyfile import PlyData

    assert len(PlyData.read(str(run / "final/mesh.ply"))["face"]) == 1
    for filename, n in (("original", 12), ("random", 9), ("random_mesh", 9)):
        count = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "csv=p=0",
                run / f"final/{filename}.mp4",
            ],
            text=True,
        )
        assert int(count) == n


def test_missing_surface_growth_and_resume(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from scripts.joint_ambisur.surface import Surface

    source = reference_runtime(tmp_path)
    marching = load_module("tetmesh", source / "joint_tetmesh.py").marching_tetrahedra
    centers = torch.tensor(
        [[-0.3, -0.3, 2.0], [0.3, -0.3, 2.0], [-0.3, 0.3, 2.0], [0.3, 0.3, 2.0], [0.0, 0.0, 2.0], [0, 0, 4.0]]
    )
    gs = SimpleNamespace(
        get_xyz=centers,
        get_opacity=torch.ones(6, 1),
        _scaling=torch.full((6, 3), -2.0),
        _rotation=torch.zeros(6, 4),
        scaling_activation=torch.exp,
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.general_utils",
        SimpleNamespace(build_rotation=lambda q: torch.eye(3).repeat(len(q), 1, 1)),
    )
    surface = Surface(gs, dict(mesh_gaussians=4, extra_sites=2), marching)
    surface.ids = torch.arange(4)
    surface.spacing = torch.full((4,), 0.2)
    p, r = surface.positions()
    surface.retriangulate(p)
    surface.sdf = torch.nn.Parameter(2.0 - p[:, 2])
    surface.observed = torch.ones_like(surface.sdf, dtype=torch.bool)
    surface.optimizer = torch.optim.Adam([surface.sdf], lr=0.01)
    n = len(surface.sdf)
    prior = torch.full((32, 32), 2.0)
    valid = torch.ones_like(prior, dtype=torch.bool)
    cam = SimpleNamespace(
        Fx=28,
        Fy=28,
        Cx=15.5,
        Cy=15.5,
        znear=0.01,
        image_width=32,
        image_height=32,
        world_view_transform=torch.eye(4),
        prior=lambda: (prior, valid),
    )
    # The missing near surface warrants a new pivot site. A rear Gaussian at
    # the same pixel conflicts with the independent depth and must be excluded.
    surface.propose_sites(cam, {"out_observe": torch.ones(6)}, prior, valid, torch.full_like(prior, 4.0))
    assert surface.pending[0].tolist() == [4]
    surface.sdf.sum().backward()
    surface.finish_step()
    assert len(surface.sdf) == n + 9
    moments = surface.optimizer.state[surface.sdf]["exp_avg"]
    assert moments[:n].abs().min() > 0 and moments[n:].count_nonzero() == 0
    restored = Surface(gs, surface.cfg, marching)
    restored.restore(surface.state())
    original = surface.mesh()
    copied = restored.mesh()
    for i in (0, 1):
        torch.testing.assert_close(original[i], copied[i])


def test_checkpoint_density_and_first_hits(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from scripts.joint_ambisur.surface import Surface

    src = reference_runtime(tmp_path)
    marching = load_module("tetmesh", src / "joint_tetmesh.py").marching_tetrahedra
    monkeypatch.setitem(
        sys.modules,
        "utils.general_utils",
        SimpleNamespace(build_rotation=lambda q: torch.eye(3).repeat(len(q), 1, 1)),
    )

    def wall(x, y):
        return 2 + 0.10 * torch.sin(7 * x) * torch.cos(5 * y)

    rng = torch.Generator().manual_seed(127)
    xy = 1.8 * (torch.rand(160, 2, generator=rng) - 0.5)
    xyz = torch.cat((xy, wall(*xy.T)[:, None]), 1)
    gs = SimpleNamespace(
        get_xyz=xyz,
        get_opacity=torch.ones(160, 1),
        _scaling=torch.full_like(xyz, -3.5),
        _rotation=torch.zeros(160, 4),
        scaling_activation=torch.exp,
    )
    y, x = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")

    def camera(i, offset):
        nx, ny = (x - 47.5) / 86, (y - 47.5) / 86
        z = torch.full_like(nx, 2.0)
        for _ in range(18):
            z = wall(offset + nx * z, ny * z)
        transform = torch.eye(4)
        transform[3, 0] = -offset
        return SimpleNamespace(
            uid=i,
            world_view_transform=transform,
            Fx=86,
            Fy=86,
            Cx=47.5,
            Cy=47.5,
            image_width=96,
            image_height=96,
            znear=0.01,
            prior=lambda: (z, torch.ones_like(z, dtype=torch.bool)),
        )

    cams = [camera(i, tx) for i, tx in enumerate((-0.18, 0.0, 0.18))]

    def render(cam):
        z, mask = cam.prior()
        return {"plane_depth": z[None], "rendered_alpha": mask[None].float()}

    # Independent pinhole-ray first intersections on a translated camera.
    held = camera(3, 0.09)
    py, px = y[24:73:3, 24:73:3].flatten(), x[24:73:3, 24:73:3].flatten()
    rays = torch.stack(((px - 47.5) / 86, (py - 47.5) / 86, torch.ones_like(px)), 1).float()
    truth = held.prior()[0][py, px]

    def measure(surface):
        verts, faces, _, _ = surface.mesh()
        a, b, c = verts.detach()[faces].unbind(1)
        ab, ac = b - a, c - a
        origin = verts.new_tensor([0.09, 0, 0])
        t = origin - a
        q = torch.linalg.cross(t, ab)
        results = []
        for d in rays.split(32):
            h = torch.linalg.cross(d[:, None, :], ac[None, :, :])
            det = (h * ab).sum(-1)
            u = (t * h).sum(-1) / det
            v = (q * d[:, None, :]).sum(-1) / det
            z = (ac * q).sum(-1) / det
            hits = (det.abs() > 1e-8) & (u >= 0) & (v >= 0) & (u + v <= 1) & (z > 0)
            results.append(torch.where(hits, z, torch.inf).amin(1))
        return len(faces), torch.cat(results)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(26)
        surface = Surface(gs, dict(mesh_gaussians=32, init_views=3), marching)
        surface.initialize(cams, render)
    surface.sdf.square().mean().backward()
    surface.finish_step()
    parents = [
        t.detach().clone()
        for t in (
            surface.ids,
            surface.spacing,
            surface.sdf,
            surface.observed,
            surface.optimizer.state[surface.sdf]["exp_avg"],
        )
    ]
    coarse, old = measure(surface)
    rng_state = torch.get_rng_state().clone()
    surface.expand_checkpoint(96, cams, render)
    dense, new = measure(surface)
    torch.testing.assert_close(torch.get_rng_state(), rng_state)
    moments = surface.optimizer.state[surface.sdf]["exp_avg"]
    for prev, cur in zip(
        parents, (surface.ids, surface.spacing, surface.sdf, surface.observed, moments), strict=True
    ):
        torch.testing.assert_close(prev, cur[: len(prev)])
    assert moments[len(parents[-1]) :].count_nonzero() == 0
    assert len(surface.ids.unique()) == 128
    assert dense > coarse
    common = old.isfinite() & new.isfinite()
    old_error = (old[common] - truth[common]).square().mean().sqrt().item()
    new_error = (new[common] - truth[common]).square().mean().sqrt().item()
    print(
        f"Corrugated wall: {coarse} -> {dense} faces; held-out hits {int(old.isfinite().sum())} -> "
        f"{int(new.isfinite().sum())}/{len(rays)}; common-ray z RMSE {old_error:.5f} -> {new_error:.5f} ({int(common.sum())} rays)"
    )
    assert common.sum() > 200 and new_error < old_error


def test_checkpoint_density_launcher(tmp_path, monkeypatch):
    from scripts import orange4f_joint_mesh as launcher

    parent = tmp_path / "original"
    (parent / "checkpoints").mkdir(parents=True)
    checkpoint = parent / "checkpoints/step_020000.pt"
    checkpoint.touch()
    (parent / "da3").mkdir()
    (parent / "cameras.json").write_text("{}")
    config = dict(
        mesh_gaussians=32000,
        mesh_from=15000,
        iterations=30000,
        source="/images",
        width=1600,
        preview_every=500,
        init_views=256,
    )
    (parent / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(launcher, "OUTPUT", parent)
    monkeypatch.setattr(launcher, "pinned", lambda *args: tmp_path)
    monkeypatch.setattr(launcher, "build_runtime", lambda run, *args: run / "runtime")
    calls = []
    monkeypatch.setattr(launcher, "invoke", lambda run, cwd, label, cmd: calls.append((label, cmd)))
    args = ["--densify-checkpoint", "20000", "--port", "0"]
    with pytest.raises(SystemExit):
        launcher.main([*args, "--run", str(parent)])
    launcher.main(args)
    out = parent.with_name("original_sites128000_from020000")
    assert [label for label, _ in calls] == ["gpu_check", "train"]
    assert calls[-1][1][-2:] == ["--start_checkpoint", str(checkpoint)]
    assert json.loads((out / "config.json").read_text())["mesh_gaussians"] == 128000
    assert (out / "da3").resolve() == parent / "da3" and (out / "cameras.json").samefile(
        parent / "cameras.json"
    )
    assert json.loads((parent / "config.json").read_text()) == config
    (out / "checkpoints").mkdir()
    local = out / "checkpoints" / checkpoint.name
    local.touch()
    launcher.main(args)
    assert calls[-1][1][-2:] == ["--start_checkpoint", str(local)]


def test_resume_expands_only_parent_checkpoint(tmp_path, monkeypatch):
    import random

    torch = pytest.importorskip("torch")
    src = reference_runtime(tmp_path)
    monkeypatch.syspath_prepend(str(src))
    monkeypatch.setitem(sys.modules, "scene.cameras", SimpleNamespace(Camera=object))
    monkeypatch.setitem(sys.modules, "scene.gaussian_model", SimpleNamespace(GaussianModel=object))
    monkeypatch.setitem(sys.modules, "utils.graphics_utils", SimpleNamespace(BasicPointCloud=object))
    runtime = load_module("density_resume_fixture", src / "joint_runtime.py")
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.tensor([0]))
    monkeypatch.setattr(runtime, "dashboard", lambda *args: None)
    path = tmp_path / "parent.pt"
    torch.save(
        dict(
            step=20000,
            gaussians=12,
            surface=32,
            scene_config=dict(mesh_gaussians=32),
            stack=[18, 13],
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.tensor([0]),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        ),
        path,
    )
    calls = []
    trainer = runtime.JointTraining.__new__(runtime.JointTraining)
    trainer.run = tmp_path
    trainer.gs = SimpleNamespace(restore=lambda value, opt: calls.append(("gs", value)), capture=lambda: 12)
    trainer.opt = SimpleNamespace(iterations=30000)
    trainer.config = dict(density_checkpoint=str(path), parent_mesh_gaussians=32, mesh_gaussians=128)
    trainer.scene = SimpleNamespace(fit=[SimpleNamespace(uid=i) for i in [13, 18]])
    trainer.render = object()
    surface = SimpleNamespace(ids=None)
    surface.restore = lambda n: setattr(surface, "ids", list(range(n)))

    def expand(count, cams, renderer):
        assert cams == trainer.scene.fit and renderer is trainer.render
        calls.append(("expand", count))
        surface.ids.extend(range(len(surface.ids), len(surface.ids) + count))

    surface.expand_checkpoint = expand
    surface.state = lambda: len(surface.ids)
    trainer.surface = surface

    def preview(step, density_stage):
        calls.append((density_stage, len(surface.ids)))
        return [], {}

    trainer.eval = SimpleNamespace(preview=preview, save_surface=lambda p: p.touch())
    for source in (path, tmp_path / "checkpoints/step_020000.pt"):
        step, stack = trainer.resume(source)
        assert step == 20000 and [c.uid for c in stack] == [18, 13]
    assert calls == [("gs", 12), ("before", 32), ("expand", 96), ("after", 128), ("gs", 12)]
    assert (tmp_path / "checkpoints/mesh_020000.ply").exists()
