from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from scripts import orange4f_splat_footprints as audit


def test_mass_uses_transmittance_and_both_masks():
    torch = pytest.importorskip("torch")
    # Third pixel is masked. Last Gaussian is behind an opaque one in every measured pixel.
    alpha = torch.tensor([[1.0, 0.25, 0.0], [0.9, 1.0, 0.0], [1.0, 1.0, 1.0]])
    transmittance = torch.vstack([torch.ones(3), (1 - alpha).cumprod(0)[:-1]])

    def render(tags):
        return ((alpha * transmittance).T @ tags).T.unsqueeze(1)

    ids, weights = audit.contribution_weights(render, torch.zeros(3, 3), (0, 0, 3, 1), [[1, 1, 0]], [[0, 1, 0]])
    assert ids.tolist() == [0, 1]
    np.testing.assert_allclose(weights, [[1.25, 0.25], [0.75, 0.75]])


def test_mass_conservation_rejects_invalid_backward():
    torch = pytest.importorskip("torch")

    def render(tag):
        return (tag.detach() + tag * 0).view(3, 1, 1)

    with pytest.raises(ValueError, match="mass mismatch"):
        audit.contribution_weights(render, torch.zeros(1, 3), (0, 0, 1, 1), [[1]], [[1]])


def test_weighted_widths_ignore_hidden_centers_and_distinguish_depth():
    w = np.array([0.86, 0.01, 0.13])
    minor, major = np.array([5.0, 50.0, 1.0]), np.array([6.0, 80.0, 8.0])
    z = np.array([2.0, 1.0, 2.1])
    offset = np.array([0.0, 0.1, -0.1])
    row = audit.summarize(w, minor, major, z, offset, 2.0, 1)
    assert row["minor_fwhm_px"]["p50"] == 5
    assert row["major_fwhm_px"]["p50"] == 6
    assert row["broad_on_plane_share"] == pytest.approx(0.86)
    assert row["front_of_plane_share"] == pytest.approx(0.01)
    assert row["behind_plane_share"] == pytest.approx(0.13)
    assert "on-plane centers" in row["verdict"]
    assert "outside" in audit.summarize(w[::-1], minor, major, z, offset, 2.0, 1)["verdict"]


def test_texture_size_and_contrast():
    rgb = np.full((80, 100, 3), 80, np.uint8)
    for y in range(8, 65, 10):
        for x in range(8, 85, 10):
            rgb[y : y + 2, x : x + 2] = 180
    mask = np.ones(rgb.shape[:2], bool)
    mask[:15] = False
    cores, stats = audit.bright_cores(rgb, mask)
    assert stats["median_diameter_px"] == pytest.approx(2 * np.sqrt(4 / np.pi))
    assert not cores[:15].any()
    assert audit.core_contrast_ratio(rgb, rgb, cores) == pytest.approx(1.0)
    assert audit.core_contrast_ratio(rgb, np.zeros_like(rgb), cores) == pytest.approx(0.0)


def test_projected_widths_have_native_pixel_scale():
    cov = audit.projected_covariance(
        [[0.0, 0.0, 2.0]], [[0.08, 0.04, 0.01]], [[1.0, 0.0, 0.0, 0.0]], np.eye(4), 200, 200, np.pi / 2, np.pi / 2
    )
    np.testing.assert_allclose(np.linalg.eigvalsh(cov), [[4.3, 16.3]])


def test_paired_report_with_common_checkpoint_and_separate_control(tmp_path, monkeypatch):
    # Exercise both frame reads, masks, photo crops, native rendering entry points,
    # checkpoint/split selection, output images, and publication on a CPU using a test renderer.
    torch = pytest.importorskip("torch")
    suite = tmp_path / "stride3_noaa_v1"
    suite.mkdir()
    sparse = tmp_path / "sparse/0"
    sparse.mkdir(parents=True)
    photo_dir = tmp_path / "images"
    photo_dir.mkdir()
    w, h, f = 400, 320, 280.0
    (sparse / "cameras.txt").write_text(f"1 PINHOLE {w} {h} {f} {f} 200 160\n")
    names = [audit.REFERENCE, audit.NEIGHBOR]
    shifts = [0.0, 0.05]
    (sparse / "images.txt").write_text(
        "".join(f"{i + 1} 1 0 0 0 {t} 0 0 1 {name}\n\n" for i, (name, t) in enumerate(zip(names, shifts)))
    )
    rgb = np.full((h, w, 3), 80, np.uint8)
    rgb[2::6, 2::6] = 180
    rgb[2::6, 3::6] = 180
    for name in names:
        Image.fromarray(rgb).save(photo_dir / name, quality=100)
    yy, xx = np.mgrid[:h:2, :w:2]
    xyz = np.stack(((xx + 0.5 - 200) * 2 / f, (yy + 0.5 - 160) * 2 / f, np.full(xx.shape, 2.0)), axis=-1)
    vertices = np.zeros(xx.size, dtype=[(k, "f4") for k in "xyz"])
    for i, k in enumerate("xyz"):
        vertices[k] = xyz[..., i].ravel()
    cloud = tmp_path / "lidar.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(cloud)
    selection = suite / "selection.json"
    selection.write_text(json.dumps({"selected_images": names}))
    (suite / "suite.json").write_text(
        json.dumps(dict(source=str(tmp_path), cloud={"cached": str(cloud)}, selection=str(selection), runs=[]))
    )
    for path, steps, train in [
        (suite / "3dgs", [1000, 3000], names),
        (suite / "3dgs_fixed_lidar", [1000, 2000], names),
        (suite.parent / "fixed_lidar_one_view_09882_knn2x", [2000], names[:1]),
    ]:
        path.mkdir()
        (path / "cfg_args").write_text(
            f"Namespace(source_path={str(tmp_path)!r}, keyframe_manifest='selected', antialiasing=False)"
        )
        (path / "keyframes.json").write_text(json.dumps({"selected_images": train}))
        for step in steps:
            d = path / f"point_cloud/iteration_{step}"
            d.mkdir(parents=True)
            (d / "point_cloud.ply").touch()

    class Gaussian:
        def __init__(self, degree):
            self.get_xyz = torch.tensor([[0.0, 0.0, 2.0]])
            self.get_rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            self.get_scaling = torch.tensor([[0.05, 0.05, 0.01]])

        def load_ply(self, path):
            assert Path(path).is_file()

    def render(cam, model, pipe, bg, override_color=None):
        color = torch.ones(1, 3) / 3 if override_color is None else override_color
        return {"render": color.reshape(3, 1, 1).expand(3, h, w) * 0.9}

    from scripts import render_views_worker

    monkeypatch.setitem(sys.modules, "scene.gaussian_model", SimpleNamespace(GaussianModel=Gaussian))
    monkeypatch.setitem(sys.modules, "gaussian_renderer", SimpleNamespace(render=render))
    monkeypatch.setattr(
        render_views_worker,
        "load_record_camera",
        lambda args, record, i: SimpleNamespace(
            FoVx=record["fov_x"], FoVy=record["fov_y"], world_view_transform=torch.eye(4)
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        audit.main(["--suite", str(suite)])
    finally:
        torch.set_num_threads(old_threads)
    (page,) = (suite / audit.REPORT).glob("iteration_*/report.json")
    rows = json.loads(page.read_text())["measurements"]
    assert [r["iteration"] for r in rows] == [1000] * 4 + [2000] * 2
    assert [r["split"] for r in rows] == ["train"] * 5 + ["held-out"]
    for row in rows:
        assert row["floor"]["mean_alpha"] == pytest.approx(0.9, abs=0.001)
        assert row["floor_z"] == pytest.approx(2.0)
        assert row["core_contrast_ratio"] == pytest.approx(0.0, abs=1e-5)
        assert all((page.parent / row[key]).is_file() for key in ["error", "mass_map", "render"])
    assert (suite / audit.REPORT / "index.html").read_text().count(page.parent.name) == 2
    assert f"{audit.REPORT}/index.html" in (suite / "index.html").read_text()


def test_native_cuda_compositing_on_concentric_splats():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Native rasterizer smoke test requires CUDA")
    from gaussian_renderer import render
    from utils.graphics_utils import getProjectionMatrix

    fov = np.pi / 2
    pc = SimpleNamespace(
        get_xyz=torch.tensor([[0, 0, 2.0], [0, 0, 3.0]], device="cuda"),
        get_rotation=torch.tensor([[1.0, 0, 0, 0]] * 2, device="cuda"),
        get_scaling=torch.full((2, 3), 0.2, device="cuda"),
        get_opacity=torch.tensor([[0.5], [0.8]], device="cuda"),
        active_sh_degree=0,
    )
    cam = SimpleNamespace(
        image_width=33,
        image_height=33,
        FoVx=fov,
        FoVy=fov,
        world_view_transform=torch.eye(4, device="cuda"),
        camera_center=torch.zeros(3, device="cuda"),
        full_proj_transform=getProjectionMatrix(0.01, 100.0, fov, fov).T.contiguous().cuda(),
    )
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)

    def tags(colors):
        return render(cam, pc, pipe, torch.zeros(3, device="cuda"), override_color=colors)["render"]

    ids, w = audit.contribution_weights(tags, pc.get_xyz, (16, 16, 1, 1), [[1]], [[1]])
    assert ids.tolist() == [0, 1]
    np.testing.assert_allclose(w, [[0.5, 0.5], [0.4, 0.4]], atol=0.001)
