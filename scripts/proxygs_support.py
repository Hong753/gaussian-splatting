"""Adapters around pinned official Proxy-GS; no changes to the source dataset."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/Visionary-Laboratory/Proxy-GS.git"
REVISION = "eac937e8f34ab46690d73cf4e4d0217ae32d645e"
from scripts.view_comparisons import atomic_json, read_colmap_records, split_names
from utils.colmap_paths import colmap_model_dir


def patch_training(text):
    """Only integration fixes, checked against the pinned release."""
    start = text.index("import subprocess\n")
    end = text.index("import torch", start)
    text = text[:start] + "# GPU selection is owned by scripts/run_proxygs.py.\n\n" + text[end:]
    old = "    all_cameras = scene.getTrainCameras() \n"
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream training depth-cache code")
    text = text.replace(old, "    all_cameras = scene.getTrainCameras() + scene.getTestCameras()\n")
    old = '        # else:\n        # scaling_reg = torch.tensor(0.0, device="cuda")'
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream empty-render handling")
    text = text.replace(old, '        else:\n            raise RuntimeError("Proxy-GS produced no Gaussians; inspect the proxy depth coverage.")')
    marker = '    # rendering\n    logger.info(f'
    if text.count(marker) != 1:
        raise ValueError("Unexpected upstream final-report entry point")
    text = text.replace(marker, '    if os.environ.get("PROXYGS_EXTERNAL_REPORT") == "1":\n        raise SystemExit(0)\n\n' + marker)
    return text


def patch_reader(text):
    old = '        frames = contents["frames"]\n'
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream transforms loader")
    return text.replace(old, old + '        if not frames:\n            return []\n')


def install(repo, environment):
    if sys.version_info < (3, 11):
        raise RuntimeError("The pinned requirements need Python >=3.11. Run setup from the existing Python 3.12 3dgs environment.")
    repo, environment = Path(repo).resolve(), Path(environment).resolve()
    if not (repo / ".git").exists():
        if repo.exists() and any(repo.iterdir()):
            raise ValueError(f"Refusing to replace a nonempty non-git directory: {repo}")
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM, str(repo)], check=True)
        subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=repo, check=True)
        subprocess.run(["git", "sparse-checkout", "set", "arguments", "scene", "utils",
                        "gaussian_renderer", "Mesh2DepthHelper", "lpipsPyTorch", "submodules"], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "--detach", REVISION], cwd=repo, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != REVISION:
        raise ValueError(f"{repo} is at {actual}; use a separate checkout pinned to {REVISION}")
    patched = ""
    for filename, patch in (("train.py", patch_training), ("scene/dataset_readers.py", patch_reader)):
        original = subprocess.check_output(["git", "show", REVISION + ":" + filename], cwd=repo, text=True)
        transformed = patch(original)
        target = repo / filename
        if target.read_text() not in (original, transformed):
            raise ValueError(f"Proxy-GS {filename} has unrelated changes; refusing to overwrite them")
        target.write_text(transformed)
        if filename == "train.py":
            patched = transformed
    if not (environment / "bin/python").is_file():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], check=True)
    python = str(environment / "bin/python")
    subprocess.run([python, "-c", "import torch; assert torch.version.cuda, 'Activate a CUDA PyTorch environment before setup'"], check=True)
    subprocess.run([python, "-m", "pip", "install", "wheel", "setuptools", "ninja", "packaging",
                    "matplotlib", "tensorboard"], check=True)
    subprocess.run([python, "-m", "pip", "install", "--no-build-isolation", "-r", str(repo / "requirements.txt")], check=True)
    versions = subprocess.check_output([python, "-c",
        "import torch,json; print(json.dumps([torch.__version__.split('+')[0],torch.version.cuda]))"], text=True)
    torch_version, cuda_version = json.loads(versions)
    major, minor = torch_version.split(".")[:2]
    wheel = f"https://data.pyg.org/whl/torch-{major}.{minor}.0+cu{cuda_version.replace('.', '')}.html"
    result = subprocess.run([python, "-m", "pip", "install", "--only-binary=:all:", "torch-scatter", "-f", wheel])
    if result.returncode:
        subprocess.run([python, "-m", "pip", "install", "--no-build-isolation", "torch-scatter"], check=True)
    for extension in ("diff-gaussian-rasterization", "simple-knn"):
        subprocess.run([python, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
                        str(repo / "submodules" / extension)], check=True)
    subprocess.run([python, "-c",
        "import torch,torch_scatter,open3d,nvdiffrast.torch;"
        "from diff_gaussian_rasterization import GaussianRasterizationSettings;"
        "assert 'depth_mesh' in GaussianRasterizationSettings._fields,"
        "'Wrong rasterizer imported: Proxy-GS needs its own extension';"
        "from simple_knn._C import distCUDA2; print('Proxy-GS imports OK')"], cwd=repo, check=True)
    atomic_json(environment / "proxygs_install.json",
                dict(upstream=str(repo), revision=REVISION, python=python,
                     training_patch_sha256=hashlib.sha256(patched.encode()).hexdigest()))
    return python


def prepare_dataset(source, keyframes, model, mesh=None, voxel_size=.05):
    import numpy as np
    from PIL import Image
    source, keyframes, model = map(lambda p: Path(p).expanduser().resolve(), (source, keyframes, model))
    manifest = json.loads(keyframes.read_text())
    if Path(manifest["source_path"]).resolve() != source:
        raise ValueError("Keyframe manifest belongs to a different source/pose set")
    records = read_colmap_records(source, manifest.get("image_directory", "images"))
    splits = split_names(records, {"eval": manifest.get("eval", False)}, manifest)
    cloud = (colmap_model_dir(source) / "points3D.ply").resolve()
    if not cloud.is_file():
        raise FileNotFoundError(f"Original LiDAR initialization PLY is missing: {cloud}")
    stage = model / "proxygs_data"
    frames = []
    transforms = {"train": [], "eval": []}
    for index, name in enumerate(sorted(splits["train"] | splits["eval"])):
        record = records[name]
        path = Path(record["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        split = "train" if name in splits["train"] else "eval"
        alias = f"{index:06d}"
        extension = path.suffix.lower()
        if extension not in (".jpg", ".jpeg", ".png"):
            raise ValueError(f"Unsupported training image extension: {path}")
        with Image.open(path) as image:
            width, height = image.size
        if abs(width / height - record["width"] / record["height"]) > 1e-3:
            raise ValueError(f"Image and calibration aspect ratios differ: {name}")
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(record["R"])
        c2w[:3, 3] = -np.array(record["R"]) @ np.array(record["T"])
        c2w[:3, 1:3] *= -1  # COLMAP camera axes -> NeRF/OpenGL.
        transforms[split].append(dict(file_path=f"images/{alias}{extension}",
                transform_matrix=c2w.tolist(), fl_x=record["fx"] * width/record["width"],
                fl_y=record["fy"] * height/record["height"],
                cx=record["cx"] * width/record["width"], cy=record["cy"] * height/record["height"]))
        frames.append({**record, "alias": alias, "split": split, "extension": extension,
                       "image_size": path.stat().st_size, "image_mtime_ns": path.stat().st_mtime_ns})
    settings = dict(version=1, source=str(source), keyframes=str(keyframes), cloud=str(cloud),
                    cloud_size=cloud.stat().st_size, cloud_mtime_ns=cloud.stat().st_mtime_ns,
                    mesh=str(Path(mesh).expanduser().resolve()) if mesh else None,
                    voxel_size=voxel_size, frames=frames,
                    projection="centered Graphdeco projection, as in the 3DGS baseline")
    if mesh:
        mesh_stat = Path(settings["mesh"]).stat()
        settings.update(mesh_size=mesh_stat.st_size, mesh_mtime_ns=mesh_stat.st_mtime_ns)
    path = stage / "preparation.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError("Existing Proxy-GS preparation differs; choose a new output directory")
        return path
    if stage.exists() and any(stage.iterdir()):
        raise ValueError(f"Incomplete/unrecognized staging directory: {stage}")
    (stage / "images").mkdir(parents=True)
    for record in frames:
        (stage / "images" / (record["alias"] + record["extension"])).symlink_to(record["image_path"])
    (stage / "input.ply").symlink_to(cloud)
    atomic_json(stage / "transforms_train.json", {"frames": transforms["train"]})
    atomic_json(stage / "transforms_test.json", {"frames": transforms["eval"]})
    atomic_json(path, settings)
    return path


def resolution(width, height, setting):
    if setting in (1, 2, 4, 8):
        return round(width / setting), round(height / setting)
    factor = max(1., width / 1600.) if setting == -1 else width / setting
    return int(width / factor), int(height / factor)


def build_proxy(preparation):
    """Use supported LiDAR triangles; do not seal unobserved space with Poisson."""
    import numpy as np
    import open3d as o3d
    path = Path(preparation)
    data = json.loads(path.read_text())
    output = path.parent / "proxy.ply"
    if output.is_file():
        return output
    if data["mesh"]:
        mesh = o3d.io.read_triangle_mesh(data["mesh"])
    else:
        cloud = o3d.io.read_point_cloud(data["cloud"]).remove_non_finite_points()
        cloud = cloud.voxel_down_sample(data["voxel_size"])
        if len(cloud.points) < 30:
            raise ValueError("Too few finite LiDAR points for a surface proxy")
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4*data["voxel_size"], max_nn=30))
        cloud.orient_normals_consistent_tangent_plane(20)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector([1.5*data["voxel_size"], 3*data["voxel_size"]]))
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if len(triangles):
            xyz = vertices[triangles]
            edges = np.linalg.norm(xyz - np.roll(xyz, 1, axis=1), axis=2).max(axis=1)
            mesh.remove_triangles_by_mask(edges > 4*data["voxel_size"])
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise ValueError("Proxy has no triangles; supply a registered mesh with --mesh or adjust --voxel-size")
    vertices = np.asarray(mesh.vertices)
    if not np.isfinite(vertices).all():
        raise ValueError("Proxy mesh contains nonfinite vertices")
    if not o3d.io.write_triangle_mesh(str(output), mesh):
        raise OSError(f"Could not write {output}")
    atomic_json(path.parent / "mesh_report.json",
                dict(vertices=len(mesh.vertices), triangles=len(mesh.triangles),
                     method="provided mesh" if data["mesh"] else "LiDAR ball pivoting; long edges removed",
                     bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()]))
    return output


def cache_depths(preparation, upstream, setting):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    sys.path.insert(0, str(upstream))
    from Mesh2DepthHelper import DepthRenderer, Load_ply_resource, Build_Ply_Render_Camera_Parameters_colmap_correct
    path = Path(preparation)
    data = json.loads(path.read_text())
    mesh_path = build_proxy(path)
    output = path.parent / "depth"
    output.mkdir(exist_ok=True)
    settings = dict(preparation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    resolution=setting, mesh_size=mesh_path.stat().st_size,
                    mesh_mtime_ns=mesh_path.stat().st_mtime_ns)
    metadata = output / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text())["settings"] != settings:
        raise ValueError("Depth cache uses different cameras, mesh, or resolution")
    atomic_json(metadata, dict(settings=settings, complete=False))
    renderer = DepthRenderer(device="cuda")
    mesh = Load_ply_resource(str(mesh_path), "cuda")
    rows = []
    with torch.no_grad():
        for index, frame in enumerate(tqdm(data["frames"], desc="Proxy mesh depth")):
            with Image.open(frame["image_path"]) as image:
                original_width, original_height = image.size
            if abs(original_width / original_height - frame["width"] / frame["height"]) > 1e-3:
                raise ValueError(f"Image and calibration aspect ratios differ: {frame['name']}")
            w, h = resolution(original_width, original_height, setting)
            target = output / (frame["alias"] + ".npy")
            if target.exists():
                depth = np.load(target)
            else:
                parameters = Build_Ply_Render_Camera_Parameters_colmap_correct(
                    frame["fx"] * w/frame["width"], frame["fy"] * h/frame["height"],
                    w*.5, h*.5, w, h, .01, 100.,
                    np.array(frame["R"]).T, np.array(frame["T"]), "cuda")
                tensor, _ = renderer.render_depth_batched(mesh, parameters, flip_y=True,
                                    max_tris_per_pass=1_000_000, linear_depth=True, verbose=False)
                depth = tensor.cpu().numpy()
                temp = target.with_suffix(".tmp.npy")
                np.save(temp, depth)
                temp.replace(target)
            if depth.shape != (h, w) or np.isnan(depth).any() or np.any(depth <= 0):
                raise ValueError(f"Invalid cached depth for {frame['name']}")
            finite = np.isfinite(depth)
            rows.append(dict(name=frame["name"], split=frame["split"], covered_fraction=float(finite.mean())))
            if index % max(1, len(data["frames"])//12) == 0:
                if finite.any():
                    near, far = np.quantile(depth[finite], [.02, .98])
                    visual = np.where(finite, np.clip((far-depth)/max(far-near, 1e-6), 0, 1), 0)
                else:
                    visual = np.zeros_like(depth)
                visual = Image.fromarray((visual*255).astype(np.uint8)).convert("RGB")
                with Image.open(frame["image_path"]) as image:
                    original = image.convert("RGB").resize((w, h))
                panel = Image.new("RGB", (2*w, h))
                panel.paste(original, (0, 0)); panel.paste(visual, (w, 0))
                panel.thumbnail((1920, 1080))
                panel.save(output / (frame["alias"] + "_rgb_depth.jpg"), quality=93)
    if not any(r["covered_fraction"] > 0 for r in rows if r["split"] == "train"):
        raise ValueError("Proxy is invisible from every training camera; verify registration")
    atomic_json(metadata, dict(settings=settings, complete=True, frames=rows,
                mean_covered_fraction=sum(r["covered_fraction"] for r in rows)/len(rows)))
    return mesh_path, output


def proxy_renderer(spec, iteration):
    """Load official saved anchors/MLPs and retain proxy culling for evaluation."""
    import numpy as np
    import torch
    from PIL import Image
    meta = spec["metadata"]
    # The CPU planning helpers may have imported this fork's utils package.
    # This process renders one model only, so give upstream its own namespaces.
    for name in list(sys.modules):
        if name.split(".")[0] in ("utils", "scene", "gaussian_renderer", "arguments", "Mesh2DepthHelper"):
            del sys.modules[name]
    sys.path.insert(0, meta["upstream"])
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import prefilter_voxel, render
    from utils.camera_utils import loadCam
    cfg = spec["cfg"]
    model = GaussianModel(*(cfg[k] for k in (
        "feat_dim", "n_offsets", "fork", "use_feat_bank", "appearance_dim",
        "add_opacity_dist", "add_cov_dist", "add_color_dist", "add_level",
        "visible_threshold", "dist2level", "base_layer", "progressive", "extend")))
    model.set_appearance(len(spec["splits"]["train"]))
    checkpoint = spec["path"] / f"point_cloud/iteration_{iteration}"
    model.load_ply_sparse_gaussian(str(checkpoint / "point_cloud.ply"), None)
    model.load_mlp_checkpoints(str(checkpoint))
    model.eval()
    pipe = SimpleNamespace(compute_cov3D_python=False, debug=False)
    background = torch.tensor([1,1,1] if cfg.get("white_background", False) else [0,0,0],
                              dtype=torch.float32, device="cuda")
    args = SimpleNamespace(resolution=cfg.get("resolution", -1), data_device="cpu")
    def make_camera(record, index):
        with Image.open(record["image_path"]) as image:
            original = image.convert("RGB").copy()
        info = SimpleNamespace(uid=index, R=np.array(record["R"]), T=np.array(record["T"]),
                FovX=record["fov_x"], FovY=record["fov_y"], image=original,
                image_name=record["alias"], image_path=record["image_path"],
                width=record["width"], height=record["height"])
        return loadCam(args, index, info, 1.)
    def render_camera(camera):
        model.set_anchor_mask(camera.camera_center, iteration, camera.resolution_scale)
        depth = np.load(Path(meta["depth_dir"]) / (camera.image_name + ".npy"))
        visible, _ = prefilter_voxel(camera, model, pipe, background, depth_map=depth)
        if not visible.any():
            return background[:, None, None].expand(3, camera.image_height, camera.image_width)
        return render(camera, model, pipe, background, visible_mask=visible)["render"]
    return render_camera, make_camera
