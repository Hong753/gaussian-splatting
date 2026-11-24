"""Build a fixed world-space textured mesh from fitting views and render camera paths."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from scripts.indoor_triangles.world_texture import WorldMesh, bake
from scripts.indoor_triangles.world_tsdf import integrate, surface
from scripts.orange4f_photo_detail import Camera, DEFAULT_RUN, camera, detail_rmse, lowpass, read_rgb, score
from scripts.orange4f_stock_triangle import NOVEL_SEED, VIDEO_FPS, frame_output_names, read_colmap_scene
from scripts.render_model_comparison import trajectory_record, write_video
from scripts.render_novel_comparison import choose_source_segment, synthesize_random_path
from scripts.view_comparisons import atomic_json


def records(run):
    inputs = json.loads((run/"split_manifest.json").read_text())
    _, frames = read_colmap_scene(Path(inputs["settings"]["colmap_model"]))
    poses = {f.name: f for f in frames.values()}
    cameras = {r["name"]: camera(r, poses[r["name"]]) for r in inputs["frames"]}
    return inputs, cameras


def scene_directory(args):
    return (args.scene or args.run/"world_texture_v1").resolve()


def build(args):
    run = args.run.resolve()
    dest = scene_directory(args)
    inputs, cameras = records(run)
    priors = inputs["settings"].get("geometry_priors")
    if not priors:
        raise ValueError("A staged DA3-guided Orange4F run supplies the training-view depth cache")
    cache = Path(priors["cache"])
    digest = hashlib.sha256((cache/"manifest.json").read_bytes()).hexdigest()
    if digest != priors["cache_sha256"]:
        raise ValueError("Depth cache differs from the cache associated with the run")
    settings = dict(version=1, run=str(run), depth_manifest=digest,
                    split=hashlib.sha256((run/"split_manifest.json").read_bytes()).hexdigest(),
                    voxel=args.voxel, far=args.far, min_views=2)
    if dest.exists():
        if json.loads((dest/"scene.json").read_text())["settings"] != settings:
            raise ValueError(f"Different build settings at {dest}; choose another --scene")
        print(f"Using {dest / 'surface.obj'}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        out = Path(tmp)/"scene"
        out.mkdir()
        views = []
        centers = []
        names = []
        volume = {}
        fitting = [r for r in inputs["frames"] if r["split"] == "train"]
        for i, row in enumerate(fitting, 1):
            cam = cameras[row["name"]]
            prior = cache/"frames"/(Path(row["staged_name"]).stem+".npz")
            with np.load(prior, allow_pickle=False) as data:
                integrate(volume, data["depth"], data["valid"], data["intrinsic"],
                          cam.rotation, cam.center, args.voxel, args.far)
            views.append((cam, prior, run/"images"/row["staged_name"]))
            centers.append(cam.center)
            names.append(row["name"])
            if i % 50 == 0 or i == len(fitting):
                print(f"Fused {i}/{len(fitting)} fitting cameras, {len(volume)} voxel blocks", flush=True)
        vertices, faces = surface(volume, args.voxel, settings["min_views"])
        del volume
        print(f"Assigning fixed texture pages to {len(faces):,} triangles", flush=True)
        stats = bake(out, vertices, faces, views, args.voxel, args.far)
        stats["model_bytes"] = (out/"surface.npz").stat().st_size + sum(p.stat().st_size for p in (out/"textures").glob("*.png"))
        atomic_json(out/"scene.json", dict(settings=settings, training_frames=names,
                                            training_centers=np.asarray(centers).tolist(), **stats))
        out.rename(dest)
    print(f"{dest / 'surface.obj'}\n{json.dumps(stats)}", flush=True)


def path_camera(row):
    w, h = row["width"], row["height"]
    if not isinstance(w, int) or not isinstance(h, int) or min(w, h) < 2:
        raise ValueError("Camera width and height must be integers of at least two pixels")
    if "intrinsics" in row:
        fx, fy, cx, cy = row["intrinsics"]
    else:
        fx, fy = w/(2*math.tan(row["fov_x"]/2)), h/(2*math.tan(row["fov_y"]/2))
        cx, cy = (w-1)/2, (h-1)/2
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    r, t = np.asarray(row["R"], float), np.asarray(row["T"], float)
    if (not np.isfinite(k).all() or min(fx, fy) <= 0 or r.shape != (3, 3) or t.shape != (3,)
            or not np.isfinite(t).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-4)):
        raise ValueError("Invalid camera calibration or pose")
    return Camera(w, h, k, r.T, -r@t)


def default_path(run):
    existing = run/"random_trajectory.json"
    if existing.is_file():
        return json.loads(existing.read_text())
    _, cameras = records(run)
    views = [SimpleNamespace(image_name=name, width=c.width, height=c.height, R=c.rotation.T,
                             T=-c.rotation@c.center, FovX=2*math.atan(c.width/(2*c.k[0, 0])),
                             FovY=2*math.atan(c.height/(2*c.k[1, 1])))
             for name, c in cameras.items()]
    segment, selection = choose_source_segment(views, window=min(600, len(views)), seed=NOVEL_SEED)
    generated, details = synthesize_random_path(segment, views, VIDEO_FPS*40, deviation=2., seed=NOVEL_SEED)
    return dict(fps=VIDEO_FPS, selection=selection, synthesis=details, cameras=trajectory_record(generated))


def render_path(args):
    directory = scene_directory(args)
    path = json.loads(args.trajectory.read_text()) if args.trajectory else default_path(args.run)
    cameras = [path_camera(c) for c in path["cameras"]]
    if not cameras:
        raise ValueError("Empty trajectory")
    w, h = cameras[0].width, cameras[0].height
    if any((c.width, c.height) != (w, h) for c in cameras):
        raise ValueError("All frames of an MP4 path must have the same dimensions")
    fps = float(path.get("fps", VIDEO_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be positive")
    path_hash = hashlib.sha256(json.dumps(path, sort_keys=True).encode()).hexdigest()[:12]
    out = directory/("path_"+path_hash)
    if out.exists():
        print(out/"render.mp4")
        return
    mesh = WorldMesh(directory)
    tree = cKDTree(json.loads((directory/"scene.json").read_text())["training_centers"])
    with TemporaryDirectory(dir=directory) as tmp:
        stage = Path(tmp)/"path"
        stage.mkdir()
        rows = []
        def frames():
            for i, cam in enumerate(cameras):
                rgb, hit, covered = mesh.render(cam)
                rows.append(dict(frame=i, nearest_fit=float(tree.query(cam.center)[0]),
                                 hit=float(hit.mean()), textured=float(covered.mean())))
                if i % 50 == 0:
                    print(f"Novel path {i+1}/{len(cameras)} · textured {covered.mean():.1%}", flush=True)
                if len(cameras) == 1:
                    Image.fromarray(rgb).save(stage/"render.png")
                yield np.pad(rgb, ((0, h%2), (0, w%2), (0, 0)), mode="edge")
        write_video(stage/"render.mp4", frames(), w+w%2, h+h%2, fps, 18)
        atomic_json(stage/"trajectory.json", path)
        atomic_json(stage/"coverage.json", rows)
        stage.rename(out)
    print(out/"render.mp4")


def evaluate(args):
    directory = scene_directory(args)
    inputs, cameras = records(args.run)
    rows = [r for r in inputs["frames"] if r["split"] == "test"]
    scene = json.loads((directory/"scene.json").read_text())
    if scene["settings"]["split"] != hashlib.sha256((args.run/"split_manifest.json").read_bytes()).hexdigest():
        raise ValueError("Evaluation split differs from the scene build")
    tree = cKDTree(scene["training_centers"])
    by_distance = sorted(rows, key=lambda r: tree.query(cameras[r["name"]].center)[0])
    # Span the distribution of translations before reading any held-out photo.
    count = len(rows) if args.all_held_out else min(16, len(rows))
    if not count:
        raise ValueError("No held-out capture views")
    selected = [by_distance[i] for i in np.linspace(0, len(rows)-1, count).round().astype(int)]
    output = directory/f"held_out_{count}"
    if output.exists():
        print(output/"scores.json")
        return
    mesh = WorldMesh(directory)
    names = frame_output_names(inputs["frames"])
    scores = []
    with TemporaryDirectory(dir=directory) as tmp:
        stage = Path(tmp)/"eval"
        stage.mkdir()
        for row in selected:
            name = row["name"]
            cam = cameras[name]
            rgb, hit, covered = mesh.render(cam)
            gt = read_rgb(args.run/"images"/row["staged_name"])
            target_detail = gt-lowpass(gt)
            record = dict(name=name, nearest_fit=float(tree.query(cam.center)[0]),
                          detail_rmse=detail_rmse(rgb, target_detail),
                          textured_psnr=score(gt[covered], rgb[covered]) if covered.any() else None,
                          textured=float(covered.mean()), hit=float(hit.mean()), full_psnr=score(gt, rgb))
            panels = [gt]
            baseline = args.run/"frames"/names[name]
            if baseline.is_file():
                stock = read_rgb(baseline)
                record["triangle_psnr"] = score(gt, stock)
                record["triangle_detail_rmse"] = detail_rmse(stock, target_detail)
                panels.append(stock)
            panels.append(rgb)
            Image.fromarray(np.concatenate(panels, axis=1)).save(stage/(Path(names[name]).stem+".jpg"), quality=95)
            scores.append(record)
        atomic_json(stage/"scores.json", dict(views=scores, mean_full_psnr=float(np.mean([r["full_psnr"] for r in scores]))))
        stage.rename(output)
    print(output/"scores.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "render", "evaluate", "run"], nargs="?", default="run")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--scene", type=Path, help="Standalone mesh package, default RUN/world_texture_v1")
    parser.add_argument("--trajectory", type=Path, help="Camera-path JSON. Omit to use the Orange4F random path.")
    parser.add_argument("--voxel", type=float, default=.04, help="TSDF grid spacing in pose units")
    parser.add_argument("--far", type=float, default=12., help="Maximum fitting depth in pose units")
    parser.add_argument("--all-held-out", action="store_true", help="Evaluate every held-out registered camera")
    args = parser.parse_args()
    if not 0 < args.voxel < args.far or not np.isfinite(args.far):
        parser.error("Require finite 0 < voxel < far")
    if args.action in ("build", "run"):
        build(args)
    if args.action in ("evaluate", "run"):
        evaluate(args)
    if args.action in ("render", "run"):
        render_path(args)


if __name__ == "__main__":
    main()
