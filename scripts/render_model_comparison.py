#!/usr/bin/env python3
"""Render 3DGS and Scaffold-GS along exactly the same registered COLMAP cameras.

Original images are read one at a time. Both renders use a shared camera
manifest and are accompanied by a three-panel video: input | 3DGS | Scaffold.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_all_frames import read_config, choose_iteration
from scripts.keyframes import natural_key


def model_spec(path, expected_type, iteration, source):
    path = Path(path).expanduser().resolve()
    cfg = read_config(path / "cfg_args")
    actual = cfg.get("model_type", "3dgs")
    if actual != expected_type:
        raise ValueError(f"{path} is model type {actual!r}, expected {expected_type!r}")
    saved_source = cfg.get("source_path")
    if not saved_source or Path(saved_source).expanduser().resolve() != source:
        raise ValueError(f"{path}: saved source {saved_source!r} does not match {source}; "
                         "camera coordinate systems must match")
    saved_iteration, ply = choose_iteration(path, iteration, actual)
    if expected_type == "scaffold":
        missing = [name for name in ("opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt")
                   if not (ply.parent / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Scaffold MLPs in {ply.parent}: {', '.join(missing)}")
    return {"path": path, "cfg": cfg, "iteration": saved_iteration, "ply": ply, "type": actual}


def make_trajectory(cameras, start, stride, max_frames):
    if start < 0 or stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("start must be nonnegative; stride and max-frames must be positive")
    names = [camera.image_name for camera in cameras]
    if len(names) != len(set(names)):
        raise ValueError("COLMAP contains duplicate image names")
    ordered = sorted(cameras, key=lambda camera: natural_key(camera.image_name))
    chosen = ordered[start::stride]
    if max_frames is not None:
        chosen = chosen[:max_frames]
    if not chosen:
        raise ValueError("No camera frames selected")
    # A single output video must keep its field-of-view and aspect well-defined.
    aspect = chosen[0].width / chosen[0].height
    if any(not math.isclose(c.width / c.height, aspect, rel_tol=1e-4) for c in chosen):
        raise ValueError("COLMAP cameras have different aspect ratios; use a same-aspect camera subset")
    if any(not all(math.isfinite(float(value)) for value in (c.FovX, c.FovY))
           or c.FovX <= 0 or c.FovY <= 0 for c in chosen):
        raise ValueError("Invalid COLMAP field of view")
    return chosen


def trajectory_record(cameras):
    return [{"image_name": c.image_name, "width": int(c.width), "height": int(c.height),
             "fov_x": float(c.FovX), "fov_y": float(c.FovY),
             "R": c.R.tolist(), "T": c.T.tolist()} for c in cameras]


def video_dimensions(cameras, width):
    if width < 2 or width % 2:
        raise ValueError("--width must be a positive even integer of at least 2 pixels")
    height = max(2, 2 * round(width * cameras[0].height / cameras[0].width / 2))
    return width, height


def ffmpeg_command(path, width, height, fps, crf):
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]


def write_video(path, frames, width, height, fps, crf):
    """Stream frames to a temporary MP4 and publish it only after encoding succeeds."""
    temp = path.with_name(path.stem + ".partial.mp4")
    proc = subprocess.Popen(ffmpeg_command(temp, width, height, fps, crf), stdin=subprocess.PIPE)
    try:
        for image in frames:
            if image.shape != (height, width, 3) or str(image.dtype) != "uint8":
                raise ValueError(f"Invalid frame shape/dtype {image.shape}/{image.dtype}")
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        proc.stdin = None
        code = proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed for {path.name} (exit {code})")
        os.replace(temp, path)
    except BaseException:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
        temp.unlink(missing_ok=True)
        raise


def compare_side_by_side(original, gs3d, scaffold, output, crf):
    temp = output.with_name(output.stem + ".partial.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(original), "-i", str(gs3d), "-i", str(scaffold),
           "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]", "-an",
           "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)]
    try:
        subprocess.run(cmd, check=True)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def validate_originals(images, cameras):
    """Fail before model rendering when a matching input image is missing."""
    missing = [str(images / info.image_name) for info in cameras if not (images / info.image_name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input frames are missing; first: {missing[0]}")


def original_frames(images, cameras, width, height):
    """Read matching RGB images in the exact camera order, one at a time."""
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    for info in tqdm(cameras, desc="Encoding original images", unit="frame"):
        path = images / info.image_name
        try:
            with Image.open(path) as image:
                yield np.asarray(image.convert("RGB").resize((width, height)), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read original image {path}: {exc}") from exc


def read_cameras(source):
    from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                     read_extrinsics_text, read_intrinsics_text)
    from scene.dataset_readers import readColmapCameras
    from utils.colmap_paths import colmap_model_dir
    model = Path(colmap_model_dir(source))
    if (model / "images.bin").is_file() and (model / "cameras.bin").is_file():
        images = read_extrinsics_binary(str(model / "images.bin"))
        cameras = read_intrinsics_binary(str(model / "cameras.bin"))
    else:
        images = read_extrinsics_text(str(model / "images.txt"))
        cameras = read_intrinsics_text(str(model / "cameras.txt"))
    result = readColmapCameras(images, cameras, None, str(source / "images"), "", [])
    del images, cameras
    return result


def load_renderer(spec, aa):
    if spec["type"] == "proxygs":
        if aa:
            raise ValueError("Proxy-GS does not support Graphdeco anti-aliasing")
        from scripts.proxygs_support import proxy_renderer
        return proxy_renderer(spec, spec["iteration"], dynamic_depth=True)[0]
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render as render_3dgs
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        separate_sh = True
    except ImportError:
        separate_sh = False
    cfg = spec["cfg"]
    if spec["type"] == "scaffold":
        from models.scaffold_gaussian import ScaffoldGaussianModel
        from models.scaffold_renderer import render as render_scaffold, prefilter_voxel
        gaussians = ScaffoldGaussianModel(feat_dim=cfg.get("scaffold_feat_dim", 32),
                        n_offsets=cfg.get("scaffold_n_offsets", 10),
                        voxel_size=cfg.get("scaffold_voxel_size", 0.001),
                        update_init_factor=16, appearance_dim=0)
        gaussians.load_ply(str(spec["ply"]))
        gaussians.eval()
    elif spec["type"] == "textured":
        if aa:
            raise ValueError("Textured Gaussians does not support Graphdeco anti-aliasing")
        from models.textured_gaussian import TexturedGaussianModel
        gaussians = TexturedGaussianModel.load(spec["ply"])
    else:
        gaussians = GaussianModel(cfg.get("sh_degree", 3))
        gaussians.load_ply(str(spec["ply"]))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=False, antialiasing=aa)
    from scripts.occlusion_options import configure_pipeline
    configure_pipeline(pipe, cfg)
    bg_color = [1, 1, 1] if cfg.get("white_background", False) else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def render(cam):
        if spec["type"] == "scaffold":
            visible = prefilter_voxel(cam, gaussians, pipe, bg)
            if not bool(visible.any()):
                return bg[:, None, None].expand(3, cam.image_height, cam.image_width)
            return render_scaffold(cam, gaussians, pipe, bg, visible_mask=visible)["render"]
        if spec["type"] == "textured":
            return gaussians.render(cam, bg)[0]
        return render_3dgs(cam, gaussians, pipe, bg, separate_sh=separate_sh,
                           use_trained_exp=False)["render"]
    return render


def rendered_frames(spec, cameras, width, height, aa):
    import torch
    from tqdm import tqdm
    render = load_renderer(spec, aa)
    from scene.cameras import MiniCam
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    try:
        with torch.no_grad():
            for info in tqdm(cameras, desc=f"Rendering {spec['type']}", unit="frame"):
                world_view = torch.tensor(getWorld2View2(info.R, info.T), dtype=torch.float32,
                                          device="cuda").transpose(0, 1)
                projection = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=info.FovX,
                                                 fovY=info.FovY).transpose(0, 1).cuda()
                cam = MiniCam(width, height, info.FovY, info.FovX, 0.01, 100.0,
                              world_view, world_view @ projection)
                # The texture renderer needs the original COLMAP pose to form
                # its world-to-camera matrix; MiniCam normally stores only GPU transforms.
                cam.R, cam.T = info.R, info.T
                cam.image_name, cam.uid, cam.resolution_scale = info.image_name, 0, 1.0
                cam.Fx, cam.Fy = width / (2 * math.tan(info.FovX / 2)), height / (2 * math.tan(info.FovY / 2))
                cam.Cx, cam.Cy = width * .5, height * .5
                cam.projection_matrix = projection
                image = render(cam).clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
                yield image
    finally:
        del render
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--images", default="images", help="Input image folder within --source, or absolute path")
    parser.add_argument("--model-3dgs", required=True, type=Path)
    parser.add_argument("--model-scaffold", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=30000, help="Use same iteration for both; -1 = latest in each")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--3dgs-aa", choices=["on", "off"], required=True, dest="aa_3dgs")
    parser.add_argument("--scaffold-aa", choices=["on", "off"], required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--start", type=int, default=0, help="Starting index in naturally sorted COLMAP images")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth registered camera")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit after start/stride")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality; lower is higher quality")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0 or not 0 <= args.crf <= 51:
        parser.error("--fps must be positive and --crf must be between 0 and 51")
    if args.iteration != -1 and args.iteration < 1:
        parser.error("--iteration must be positive or -1")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required (including an H.264 libx264 encoder)")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Scene not found: {source}")
    output = args.output_dir.expanduser().resolve()
    original_video = output / "original.mp4"
    gs_video = output / "3dgs.mp4"
    scaffold_video = output / "scaffold.mp4"
    comparison_video = output / "side_by_side.mp4"
    trajectory_file = output / "trajectory.json"
    output_files = [original_video, gs_video, scaffold_video, comparison_video, trajectory_file]
    if not args.overwrite and any(path.exists() for path in output_files):
        parser.error(f"Comparison outputs already exist in {output}; choose a new folder or use --overwrite")
    try:
        specs = [model_spec(args.model_3dgs, "3dgs", args.iteration, source),
                 model_spec(args.model_scaffold, "scaffold", args.iteration, source)]
        if specs[0]["cfg"].get("white_background", False) != specs[1]["cfg"].get("white_background", False):
            raise ValueError("The runs use different background colors; comparisons require the same background")
        cameras = make_trajectory(read_cameras(source), args.start, args.stride, args.max_frames)
        width, height = video_dimensions(cameras, args.width)
        images = (source / Path(args.images).expanduser()).resolve()
        validate_originals(images, cameras)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    records = trajectory_record(cameras)
    trajectory_hash = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    details = {"source": str(source), "ordering": "natural sort of COLMAP image names",
               "start_index": args.start, "stride": args.stride, "frame_count": len(records),
               "fps": args.fps, "width": width, "height": height, "trajectory_sha256": trajectory_hash,
               "color_protocol": "base model appearance; no per-image exposure transforms",
               "original_images": str(images),
               "side_by_side": {"left": "original", "middle": "3dgs", "right": "scaffold"},
               "models": {s["type"]: {"path": str(s["path"]), "iteration": s["iteration"],
                          "antialiasing": aa == "on"} for s, aa in zip(specs, [args.aa_3dgs, args.scaffold_aa])},
               "cameras": records}
    trajectory_file.write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    print(f"Rendering {len(cameras)} frames at {width}x{height}, {args.fps:g} fps on GPU {args.gpu}", flush=True)
    write_video(original_video, original_frames(images, cameras, width, height), width, height, args.fps, args.crf)
    for spec, aa, destination in zip(specs, [args.aa_3dgs, args.scaffold_aa], [gs_video, scaffold_video]):
        frames = rendered_frames(spec, cameras, width, height, aa == "on")
        write_video(destination, frames, width, height, args.fps, args.crf)
    compare_side_by_side(original_video, gs_video, scaffold_video, comparison_video, args.crf)
    for path in output_files:
        print(path)


if __name__ == "__main__":
    main()


