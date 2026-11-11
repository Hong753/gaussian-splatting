"""CPU-only configuration shared by training, image evaluation, and videos."""
import hashlib
import json
from pathlib import Path

DEFAULTS = dict(occlusion_mode="off", occlusion_geometry="", occlusion_cache="",
                occlusion_margin=0.05, occlusion_sigma=3.0, occlusion_edge_pixels=1,
                occlusion_chunk_size=131072)


def add_arguments(parser, per_model=False):
    parser.add_argument("--occlusion", choices=["saved", "on", "off"],
                        nargs="+" if per_model else None, default=None)
    parser.add_argument("--occlusion-geometry", type=Path, help="Fixed LiDAR surfels (.npz) or scene mesh manifest (scene.json)")
    parser.add_argument("--occlusion-cache", type=Path)
    parser.add_argument("--occlusion-margin", type=float)


def settings(cfg):
    return {k: cfg.get(k, v) for k, v in DEFAULTS.items()}


def apply_overrides(spec, args, mode=None):
    saved = settings(spec["cfg"])
    current = dict(saved)
    mode = mode or getattr(args, "occlusion", None) or "saved"
    if mode != "saved":
        current["occlusion_mode"] = (saved['occlusion_mode'] if saved['occlusion_mode'] != 'off' else 'lidar') if mode == 'on' else 'off'
    for key in ("occlusion_geometry", "occlusion_cache", "occlusion_margin"):
        value = getattr(args, key, None)
        if value is not None:
            current[key] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    if mode == 'on' and Path(current['occlusion_geometry']).suffix == '.json':
        current['occlusion_mode'] = 'mesh'
    if current["occlusion_mode"] != "off" and spec["type"] != "3dgs":
        raise ValueError("External occlusion culling currently supports native 3DGS only")
    validate(current)
    spec["cfg"] = {**spec["cfg"], **current}
    # Separate overridden evaluations from the saved model's default protocol.
    comparable = lambda c: {k: v for k, v in c.items() if k != "occlusion_cache"}
    if comparable(current) != comparable(saved):
        suffix = hashlib.sha256(json.dumps(comparable(current), sort_keys=True).encode()).hexdigest()[:10]
        spec["render_variant"] = "occlusion_off" if current["occlusion_mode"] == "off" else "occlusion_on_" + suffix
    return spec


def validate(cfg):
    import math
    c = settings(cfg)
    if c["occlusion_mode"] not in ("off", "lidar", "mesh"):
        raise ValueError("occlusion_mode must be off, lidar, or mesh")
    if c["occlusion_mode"] == "off":
        return
    if not c["occlusion_geometry"] or not Path(c["occlusion_geometry"]).is_file():
        raise FileNotFoundError("Missing LiDAR occlusion geometry; native training prepares it automatically, "
                                "or use scripts/prepare_occlusion.py for an existing checkpoint")
    if not math.isfinite(c["occlusion_margin"]) or c["occlusion_margin"] < 0:
        raise ValueError("Occlusion margin must be finite and nonnegative (scene units)")
    if not math.isfinite(c["occlusion_sigma"]) or c["occlusion_sigma"] < 3:
        raise ValueError("Conservative occlusion support requires occlusion_sigma >= 3")
    if c["occlusion_edge_pixels"] < 0 or c["occlusion_chunk_size"] < 1:
        raise ValueError("Invalid occlusion edge padding or chunk size")
    if c['occlusion_mode'] == 'mesh':
        from utils.scene_occlusion import load_scene
        load_scene(c['occlusion_geometry'])


def configure_pipeline(pipe, cfg):
    values = settings(cfg if isinstance(cfg, dict) else vars(cfg))
    validate(values)
    for key, value in values.items():
        setattr(pipe, key, value)
    return pipe


def worker_arguments(args):
    result = []
    for key in ("occlusion", "occlusion_geometry", "occlusion_cache", "occlusion_margin"):
        value = getattr(args, key, None)
        if value is not None:
            result += ["--" + key.replace("_", "-"), str(value)]
    return result


def metric_directory(model, iteration, variant=""):
    root = Path(model) / "psnr_all_frames" / f"iteration_{iteration}"
    return root / variant if variant else root


def training_paths(dataset):
    """Set model-local asset paths before cfg_args is written."""
    if dataset.occlusion_mode == 'off':
        return
    if dataset.occlusion_mode not in ('lidar', 'mesh'):
        raise ValueError('occlusion_mode must be off, lidar, or mesh')
    if dataset.occlusion_mode == 'mesh' and not dataset.occlusion_geometry:
        raise ValueError('Mesh occlusion requires a prepared scene.json')
    model = Path(dataset.model_path).expanduser().resolve()
    dataset.occlusion_geometry = str(Path(dataset.occlusion_geometry).expanduser().resolve() if dataset.occlusion_geometry else model/'occlusion_geometry.npz')
    dataset.occlusion_cache = str(Path(dataset.occlusion_cache).expanduser().resolve() if dataset.occlusion_cache else model/'occlusion_depth_cache')


def prepare_training_geometry(dataset, cameras=None):
    if dataset.occlusion_mode == 'off':
        return
    if dataset.occlusion_mode == 'mesh':
        from utils.scene_occlusion import load_scene
        report, _ = load_scene(dataset.occlusion_geometry, Path(dataset.model_path)/'input.ply', dataset.source_path)
        selected = set(json.loads(Path(dataset.keyframe_manifest).read_text())['selected_images'])
        if set(report['training_images']) != selected:
            raise ValueError('Scene surface fusion and training keyframes differ')
        if cameras is not None:
            from utils.occlusion_depth import camera_signature
            signatures = report.get('camera_signatures', {})
            if any(signatures.get(camera.image_name) != camera_signature(camera, 1) for camera in cameras):
                raise ValueError('Training cameras differ from the fixed scene fusion cameras')
        return
    from utils.lidar_occlusion import prepare_geometry
    # Scene has already copied the actual initialization cloud to input.ply.
    points = dataset.occlusion_points or str(Path(dataset.model_path)/'input.ply')
    prepare_geometry(points, dataset.occlusion_geometry, voxel_size=dataset.occlusion_voxel_size,
                     max_radius=dataset.occlusion_max_radius, neighbors=dataset.occlusion_neighbors,
                     confidence=dataset.occlusion_confidence)
