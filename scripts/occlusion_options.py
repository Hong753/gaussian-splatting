"""CPU-only configuration shared by training, image evaluation, and videos."""
import hashlib
import json
from pathlib import Path

DEFAULTS = dict(occlusion_mode="off", occlusion_mesh="", occlusion_cache="",
                occlusion_margin=0.05, occlusion_sigma=3.0, occlusion_edge_pixels=1,
                occlusion_chunk_size=131072)


def add_arguments(parser, per_model=False):
    parser.add_argument("--occlusion", choices=["saved", "on", "off"],
                        nargs="+" if per_model else None, default=None)
    parser.add_argument("--occlusion-mesh", type=Path)
    parser.add_argument("--occlusion-cache", type=Path)
    parser.add_argument("--occlusion-margin", type=float)


def settings(cfg):
    return {k: cfg.get(k, v) for k, v in DEFAULTS.items()}


def apply_overrides(spec, args, mode=None):
    saved = settings(spec["cfg"])
    current = dict(saved)
    mode = mode or getattr(args, "occlusion", None) or "saved"
    if mode != "saved":
        current["occlusion_mode"] = "proxy" if mode == "on" else "off"
    for key in ("occlusion_mesh", "occlusion_cache", "occlusion_margin"):
        value = getattr(args, key, None)
        if value is not None:
            current[key] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
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
    if c["occlusion_mode"] not in ("off", "proxy"):
        raise ValueError("occlusion_mode must be off or proxy")
    if c["occlusion_mode"] == "off":
        return
    if not c["occlusion_mesh"] or not Path(c["occlusion_mesh"]).is_file():
        raise FileNotFoundError("Occlusion culling requires an existing registered triangle mesh (--occlusion_mesh)")
    if not math.isfinite(c["occlusion_margin"]) or c["occlusion_margin"] < 0:
        raise ValueError("Occlusion margin must be finite and nonnegative (scene units)")
    if not math.isfinite(c["occlusion_sigma"]) or c["occlusion_sigma"] < 3:
        raise ValueError("Conservative occlusion support requires occlusion_sigma >= 3")
    if c["occlusion_edge_pixels"] < 0 or c["occlusion_chunk_size"] < 1:
        raise ValueError("Invalid occlusion edge padding or chunk size")


def configure_pipeline(pipe, cfg):
    values = settings(cfg if isinstance(cfg, dict) else vars(cfg))
    validate(values)
    for key, value in values.items():
        setattr(pipe, key, value)
    return pipe


def worker_arguments(args):
    result = []
    for key in ("occlusion", "occlusion_mesh", "occlusion_cache", "occlusion_margin"):
        value = getattr(args, key, None)
        if value is not None:
            result += ["--" + key.replace("_", "-"), str(value)]
    return result


def metric_directory(model, iteration, variant=""):
    root = Path(model) / "psnr_all_frames" / f"iteration_{iteration}"
    return root / variant if variant else root
