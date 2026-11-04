"""Resolve the location of a scene's COLMAP sparse reconstruction."""

from pathlib import Path


def colmap_model_dir(source_path):
    """Return sparse/0 or sparse containing a complete image/camera model.

    When both locations have a complete model, use the original 3DGS
    convention (sparse/0). The point cloud is resolved in the same directory.
    """
    sparse = Path(source_path) / "sparse"
    for model in (sparse / "0", sparse):
        if any((model / ("images." + extension)).is_file() and
               (model / ("cameras." + extension)).is_file()
               for extension in ("bin", "txt")):
            return model
    raise FileNotFoundError(
        f"No COLMAP image/camera pair found in {sparse / '0'} or {sparse}; "
        "expected matching images.bin + cameras.bin or images.txt + cameras.txt"
    )
