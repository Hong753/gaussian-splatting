"""Use the shared preview writer with upstream Triangle Splatting cameras."""

import json
from pathlib import Path
from types import SimpleNamespace

from scene.dataset_readers import readColmapCameras, read_extrinsics_text, read_intrinsics_text
from utils.camera_utils import cameraList_from_camInfos
from utils.training_previews import TrainingPreviewer, archive_later_previews


def create_triangle_previewer(path, scene, dataset, start_iteration):
    if path is None:
        return None
    path = Path(path)
    selection = json.loads((path / "selection.json").read_text())
    train = {camera.image_name: camera for camera in scene.getTrainCameras()}
    heldout = {}
    if any(row["split"] == "held-out" for row in selection["frames"]):
        sparse = path / "sparse/0"
        infos = readColmapCameras(read_extrinsics_text(sparse / "images.txt"),
                                  read_intrinsics_text(sparse / "cameras.txt"), path / "images")
        try:
            options = SimpleNamespace(resolution=dataset.resolution, data_device="cpu")
            heldout = {c.image_name: c for c in cameraList_from_camInfos(infos, 1., options)}
        finally:
            for info in infos:
                info.image.close()
    frames = []
    for row in selection["frames"]:
        cameras = train if row["split"] == "train" else heldout
        camera = cameras[Path(row["staged_name"]).stem]
        view = SimpleNamespace(camera=camera, original_image=camera.original_image, image_name=row["name"])
        frames.append((row["split"], view))

    root = Path(scene.model_path) / "debug_previews"
    archive_later_previews(root, start_iteration)
    metadata = dict(heldout_protocol="Original held-out cameras from the Orange4F split",
                    state_timing="before optimizer step and densification, aligned with the upstream checkpoint")
    preview = TrainingPreviewer(root, frames, selection["config"], metadata)
    print(f"Live previews: {len(frames)} fixed frames, every {preview.config['every']} steps. {root / 'index.html'}",
          flush=True)
    return preview


def capture_triangle_preview(preview, iteration, triangles, render, pipe, background, tb_writer):
    def render_adapter(view, model, pipeline, bg, **_):
        return render(view.camera, model, pipeline, bg)

    preview.capture(iteration, triangles, render_adapter, pipe, background, tb_writer=tb_writer)
