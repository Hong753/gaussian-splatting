#!/usr/bin/env python3
"""Browse and view saved Scaffold-GS models on localhost.

The file picker browses the viewer machine's filesystem; it does not upload PLYs.
The selected point_cloud.ply must have the Scaffold MLP files beside it.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_all_frames import read_config

MLP_FILES = ("opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt")


def model_paths(ply):
    ply = Path(ply).expanduser().resolve()
    if not ply.is_file() or ply.suffix.lower() != ".ply":
        raise ValueError(f"Select an existing Scaffold .ply file: {ply}")
    missing = [name for name in MLP_FILES if not (ply.parent / name).is_file()]
    if missing:
        raise ValueError("Scaffold MLP files are missing beside the PLY: " + ", ".join(missing))
    root = next((p for p in ply.parents if (p / "cfg_args").is_file()), None)
    if root is None:
        raise ValueError("Could not find cfg_args in the model's parent directories")
    cfg = read_config(root / "cfg_args")
    if cfg.get("model_type") != "scaffold":
        raise ValueError("The selected run is not marked as a Scaffold model in cfg_args")
    return ply, root, cfg


def first_camera(path):
    """Get the initial OpenCV camera-to-world pose saved by camera_to_JSON."""
    data = json.loads(Path(path).read_text())
    if not data:
        raise ValueError("cameras.json contains no camera poses")
    camera = data[0]
    pos = np.asarray(camera["position"], dtype=float)
    rot = np.asarray(camera["rotation"], dtype=float)
    if pos.shape != (3,) or rot.shape != (3, 3) or not np.isfinite(pos).all() or not np.isfinite(rot).all():
        raise ValueError("Invalid first camera pose in cameras.json")
    fy, height = float(camera["fy"]), float(camera["height"])
    if fy <= 0 or height <= 0:
        raise ValueError("Invalid camera intrinsics in cameras.json")
    fovy = 2 * math.atan(height / (2 * fy))
    # A look-at distance in the scale of the training camera trajectory.
    positions = np.asarray([c["position"] for c in data if "position" in c], dtype=float)
    distance = float(np.median(np.linalg.norm(positions - pos, axis=1))) if len(positions) else 1.0
    if not math.isfinite(distance) or distance < 1e-4:
        distance = 1.0
    return pos, rot, fovy, distance


def camera_matrices(wxyz, position, fovy, width, height):
    """Convert Viser's OpenCV camera-to-world pose to 3DGS matrices."""
    import viser.transforms as vtf
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = vtf.SO3(np.asarray(wxyz)).as_matrix()
    c2w[:3, 3] = position
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    return np.linalg.inv(c2w).astype(np.float32), fovx


class ScaffoldBackend:
    def __init__(self, ply, cfg):
        import torch
        from models.scaffold_gaussian import ScaffoldGaussianModel
        from models.scaffold_renderer import prefilter_voxel, render
        from scene.cameras import MiniCam
        from utils.graphics_utils import getProjectionMatrix
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to render Scaffold-GS on this machine")
        self.torch, self.render, self.prefilter = torch, render, prefilter_voxel
        self.MiniCam, self.projection = MiniCam, getProjectionMatrix
        self.model = ScaffoldGaussianModel(feat_dim=cfg.get("scaffold_feat_dim", 32),
                    n_offsets=cfg.get("scaffold_n_offsets", 10),
                    voxel_size=cfg.get("scaffold_voxel_size", 0.001),
                    update_init_factor=16, appearance_dim=0)
        self.model.load_ply(str(ply))
        self.model.eval()
        self.background = torch.tensor([1, 1, 1] if cfg.get("white_background", False) else [0, 0, 0],
                                       dtype=torch.float32, device="cuda")

    def render_image(self, wxyz, pos, fovy, width, height, antialiasing):
        torch = self.torch
        w2c, fovx = camera_matrices(wxyz, pos, fovy, width, height)
        with torch.no_grad():
            view = torch.as_tensor(w2c, device="cuda").T.contiguous()
            proj = self.projection(0.01, 100.0, fovx, fovy).T.cuda()
            cam = self.MiniCam(width, height, fovy, fovx, 0.01, 100.0, view, view @ proj)
            pipe = SimpleNamespace(debug=False, antialiasing=antialiasing, compute_cov3D_python=False)
            visible = self.prefilter(cam, self.model, pipe, self.background)
            if not bool(visible.any()):
                bg = np.rint(self.background.cpu().numpy() * 255).astype(np.uint8)
                return np.broadcast_to(bg, (height, width, 3)).copy()
            rgb = self.render(cam, self.model, pipe, self.background, visible_mask=visible)["render"]
            return (rgb.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy())

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


class Viewer:
    def __init__(self, server, start_dir, backend_factory=ScaffoldBackend):
        self.server, self.backend_factory = server, backend_factory
        self.backend = None
        self.generation = 0
        self.model_lock = threading.RLock()
        self.work_lock = threading.Lock()
        self.pending = {}
        self.seq = {}
        self.wakeup = threading.Event()
        self.last_camera = None

        gui = server.gui
        gui.add_markdown("## Scaffold-GS Viewer\nBrowse the viewer machine's files and open a saved Scaffold PLY.")
        self.directory = gui.add_text("Folder", str(Path(start_dir).expanduser().resolve()))
        self.entries = gui.add_dropdown("Folder / PLY", ("(empty)",), initial_value="(empty)")
        self.open_button = gui.add_button("Open selected")
        self.parent_button = gui.add_button("Parent folder")
        self.refresh_button = gui.add_button("Refresh folder")
        self.path_text = gui.add_text("Or enter PLY path", "", hint="Path on the machine running this viewer")
        self.load_button = gui.add_button("Load PLY path")
        self.status = gui.add_markdown("Choose a `point_cloud.ply` to begin.")
        self.resolution = gui.add_slider("Render width", min=320, max=1920, step=32, initial_value=960)
        self.antialiasing = gui.add_checkbox("Anti-aliasing", initial_value=True,
                                            hint="Match the setting used when training the run")
        self.reset_camera = gui.add_button("Go to first training camera")
        self.gui_help = gui.add_markdown("Drag to orbit; scroll to zoom. This viewer renders on the GPU.\n\n"
                         "Select a saved PLY with the three MLP files in the same folder.")

        @self.open_button.on_click
        def _open(_):
            chosen = self.entries.value
            if chosen == "(empty)":
                return
            target = Path(self.directory.value).expanduser() / chosen.rstrip("/")
            if target.is_dir():
                self.directory.value = str(target.resolve())
                self.refresh()
            else:
                self.path_text.value = str(target.resolve())
                self.load(target)

        @self.parent_button.on_click
        def _parent(_):
            self.directory.value = str(Path(self.directory.value).expanduser().resolve().parent)
            self.refresh()

        @self.refresh_button.on_click
        def _refresh(_):
            self.refresh()

        @self.load_button.on_click
        def _load(_):
            self.load(Path(self.path_text.value).expanduser())

        @self.resolution.on_update
        @self.antialiasing.on_update
        def _settings(_):
            self.schedule_all()

        @self.reset_camera.on_click
        def _reset(event):
            if self.last_camera is not None and event.client is not None:
                self.set_initial_camera(event.client)

        @server.on_client_connect
        def _connect(client):
            if self.last_camera is not None:
                self.set_initial_camera(client)
            @client.camera.on_update
            def _camera(_):
                self.schedule(client)
            self.schedule(client)

        @server.on_client_disconnect
        def _disconnect(client):
            with self.work_lock:
                self.pending.pop(client.client_id, None)
                self.seq.pop(client.client_id, None)

        self.refresh()
        threading.Thread(target=self.worker, name="scaffold-renderer", daemon=True).start()

    def refresh(self):
        try:
            path = Path(self.directory.value).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"Not a folder: {path}")
            children = list(path.iterdir())
            folders = sorted((p.name + "/" for p in children if p.is_dir() and not p.name.startswith(".")), key=str.casefold)
            plys = sorted((p.name for p in children if p.is_file() and p.suffix.lower() == ".ply"), key=str.casefold)
            options = folders + plys or ["(empty)"]
            self.entries.options = options
            self.entries.value = options[0]
            self.directory.value = str(path)
        except (OSError, ValueError) as exc:
            self.status.content = f"**Folder error:** {exc}"

    def load(self, selected):
        try:
            ply, root, cfg = model_paths(selected)
            cameras = first_camera(root / "cameras.json")
            self.status.content = f"Loading `{ply}` on the GPU…"
            with self.model_lock:
                if self.backend is not None:
                    self.backend.close()
                    self.backend = None
                self.generation += 1
                self.last_camera = None
                self.backend = self.backend_factory(ply, cfg)
                self.last_camera = cameras
            for client in self.server.get_clients().values():
                self.set_initial_camera(client)
            self.status.content = f"**Loaded:** `{ply}`\n\nAA is controlled below. Showing base appearance."
            self.schedule_all()
        except Exception as exc:
            self.status.content = f"**Load failed:** {type(exc).__name__}: {exc}"
            print(f"Load failed: {type(exc).__name__}: {exc}", flush=True)

    def set_initial_camera(self, client):
        pos, rot, fovy, distance = self.last_camera
        client.camera.position = pos
        client.camera.up_direction = -rot[:, 1]
        client.camera.look_at = pos + rot[:, 2] * distance
        client.camera.fov = fovy

    def schedule_all(self):
        for client in self.server.get_clients().values():
            self.schedule(client)

    def schedule(self, client):
        try:
            camera = client.camera
            aspect = float(camera.aspect)
            if not math.isfinite(aspect) or aspect <= 0:
                return
            width = int(self.resolution.value)
            height = max(1, min(2160, round(width / aspect)))
            task = (client, np.asarray(camera.wxyz).copy(), np.asarray(camera.position).copy(),
                    float(camera.fov), width, height, bool(self.antialiasing.value), self.generation)
            with self.work_lock:
                client_id = client.client_id
                self.seq[client_id] = self.seq.get(client_id, 0) + 1
                self.pending[client_id] = (self.seq[client_id], task)
                self.wakeup.set()
        except (AttributeError, ValueError):
            return

    def worker(self):
        while True:
            self.wakeup.wait()
            with self.work_lock:
                tasks = list(self.pending.items())
                self.pending.clear()
                self.wakeup.clear()
            for client_id, (seq, task) in tasks:
                client, wxyz, pos, fov, width, height, aa, generation = task
                try:
                    with self.model_lock:
                        if self.backend is None or generation != self.generation:
                            continue
                        image = self.backend.render_image(wxyz, pos, fov, width, height, aa)
                    with self.work_lock:
                        current = self.seq.get(client_id) == seq
                    if current and generation == self.generation:
                        client.scene.set_background_image(image, format="jpeg", jpeg_quality=85)
                except Exception as exc:
                    self.status.content = f"**Render failed:** {type(exc).__name__}: {exc}"
                    print(f"Render failed: {type(exc).__name__}: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="CUDA GPU index")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--browse-root", type=Path, default=Path("/data/3dgs_outputs"), help="Initial folder in the file picker")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not args.browse_root.expanduser().is_dir():
        parser.error(f"Initial folder does not exist: {args.browse_root}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        import viser
    except ImportError:
        parser.exit(1, "Install the viewer dependency: python -m pip install 'viser==1.0.30'\n")
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="Scaffold-GS Viewer")
    Viewer(server, args.browse_root)
    print(f"Open http://127.0.0.1:{server.get_port()} in a browser on this machine.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
