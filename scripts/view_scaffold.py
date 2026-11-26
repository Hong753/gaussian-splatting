#!/usr/bin/env python3
"""Explore saved Scaffold-GS or native 3DGS checkpoints in a browser.

Open a native run with --native-model or browse Scaffold runs with --suite.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_all_frames import read_config
from scripts.scaffold_viewer_render import MODES, NativeBackend, ScaffoldBackend, UpstreamBackend, camera_matrices

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


def camera_pose(camera, distance):
    pos = np.asarray(camera["position"], dtype=float)
    rot = np.asarray(camera["rotation"], dtype=float)
    if pos.shape != (3,) or rot.shape != (3, 3) or not np.isfinite(pos).all() or not np.isfinite(rot).all():
        raise ValueError("Invalid camera pose in cameras.json")
    fy, height = float(camera["fy"]), float(camera["height"])
    if not (fy > 0 and height > 0):
        raise ValueError("Invalid camera intrinsics in cameras.json")
    return pos, rot, 2*math.atan(height/(2*fy)), distance


def saved_cameras(path):
    data = json.loads(Path(path).read_text())
    if not data:
        raise ValueError("cameras.json contains no poses")
    positions = np.array([c["position"] for c in data], dtype=float)
    distance = float(np.median(np.linalg.norm(positions - positions[0], axis=1)))
    if not np.isfinite(distance) or distance < 1e-4:
        distance = 1.
    return {f'{i+1} · {c.get("img_name", "camera")}': camera_pose(c, distance) for i,c in enumerate(data)}


def first_camera(path):
    return next(iter(saved_cameras(path).values()))


def discover_checkpoints(suite):
    """Discover model directories and complete iteration bundles, newest step first."""
    runs = {}
    for root in sorted(suite.iterdir(), key=lambda path: path.name):
        if not root.is_dir() or not root.resolve().is_relative_to(suite):
            continue
        cfg = root/'cfg_args'
        if cfg.is_file():
            try:
                if read_config(cfg).get('model_type') != 'scaffold':
                    continue
            except (OSError, ValueError, SyntaxError):
                continue
        elif not (root/'upstream.json').is_file():
            continue
        steps = {}
        runs[root.name] = steps
        if not cfg.is_file() or not (root / "cameras.json").is_file():
            continue
        for folder in (root/'point_cloud').glob('iteration_*'):
            number = folder.name.removeprefix('iteration_')
            ply = folder/'point_cloud.ply'
            if not number.isdecimal() or not ply.is_file() or not ply.stat().st_size:
                continue
            publishes = ((root/'periodic_evaluation/plan.json').is_file()
                         or (root/'scaffold_exposure_plan.json').is_file())
            if publishes and not (folder/'ready.json').is_file():
                continue
            # A traced MLP is a zip archive. Its closing directory is written last.
            if all((folder/n).is_file() and zipfile.is_zipfile(folder/n) for n in MLP_FILES):
                steps[int(number)] = ply
        runs[root.name] = dict(sorted(steps.items(), reverse=True))
    return runs


def native_checkpoints(root):
    steps = {}
    if not (root / 'cfg_args').is_file() or not (root / 'cameras.json').is_file():
        return steps
    publishes_snapshots = (root / 'periodic_evaluation/plan.json').is_file()
    for folder in (root / 'point_cloud').glob('iteration_*'):
        number = folder.name.removeprefix('iteration_')
        ply = folder / 'point_cloud.ply'
        if not number.isdecimal() or not ply.resolve().is_relative_to(root):
            continue
        if publishes_snapshots and not (folder / 'ready.json').is_file():
            continue
        if ply.is_file() and ply.stat().st_size:
            steps[int(number)] = ply
    return dict(sorted(steps.items(), reverse=True))


def backend_for(ply, cfg):
    return (UpstreamBackend if cfg.get("scaffold_upstream") else ScaffoldBackend)(ply, cfg)


class Viewer:
    def __init__(self, server, start_dir, backend_factory=None, *, suite=None, native_model=None):
        self.server = server
        self.backend_factory = backend_factory or (NativeBackend if native_model is not None else backend_for)
        self.backend = None
        self.generation = 0
        self.model_lock = threading.RLock()
        self.work_lock = threading.Lock()
        self.pending = {}
        self.active_clients = set()
        self.wakeup = threading.Event()
        self.last_camera = None
        self.camera_poses = {}
        self.scene_source = None
        self.suite = Path(suite).resolve() if suite is not None else None
        self.native_model = Path(native_model).resolve() if native_model is not None else None
        self.catalog_lock = threading.RLock()
        self.runs = {}
        self.attempted = None
        self.closed = False

        gui = server.gui
        gui.configure_theme(dark_mode=True, control_layout="collapsible", show_share_button=False)
        with gui.add_folder("Scene"):
            if self.suite is None and self.native_model is None:
                self.add_file_picker(start_dir)
            else:
                self.run = gui.add_dropdown("Run", ("Waiting for runs",))
                self.step = gui.add_dropdown("Checkpoint", ("Waiting for checkpoint",))
                self.load_step = gui.add_button("Open checkpoint")
                self.refresh_runs = gui.add_button("Refresh runs")
                self.run.on_update(lambda _: self.refresh_steps())
                self.load_step.on_click(lambda _: self.open_selected_step())
                self.refresh_runs.on_click(lambda _: self.refresh_catalog())
                if self.native_model is not None:
                    self.run.disabled = True
                    self.follow_latest = gui.add_checkbox("Follow latest checkpoint", initial_value=True)
                    self.follow_latest.on_update(lambda _: self.refresh_catalog())
            model_type = "3DGS" if self.native_model is not None else "Scaffold"
            self.status = gui.add_markdown(f"Choose a saved {model_type} checkpoint.")
        with gui.add_folder("Viewport"):
            if self.native_model is not None:
                self.mode = gui.add_dropdown("Display", ("RGB",))
                self.mode.disabled = True
            else:
                self.mode = gui.add_dropdown("Display", MODES, hint="Rings show sampled one-sigma child footprints. Centers show sampled voxel anchors.")
            self.resolution = gui.add_slider("Render width", min=320, max=1920, step=32, initial_value=960)
            self.antialiasing = gui.add_checkbox("Anti-aliasing", initial_value=True)
            self.lossless = gui.add_checkbox("Lossless PNG", initial_value=False,
                                             hint="Preserve fine texture when inspecting blur.")
        with gui.add_folder("Cameras"):
            self.bookmark = gui.add_dropdown("Saved view", ("No checkpoint",))
            self.jump = gui.add_button("Go to saved view")
            self.reset_camera = gui.add_button("Go to first camera")
        gui.add_markdown("Drag to orbit · Right-drag to pan · Scroll to zoom")

        @self.mode.on_update
        @self.resolution.on_update
        @self.antialiasing.on_update
        @self.lossless.on_update
        def _settings(_):
            self.schedule_all()

        @self.reset_camera.on_click
        def _reset(event):
            if self.last_camera is not None and event.client is not None:
                self.set_initial_camera(event.client)

        @self.jump.on_click
        def _jump(event):
            pose = self.camera_poses.get(self.bookmark.value)
            if pose is not None and event.client is not None:
                self.set_initial_camera(event.client, pose)

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
                self.active_clients.discard(client.client_id)

        if self.suite is None and self.native_model is None:
            self.refresh()
        else:
            self.refresh_catalog()
        threading.Thread(target=self.worker, name="scaffold-renderer", daemon=True).start()

    def add_file_picker(self, start_dir):
        gui = self.server.gui
        self.directory = gui.add_text("Folder", str(Path(start_dir).expanduser().resolve()),
                                      hint="Enter a folder path, then click Open folder")
        self.entries = gui.add_dropdown("Folder / PLY", ("(empty)",), initial_value="(empty)")
        self.open_button = gui.add_button("Open selected")
        self.parent_button = gui.add_button("Parent folder")
        self.refresh_button = gui.add_button("Open folder / refresh")
        self.path_text = gui.add_text("Or enter PLY path", "", hint="Path on the machine running this viewer")
        self.load_button = gui.add_button("Load PLY path")
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

    def refresh_catalog(self):
        with self.catalog_lock:
            if self.native_model is not None:
                self.runs = {self.native_model.name: native_checkpoints(self.native_model)}
            else:
                self.runs = discover_checkpoints(self.suite)
            names = tuple(self.runs) or ("Waiting for runs",)
            if tuple(self.run.options) != names:
                previous = self.run.value
                self.run.options = names
                preferred = next((n for n in ("scaffold_upstream_fixed_v1", "scaffold_upstream_adaptive_v1")
                                  if self.runs.get(n)), next((n for n in names if self.runs.get(n)), names[0]))
                self.run.value = previous if previous in names else preferred
            self.refresh_steps()
            steps = self.runs.get(self.run.value, {})
            auto_load = self.backend is None if self.native_model is None else self.follow_latest.value
            if auto_load and steps:
                number, latest = next(iter(steps.items()))
                if latest != self.attempted:
                    self.step.value = str(number)
                    self.load(latest)
            elif self.backend is None and not steps:
                self.status.content = "Waiting for a complete saved checkpoint. The first save depends on this run's schedule."

    def refresh_steps(self):
        with self.catalog_lock:
            steps = tuple(map(str, self.runs.get(self.run.value, {})))
            options = steps or ("Waiting for checkpoint",)
            if tuple(self.step.options) != options:
                self.step.options = options
                self.step.value = options[0]
            self.load_step.disabled = not bool(steps)

    def open_selected_step(self):
        with self.catalog_lock:
            if self.step.value.isdecimal():
                ply = self.runs.get(self.run.value, {}).get(int(self.step.value))
                if ply is not None:
                    if self.native_model is not None:
                        self.follow_latest.value = False
                    self.load(ply)

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
        self.attempted = selected
        try:
            if self.native_model is not None:
                root = self.native_model
                if selected not in self.runs.get(root.name, {}).values():
                    raise ValueError("Choose a checkpoint from the specified native 3DGS run")
                ply, cfg = selected, read_config(root / "cfg_args")
                if cfg.get("model_type", "3dgs") != "3dgs":
                    raise ValueError("The specified run is not native 3DGS")
            else:
                ply, root, cfg = model_paths(selected)
            if self.suite is not None and (root.parent != self.suite or ply not in self.runs.get(root.name, {}).values()):
                raise ValueError("Choose a checkpoint from the suite")
            poses = saved_cameras(root / "cameras.json")
            source = cfg.get("source_path")
            self.status.content = f"Loading `{ply}` on the GPU…"
            with self.model_lock:
                keep_pose = self.backend is not None and source is not None and source == self.scene_source
                if self.backend is not None:
                    self.backend.close()
                    self.backend = None
                self.generation += 1
                self.last_camera = None
                self.camera_poses = {}
                self.bookmark.options = ("No checkpoint",)
                self.backend = self.backend_factory(ply, cfg)
                self.antialiasing.disabled = bool(cfg.get("scaffold_upstream"))
                self.antialiasing.value = bool(cfg.get("antialiasing", True)) and not self.antialiasing.disabled
                self.camera_poses = poses
                self.last_camera = next(iter(poses.values()))
                self.scene_source = source
                self.bookmark.options = tuple(poses)
            if not keep_pose:
                for client in self.server.get_clients().values():
                    self.set_initial_camera(client)
            self.status.content = f"**Loaded** · {root.name} · {ply.parent.name}"
            if self.native_model is not None:
                print(f"Loaded {root.name} · {ply.parent.name}", flush=True)
            self.schedule_all()
        except Exception as exc:
            self.status.content = f"**Load failed:** {type(exc).__name__}: {exc}"
            print(f"Load failed: {type(exc).__name__}: {exc}", flush=True)

    def set_initial_camera(self, client, pose=None):
        pos, rot, fovy, distance = self.last_camera if pose is None else pose
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
                    float(camera.fov), width, height, bool(self.antialiasing.value), self.generation, self.mode.value, self.lossless.value)
            with self.work_lock:
                client_id = client.client_id
                self.active_clients.add(client_id)
                self.pending[client_id] = task
                self.wakeup.set()
        except (AttributeError, ValueError):
            return

    def worker(self):
        while not self.closed:
            self.wakeup.wait()
            with self.work_lock:
                tasks = list(self.pending.items())
                self.pending.clear()
                self.wakeup.clear()
            for client_id, task in tasks:
                client, wxyz, pos, fov, width, height, aa, generation, mode, lossless = task
                try:
                    with self.model_lock:
                        if self.backend is None or generation != self.generation:
                            continue
                        image = self.backend.render_image(wxyz, pos, fov, width, height, aa, mode)
                        # Deliver the finished frame during motion. The next pass picks up the newest pose.
                        with self.work_lock:
                            connected = client_id in self.active_clients
                        if connected:
                            client.scene.set_background_image(image, format="png" if lossless else "jpeg", jpeg_quality=85)
                except Exception as exc:
                    self.status.content = f"**Render failed:** {type(exc).__name__}: {exc}"
                    print(f"Render failed: {type(exc).__name__}: {exc}", flush=True)


    def close(self):
        self.closed = True
        self.wakeup.set()
        with self.model_lock:
            if self.backend is not None:
                self.backend.close()
                self.backend = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="CUDA GPU index")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 with --suite or --native-model for remote access.")
    group = parser.add_mutually_exclusive_group()
    default_suite = Path(os.environ.get("ORANGE4F_OUTPUT_ROOT", "/data/3dgs_outputs/orange4f/3dgs_finetuning"))/"stride3_noaa_v1"
    group.add_argument("--suite", nargs="?", type=Path, const=default_suite, help="Pick saved runs and iterations (default Orange4F suite)")
    group.add_argument("--native-model", type=Path, help="Open and follow saved checkpoints from a single native 3DGS run")
    group.add_argument("--browse-root", type=Path, default=None,
                        help="Optional initial folder (default: home directory)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.host not in ("127.0.0.1", "::1", "localhost") and args.suite is None and args.native_model is None:
        parser.error("Use --suite or --native-model for a remote viewer.")
    if args.suite is not None and not args.suite.expanduser().is_dir():
        parser.error(f"Suite directory does not exist: {args.suite}")
    if args.native_model is not None and not args.native_model.expanduser().is_dir():
        parser.error(f"Run directory does not exist: {args.native_model}")
    start_dir = args.browse_root.expanduser() if args.browse_root is not None else Path.home()
    if not start_dir.is_dir():
        fallback = Path.cwd()
        print(f"Initial folder {start_dir} not available; starting in {fallback}.", flush=True)
        start_dir = fallback
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        import viser
    except ImportError:
        parser.exit(1, "Install the viewer dependency: python -m pip install 'viser==1.0.30'\n")
    label = "3DGS Viewer" if args.native_model is not None else "Scaffold-GS Viewer"
    server = viser.ViserServer(host=args.host, port=args.port, label=label)
    viewer = Viewer(server, start_dir, suite=args.suite.expanduser() if args.suite else None,
                    native_model=args.native_model.expanduser() if args.native_model else None)
    host = "<server-ip>" if args.host == "0.0.0.0" else args.host
    port = server.get_port()
    print(f"Open http://{host}:{port}/", flush=True)
    if args.suite and args.host == "0.0.0.0" and (args.suite.expanduser() / "suite.json").is_file():
        from scripts.refresh_orange4f_dashboard import refresh
        try:
            refresh(args.suite.expanduser(), viewer_port=port)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Could not add the viewer link to the suite dashboard: {exc}", flush=True)
    try:
        while True:
            time.sleep(10)
            if args.suite or args.native_model:
                try:
                    viewer.refresh_catalog()
                except OSError as exc:
                    print(f"Could not refresh checkpoints: {exc}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        server.stop()


if __name__ == "__main__":
    main()

