#!/usr/bin/env python3
"""Per-image 3D inspection of actual native culling masks and image gradients."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.occlusion_diagnostics import COLORS, LABELS, display_indices, nearest_ray_point, write_colored_ply


def thumbnail(array, width=600):
    from PIL import Image
    image = Image.fromarray(array)
    image.thumbnail((width, width))
    return np.asarray(image)


class InspectorViewer:
    def __init__(self, server, backend, max_points=0):
        self.server, self.backend = server, backend
        self.lock = threading.Lock()
        self.nodes, self.rows, self.images = {}, {}, {}
        self.ready = False
        gui = server.gui
        gui.add_markdown('## Occlusion inspector\n**Green:** kept with raster footprint. **Red:** depth-culled. '
                         '**Gray:** other. Markers show Gaussian centers and ignore learned opacity. '
                         'A raster footprint does not guarantee a final pixel contribution.')
        self.selection = gui.add_dropdown('Image', tuple(backend.names),
            initial_value=backend.args.image or backend.names[len(backend.names)//2])
        self.reference = gui.add_dropdown('Occluding reference', tuple(backend.source_options.values()),
            initial_value=backend.source_options[backend.source_key])
        self.comparison = gui.add_markdown('')
        self.load = gui.add_button('Inspect image')
        self.probe = gui.add_button('Check image gradients')
        self.status = gui.add_markdown('Choose an image to inspect.')
        self.counts = gui.add_markdown('')
        with gui.add_folder('3D layers'):
            self.layers = {name: gui.add_checkbox(label, initial_value=initial) for name, label, initial in (
                ('active', 'Green: kept / in view', True), ('culled', 'Red: depth-culled', True),
                ('other', 'Gray: other Gaussians', False), ('surface', 'Blue: trusted depth surface', True))}
            self.limit = gui.add_number('Display cap per layer (0 = all)', max_points, min=0, step=100000,
                hint='Only changes browser display; mask counts and exported PLY remain complete.')
            self.point_size = gui.add_slider('Center size (scene units)', min=.001, max=.06, step=.001, initial_value=.012)
            self.view_camera = gui.add_button('Go to image camera')
            self.view_side = gui.add_button('Side view')
            self.reference_only = gui.add_button('Show reference surface only')
            self.export = gui.add_button('Export all green + red centers to PLY')
            gui.add_markdown('Hide green to inspect red points behind surfaces; orbit to view their depth. '
                             'Depth surface is sampled every 3 pixels; Gaussian layers are complete at cap 0.')
        with gui.add_folder('Selected Gaussian'):
            self.identifier = gui.add_number('Checkpoint Gaussian ID', 0, min=0, max=len(backend.xyz)-1, step=1)
            self.inspect_button = gui.add_button('Inspect Gaussian')
            self.details = gui.add_markdown('Shift-click near a colored point to pick the nearest displayed center to its ray. '
                         'Yellow rings show the selected Gaussian at the configured sigma support.')
            self.footprint = gui.add_image(np.zeros((32, 32, 3), np.uint8), label='Conservative tested tile footprint')
        with gui.add_folder('Image comparisons'):
            for key, label in (('gt', 'Actual image'), ('culling_off', 'Culling OFF'), ('culling_on', 'Culling ON'),
                               ('difference', 'Absolute difference x10'), ('centers', 'Centers over GT (no z-buffer)'),
                               ('depth', 'Conservative depth bounds; black = unknown'),
                               ('reference_raw', 'Raw reference depth (may include rejected pixels)'),
                               ('reference_trusted', 'Trusted depth coverage over GT'),
                               ('coverage', 'Green usable tiles / blue trusted pixels / black unknown'),
                               ('anchors', 'LiDAR anchors: green agree / red disagree / gold untrusted')):
                self.images[key] = gui.add_image(np.zeros((32, 32, 3), np.uint8), label=label)
        self.load.on_click(lambda _: self.inspect(self.selection.value))
        self.probe.on_click(lambda _: self.inspect(self.backend.current['name'] if self.ready else self.selection.value, True))
        self.reference.on_update(lambda _: self.inspect(self.backend.current['name'] if self.ready else self.selection.value))
        for name, handle in self.layers.items():
            handle.on_update(lambda _, name=name: self.update_layer(name))
        self.limit.on_update(lambda _: self.update_layers())
        self.point_size.on_update(lambda _: self.update_layers())
        self.view_camera.on_click(lambda event: self.set_camera(event.client, side=False))
        self.view_side.on_click(lambda event: self.set_camera(event.client, side=True))
        self.reference_only.on_click(lambda _: self.show_reference_only())
        self.inspect_button.on_click(lambda _: self.inspect_gaussian(int(self.identifier.value)))
        self.export.on_click(lambda _: self.export_ply())
        server.on_client_connect(lambda client: self.set_camera(client, side=True))
        server.scene.on_click(modifier='shift')(self.pick_gaussian)

    def work(self, action):
        def run():
            if not self.lock.acquire(blocking=False):
                return
            self.load.disabled = self.probe.disabled = self.export.disabled = self.inspect_button.disabled = self.reference.disabled = True
            try:
                action()
            except Exception as exc:
                self.status.content = f'**Failed:** {exc}'
                traceback.print_exc()
            finally:
                self.load.disabled = self.probe.disabled = self.export.disabled = self.inspect_button.disabled = self.reference.disabled = False
                self.lock.release()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def inspect(self, name, gradients=False):
        def action():
            self.ready = False
            self.status.content = f'Computing native culling {"and image gradients " if gradients else ""}for `{name}`…'
            source = next(key for key, label in self.backend.source_options.items() if label == self.reference.value)
            result = self.backend.analyze(name, gradients, source)
            self.ready = True
            for node in self.nodes.values(): node.remove()
            self.nodes.clear()
            self.rows.clear()
            for key, handle in self.images.items():
                handle.image = thumbnail(result['images'][key][1]) if key in result['images'] else np.zeros((32, 32, 3), np.uint8)
            counts = result['summary']['counts']
            self.counts.content = '\n'.join(f'- {name}: **{counts[name]:,}**' for name in ('active', 'culled', 'other', 'total'))
            summary = result['summary']
            comparison = summary.get('reference_comparison', {})
            lines = ['| Reference | Trusted pixels | Usable tiles | Culled candidates |',
                     '| --- | ---: | ---: | ---: |']
            for row in comparison.values():
                lines.append(f"| {row['culling_source']} | {row['coverage']['trusted_pixel_fraction']:.1%} | "
                             f"{row['reliable_tile_fraction']:.1%} | {row['culled_fraction_raster_candidates']:.1%} |")
            self.comparison.content = '\n'.join(lines) if comparison else ''
            message = (f"**{name} — iteration {self.backend.iteration}**\n\n"
                f"**Reference: {summary.get('culling_source', self.reference.value)}**\n\n"
                f"Depth-culled: {100*summary['culled_fraction_all']:.2f}% of all Gaussians; "
                f"{100*summary['culled_fraction_raster_candidates']:.2f}% of OFF raster candidates. "
                f"Reliable depth tiles: {100*summary['reliable_tile_fraction']:.1f}%.\n\n")
            if summary.get('culling_lookup'):
                message += f"Depth lookup: `{summary['culling_lookup']}`\n\n"
            if summary.get('diagnostic_override'):
                message += f"Diagnostic override; this checkpoint was trained with **{summary['saved_training_reference']}**.\n\n"
            if summary.get('candidate_reasons'):
                message += 'OFF raster candidates:\n\n'+'\n'.join(
                    f'- {key.replace("_", " ")}: **{value:,}**' for key, value in summary['candidate_reasons'].items())+'\n\n'
                if not summary['candidate_reasons']['culled']:
                    message += '**No raster candidates were removed for this view.**\n\n'
            if summary['gradient_probe']:
                message += summary['gradient_probe']['verdict']+'\n\n'
            else:
                message += 'Gradient check has not been run for this view.\n\n'
            self.status.content = message + f"Report: `{result['directory']}/index.html`"
            import viser.transforms as vtf
            c2w = result['c2w']
            self.nodes['camera'] = self.server.scene.add_camera_frustum('/reference_camera',
                fov=result['camera'].FoVy, aspect=result['camera'].image_width/result['camera'].image_height,
                scale=.4, image=thumbnail(result['images']['gt'][1], 400),
                wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz, position=c2w[:3, 3])
            self.update_layers()
            for client in self.server.get_clients().values(): self.set_camera(client, side=True)
            self.details.content = 'Shift-click near a colored center, or enter a Gaussian ID, to inspect its culling decision.'
        return self.work(action)

    def update_layers(self):
        for name in self.layers: self.update_layer(name)

    def show_reference_only(self):
        for name, handle in self.layers.items():
            handle.value = name == 'surface'
        self.update_layers()

    def update_layer(self, name):
        if not self.ready:
            return
        visible = self.layers[name].value
        if name in self.nodes:
            self.nodes[name].remove()
            del self.nodes[name]
        if not visible:
            return
        current = self.backend.current
        if name == 'surface':
            points, color = current['surface'], (85, 155, 255)
        else:
            indices = display_indices(current['labels'], LABELS.index(name), int(self.limit.value))
            self.rows[name] = indices
            points, color = self.backend.xyz[indices], tuple(int(v) for v in COLORS[LABELS.index(name)])
            self.layers[name].label = f"{name}: {len(indices):,} shown / {current['summary']['counts'][name]:,} total"
        if not len(points):
            return
        node = self.server.scene.add_point_cloud('/'+name, points, color, point_size=float(self.point_size.value),
            point_shape='circle', point_shading='flat', precision='float32')
        self.nodes[name] = node

    def pick_gaussian(self, event):
        if not self.ready or self.lock.locked():
            return
        candidates = []
        for name, indices in self.rows.items():
            if name in self.nodes and self.layers[name].value:
                idx = nearest_ray_point(self.backend.xyz, indices, event.ray_origin, event.ray_direction)
                if idx is not None: candidates.append(idx)
        if candidates:
            idx = nearest_ray_point(self.backend.xyz, np.asarray(candidates), event.ray_origin, event.ray_direction)
            self.identifier.value = idx
            self.inspect_gaussian(idx)

    def set_camera(self, client, side):
        if client is None or not self.ready:
            return
        current = self.backend.current
        c2w = current['c2w']
        pos, right, down, forward = c2w[:3, 3], c2w[:3, 0], c2w[:3, 1], c2w[:3, 2]
        surface = current['surface']
        distance = float(np.median(np.linalg.norm(surface-pos, axis=1))) if len(surface) else 3.
        distance = np.clip(distance, .3, 30.)
        client.camera.position = pos + distance*(.7*right-.35*down-.3*forward) if side else pos
        client.camera.up_direction = -down
        client.camera.look_at = pos+forward*distance
        client.camera.fov = current['camera'].FoVy

    def inspect_gaussian(self, index):
        if not self.ready:
            return
        def action():
            values, matrix = self.backend.gaussian_details(index)
            center = self.backend.xyz[index]
            angles = np.linspace(0, 2*np.pi, 65)
            circles = []
            for i, j in ((0, 1), (0, 2), (1, 2)):
                p = np.zeros((65, 3)); p[:, i], p[:, j] = np.cos(angles), np.sin(angles)
                p = p@(values['sigma']*matrix).T+center
                circles.append(np.stack((p[:-1], p[1:]), 1))
            for key in ('selected', 'ray'):
                if key in self.nodes: self.nodes[key].remove()
            self.nodes['selected'] = self.server.scene.add_line_segments('/selected', np.concatenate(circles), (255, 220, 40), line_width=2)
            self.nodes['ray'] = self.server.scene.add_line_segments('/ray',
                np.array([[self.backend.current['c2w'][:3, 3], center]]), (255, 180, 40), line_width=1)
            self.details.content = values['reason']+'\n\n```json\n'+json.dumps(values, indent=2)+'\n```'
            from PIL import Image, ImageDraw
            picture = Image.fromarray(self.backend.current['images']['gt'][1])
            lo, hi = np.asarray(values['tile_lo'])*16, (np.asarray(values['tile_hi'])+1)*16-1
            ImageDraw.Draw(picture).rectangle((*lo, *hi), outline=(255, 220, 40), width=3)
            self.footprint.image = thumbnail(np.asarray(picture))
        return self.work(action)

    def export_ply(self):
        if not self.ready:
            return
        def action():
            current = self.backend.current
            path = current['directory']/'active_and_culled.ply'
            count = write_colored_ply(path, self.backend.xyz, current['labels'])
            self.status.content += f'\n\nExported **all {count:,} green/red centers** with checkpoint IDs: `{path}`'
            print(f'Exported {count:,} Gaussian centers: {path}', flush=True)
        return self.work(action)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iteration', type=int, default=-1, help='Latest saved PLY by default')
    p.add_argument('--image', help='Exact registered image name; otherwise the middle training image')
    p.add_argument('--split', choices=('train', 'eval', 'all'), default='train')
    p.add_argument('--depth-dir', type=Path,
                   help='Compare a LiDAR-calibrated DA3 cache against the saved reference on the same checkpoint')
    p.add_argument('--reference', choices=('saved', 'da3'),
                   help='Initial displayed reference; defaults to DA3 when --depth-dir is supplied')
    p.add_argument('--gpu', default='5')
    p.add_argument('--port', type=int, default=8082)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--max-points', type=int, default=0, help='Display cap per category; 0 shows all (default)')
    p.add_argument('--check-gradients', action='store_true', help='Run native culling ON/OFF backward checks for the initial image')
    p.add_argument('--output', type=Path)
    p.add_argument('--export-only', action='store_true', help='Save the report, exact masks and full green/red PLY, then exit')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_points < 0 or not 1 <= args.port <= 65535:
        raise ValueError('Invalid display cap or port')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not args.export_only:
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError("Install the viewer in 3dgs: python -m pip install 'viser==1.0.30'") from exc
    from scripts.occlusion_inspector_backend import InspectorBackend
    backend = InspectorBackend(args)
    name = args.image or backend.names[len(backend.names)//2]
    if args.export_only:
        current = backend.analyze(name, args.check_gradients)
        write_colored_ply(current['directory']/'active_and_culled.ply', backend.xyz, current['labels'])
        return 0
    server = viser.ViserServer(host=args.host, port=args.port, label='GS Occlusion Inspector')
    viewer = InspectorViewer(server, backend, args.max_points)
    viewer.inspect(name, args.check_gradients)
    print(f'Listening on {args.host}:{server.get_port()}; use this host IP for a remote connection when bound to 0.0.0.0.', flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
