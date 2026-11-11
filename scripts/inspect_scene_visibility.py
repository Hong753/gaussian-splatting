#!/usr/bin/env python3
"""Inspect precomputed LiDAR point visibility and the fixed occluding mesh."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--point-cloud', type=Path, required=True)
    p.add_argument('--visibility', type=Path)
    p.add_argument('--image')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8083)
    p.add_argument('--max-points', type=int, default=0, help='Display cap per class; 0 displays every in-view point')
    args = p.parse_args(argv)
    if args.max_points < 0 or not 1 <= args.port <= 65535:
        raise ValueError('Invalid display limit/port')
    import viser
    import viser.transforms as vtf
    from PIL import Image
    from utils.scene_occlusion import load_scene, mesh_arrays
    from utils.lidar_occlusion import file_digest
    from utils.sparse_lidar import frame_filename
    from scripts.prepare_scene_occlusion import read_points
    from scripts.view_comparisons import read_colmap_records
    report, mesh_path = load_scene(args.scene, args.point_cloud)
    directory = args.visibility or args.scene.parent/'visibility'
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata['scene_sha256'] != file_digest(args.scene):
        raise ValueError('Visibility labels belong to a different scene surface')
    names = [name for name in metadata['images'] if (directory/'frames'/frame_filename(name)).exists()]
    if not names or (args.image and args.image not in names):
        raise ValueError('Precompute the requested view first')
    points = read_points(args.point_cloud)
    records = read_colmap_records(Path(report['source']))
    vertices, faces = mesh_arrays(mesh_path)
    server = viser.ViserServer(host=args.host, port=args.port, label='Fixed scene / LiDAR visibility')
    mesh_handle = server.scene.add_mesh_simple('/occluding_surface', vertices, faces, color=(80, 150, 250),
        opacity=.20, side='double')
    controls = server.gui
    controls.add_markdown('**Original LiDAR points, fixed mesh visibility.**\n\n'
        'Green: on the first surface. Red: behind it. Gray: no intersection. '
        'Orange: in front of the mesh (geometry disagreement). Blue: occluding mesh. '
        'Out-of-view points are omitted. These are point labels, not optimized Gaussian labels.')
    select = controls.add_dropdown('Reference image', names, initial_value=args.image or names[len(names)//2])
    status = controls.add_markdown('Loading…')
    colors = dict(visible=(35, 210, 105), occluded=(245, 55, 65), unknown=(150, 150, 150), in_front=(245, 160, 40))
    toggles = {key: controls.add_checkbox('Show '+key.replace('_', ' '), initial_value=key != 'unknown') for key in colors}
    mesh_toggle = controls.add_checkbox('Show fixed surface', initial_value=True)
    recenter = controls.add_button('Overview of reference camera')
    server.scene.add_frame('/reference', show_axes=False)
    nodes = {}
    state = {}
    def place_client(client):
        if 'c2w' not in state:
            return
        pose = state['c2w']
        origin, right, down, forward = pose[:3, 3], pose[:3, 0], pose[:3, 1], pose[:3, 2]
        client.camera.position = origin+4*right-2*down-2*forward
        client.camera.look_at = origin+2*forward
        client.camera.up_direction = -down
    server.on_client_connect(place_client)
    recenter.on_click(lambda _: [place_client(client) for client in server.get_clients().values()])
    def show():
        with np.load(directory/'frames'/frame_filename(select.value), allow_pickle=False) as data:
            row = json.loads(str(data['report']))
            for key, color in colors.items():
                if key in nodes:
                    nodes[key].remove()
                ids = data[key+'_ids']
                if args.max_points and len(ids) > args.max_points:
                    ids = ids[np.linspace(0, len(ids)-1, args.max_points, dtype=int)]
                nodes[key] = server.scene.add_point_cloud('/lidar/'+key, points[ids], colors=color,
                    point_size=.012, visible=toggles[key].value)
            k, e = data['intrinsic'], data['extrinsic']
            w, h = int(data['width']), int(data['height'])
        c2w = np.linalg.inv(e)
        state['c2w'] = c2w
        with Image.open(records[select.value]['image_path']) as im:
            im.thumbnail((640, 640))
            rgb = np.asarray(im.convert('RGB'))
        if 'camera' in nodes:
            nodes['camera'].remove()
        nodes['camera'] = server.scene.add_camera_frustum('/reference/camera',
            fov=2*np.arctan(h/(2*k[1, 1])), aspect=w/h, scale=.35, image=rgb,
            wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz, position=c2w[:3, 3])
        status.content = (f"**{select.value}**\n\nDepth coverage: {row['depth_coverage']:.1%}; "
            f"usable Gaussian-culling tiles: {row['usable_tile_fraction']:.1%}.\n\n"+
            '\n'.join(f'- {key}: **{value:,}**' for key, value in row['counts'].items()))
    select.on_update(lambda _: show())
    for key, toggle in toggles.items():
        toggle.on_update(lambda _, key=key: setattr(nodes[key], 'visible', toggles[key].value))
    mesh_toggle.on_update(lambda _: setattr(mesh_handle, 'visible', mesh_toggle.value))
    show()
    print(f'LiDAR visibility viewer: http://{args.host}:{server.get_port()}', flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
