#!/usr/bin/env python3
"""Select a LiDAR region, inspect overlapping RGB views, export original poses."""
import argparse
import json
from pathlib import Path
import sys
import threading

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.region_dataset import RegionData, inside


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--baseline', type=Path, help='Supplies original input.ply and keyframes.json')
    p.add_argument('--point-cloud', type=Path)
    p.add_argument('--keyframes', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--anchor', default='06663.jpg')
    p.add_argument('--center', type=float, nargs=3)
    p.add_argument('--size', type=float, nargs=3, default=[4.,4.,4.])
    p.add_argument('--halo', type=float, default=1.)
    p.add_argument('--min-overlap', type=float, default=.02)
    p.add_argument('--max-camera-distance', type=float, default=8.)
    p.add_argument('--frame-start', default='', help='Optional inclusive lexical image-name bound')
    p.add_argument('--frame-end', default='')
    p.add_argument('--exclude', nargs='*', default=[])
    p.add_argument('--sample-points', type=int, default=1000000)
    p.add_argument('--port', type=int, default=8091)
    p.add_argument('--export', action='store_true', help='Headless export with the supplied box/filter')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.baseline:
        args.point_cloud = args.point_cloud or args.baseline/'input.ply'
        args.keyframes = args.keyframes or args.baseline/'keyframes.json'
    if args.point_cloud is None or args.keyframes is None:
        raise ValueError('Supply --baseline or both --point-cloud and --keyframes')
    if args.output.exists():
        raise FileExistsError(f'Use a new region directory: {args.output}')
    data = RegionData(args.source, args.point_cloud, args.keyframes, args.sample_points)
    names = [im.name for im in data.images]
    if args.anchor not in names:
        raise ValueError(f'Anchor {args.anchor!r} not registered; specify --anchor IMAGE_NAME')
    center = args.center if args.center is not None else data.anchor_center(args.anchor)
    settings = dict(halo=args.halo,min_overlap=args.min_overlap,max_camera_distance=args.max_camera_distance,
                    frame_start=args.frame_start,frame_end=args.frame_end,exclude=args.exclude)
    if args.export:
        rows = data.find(center,args.size,**settings,progress=lambda n,total,kept: print(f'Checked {n}/{total}; {kept} overlapping views',flush=True))
        data.export(args.output,center,args.size,rows,**settings)
        return 0
    import viser
    server = viser.ViserServer(host='0.0.0.0',port=args.port)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## Regional reconstruction ablation\nGray: scene LiDAR. Yellow: box LiDAR. Blue: retained halo. Camera dots: green=train, cyan=held-out after selection. Bright RGB pixels are supervised; dim pixels are excluded. Box coordinates use original scene units. No ray culling.')
    c = server.gui.add_vector3('Box center',initial_value=tuple(center),step=.1)
    size = server.gui.add_vector3('Box XYZ size',initial_value=tuple(args.size),min=(.1,.1,.1),step=.1)
    halo = server.gui.add_number('LiDAR halo',initial_value=args.halo,min=0.,step=.1)
    overlap = server.gui.add_number('Minimum image fraction',initial_value=args.min_overlap,min=.001,max=1.,step=.01)
    distance = server.gui.add_number('Camera distance to box (0=all)',initial_value=args.max_camera_distance,min=0.,step=1.)
    first = server.gui.add_text('First image (optional)',initial_value=args.frame_start)
    last = server.gui.add_text('Last image (optional)',initial_value=args.frame_end)
    excluded = server.gui.add_text('Exclude names (comma/newline)',initial_value=','.join(args.exclude),multiline=True)
    current = server.gui.add_dropdown('Inspect image',options=names,initial_value=args.anchor)
    recenter = server.gui.add_button('Center box on inspected image')
    find = server.gui.add_button('Find overlapping views')
    review = server.gui.add_dropdown('Selected image review',options=['(run selection first)'])
    export = server.gui.add_button('Export region',disabled=True)
    status = server.gui.add_markdown('Adjust the box, then find overlapping views.')
    rgb = server.gui.add_image(np.zeros((128,192,3),dtype=np.uint8),label='RGB supervision overlay')
    result = {'rows':None, 'config':None}
    lock = threading.Lock()
    server.scene.add_point_cloud('/scene_lidar',points=data.points[::3],colors=(110,120,135),point_size=.012,precision='float32')
    server.scene.add_point_cloud('/all_cameras',points=np.array([im.C for im in data.images]),colors=(170,170,190),point_size=.035,precision='float32')

    def config():
        return dict(center=list(c.value),size=list(size.value),halo=halo.value,min_overlap=overlap.value,
                    max_camera_distance=distance.value,frame_start=first.value,frame_end=last.value,
                    exclude=[x.strip() for x in excluded.value.replace('\n',',').split(',') if x.strip()])

    def draw():
        core = inside(data.points,c.value,size.value)
        support = inside(data.points,c.value,size.value,halo.value) & ~core
        server.scene.add_box('/region_box',dimensions=size.value,position=c.value,color=(255,210,40),wireframe=True)
        server.scene.add_box('/halo_box',dimensions=np.array(size.value)+2*halo.value,position=c.value,color=(90,140,250),wireframe=True)
        server.scene.add_point_cloud('/core_lidar',points=data.points[core],colors=(255,205,40),point_size=.015,precision='float32')
        server.scene.add_point_cloud('/halo_lidar',points=data.points[support],colors=(80,140,250),point_size=.012,precision='float32')
        _, image, info = data.overlay(current.value,c.value,size.value)
        rgb.image = image
        im = next(im for im in data.images if im.name == current.value)
        cam = data.cameras[im.camera_id]
        server.scene.add_camera_frustum('/inspected_camera',fov=2*np.arctan(cam.height/(2*cam.params[1])),
            aspect=cam.width/cam.height,wxyz=Rotation.from_matrix(im.R.T).as_quat()[[3,0,1,2]],position=im.C,
            scale=.4,color=(255,180,40),image=image)

    def changed(_):
        export.disabled=True
        result['rows']=None
        status.content='Settings changed. Run **Find overlapping views** before export.'
        try: draw()
        except Exception as exc: status.content=str(exc)
    for control in (c,size,halo,overlap,distance,first,last,excluded):
        control.on_update(changed)
    @current.on_update
    def inspect(_):
        try: draw()
        except Exception as exc: status.content=str(exc)
    @review.on_update
    def inspect_selected(_):
        if review.value in names: current.value=review.value
    @recenter.on_click
    def center_on_camera(_):
        try: c.value=tuple(data.anchor_center(current.value))
        except Exception as exc: status.content=str(exc)
    @find.on_click
    def select(_):
        if not lock.acquire(blocking=False): return
        find.disabled=True; export.disabled=True
        try:
            cfg=config()
            status.content='Projecting sampled LiDAR into candidate images…'
            rows=data.find(**cfg,progress=lambda n,total,kept: setattr(status,'content',f'Checked {n}/{total} cameras; {kept} overlapping images.'))
            if cfg != config():
                raise ValueError('Box/filter changed during selection; run selection again')
            result.update(rows=rows,config=cfg)
            review.options=[r['name'] for r in rows]
            review.value=rows[0]['name']
            train=sum(r['split']=='train' for r in rows)
            status.content=f'**{train} training + {len(rows)-train} held-out views.** Review overlays, then export to `{args.output}`.'
            if train < 2: status.content+=' Need at least two training views.'
            export.disabled=train<2
            lookup={im.name:im for im in data.images}
            server.scene.add_point_cloud('/selected_cameras',points=np.array([lookup[r['name']].C for r in rows]),
                colors=np.array([(50,230,120) if r['split']=='train' else (50,210,255) for r in rows],dtype=np.uint8),point_size=.09,precision='float32')
        except Exception as exc: status.content=f'Selection failed: {exc}'
        finally: find.disabled=False; lock.release()
    @export.on_click
    def save(_):
        if not lock.acquire(blocking=False): return
        export.disabled=True
        try:
            if result['config'] != config(): raise ValueError('Selection is stale; find overlapping views again')
            status.content='Exporting original poses, RGB links, all regional LiDAR, and masks…'
            data.export(args.output,rows=result['rows'],**result['config'])
            status.content=f'**Export complete:** `{args.output}`. Inspect `index.html`, then run `bash scripts/run_region_ablation.sh 0 {args.output} crop`.'
        except Exception as exc:
            status.content=f'Export failed: {exc}'
            export.disabled=False
        finally: lock.release()
    @server.on_client_connect
    def connect(client):
        client.camera.look_at=np.asarray(c.value)
        client.camera.position=np.asarray(c.value)+np.array([6.,-6.,4.])
    draw()
    print(f'Region selector: http://0.0.0.0:{args.port} (CPU only)',flush=True)
    threading.Event().wait()


if __name__ == '__main__':
    raise SystemExit(main())
