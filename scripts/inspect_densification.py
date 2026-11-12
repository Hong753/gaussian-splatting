#!/usr/bin/env python3
"""Viser: LiDAR, Gaussian size branches, 3-sigma ellipsoids and CUDA contribution images."""
import argparse
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.densification_diagnostics import COLORS, LABELS


def sample_ids(ids, limit):
    return ids if not limit or len(ids)<=limit else ids[np.linspace(0,len(ids)-1,limit,dtype=np.int64)]


def build_viewer(backend,args):
    import viser
    import viser.transforms as vtf
    server=viser.ViserServer(host=args.host,port=args.port,label='Gaussian size / densification audit')
    gui=server.gui;lock=threading.RLock();nodes={}
    state=dict(current=backend.analyze(args.image,args.large_radius,args.roi))
    gui.add_markdown('## Gaussian size audit\n**Orange:** large screen footprint, clone-size branch. '
        '**Magenta:** large screen footprint, split-size branch. **Teal:** smaller footprint. '
        '**Gray:** outside native raster candidates. These are **size classes**, not culling labels. '
        'The contribution image measures what actually reaches the rendered pixels. '
        'PLY files do not contain historical densification gradients.')
    camera=gui.add_dropdown('Camera',backend.names,initial_value=args.image)
    prev=gui.add_button('Previous frame');nxt=gui.add_button('Next frame')
    prev.on_click(lambda _:setattr(camera,'value',backend.names[max(0,backend.names.index(camera.value)-1)]))
    nxt.on_click(lambda _:setattr(camera,'value',backend.names[min(len(backend.names)-1,backend.names.index(camera.value)+1)]))
    status=gui.add_markdown('')
    with gui.add_folder('Region and size'):
        radius=gui.add_slider('Large footprint radius (pixels)',min=1.,max=max(200.,args.large_radius),step=1.,initial_value=args.large_radius)
        roi=[gui.add_slider(label,min=0.,max=1.,step=.01,initial_value=value)
             for label,value in zip(('ROI left','ROI top','ROI right','ROI bottom'),args.roi)]
        focus=gui.add_dropdown('Ellipsoid selection',('large_clone_size','large_split_size','all'),initial_value='large_clone_size')
        analyze=gui.add_button('Analyze selected region')
        gui.add_markdown('Radius changes only this diagnostic. White rectangle = ROI. '
            'Counts include CUDA bounding footprints intersecting the ROI, even if their centers are outside it. '
            'Ellipsoids show the largest selected footprints; they may still be naturally occluded.')
    with gui.add_folder('Independent 3D layers'):
        show_lidar=gui.add_checkbox('Show LiDAR',initial_value=True)
        lidar_all=gui.add_checkbox('LiDAR: whole scene',initial_value=False)
        lidar_size=gui.add_slider('LiDAR point size',min=.001,max=.05,step=.001,initial_value=.003)
        show_gs=gui.add_checkbox('Show Gaussian centers',initial_value=True)
        scope=gui.add_dropdown('Gaussian population',('Current ROI','Camera frustum','All scene'),initial_value='Current ROI')
        gs_size=gui.add_slider('Gaussian point size',min=.001,max=.1,step=.001,initial_value=.012)
        ellipsoids=gui.add_checkbox('Show selected 3-sigma ellipsoids',initial_value=True)
    show_mesh=None;mesh=None
    if args.scene:
        from utils.scene_occlusion import load_scene,mesh_arrays
        _,path=load_scene(args.scene)
        vertices,faces=mesh_arrays(path)
        mesh=server.scene.add_mesh_simple('/reference_mesh',vertices,faces,color=(70,130,220),opacity=.10,
            wireframe=True,visible=False,side='double',cast_shadow=False,receive_shadow=False)
        show_mesh=gui.add_checkbox('Reference mesh (visual only)',initial_value=False)
        show_mesh.on_update(lambda _:setattr(mesh,'visible',show_mesh.value))
    overview=gui.add_button('Scene overview');reference=gui.add_button('Look from reference camera')
    export=gui.add_button('Export this diagnostic and ZIP')
    with gui.add_folder('Actual / render / footprint / contribution'):
        panels={key:gui.add_image(pixels,label=title) for key,(title,pixels) in state['current']['images'].items()}

    def pose(client,close=False):
        c=state['current']['c2w'];origin=c[:3,3]
        client.camera.position=origin if close else origin+3*c[:3,0]-2*c[:3,1]-2*c[:3,2]
        client.camera.look_at=origin+c[:3,2]*(1 if close else 3)
        client.camera.up_direction=-c[:3,1]
    server.on_client_connect(lambda client:pose(client))
    overview.on_click(lambda _: [pose(c) for c in server.get_clients().values()])
    reference.on_click(lambda _: [pose(c,True) for c in server.get_clients().values()])

    def update_clouds(_=None):
        with lock:
            current=state['current']
            mask=current['roi_mask'] if scope.value=='Current ROI' else current['frustum'] if scope.value=='Camera frustum' else np.ones(len(backend.points),bool)
            ids=sample_ids(np.flatnonzero(mask),args.max_points)
            lidar_ids=sample_ids(np.arange(len(backend.lidar)) if lidar_all.value else np.flatnonzero(current['lidar_frustum']),args.max_points)
            # Display-only controls never rerender CUDA or change statistics.
            for key in ('Gaussians','LiDAR','Ellipsoids'):
                if key in nodes:nodes.pop(key).remove()
            nodes['Gaussians']=server.scene.add_point_cloud('/Gaussians',backend.points[ids],
                colors=COLORS[current['labels'][ids]],point_size=gs_size.value,visible=show_gs.value)
            nodes['LiDAR']=server.scene.add_point_cloud('/LiDAR',backend.lidar[lidar_ids],
                colors=backend.lidar_colors[lidar_ids] if backend.lidar_colors is not None else (140,155,175),
                point_size=lidar_size.value,visible=show_lidar.value)
            if len(current['segments']):
                nodes['Ellipsoids']=server.scene.add_line_segments('/Ellipsoids',current['segments'],
                    colors=current['segment_colors'],line_width=1.5,visible=ellipsoids.value)
            r=current['report'];counts=r['size_branch_counts'];share=r['compositing_mass_fraction']['large_clone_size']
            share_text='undefined (no accumulated alpha)' if share is None else f'{share:.1%}'
            probe=r.get('single_view_gradient_probe')
            probe_text=(f"\n\nCurrent-view high-gradient orange Gaussians: **{probe['high_gradient_large_clone_size']:,}** "
                '(photometric gradient only; not the training-window average).' if probe else '')
            status.content=(f"**{r['image']} · {r['iteration']:,} steps**\n\n"
                f"Scene extent: **{r['scene_extent']:.4g}**; clone/split sigma cutoff: **{r['split_sigma_threshold']:.4g} scene units**. "
                f"Growth window at checkpoint: **{'open' if r['densification_window_open_at_checkpoint'] else 'closed'}**.\n\n"
                f"**{r['total_gaussians']:,} Gaussians loaded**, {r['total_lidar']:,} LiDAR points. "
                f"ROI raster candidates: **{r['raster_footprints_overlapping_roi']:,}**.\n\n"
                f"Orange large clone-size: **{counts['large_clone_size']:,}**; magenta large split-size: **{counts['large_split_size']:,}**. "
                f"Orange share of ROI compositing weight: **{share_text}**. ROI PSNR: **{r['roi_psnr_db']:.2f} dB**.\n\n"
                f"Display: **{len(ids):,}/{int(mask.sum()):,} Gaussian centers**, **{len(lidar_ids):,} LiDAR points**, "
                f"**{len(current['draw_ids'])} ellipsoids**. Counts and render contributions use the complete checkpoint.\n\n"
                f"Recorded growth events: **{r['recorded_growth']['events']}**. "
                'A size class alone does not establish that cloning occurred. No hard culling is applied in these renders.'+probe_text)

    def refresh():
        current=state['current'];c=current['c2w'];cam=backend.frame['camera']
        if 'camera' in nodes:nodes['camera'].remove()
        rgb=current['images']['gt'][1];stride=max(1,rgb.shape[1]//640)
        nodes['camera']=server.scene.add_camera_frustum('/camera',fov=cam.FoVy,aspect=cam.image_width/cam.image_height,
            scale=.5,image=rgb[::stride,::stride],wxyz=vtf.SO3.from_matrix(c[:3,:3]).wxyz,position=c[:3,3])
        update_clouds()
        for key,(_,pixels) in current['images'].items():panels[key].image=pixels

    def analyze_view(_):
        with lock:
            status.content='Computing exact CUDA radii and compositing weights…'
            try:
                state['current']=backend.analyze(camera.value,radius.value,tuple(x.value for x in roi),focus.value)
                refresh()
            except Exception as exc:
                status.content=f'Diagnostic failed: {type(exc).__name__}: {exc}'
                raise
    camera.on_update(analyze_view);analyze.on_click(analyze_view)
    def display_only(_):
        with lock:
            nodes['LiDAR'].visible=show_lidar.value;nodes['LiDAR'].point_size=lidar_size.value
            nodes['Gaussians'].visible=show_gs.value;nodes['Gaussians'].point_size=gs_size.value
            if 'Ellipsoids' in nodes:nodes['Ellipsoids'].visible=ellipsoids.value
    for control in (show_lidar,lidar_size,show_gs,gs_size,ellipsoids):control.on_update(display_only)
    for control in (lidar_all,scope):control.on_update(update_clouds)
    def save(_):
        with lock:
            folder,archive=backend.export(state['current'])
            gui.add_markdown(f'Saved: `{folder}/index.html`\n\nShare: `{archive}`')
    export.on_click(save)
    refresh()
    return server


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--iteration',type=int,default=30000)
    p.add_argument('--image',default='06663.jpg')
    p.add_argument('--point-cloud',type=Path,help='Default: MODEL/input.ply')
    p.add_argument('--scene',type=Path,help='Optional reference mesh, visual only; never used for culling')
    p.add_argument('--gpu',default='0')
    p.add_argument('--large-radius',type=float,default=20.,help='Diagnostic CUDA bounding-radius threshold in pixels')
    p.add_argument('--roi',type=float,nargs=4,default=(0,0,1,1),metavar=('X0','Y0','X1','Y1'))
    p.add_argument('--ellipsoids',type=int,default=100)
    p.add_argument('--max-points',type=int,default=250000,help='Display cap per cloud; 0 displays all. Statistics always use all points.')
    p.add_argument('--probe-gradients',action='store_true',help='Also measure this view\'s photometric screen-position gradients')
    p.add_argument('--output',type=Path)
    p.add_argument('--export-only',action='store_true',help='Write images/JSON/CSV/ZIP without starting Viser')
    p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8087)
    args=p.parse_args(argv)
    if args.ellipsoids<1 or args.max_points<0 or not np.isfinite(args.large_radius) or args.large_radius<=0:
        p.error('Invalid footprint/display limits')
    args.model=args.model.expanduser().resolve()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    from scripts.densification_inspector_backend import DensificationInspector
    print(f'Loading checkpoint and LiDAR for size diagnostics: {args.model}',flush=True)
    backend=DensificationInspector(args)
    if args.image not in backend.records:raise ValueError(f'Unregistered image: {args.image}')
    if args.export_only:
        backend.export(backend.analyze(args.image,args.large_radius,args.roi))
        return 0
    server=build_viewer(backend,args)
    print(f'Densification audit: http://localhost:{server.get_port()}',flush=True)
    try:
        while True:time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
