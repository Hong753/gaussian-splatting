#!/usr/bin/env python3
"""Inspect scene/frustum/surface/occluded points and actual CUDA ray clipping."""
import argparse
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

LAYERS = {
    'all':('All scene points (overlays classifications)',(135,140,150),False),
    'outside':('Points outside camera frustum',(135,140,150),True),
    'frustum':('Points inside camera frustum',(30,195,235),False),
    'culled':('Points behind first surface',(245,60,65),True),
    'on_surface':('Points within narrow surface band',(245,200,35),True),
    'uncertain':('Behind surface but within culling margin',(235,135,45),True),
    'unknown':('In-frustum points without surface depth',(155,85,205),True),
    'in_front':('Measured points in front of mesh (inspect gaps)',(65,195,100),True),
    'fragment_blocked':('Gaussians fully blocked by CUDA',(245,45,45),False),
    'fragment_partial':('Gaussians partly blocked by CUDA',(255,135,35),False),
    'contributing':('Contributing Gaussians with no blocked fragments',(35,210,95),False),
    'fragment_inactive':('Gaussians with no encountered contribution',(135,140,150),False),
    'fragment_frustum':('Gaussian image overlap before clipping (overlay)',(30,195,235),False),
}


def default_layers(point_set):
    if point_set == 'Gaussians':
        return {'fragment_blocked', 'fragment_partial', 'contributing', 'fragment_inactive'}
    return {key for key, (_, _, enabled) in LAYERS.items() if enabled}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--point-cloud',type=Path,required=True)
    p.add_argument('--model',type=Path,help='Native checkpoint for actual ON/OFF render and fragment flags')
    p.add_argument('--iteration',type=int,default=-1)
    p.add_argument('--point-set',choices=('LiDAR','Gaussians'),
                   help='Defaults to actual Gaussian CUDA states when a checkpoint is supplied')
    p.add_argument('--image',required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--margin',type=float,default=.05)
    p.add_argument('--surface-band',type=float,default=.03,help='Yellow proximity band, independent of culling tolerance')
    p.add_argument('--output',type=Path)
    p.add_argument('--check-gradients',action='store_true')
    p.add_argument('--host',default='0.0.0.0')
    p.add_argument('--port',type=int,default=8085)
    p.add_argument('--max-points',type=int,default=0,help='Display cap per layer; 0 displays every point')
    args=p.parse_args(argv)
    if (args.max_points<0 or not np.isfinite([args.margin,args.surface_band]).all() or
            min(args.margin,args.surface_band)<0):
        raise ValueError('Invalid display count or margin')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import viser
    import viser.transforms as vtf
    from scripts.ray_inspector_backend import RayInspector
    backend=RayInspector(args)
    if args.image not in backend.names: raise ValueError('Requested image is not registered')
    initial_set=args.point_set or ('Gaussians' if 'Gaussians' in backend.point_sets else 'LiDAR')
    if initial_set not in backend.point_sets: raise ValueError('Gaussian inspection requires --model and a saved checkpoint')
    state={'current':backend.analyze(args.image)}
    lock=threading.Lock()
    server=viser.ViserServer(host=args.host,port=args.port,label='CUDA ray occlusion / fixed surface')
    mesh=server.scene.add_mesh_simple('/surface',backend.vertices,backend.faces,color=(65,130,240),
        opacity=.25,side='double')
    supplement=None
    if getattr(backend,'supplement_arrays',None) is not None:
        supplement=server.scene.add_mesh_simple('/supplement',*backend.supplement_arrays,color=(240,145,65),
            opacity=.2,side='double',visible=False)
    gui=server.gui
    gui.add_markdown('## CUDA ray occlusion\n'
        '**Blue mesh:** fixed occlusion surface. The small RGB plane displays the camera image. '
        'Gaussian colors describe the actual CUDA compositor for the selected camera; '
        'LiDAR colors describe point-center depth tests. Optional center/scene overlays can overlap these colors.')
    image=gui.add_dropdown('Camera',backend.names,initial_value=args.image)
    previous=gui.add_button('Previous frame')
    following=gui.add_button('Next frame')
    previous.on_click(lambda _:setattr(image,'value',backend.names[max(0,backend.names.index(image.value)-1)]))
    following.on_click(lambda _:setattr(image,'value',backend.names[min(len(backend.names)-1,backend.names.index(image.value)+1)]))
    point_set=gui.add_dropdown('Point set',list(backend.point_sets),initial_value=initial_set)
    legend=gui.add_markdown('')
    status=gui.add_markdown('')
    with gui.add_folder('Point layers'):
        toggles={key:gui.add_checkbox(label,initial_value=key in default_layers(initial_set)) for key,(label,color,visible) in LAYERS.items()}
        size=gui.add_slider('Point size',min=.002,max=.04,step=.002,initial_value=.008)
    mesh_toggle=gui.add_checkbox('Show occlusion surface',initial_value=True)
    if supplement is not None:
        extra_toggle=gui.add_checkbox('Show DA3 candidates (used only on rays missing LiDAR)',initial_value=False)
        extra_toggle.on_update(lambda _:setattr(supplement,'visible',extra_toggle.value))
    opacity=gui.add_slider('Surface opacity',min=.05,max=1.,step=.05,initial_value=.25)
    overview=gui.add_button('Scene overview')
    reference=gui.add_button('Look from reference camera')
    panels={}
    with gui.add_folder('Actual image / CUDA render / depth'):
        for key,(title,pixels) in state['current']['images'].items():panels[key]=gui.add_image(pixels,label=title)
    nodes={}
    def overview_pose(client):
        c2w=state['current']['c2w'];origin=c2w[:3,3]
        client.camera.position=origin+4*c2w[:3,0]-3*c2w[:3,1]-3*c2w[:3,2]
        client.camera.look_at=origin+3*c2w[:3,2]
        client.camera.up_direction=-c2w[:3,1]
    def reference_pose(client):
        c2w=state['current']['c2w'];origin=c2w[:3,3]
        client.camera.position=origin
        client.camera.look_at=origin+c2w[:3,2]
        client.camera.up_direction=-c2w[:3,1]
    server.on_client_connect(overview_pose)
    overview.on_click(lambda _:[overview_pose(c) for c in server.get_clients().values()])
    reference.on_click(lambda _:[reference_pose(c) for c in server.get_clients().values()])
    def update_layer(key):
        if key in nodes:nodes.pop(key).remove()
        current=state['current'];points=backend.point_sets[point_set.value]
        ids=current['groups'][point_set.value].get(key+'_ids')
        if key=='all':ids=np.flatnonzero(np.isfinite(points).all(1))
        if not toggles[key].value or ids is None or not len(ids):return
        if args.max_points and len(ids)>args.max_points:
            ids=ids[np.linspace(0,len(ids)-1,args.max_points,dtype=int)]
        nodes[key]=server.scene.add_point_cloud('/points/'+key,points[ids],colors=LAYERS[key][1],
            point_size=size.value*(.65 if key in ('all','outside') else 1.25))
    def refresh():
        current=state['current']
        if point_set.value == 'Gaussians':
            legend.content=('**Red:** fully blocked by CUDA. **Orange:** partly blocked, still contributing. '
                '**Green:** contributing, no blocked fragments. **Gray:** no encountered contribution '
                '(may be outside the image, below the opacity threshold, or skipped by early termination). '
                'These four default groups are disjoint.')
        else:
            legend.content=('**Gray:** outside frustum. **Red:** behind the first surface. '
                f'**Yellow:** within {args.surface_band:g} scene units of it. '
                '**Orange:** culling tolerance band. **Purple:** missing depth. **Green:** in front of mesh.')
        for key in toggles:update_layer(key)
        k,c2w=current['k'],current['c2w']
        rgb=current['images']['gt'][1];h,w=rgb.shape[:2]
        if 'camera' in nodes:nodes.pop('camera').remove()
        nodes['camera']=server.scene.add_camera_frustum('/camera',fov=2*np.arctan(h/(2*k[1,1])),
            aspect=w/h,scale=.5,image=rgb[::max(1,w//640),::max(1,w//640)],
            wxyz=vtf.SO3.from_matrix(c2w[:3,:3]).wxyz,position=c2w[:3,3])
        # Draw a frustum extending into the scene, rather than only the RGB plane.
        length=5.
        rays=np.array([[-.5,-.5],[w-.5,-.5],[w-.5,h-.5],[-.5,h-.5]],np.float32)
        corners=np.column_stack(((rays[:,0]-k[0,2])/k[0,0],(rays[:,1]-k[1,2])/k[1,1],np.ones(4)))*length
        corners=corners@c2w[:3,:3].T+c2w[:3,3]
        segments=np.array([[c2w[:3,3],v] for v in corners]+[[corners[i],corners[(i+1)%4]] for i in range(4)])
        if 'frustum_lines' in nodes:nodes.pop('frustum_lines').remove()
        nodes['frustum_lines']=server.scene.add_line_segments('/frustum_lines',segments,colors=(30,195,235),line_width=1.5)
        row=current['report']
        if row.get('geometry_source')=='full_initialization_lidar':
            stats=row['lidar_surface']
            origin_text=(f"Primary: **full LiDAR mesh**, {stats['triangles']:,} triangles; "
                f"{stats['meshed_representative_fraction']:.1%} of measured representatives connected.")
        else:
            origin_text=f"Surface fused from **{row['integrated_frames']} views**."
        audit=row.get('first_surface_audit')
        audit_text=(f"\n\nForeground audit: **{audit['foreground_miss_fraction_of_measured']:.1%}** of measured pixels "
                    'have no mesh hit or a hit behind the measured foreground. See the RGB overlay.' if audit else '')
        checkpoint=row.get('checkpoint')
        checkpoint_text=(f"\n\n**Saved checkpoint: {checkpoint['iteration']:,} steps.** "
            'All Gaussian rows in this checkpoint are tested, including densified children. '
            'Restart the viewer to load a newer checkpoint; this is not live training state.' if checkpoint else '')
        fragments=row.get('fragments')
        fragment_text=(f"\n\nCUDA fully blocked: **{fragments['fully_blocked']:,}** "
            f"(**{fragments['fully_blocked_fraction_in_image']:.1%}** of pre-culling image candidates); "
            f"partly blocked: **{fragments['partially_blocked']:,}** "
            f"(**{fragments['partially_blocked_fraction_in_image']:.1%}**)." if fragments else '')
        status.content=(f"**{current['name']} · {point_set.value}**\n\n"
            f"{origin_text} Pixel depth coverage **{row['depth_coverage']:.1%}**.\n\n"+
            '\n'.join(f'- {key}: **{count:,}**' for key,count in current['counts'][point_set.value].items())+
            audit_text+checkpoint_text+
            ('\n\n**Display:** all points.' if not args.max_points else f'\n\nDisplay cap: {args.max_points:,} per layer; counts are exact.')+
            ('\n\n'+row['gradient_probe']['verdict'] if 'gradient_probe' in row else '')+
            fragment_text)
        for key,(_,pixels) in current['images'].items():panels[key].image=pixels
    def change_view(_):
        with lock:
            status.content='Computing CUDA depth, point labels, and render checks…'
            try:state['current']=backend.analyze(image.value);refresh()
            except Exception as error:
                status.content=f'Inspection failed: {type(error).__name__}: {error}'
                raise
    image.on_update(change_view)
    def change_point_set(_):
        for key,toggle in toggles.items():toggle.value=key in default_layers(point_set.value)
        refresh()
    point_set.on_update(change_point_set)
    size.on_update(lambda _:refresh())
    for key,toggle in toggles.items():toggle.on_update(lambda _,key=key:update_layer(key))
    mesh_toggle.on_update(lambda _:setattr(mesh,'visible',mesh_toggle.value))
    opacity.on_update(lambda _:setattr(mesh,'opacity',opacity.value))
    refresh()
    print(f'CUDA surface viewer: http://{args.host}:{server.get_port()}',flush=True)
    try:
        while True:time.sleep(1)
    except KeyboardInterrupt:server.stop()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
