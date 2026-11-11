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
LIDAR_LAYERS=('all','outside','frustum','culled','on_surface','uncertain','unknown','in_front')
GAUSSIAN_LAYERS=('fragment_blocked','fragment_partial','contributing','fragment_inactive',
                 'fragment_frustum','on_surface')


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
    p.add_argument('--point-set',choices=('Both','LiDAR','Gaussians'),
                   help='Initial visibility only; defaults to Both with a checkpoint. Toggles are independent.')
    p.add_argument('--mesh-style',choices=('hidden','wireframe','filled'),default='hidden',
                   help='Visual mesh overlay only; does not change CUDA culling')
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
    initial_set=args.point_set or ('Both' if 'Gaussians' in backend.point_sets else 'LiDAR')
    if initial_set in ('Both','Gaussians') and 'Gaussians' not in backend.point_sets:
        raise ValueError('Gaussian inspection requires --model and a saved checkpoint')
    state={'current':backend.analyze(args.image)}
    lock=threading.RLock()
    server=viser.ViserServer(host=args.host,port=args.port,label='CUDA ray occlusion / fixed surface')
    mesh=server.scene.add_mesh_simple('/surface',backend.vertices,backend.faces,color=(65,130,240),
        opacity=.15,side='double',wireframe=args.mesh_style=='wireframe',
        visible=args.mesh_style!='hidden',cast_shadow=False,receive_shadow=False)
    supplement=None
    if getattr(backend,'supplement_arrays',None) is not None:
        supplement=server.scene.add_mesh_simple('/supplement',*backend.supplement_arrays,color=(240,145,65),
            opacity=.2,side='double',visible=False)
    gui=server.gui
    gui.add_markdown('## CUDA ray occlusion\n'
        'LiDAR and Gaussian layers have **independent visibility**. Gaussians are shown as colored centers. '
        'The small RGB plane displays the camera image. **Blue mesh:** the fixed triangulated occluder; '
        'its display is initially hidden so it cannot cover the points. Changing its display does not change culling.')
    image=gui.add_dropdown('Camera',backend.names,initial_value=args.image)
    previous=gui.add_button('Previous frame')
    following=gui.add_button('Next frame')
    previous.on_click(lambda _:setattr(image,'value',backend.names[max(0,backend.names.index(image.value)-1)]))
    following.on_click(lambda _:setattr(image,'value',backend.names[min(len(backend.names)-1,backend.names.index(image.value)+1)]))
    show={'LiDAR':gui.add_checkbox('Show LiDAR',initial_value=initial_set in ('Both','LiDAR'))}
    if 'Gaussians' in backend.point_sets:
        show['Gaussians']=gui.add_checkbox('Show Gaussians',initial_value=initial_set in ('Both','Gaussians'))
    else:
        gui.add_markdown('**No Gaussian checkpoint loaded.** Supply `--model` to show trained Gaussians.')
    gui.add_markdown('**Gaussian states:** red = fully blocked; orange = partially blocked; '
        'green = contributing with no blocked fragments; gray = no encountered contribution. '
        'Raw LiDAR uses its RGB colors when available. These clouds may overlap at the same locations; '
        'toggle either one off to inspect the other.')
    status=gui.add_markdown('')
    toggles,sizes={},{}
    with gui.add_folder('LiDAR layer'):
        lidar_style=gui.add_dropdown('LiDAR colors',('Original / neutral','Depth classes'),initial_value='Original / neutral')
        sizes['LiDAR']=gui.add_slider('LiDAR point size',min=.001,max=.04,step=.001,initial_value=.004)
        gui.add_markdown('Depth classes: yellow = near surface, red = behind, orange = tolerance band, '
            'purple = missing depth, green = in front, gray = outside frustum. Checks below apply to Depth classes.')
        toggles['LiDAR']={key:gui.add_checkbox(LAYERS[key][0],initial_value=key in default_layers('LiDAR')) for key in LIDAR_LAYERS}
    if 'Gaussians' in show:
        with gui.add_folder('Gaussian layer'):
            sizes['Gaussians']=gui.add_slider('Gaussian point size',min=.001,max=.04,step=.001,initial_value=.008)
            toggles['Gaussians']={key:gui.add_checkbox(LAYERS[key][0],initial_value=key in default_layers('Gaussians')) for key in GAUSSIAN_LAYERS}
    mesh_style=gui.add_dropdown('Surface display (visual only)',('hidden','wireframe','filled'),initial_value=args.mesh_style)
    isolate=gui.add_button('Show contributing Gaussians only') if 'Gaussians' in show else None
    if supplement is not None:
        extra_toggle=gui.add_checkbox('Show DA3 candidates (used only on rays missing LiDAR)',initial_value=False)
        extra_toggle.on_update(lambda _:setattr(supplement,'visible',extra_toggle.value))
    opacity=gui.add_slider('Surface opacity',min=.05,max=1.,step=.05,initial_value=.15)
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
    def update_layer(point_set,key):
        node_key=(point_set,key)
        visible=show[point_set].value and toggles[point_set][key].value
        if point_set=='LiDAR':visible=visible and lidar_style.value=='Depth classes'
        point_size=sizes[point_set].value*(.65 if key in ('all','outside') else 1.25)
        if node_key in nodes:
            nodes[node_key].visible=visible
            nodes[node_key].point_size=point_size
            return
        if not visible:return
        current=state['current'];points=backend.point_sets[point_set]
        ids=current['groups'][point_set].get(key+'_ids')
        if key=='all':ids=np.flatnonzero(np.isfinite(points).all(1))
        if ids is None or not len(ids):return
        if args.max_points and len(ids)>args.max_points:
            ids=ids[np.linspace(0,len(ids)-1,args.max_points,dtype=int)]
        nodes[node_key]=server.scene.add_point_cloud('/points/'+point_set+'/'+key,points[ids],colors=LAYERS[key][1],
            point_size=point_size)
    def refresh_clouds():
        with lock:
            raw_visible=show['LiDAR'].value and lidar_style.value!='Depth classes'
            raw=nodes.get(('LiDAR','raw'))
            if raw is None and raw_visible:
                points=backend.point_sets['LiDAR']
                ids=np.flatnonzero(np.isfinite(points).all(1))
                if args.max_points and len(ids)>args.max_points:
                    ids=ids[np.linspace(0,len(ids)-1,args.max_points,dtype=int)]
                colors=getattr(backend,'lidar_colors',None)
                raw=server.scene.add_point_cloud('/points/LiDAR/raw',points[ids],
                    colors=colors[ids] if colors is not None else (120,145,180),point_size=sizes['LiDAR'].value)
                nodes[('LiDAR','raw')]=raw
            if raw is not None:
                raw.visible=raw_visible
                raw.point_size=sizes['LiDAR'].value
            for point_set,controls in toggles.items():
                for key in controls:update_layer(point_set,key)
    def refresh_unlocked():
        current=state['current']
        # Classification changes per camera. Visibility/size controls reuse
        # existing buffers instead of resending millions of unchanged points.
        for node_key in list(nodes):
            if isinstance(node_key,tuple) and node_key!=('LiDAR','raw'):
                nodes.pop(node_key).remove()
        refresh_clouds()
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
        fragment_text=(f"\n\n**CUDA contributing: {fragments['contributing_only']+fragments['partially_blocked']:,}** "
            '(green + orange; actual post-culling contributions). '
            f"Fully blocked: **{fragments['fully_blocked']:,}** "
            f"(**{fragments['fully_blocked_fraction_in_image']:.1%}** of pre-culling image candidates); "
            f"partly blocked: **{fragments['partially_blocked']:,}** "
            f"(**{fragments['partially_blocked_fraction_in_image']:.1%}**)." if fragments else '')
        counts_text='\n'.join(f"- **{name}: {values['total']:,} loaded**; "
            f"{values.get('frustum',0):,} centers in frustum; {values.get('on_surface',0):,} near surface."
            for name,values in current['counts'].items())
        status.content=(f"**{current['name']}**\n\n"
            f"{origin_text} Pixel depth coverage **{row['depth_coverage']:.1%}**.\n\n"+
            counts_text+
            audit_text+checkpoint_text+
            ('\n\n**Display:** all points.' if not args.max_points else f'\n\nDisplay cap: {args.max_points:,} per layer; counts are exact.')+
            ('\n\n'+row['gradient_probe']['verdict'] if 'gradient_probe' in row else '')+
            fragment_text)
        for key,(_,pixels) in current['images'].items():panels[key].image=pixels
    def refresh():
        with lock:refresh_unlocked()
    def change_view(_):
        with lock:
            status.content='Computing CUDA depth, point labels, and render checks…'
            try:state['current']=backend.analyze(image.value);refresh()
            except Exception as error:
                status.content=f'Inspection failed: {type(error).__name__}: {error}'
                raise
    image.on_update(change_view)
    for toggle in show.values():toggle.on_update(lambda _:refresh_clouds())
    lidar_style.on_update(lambda _:refresh_clouds())
    for size in sizes.values():size.on_update(lambda _:refresh_clouds())
    for controls in toggles.values():
        for toggle in controls.values():toggle.on_update(lambda _:refresh_clouds())
    def change_mesh_style(_):
        mesh.visible=mesh_style.value!='hidden'
        mesh.wireframe=mesh_style.value=='wireframe'
    mesh_style.on_update(change_mesh_style)
    opacity.on_update(lambda _:setattr(mesh,'opacity',opacity.value))
    if isolate is not None:
        def isolate_contributing(_):
            with lock:
                show['LiDAR'].value=False
                show['Gaussians'].value=True
                mesh_style.value='hidden'
                if supplement is not None:extra_toggle.value=False
                for key,toggle in toggles['Gaussians'].items():
                    toggle.value=key in ('contributing','fragment_partial')
        isolate.on_click(isolate_contributing)
    refresh()
    print(f'CUDA surface viewer: http://{args.host}:{server.get_port()}',flush=True)
    try:
        while True:time.sleep(1)
    except KeyboardInterrupt:server.stop()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
