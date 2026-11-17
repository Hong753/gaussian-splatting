"""Viser controls for reviewing automatic camera candidates before export."""
from functools import lru_cache
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from scripts.region_dataset import inside
from scripts.region_review import CameraReview


def create_app(server, data, output, review_file, identity, initial, session=None, anchor=None):
    """Return controls/actions as well as registering UI callbacks for testability."""
    server.scene.set_up_direction('+z')
    server.gui.configure_theme(control_width='large')
    server.gui.add_markdown('## Review region camera candidates\nAutomatic projection only proposes candidates; visibility is not verified. Review the **actual RGB**, then keep or reject each image. Only explicitly kept images will be exported. Choices save automatically.')
    status = server.gui.add_markdown('Find candidates, then review their images.')
    find = server.gui.add_button('Find candidate views')
    review_status = server.gui.add_markdown('No candidates yet.')
    filter_control = server.gui.add_dropdown('Review filter',options=['All','Unreviewed','Kept','Rejected'])
    candidate = server.gui.add_dropdown('Candidate image',options=['(find candidates first)'],disabled=True)
    previous = server.gui.add_button('Previous image',disabled=True)
    following = server.gui.add_button('Next image',disabled=True)
    keep = server.gui.add_button('Keep and next',disabled=True,color='green')
    reject = server.gui.add_button('Reject and next',disabled=True,color='red')
    unreviewed = server.gui.add_button('Mark unreviewed',disabled=True)
    export = server.gui.add_button('Export kept images',disabled=True)
    rgb = server.gui.add_image(np.zeros((128,192,3),np.uint8),label='Actual RGB — unmasked',format='jpeg')
    show_mask = server.gui.add_checkbox('Show approximate ROI mask',initial_value=False)
    overlay = server.gui.add_image(np.zeros((128,192,3),np.uint8),label='Approximate ROI — NOT verified visibility',visible=False)
    with server.gui.add_folder('Region box and candidate search'):
        server.gui.add_markdown('Gray: scene LiDAR. Yellow: box LiDAR. Blue: retained halo. Camera dots: gray=unreviewed, red=rejected, green=kept train, cyan=kept held-out. Changing the box/search starts a new review after finding candidates again.')
        center = server.gui.add_vector3('Box center',initial_value=tuple(initial['center']),step=.1)
        size = server.gui.add_vector3('Box XYZ size',initial_value=tuple(initial['size']),min=(.1,.1,.1),step=.1)
        halo = server.gui.add_number('LiDAR halo',initial_value=initial['halo'],min=0.,step=.1)
        overlap = server.gui.add_number('Minimum image fraction',initial_value=initial['min_overlap'],min=.001,max=1.,step=.01)
        distance = server.gui.add_number('Camera distance to box (0=all)',initial_value=initial['max_camera_distance'],min=0.,step=1.)
        first = server.gui.add_text('First image (optional)',initial_value=initial['frame_start'])
        last = server.gui.add_text('Last image (optional)',initial_value=initial['frame_end'])
        excluded = server.gui.add_text('Exclude names before search',initial_value=','.join(initial['exclude']),multiline=True)
        inspector = server.gui.add_dropdown('Inspect any source image',options=[im.name for im in data.images],initial_value=anchor or data.images[0].name)
        recenter = server.gui.add_button('Center box on inspected image')
    state = dict(session=session,busy=False,completed=False)
    lock = threading.RLock()
    filters = dict(All='all',Unreviewed='unreviewed',Kept='keep',Rejected='reject')
    lookup = {im.name:im for im in data.images}
    geometric = (center,size,halo,overlap,distance,first,last,excluded,recenter)
    server.scene.add_point_cloud('/scene_lidar',points=data.points[::3],colors=(110,120,135),point_size=.012,precision='float32')
    server.scene.add_point_cloud('/all_cameras',points=np.array([im.C for im in data.images]),colors=(170,170,190),point_size=.035,precision='float32')

    def config():
        return dict(center=list(center.value),size=list(size.value),halo=halo.value,min_overlap=overlap.value,
                    max_camera_distance=distance.value,frame_start=first.value,frame_end=last.value,
                    exclude=[x.strip() for x in excluded.value.replace('\n',',').split(',') if x.strip()])

    @lru_cache(maxsize=4)
    def raw_image(name):
        return data.rgb(name)

    def show_image(name):
        actual=raw_image(name)
        rgb.image=actual
        overlay.visible=show_mask.value
        if show_mask.value:
            overlay.image=data.overlay(name,center.value,size.value)[1]
        im=lookup[name]; cam=data.cameras[im.camera_id]
        thumbnail=Image.fromarray(actual); thumbnail.thumbnail((320,320))
        server.scene.add_camera_frustum('/inspected_camera',fov=2*np.arctan(cam.height/(2*cam.params[1])),
            aspect=cam.width/cam.height,wxyz=Rotation.from_matrix(im.R.T).as_quat()[[3,0,1,2]],position=im.C,
            scale=.4,color=(255,180,40),image=np.asarray(thumbnail))

    def draw_box():
        core=inside(data.points,center.value,size.value)
        support=inside(data.points,center.value,size.value,halo.value)&~core
        server.scene.add_box('/region_box',dimensions=size.value,position=center.value,color=(255,210,40),wireframe=True)
        server.scene.add_box('/halo_box',dimensions=np.array(size.value)+2*halo.value,position=center.value,color=(90,140,250),wireframe=True)
        server.scene.add_point_cloud('/core_lidar',points=data.points[core],colors=(255,205,40),point_size=.015,precision='float32')
        server.scene.add_point_cloud('/halo_lidar',points=data.points[support],colors=(80,140,250),point_size=.012,precision='float32')

    def refresh(name=None):
        s=state['session']
        if s is None:
            for control in (candidate,previous,following,keep,reject,unreviewed,export): control.disabled=True
            review_status.content='No current candidate review. Find candidate views first.'
            server.scene.add_point_cloud('/candidate_cameras',points=np.empty((0,3)),colors=(150,150,150),point_size=.09,precision='float32')
            return
        available=s.filtered(filters[filter_control.value])
        chosen=name if name in available else (s.current if s.current in available else (available[0] if available else None))
        if chosen: s.current=chosen
        candidate.options=available or ['(no images in this filter)']
        candidate.value=chosen or '(no images in this filter)'
        for control in (candidate,previous,following): control.disabled=not available or state['busy']
        for control in (keep,reject,unreviewed): control.disabled=not available or state['busy'] or state['completed']
        counts=s.counts()
        export.disabled=counts['train']<2 or state['busy'] or state['completed']
        current = f"\n**{s.current}** · {s.decisions[s.current]}" if chosen else '\nNo images in this filter; select another filter to revisit decisions.'
        review_status.content=(f"**{counts['kept']} kept** ({counts['train']} train + {counts['heldout']} held-out), "
            f"**{counts['rejected']} rejected**, **{counts['unreviewed']} unreviewed** / {counts['candidates']} candidates."+current+
            '\nRejected and unreviewed images are excluded from BOTH training and evaluation. At least two kept training views are needed.')
        colors=[(240,70,70) if s.decisions[r['name']]=='reject' else
                ((50,230,120) if r['split']=='train' else (50,210,255)) if s.decisions[r['name']]=='keep' else (160,160,170)
                for r in s.rows]
        server.scene.add_point_cloud('/candidate_cameras',points=np.array([lookup[n].C for n in s.names]),
            colors=np.array(colors,np.uint8),point_size=.09,precision='float32')
        if chosen: show_image(chosen)

    def busy(value):
        state['busy']=value
        for control in (*geometric,find,inspector): control.disabled=value or state['completed']
        refresh()

    def decide(value):
        with lock:
            s=state['session']; name=candidate.value
            if s is None or state['busy'] or state['completed'] or name not in s.names: return
            s.choose(name,value)
            next_name=s.neighbor(name,1,filters[filter_control.value]) if value!='unreviewed' else name
            if next_name: s.current=next_name
            s.save(review_file)
            refresh(next_name)

    def move(direction):
        with lock:
            s=state['session']
            if s is None or state['busy']: return
            name=s.neighbor(candidate.value,direction,filters[filter_control.value])
            if name:
                s.current=name; s.save(review_file); refresh(name)

    @keep.on_click
    def keep_image(_): decide('keep')
    @reject.on_click
    def reject_image(_): decide('reject')
    @unreviewed.on_click
    def reset_image(_): decide('unreviewed')
    @previous.on_click
    def previous_image(_): move(-1)
    @following.on_click
    def next_image(_): move(1)
    @candidate.on_update
    def inspect_candidate(_):
        with lock:
            s=state['session']
            if s is not None and not state['busy'] and candidate.value in s.names and s.current!=candidate.value:
                s.current=candidate.value; s.save(review_file); refresh(s.current)
    @filter_control.on_update
    def change_filter(_):
        with lock: refresh()
    @show_mask.on_update
    def toggle_mask(_):
        with lock:
            s=state['session']
            show_image(s.current if s is not None else inspector.value)
    @inspector.on_update
    def inspect_source(_):
        with lock:
            # Keep the displayed image and Keep/Reject target unambiguous.
            s=state['session']
            if s is not None:
                status.content='Review is active: use Candidate image to inspect and classify candidates. The source-image selector is only used to recenter the box.'
                return
            show_image(inspector.value)
    @recenter.on_click
    def center_on_camera(_):
        with lock:
            try: center.value=tuple(data.anchor_center(inspector.value))
            except Exception as exc: status.content=str(exc)

    def changed(_):
        with lock:
            if state['busy'] or state['completed']: return
            state['session']=None
            refresh()
            status.content='Region/search changed. Find candidate views again to start a new review. The last saved review is retained until then.'
            try: draw_box(); show_image(inspector.value)
            except Exception as exc: status.content=str(exc)
    for control in geometric[:-1]: control.on_update(changed)

    @find.on_click
    def select(_):
        with lock:
            if state['busy'] or state['completed']: return
            busy(True)
            try:
                cfg=config()
                status.content='Finding approximate camera candidates…'
                rows=data.find(**cfg,progress=lambda n,total,kept: setattr(status,'content',f'Checked {n}/{total} cameras; {kept} candidates.'))
                if cfg!=config(): raise ValueError('Search changed; find candidates again')
                # Repeating the same search preserves prior decisions.
                old=state['session']
                if old is None or old.config!=cfg or old.rows!=rows:
                    state['session']=CameraReview(identity,cfg,rows)
                state['session'].save(review_file)
                status.content=f'Review actual RGB images. Autosaved to `{review_file}`. Keep only useful, visible views of your region.'
            except Exception as exc: status.content=f'Candidate search failed: {exc}'
            finally: busy(False)

    @export.on_click
    def save(_):
        with lock:
            s=state['session']
            if s is None or state['busy'] or state['completed']: return
            busy(True)
            try:
                if s.config!=config(): raise ValueError('Review is stale; find candidates again')
                if s.counts()['train']<2: raise ValueError('Keep at least two training views')
                status.content='Exporting only kept images and their original poses, plus regional LiDAR…'
                s.save(review_file)
                data.export(output,rows=s.rows,review=s.decisions,**s.config)
                state['completed']=True
                status.content=f'**Export complete:** `{output}`. Only kept images were exported. Review decisions are in `review.json`; original source files are unchanged.'
            except Exception as exc: status.content=f'Export failed: {exc}'
            finally: busy(False)
    @server.on_client_connect
    def connect(client):
        client.camera.look_at=np.asarray(center.value)
        client.camera.position=np.asarray(center.value)+np.array([6.,-6.,4.])
    draw_box()
    if session is not None:
        status.content=f'Restored saved review from `{review_file}`. Continue reviewing or export kept images.'
    else: show_image(inspector.value)
    refresh()
    return dict(state=state,select=select,save=save,decide=decide,move=move,refresh=refresh,
                controls=dict(candidate=candidate,filter=filter_control,keep=keep,reject=reject,
                              unreviewed=unreviewed,export=export,status=status,summary=review_status,
                              rgb=rgb,overlay=overlay,center=center,inspector=inspector))
