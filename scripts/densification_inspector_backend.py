"""CUDA renders and complete-population footprint statistics for saved native GS."""
import csv
import html
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from utils.densification_diagnostics import (COLORS, LABELS, training_extent, classify_sizes,
    project_centers, roi_pixels, footprint_roi, center_frustum, size_report, ellipsoid_segments, projected_ellipses)


class DensificationInspector:
    def __init__(self, args):
        import torch
        from scene.gaussian_model import GaussianModel
        from scripts.view_comparisons import model_info, choose_iteration
        from scripts.ray_inspector_backend import read_lidar_cloud
        self.args=args
        if not torch.cuda.is_available():
            raise RuntimeError('Activate the CUDA-enabled 3dgs environment for this diagnostic')
        self.spec=model_info(args.model)
        if self.spec['type']!='3dgs':
            raise ValueError('Choose a native 3DGS/soft-surface checkpoint')
        self.cfg=self.spec['cfg'];self.records=self.spec['records'];self.names=sorted(self.records)
        self.iteration,self.ply=choose_iteration(self.spec['path'],args.iteration)
        from utils.checkpoint_snapshots import digest
        self.identity=dict(ply_sha256=digest(self.ply),cfg_sha256=digest(args.model/'cfg_args'))
        self.model=GaussianModel(self.cfg.get('sh_degree',3));self.model.load_ply(str(self.ply))
        for value in vars(self.model).values():
            if torch.is_tensor(value) and value.requires_grad:value.requires_grad_(False)
        self.points=self.model.get_xyz.detach().cpu().numpy()
        self.max_sigma=self.model.get_scaling.amax(dim=1).cpu().numpy()
        self.opacities=self.model.get_opacity[:,0].cpu().numpy()
        self.opt=self.cfg.get('optimization_settings',{})
        self.extent=training_extent(self.records,self.spec['splits']['train'])
        self.percent_dense=self.opt.get('percent_dense',.01)
        self.cutoff=self.extent*self.percent_dense
        self.lidar,self.lidar_colors=read_lidar_cloud(args.point_cloud or args.model/'input.ply')
        self.pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,
                                  antialiasing=True,occlusion_mode='off')
        self.bg=torch.tensor([1.,1.,1.] if self.cfg.get('white_background') else [0.,0.,0.],device='cuda')
        self.frame=None;self.output=args.output or args.model/'densification_inspection'/f'iteration_{self.iteration}'

    def load_view(self,name):
        import torch
        from gaussian_renderer import render
        from utils.camera_utils import loadCam
        from utils.loss_utils import l1_loss,ssim
        record=self.records[name]
        info=SimpleNamespace(uid=0,image_path=record['image_path'],image_name=name,R=np.array(record['R']),
            T=np.array(record['T']),FovX=record['fov_x'],FovY=record['fov_y'],depth_path='',depth_params=None,is_test=False)
        camera=loadCam(SimpleNamespace(resolution=self.cfg.get('resolution',-1),data_device='cpu',train_test_exp=False),
                       0,info,1.,False,False)
        gradient=None
        with torch.enable_grad() if self.args.probe_gradients else torch.no_grad():
            result=render(camera,self.model,self.pipe,self.bg,use_trained_exp=False)
            if self.args.probe_gradients:
                gt=camera.original_image[:3].to('cuda')
                weight=self.opt.get('lambda_dssim',.2)
                loss=(1-weight)*l1_loss(result['render'],gt)+weight*(1-ssim(result['render'],gt))
                grad=torch.autograd.grad(loss,result['viewspace_points'])[0]
                gradient=grad[:,:2].norm(dim=-1).detach().cpu().numpy()
            pred=result['render'].detach().cpu().numpy()
            radii=result['radii'].detach().cpu().numpy()
        del result
        target=camera.original_image[:3].cpu().numpy()
        view=camera.world_view_transform.cpu().numpy();projection=camera.full_proj_transform.cpu().numpy()
        w,h=camera.image_width,camera.image_height
        projected=project_centers(self.points,view,projection,w,h)
        lidar_projection=project_centers(self.lidar,view,projection,w,h)
        self.frame=dict(name=name,camera=camera,view=view,width=w,height=h,pred=pred,target=target,radii=radii,
            projected=projected,gradient=gradient,frustum=center_frustum(projected,w,h),
            lidar_frustum=center_frustum(lidar_projection,w,h),c2w=np.linalg.inv(view.T),radius=None)

    def analyze(self,name,large_radius=20.,roi=(0,0,1,1),focus='large_clone_size'):
        import torch
        from gaussian_renderer import render
        from PIL import Image,ImageDraw
        if self.frame is None or self.frame['name']!=name:self.load_view(name)
        f=self.frame;w,h=f['width'],f['height']
        rectangle=roi_pixels(roi,w,h)
        if f['radius']!=large_radius:
            labels=classify_sizes(self.max_sigma,f['radii'],self.cutoff,large_radius)
            # RGB channels encode compositing weights, independent of SH colors.
            # All Gaussians remain in transmittance, so occluded candidates do
            # not inflate this measure of what actually contributes to the image.
            with torch.no_grad():
                color=torch.zeros_like(self.model.get_xyz)
                classes=torch.as_tensor(labels,device=self.model.get_xyz.device)
                color[:,0]=(classes==2);color[:,1]=(classes==3);color[:,2]=(classes<2)
                result=render(f['camera'],self.model,self.pipe,self.bg.new_zeros(3),override_color=color)
                weights=result['render'].cpu().numpy()
            f.update(radius=large_radius,labels=labels,weights=weights)
        mask=footprint_roi(f['projected'],f['radii'],rectangle)
        report=size_report(self.max_sigma,f['radii'],f['labels'],mask,f['weights'],rectangle)
        phase=self.iteration-self.opt.get('densification_phase_offset',self.cfg.get('densification_phase_offset',0))
        interval=self.opt.get('densification_interval',100)
        active=self.opt.get('densify_from_iter',500)<phase<self.opt.get('densify_until_iter',15000)
        report.update(image=name,iteration=self.iteration,model=str(self.args.model.resolve()),ply=str(self.ply),
            selected_for_training=name in self.spec['splits']['train'],antialiasing=True,hard_culling=False,
            total_gaussians=len(self.points),total_lidar=len(self.lidar),gaussian_centers_in_frustum=int(f['frustum'].sum()),
            lidar_centers_in_frustum=int(f['lidar_frustum'].sum()),raster_candidates=int((f['radii']>0).sum()),
            scene_extent=self.extent,percent_dense=self.percent_dense,split_sigma_threshold=self.cutoff,
            large_radius_pixels=large_radius,roi_normalized=list(roi),
            densification_window_open_at_checkpoint=active,phase_step=phase,densification_interval=interval,
            branch_rule='max(scale) <= percent_dense * selected-camera scene extent: clone-size; otherwise split-size',
            caveat='PLY size branches are hypothetical, not recorded growth events; gradient history is absent. '
                   'Footprints are CUDA 3-sigma bounding radii, not visibility. Colored weight images use native alpha compositing.',
            saved_surface_mode=self.opt.get('surface_mode',self.cfg.get('surface_mode','off')),
            assumed_upstream_percent_dense='percent_dense' not in self.opt)
        report['roi_size_branch_counts'] = dict(clone_size=int((mask & (self.max_sigma<=self.cutoff)).sum()),
                                               split_size=int((mask & (self.max_sigma>self.cutoff)).sum()))
        report['checkpoint_identity']=getattr(self,'identity',{})
        x0,y0,x1,y1=rectangle
        mse=((f['pred'][:,y0:y1,x0:x1]-f['target'][:,y0:y1,x0:x1])**2).mean(axis=(1,2))
        report['roi_psnr_db']=float((-10*np.log10(np.maximum(mse,1e-12))).mean())
        if f['gradient'] is not None:
            threshold=self.opt.get('densify_grad_threshold',.0002)
            high=f['gradient']>=threshold
            report['single_view_gradient_probe']=dict(threshold=threshold,
                high_gradient_clone_size=int((mask & high & (self.max_sigma<=self.cutoff)).sum()),
                high_gradient_split_size=int((mask & high & (self.max_sigma>self.cutoff)).sum()),
                high_gradient_large_clone_size=int((mask & high & (f['labels']==2)).sum()),
                note='This view only: RGB L1/SSIM screen-position gradient, not the training-window average')
        ids=np.flatnonzero(mask)
        if focus!='all':ids=ids[f['labels'][ids]==LABELS.index(focus)]
        order=np.argsort(-f['radii'][ids],kind='stable')[:self.args.ellipsoids]
        draw_ids=ids[order]
        scales=self.model.get_scaling[torch.as_tensor(draw_ids,device=self.model.get_xyz.device)].cpu().numpy()
        quaternion=self.model.get_rotation[torch.as_tensor(draw_ids,device=self.model.get_xyz.device)].cpu().numpy()
        segments=ellipsoid_segments(self.points[draw_ids],scales,quaternion)
        colors=np.repeat(COLORS[f['labels'][draw_ids]],3*32,axis=0)[:,None,:].repeat(2,axis=1)
        def rgb(array):return np.clip(array.transpose(1,2,0)*255,0,255).round().astype(np.uint8)
        gt,pred=rgb(f['target']),rgb(f['pred'])
        overlay=Image.fromarray(gt)
        draw=ImageDraw.Draw(overlay)
        contours=projected_ellipses(self.points[draw_ids],scales,quaternion,f['view'],w,h,f['camera'].FoVx,f['camera'].FoVy)
        for idx,contour in zip(draw_ids,contours):
            pixels=(contour+f['projected'][idx,:2]).clip(-1e6,1e6)
            draw.line([tuple(p) for p in pixels],fill=tuple(map(int,COLORS[f['labels'][idx]])),width=1)
        draw.rectangle((x0,y0,x1-1,y1-1),outline='white',width=3)
        palette=COLORS[[2,3,1]].astype(np.float32)
        weight_rgb=np.einsum('chw,cd->hwd',f['weights'],palette).clip(0,255).astype(np.uint8)
        images=dict(gt=('Actual image',gt),render=('Native 3DGS + AA, no hard culling',pred),
            error=('Absolute RGB error x4',np.clip(np.abs(gt.astype(float)-pred)*4,0,255).astype(np.uint8)),
            contributions=('Compositing weight: orange large clone-size, magenta large split-size, teal other',weight_rgb),
            footprints=('Selected 3-sigma footprints on actual image; white ROI',np.array(overlay)))
        growth=self.args.model/'densification.jsonl'
        events=[]
        if growth.is_file():
            for line in growth.read_text().splitlines():
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                if event['iteration']<=self.iteration:events.append(event)
        report['recorded_growth']=dict(events=len(events),
            cloned_parents=sum(r['cloned_parents'] for r in events),split_parents=sum(r['split_parents'] for r in events),
            note='No historical events can be recovered from a PLY' if not events else 'Logged actual native growth events')
        return dict(report=report,images=images,roi_mask=mask,draw_ids=draw_ids,segments=segments,segment_colors=colors,
                    c2w=f['c2w'],labels=f['labels'],frustum=f['frustum'],lidar_frustum=f['lidar_frustum'])

    def export(self,current):
        from PIL import Image
        from scripts.view_comparisons import image_id
        from utils.checkpoint_snapshots import atomic_json
        report=current['report'];folder=self.output/image_id(report['image'])
        folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'summary.json',report)
        ids=np.flatnonzero(current['roi_mask']);ids=ids[np.argsort(-self.frame['radii'][ids],kind='stable')[:1000]]
        with (folder/'largest_footprints.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['gaussian_id','radius_px','max_sigma','size_branch','opacity','u','v','z','single_view_screen_gradient'])
            for idx in ids:
                writer.writerow([int(idx),int(self.frame['radii'][idx]),float(self.max_sigma[idx]),
                    'clone' if self.max_sigma[idx]<=self.cutoff else 'split',float(self.opacities[idx]),
                    *map(float,self.frame['projected'][idx]),float(self.frame['gradient'][idx]) if self.frame['gradient'] is not None else ''])
        panels=[]
        for key,(title,pixels) in current['images'].items():
            Image.fromarray(pixels).save(folder/(key+'.png'))
            panels.append(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{key}.png"></figure>')
        page=('<!doctype html><meta charset="utf-8"><title>Densification diagnostic</title>'
              '<style>body{background:#111827;color:white;font:16px sans-serif;margin:2em}section{display:grid;grid-template-columns:1fr 1fr;gap:1em}img{width:100%}figure{margin:0}pre{white-space:pre-wrap}</style>'
              f'<h1>{html.escape(report["image"])}</h1><p>{html.escape(report["caveat"])}</p><section>'+''.join(panels)+
              '</section><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>')
        (folder/'index.html').write_text(page)
        archive=self.output/'densification_debug.zip'
        with ZipFile(archive,'w',ZIP_DEFLATED) as bundle:
            for path in folder.iterdir():
                if path.is_file():bundle.write(path,path.name)
        print(f'Diagnostic: {folder / "index.html"}\nShare: {archive}',flush=True)
        return folder,archive
