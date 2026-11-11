"""GPU surface depth, original point labels, and actual CUDA fragment diagnostics."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def read_lidar_cloud(path):
    """Keep original PLY row identities and optional RGB for the independent layer."""
    from plyfile import PlyData
    vertex=PlyData.read(str(path))['vertex']
    points=np.stack([vertex[k] for k in ('x','y','z')],axis=1).astype(np.float32)
    colors=None
    if all(k in vertex.data.dtype.names for k in ('red','green','blue')):
        colors=np.stack([vertex[k] for k in ('red','green','blue')],axis=1)
        if colors.dtype.kind=='f' and np.isfinite(colors).all() and colors.size and colors.max()<=1:
            colors=colors*255
        colors=np.nan_to_num(colors).clip(0,255).astype(np.uint8)
    return points,colors


def fragment_layers(flags, raster_radii):
    """Disjoint CUDA states; percentages use pre-culling image candidates."""
    flags, raster_radii = np.asarray(flags).reshape(-1), np.asarray(raster_radii).reshape(-1)
    if flags.shape != raster_radii.shape or np.any((flags < 0) | (flags > 3)):
        raise ValueError('Invalid Gaussian fragment flags/radii')
    candidates = raster_radii > 0
    ids = dict(fragment_blocked_ids=np.flatnonzero(flags==2),
               fragment_partial_ids=np.flatnonzero(flags==3),
               contributing_ids=np.flatnonzero(flags==1),
               fragment_inactive_ids=np.flatnonzero(flags==0),
               fragment_frustum_ids=np.flatnonzero(candidates))
    count = int(candidates.sum())
    report = dict(total=len(flags), image_candidates=count,
        no_encountered_contribution=len(ids['fragment_inactive_ids']),
        contributing_only=len(ids['contributing_ids']), fully_blocked=len(ids['fragment_blocked_ids']),
        partially_blocked=len(ids['fragment_partial_ids']),
        fully_blocked_fraction_in_image=float(((flags==2)&candidates).sum()/max(count,1)),
        partially_blocked_fraction_in_image=float(((flags==3)&candidates).sum()/max(count,1)),
        note='Flags record encountered fragments before native opacity early termination; zero does not prove geometric occlusion')
    return ids, report


class RayInspector:
    def __init__(self, args):
        import torch
        from utils.scene_occlusion import MeshDepthCache, load_scene, mesh_arrays
        from scripts.view_comparisons import read_colmap_records, model_info, choose_iteration
        self.args,self.torch=args,torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this viewer; activate 3dgs on the GPU host')
        self.scene,self.mesh_path=load_scene(args.scene,args.point_cloud)
        self.vertices,self.faces=mesh_arrays(self.mesh_path)
        self.supplement_arrays=(mesh_arrays(args.scene.parent/self.scene['supplement_mesh'])
                               if self.scene.get('supplement_mesh') else None)
        self.provider=MeshDepthCache(args.scene,edge_pixels=1)
        self.points,self.lidar_colors=read_lidar_cloud(args.point_cloud)
        self.records=read_colmap_records(Path(self.scene['source']))
        self.names=sorted(self.records)
        self.model=None
        self.point_sets={'LiDAR':self.points}
        self.output=args.output or args.scene.parent/'ray_inspection'
        if args.model:
            from scene.gaussian_model import GaussianModel
            from scripts.occlusion_options import configure_pipeline
            spec=model_info(args.model)
            if spec['type']!='3dgs' or Path(spec['cfg']['source_path']).resolve()!=Path(self.scene['source']).resolve():
                raise ValueError('Choose a native checkpoint for the same camera scene')
            self.iteration,self.ply=choose_iteration(spec['path'],args.iteration)
            self.output=args.output or spec['path']/'ray_inspection'/f'iteration_{self.iteration}'
            self.cfg=spec['cfg']
            self.model=GaussianModel(self.cfg.get('sh_degree',3))
            self.model.load_ply(str(self.ply))
            self.point_sets['Gaussians']=self.model.get_xyz.detach().cpu().numpy()
            from scripts.occlusion_inspector_backend import PARAMETERS
            self.parameters=[getattr(self.model,n) for n in PARAMETERS]
            for parameter in self.parameters: parameter.requires_grad_(False)
            common=dict(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=True)
            self.pipe=configure_pipeline(SimpleNamespace(**common),dict(occlusion_mode='off',
                occlusion_backend='ray',occlusion_margin=args.margin))
            self.bg=torch.tensor([1.,1.,1.] if self.cfg.get('white_background') else [0.,0.,0.],device='cuda')

    def analyze(self,name):
        import torch
        from PIL import Image
        from utils.occlusion_depth import record_camera
        from utils.view_tangent import camera_intrinsics
        from utils.ray_occlusion import classify_points_depth, projected_lidar_depth, first_surface_audit
        from utils.occlusion_diagnostics import depth_rgb,write_report
        from scripts.view_comparisons import image_id
        from scripts.prepare_da3_occlusion import save_npz
        r=self.records[name]
        resolution=self.cfg.get('resolution',-1) if self.model else self.scene.get('resolution',-1)
        camera=record_camera(r,resolution,device='cuda')
        camera.camera_center=camera.world_view_transform.inverse()[3,:3]
        k=camera_intrinsics(camera,'cuda')
        e=camera.world_view_transform.T
        raw=self.provider.raw_depth(camera,'cuda')
        bound=self.provider.pixel_depth(camera,'cuda')
        with Image.open(r['image_path']) as im:
            rgb=np.asarray(im.convert('RGB').resize((camera.image_width,camera.image_height)))
        groups={key:classify_points_depth(points,k,e,raw,self.args.margin,self.scene['relative_margin'],
                                          surface_band=self.args.surface_band)
                for key,points in self.point_sets.items()}
        counts={key:{name[:-4]:len(ids) for name,ids in value.items()} for key,value in groups.items()}
        counts={key:dict(value,total=len(self.point_sets[key])) for key,value in counts.items()}
        colored,_=depth_rgb(raw.cpu().numpy(),camera.image_width,camera.image_height,pixel_size=1)
        images={'gt':('Actual image',rgb),'depth':('CUDA first-hit surface depth; black unknown',colored)}
        lidar_depth=projected_lidar_depth(self.points,k,e,camera.image_height,camera.image_width,'cuda')
        audit,overlay=first_surface_audit(lidar_depth.cpu().numpy(),raw.cpu().numpy(),rgb,self.args.margin)
        images['foreground_audit']=('LiDAR/mesh: green agrees; red mesh behind LiDAR; purple missing; cyan mesh in front',overlay)
        report=dict(image=name,depth_coverage=float(torch.isfinite(raw).float().mean()),counts=counts,
            backend='CUDA mesh rasterization + CUDA point projection/depth lookup + CUDA splat clipping',
            point_rule='nearest image pixel first-hit z; points are not Gaussian fragment masks',
            splat_rule='native center camera-z per fragment; fixed hard visibility gate',
            mesh=str(self.args.scene),diagnostic_only=self.scene.get('diagnostic_only',False),
            integrated_frames=self.scene.get('integrated_frames',0),training_started=False,
            geometry_source=self.scene.get('geometry_source','validated_DA3_TSDF'),
            lidar_surface=self.scene.get('lidar_surface'), surface_band=self.args.surface_band,
            culling_margin=self.args.margin, relative_margin=self.scene['relative_margin'],
            first_surface_audit=audit)
        if self.model:
            from gaussian_renderer import render
            from scripts.occlusion_inspector_backend import PARAMETERS,gradient_statistics,probe_verdict
            from utils.loss_utils import l1_loss,ssim
            def render_one(enabled, hidden=None):
                for p in self.parameters:p.requires_grad_(self.args.check_gradients)
                with torch.enable_grad() if self.args.check_gradients else torch.no_grad():
                    result=render(camera,self.model,self.pipe,self.bg,occlusion_depth=bound if enabled else None)
                    predicted=result['render']
                    flags=result.get('occlusion_fragment_flags')
                    if enabled: hidden=flags==2
                    stats=None
                    if self.args.check_gradients:
                        gt=torch.tensor(rgb.copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
                        loss=.8*l1_loss(predicted,gt)+.2*(1-ssim(predicted,gt))
                        gradients=torch.autograd.grad(loss,self.parameters,allow_unused=True)
                        stats=gradient_statistics(PARAMETERS,gradients,hidden)
                    pixels=(predicted.detach().permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
                for p in self.parameters:p.requires_grad_(False)
                return pixels,flags,stats,result.get('raster_radii')
            on,flags,on_grad,raster_radii=render_one(True)
            off,_,off_grad,_=render_one(False,flags==2)
            flags=flags.cpu().numpy()
            images.update(on=('CUDA ray culling ON',on),off=('Culling OFF',off),
                difference=('Absolute ON/OFF difference x10',np.minimum(np.abs(on.astype(int)-off.astype(int))*10,255).astype(np.uint8)))
            fragment_ids,report['fragments']=fragment_layers(flags,raster_radii.cpu().numpy())
            report['checkpoint']=dict(model=str(self.args.model),iteration=self.iteration,ply=str(self.ply),
                saved_inference_occlusion=self.cfg.get('occlusion_inference_mode','saved'),
                display='Fixed saved checkpoint; change camera freely, restart to load newer weights')
            if self.args.check_gradients:
                report['gradient_probe']=dict(on=on_grad,off=off_grad,
                    verdict=probe_verdict(on_grad,off_grad,int((flags==2).sum())))
            groups['Gaussians'].update(fragment_ids)
        directory=self.output/image_id(name)
        directory.mkdir(parents=True,exist_ok=True)
        for key,value in groups.items():save_npz(directory/(key.lower()+'_ids.npz'),**value)
        save_npz(directory/'depth_audit.npz',lidar_depth=lidar_depth.cpu().numpy(),mesh_depth=raw.cpu().numpy(),
                 cutoff_depth=(bound+self.args.margin).cpu().numpy(),intrinsics=k.cpu().numpy(),extrinsics=e.cpu().numpy())
        write_report(directory,report,images)
        print(json.dumps(report,indent=2),flush=True)
        print(f'Ray inspection: {directory}/index.html',flush=True)
        return dict(name=name,groups=groups,counts=counts,report=report,images=images,k=k.cpu().numpy(),
                    c2w=torch.linalg.inv(e).cpu().numpy(),directory=directory)
