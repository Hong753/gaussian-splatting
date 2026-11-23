"""Direct mesh export and evaluator-only surface, visibility and RGB scores."""

from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .model import Mesh
from .render import normals, OFFSETS, pixels, render
from .synthetic import oracle
from .train import chunks


def write_mesh(path, vertices, faces):
    path = Path(path)
    with path.open("w") as file:
        file.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n"
                   "property double x\nproperty double y\nproperty double z\n"
                   f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        np.savetxt(file,vertices,fmt="%.16g")
        np.savetxt(file,np.column_stack((np.repeat(3,len(faces)),faces)),fmt="%d")


def read_ascii_mesh(path):
    with Path(path).open() as file:
        if file.readline().strip()!="ply" or file.readline().strip()!="format ascii 1.0":
            raise ValueError(f"{path} requires Open3D or the reference ASCII PLY format.")
        meta = {}
        for line in file:
            fields = line.split()
            if fields[0]=="element":
                meta[fields[1]]=int(fields[2])
            if fields[0]=="end_header":
                break
        n,m = meta["vertex"],meta["face"]
        vertices=np.loadtxt(file,max_rows=n,ndmin=2)[:,:3] if n else np.empty((0,3))
        records=np.loadtxt(file,max_rows=m,dtype=int,ndmin=2) if m else np.empty((0,4),int)
        if records.shape[1]!=4 or np.any(records[:,0]!=3):
            raise ValueError("Exported faces must be triangles.")
    return vertices,records[:,1:]


def face_mask(mesh, mode, threshold):
    if mode=="all":
        return np.ones(len(mesh.faces),bool)
    if mode=="thresholded":
        return mesh.hard | (mesh.opacity>=threshold)
    if mode=="committed":
        return mesh.hard.copy()
    raise ValueError(mode)


def export(checkpoint, folder, threshold):
    from .data import compact
    model = Mesh.read(checkpoint)
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    exports = {}
    for label in ("all","thresholded","committed"):
        keep = face_mask(model,label,threshold)
        selected = model.faces[keep]
        v,f,_ = compact(model.vertices,selected)
        write_mesh(folder/f"{label}.ply",v,f)
        exports[label] = (v,f)
    if not model.vertex_color and model.shader=="atlas":
        # OBJ/MTL exposes the same surface-fixed atlas, with conventional v-up UVs.
        Image.fromarray((255*model.colors).round().astype(np.uint8)).save(folder/"atlas.png")
        (folder/"mesh.mtl").write_text("newmtl atlas\nKd 1 1 1\nmap_Kd atlas.png\n")
        with (folder/"textured.obj").open("w") as file:
            file.write("mtllib mesh.mtl\n")
            for v in model.vertices:
                file.write("v %.15g %.15g %.15g\n"%tuple(v))
            for u,v in model.uv.reshape(-1,2):
                file.write(f"vt {u:.15g} {1-v:.15g}\n")
            file.write("usemtl atlas\n")
            for i,f in enumerate(model.faces):
                file.write("f "+" ".join(f"{j+1}/{3*i+k+1}" for k,j in enumerate(f))+"\n")
    return exports


def render_view(mesh, cam, tau, direct=False):
    grid = pixels(cam)
    sample = grid.reshape(-1,2)
    if not direct:
        sample=(sample[:,None]+OFFSETS).reshape(-1,2)
    data={"rgb":[],"range_m":[],"normal":[],"hard":[],"wire":[]}
    mesh_normals=np.vstack((normals(mesh.vertices,mesh.faces),np.zeros((1,3))))
    for b in chunks(sample):
        pred=render(mesh,cam,b,tau)
        idx=pred["hit"]
        r=np.arange(len(b))
        distance=pred["depth"][r,idx] if len(mesh.faces) else np.full(len(b),np.inf)
        n=mesh_normals[idx]
        n*=np.where((n*cam.rays(b)).sum(1)>0,-1.,1.)[:,None]
        data["rgb"].extend(pred["rgb"])
        data["range_m"].extend(np.where(idx>=0,distance,np.inf))
        data["normal"].extend(np.where(idx[:,None]>=0,n,0))
        data["hard"].extend(pred["hard_hit"]>=0)
        edge=pred["beta"][r,idx].min(1) if len(mesh.faces) else np.full(len(b),np.inf)
        data["wire"].extend((edge<.032)&(idx>=0))
    factor=1 if direct else len(OFFSETS)
    h,w=cam.height,cam.width
    return {k:np.asarray(v).reshape(h,w,factor,*np.asarray(v).shape[1:]) for k,v in data.items()}


def ref_view(reference, camera):
    coords=pixels(camera).reshape(-1,2)
    values=[oracle(*reference,camera,b) for b in chunks(coords)]
    h,w=camera.height,camera.width
    return [np.concatenate([v[i] for v in values]).reshape(h,w,*values[0][i].shape[1:])
            for i in range(4)]


def ssim(pred, target):
    # Standard Gaussian spatial window with an 11x11 valid support.
    smooth=lambda x:gaussian_filter(x,(1.5,1.5,0),truncate=10/3)
    a,b=smooth(pred),smooth(target)
    xy=smooth(pred*target)-a*b
    xx=smooth(pred*pred)-a*a
    yy=smooth(target*target)-b*b
    score=((2*a*b+.01**2)*(2*xy+.03**2))/((a*a+b*b+.01**2)*(xx+yy+.03**2))
    return float(score[5:-5,5:-5].mean())


def area_samples(vertices,faces,density,seed):
    rng=np.random.default_rng(seed)
    a,b,c=np.moveaxis(vertices[faces],1,0)
    cross=np.cross(b-a,c-a)
    twice=np.linalg.norm(cross,axis=1)
    count=rng.poisson(twice*.5*density)
    ids=np.repeat(np.arange(len(faces)),count)
    t=rng.uniform(size=(len(ids),2))
    t=np.where((t.sum(1)>1)[:,None],1-t,t)
    xyz=a[ids]+(b-a)[ids]*t[:,:1]+(c-a)[ids]*t[:,1:]
    return xyz,cross[ids]/twice[ids,None]


def observable(points,cameras,depths,margin,target=False):
    """One frozen reference/frustum mask. A premature surface remains a candidate error."""
    keep=np.zeros(len(points),bool)
    for cam,depth in zip(cameras,depths):
        xyz=cam.local(points)
        pos=xyz[:,2]>.05
        proj=np.zeros_like(xyz[:,:2])
        proj[pos]=cam.project(points[pos])
        px=np.rint(proj).astype(int)
        valid=pos & ((px>=0)&(px<[cam.width,cam.height])).all(1)
        x,y=px[valid].T
        observed=depth[y,x]
        delta=np.linalg.norm(xyz[valid],axis=1)-observed
        mask=np.isfinite(observed)&(delta<=margin)
        if target:
            mask &= delta>=-margin
        keep[valid] |= mask
    return keep


def surface_context(vertices,faces,cameras,depths,cfg):
    samples,norm=area_samples(vertices,faces,cfg["min_samples_per_m2"],20261123)
    valid=observable(samples,cameras,depths,.075,target=True)
    if not valid.any():
        raise ValueError("Reference contains no sampled surface visible on the fixed evaluation cameras.")
    return samples[valid],norm[valid],cKDTree(samples[valid])


def surface_score(vertices,faces,cameras,depths,target,cfg):
    points,n=area_samples(vertices,faces,cfg["min_samples_per_m2"],20261124)
    keep=observable(points,cameras,depths,cfg["observation_depth_guard_m"])
    points,n=points[keep],n[keep]
    ref,rn,tree=target
    if not len(points):
        return dict(accuracy_m=None,completeness_m=None,chamfer_l1_m=None,fscore=0.,
                    precision=0.,recall=0.,normal_consistency=None,samples=0,
                    reference_samples=len(ref),distance_threshold_m=cfg["score_threshold_m"])
    distance,to_ref=tree.query(points)
    comp,to_pred=cKDTree(points).query(ref)
    at=cfg["score_threshold_m"]
    precision=float(np.mean(distance<=at))
    recall=float(np.mean(comp<=at))
    consistency=.5*(np.abs((n*rn[to_ref]).sum(1)).mean()+np.abs((rn*n[to_pred]).sum(1)).mean())
    return dict(accuracy_m=float(distance.mean()),completeness_m=float(comp.mean()),
                chamfer_l1_m=float((distance.mean()+comp.mean())*.5),
                fscore=2*precision*recall/max(precision+recall,1e-12),
                precision=precision,recall=recall,normal_consistency=float(consistency),
                samples=len(points),reference_samples=len(ref),distance_threshold_m=at)


def region_depth(pred,target,mask,block_threshold=.18):
    valid=mask&np.isfinite(target)
    hit=np.isfinite(pred)&valid
    err=pred[hit]-target[hit]
    n=int(valid.sum())
    return dict(reference_pixels=n,valid_predictions=int(hit.sum()),missed=int(n-hit.sum()),
                mae_m=float(np.abs(err).mean()) if len(err) else None,
                signed_mean_m=float(err.mean()) if len(err) else None,
                early=int((err < -block_threshold).sum()),
                early_fraction=float((err < -block_threshold).sum()/max(n,1)))


def as_image(rgb):
    return Image.fromarray((np.clip(rgb,0,1)*255).round().astype(np.uint8))


def montage(view_image,target,near_d,near_n,ref_n,wire,truth_d,goal,path):
    valid=np.isfinite(near_d)&np.isfinite(truth_d)
    error=np.zeros_like(target)
    delta=np.zeros_like(near_d)
    delta[valid]=near_d[valid]-truth_d[valid]
    error[...,0]=np.clip(-delta/.4,0,1)
    error[...,2]=np.clip(delta/.4,0,1)
    error[~np.isfinite(near_d)&np.isfinite(truth_d)]=[1.,.84,.14]
    with_wire=np.where(wire[...,None],[.09,.95,.78],target*.65)
    gt_norm=np.where(np.isfinite(truth_d)[...,None],.5+.5*ref_n,0.)
    geom_norm=np.where(np.isfinite(near_d)[...,None],.5+.5*near_n,0.)
    h,w=target.shape[:2]
    up=3
    items=[("Ground-truth RGB",view_image),("Rendered RGB",target),("Snapshot wireframe",with_wire),
           ("Ground-truth normals",gt_norm),("Mesh normals",geom_norm),("Range error / misses",error)]
    out=Image.new("RGB",(w*up*3,(h*up+26)*2+58),"#161d26")
    draw=ImageDraw.Draw(out)
    for k,(label,img) in enumerate(items):
        x,y=k%3*w*up,k//3*(h*up+26)
        draw.text((x+7,y+6),label,fill="#edf3fa")
        out.paste(as_image(img).resize((w*up,h*up),Image.Resampling.NEAREST),(x,y+26))
    draw.text((8,(h*up+26)*2+8),goal,fill="#edf3fa")
    draw.text((8,(h*up+26)*2+29),"Red = early hit   Blue = late hit   Yellow = missing geometry   Full scale 40 cm",
              fill="#d6dde8")
    out.save(path)


def rgb_montage(image, rgb, distance, normals, wire, title, path):
    hit = np.isfinite(distance)
    gray = np.zeros_like(distance)
    if hit.any():
        near, far = np.percentile(distance[hit], [5, 95])
        gray[hit] = 1. - .85 * np.clip((distance[hit]-near)/max(far-near,1e-9),0,1)
    error = np.abs(rgb-image).mean(axis=-1)
    error_rgb = np.stack((np.minimum(1,4*error),np.clip(4*error-1,0,1),np.zeros_like(error)),axis=-1)
    panels = (
        ("Reference RGB", image), ("Rendered RGB", rgb), ("RGB error", error_rgb),
        ("Mesh range", np.repeat(gray[...,None],3,axis=-1)),
        ("Mesh normals", np.where(hit[...,None],.5+.5*normals,0.)),
        ("Mesh wireframe", np.where(wire[...,None],[.09,.95,.78],rgb*.65)),
    )
    height,width = image.shape[:2]
    canvas = Image.new("RGB",(3*width,36+2*(height+24)),"#161d26")
    draw = ImageDraw.Draw(canvas)
    draw.text((8,10),title,fill="#edf3fa")
    for k,(label,panel) in enumerate(panels):
        x,y = k%3*width,36+k//3*(height+24)
        draw.text((x+6,y+4),label,fill="#edf3fa")
        canvas.paste(as_image(panel),(x,y+24))
    canvas.save(path)


def compare_arm(checkpoint,images,cameras,refs,folder,cfg,target,*,first_hit=False):
    start=perf_counter()
    mesh=Mesh.read(checkpoint)
    arm=Path(checkpoint).stem
    local=Path(folder)/arm
    local.mkdir(parents=True,exist_ok=True)
    tick=perf_counter()
    exports=export(checkpoint,local,cfg["soft_export_threshold"])
    extraction=perf_counter()-tick
    direct=mesh.clone(hard=np.ones(len(mesh.faces),bool))
    photometric = mesh.clone(hard=~mesh.transmitting, elastic_sigma=0.) if first_hit else mesh
    depths=[item[1] for item in refs] if refs is not None else None
    render_metrics=[]
    for index,(image,cam) in enumerate(zip(images,cameras)):
        pred=render_view(photometric,cam,cfg["tau_px"])
        geom=render_view(direct,cam,cfg["tau_px"],direct=True)
        rgb=pred["rgb"].mean(axis=2)
        distance=geom["range_m"][...,0]
        mesh_normals=geom["normal"][...,0,:]
        wire=geom["wire"][...,0]
        mse=float(np.mean((rgb-image)**2))
        score=dict(view=index,psnr_db=float(-10*np.log10(max(mse,1e-14))),
                   ssim=ssim(rgb,image),lpips=None,
                   mesh_hit_fraction=float(np.isfinite(distance).mean()),
                   opaque_subsample_fraction=float(pred["hard"].mean()))
        title=f"{arm}  |  final view {index:02}  |  {score['psnr_db']:.2f} dB PSNR"
        preview=local/f"final_{index:02}.png"
        arrays=dict(rgb=rgb,reference_rgb=image,mesh_range=distance,normals=mesh_normals,
                    wire=wire,hard_coverage=pred["hard"].mean(-1))
        if refs is None:
            rgb_montage(image,rgb,distance,mesh_normals,wire,title,preview)
        else:
            _,truth_d,kind,truth_n=refs[index]
            score.update(floor=region_depth(distance,truth_d,kind==0),
                         wall=region_depth(distance,truth_d,kind==1),
                         doorway=region_depth(distance,truth_d,kind==2),
                         doorway_render_event=region_depth(pred["range_m"][...,0],truth_d,kind==2))
            montage(image,rgb,distance,mesh_normals,truth_n,wire,truth_d,title,preview)
            arrays.update(reference_range=truth_d,reference_kind=kind,reference_normals=truth_n)
        render_metrics.append(score)
        np.savez_compressed(local/f"final_{index:02}.npz",**arrays)
    surfaces={}
    for label,(_,faces) in exports.items():
        if target is None:
            surfaces[label]={"exported_faces":len(faces)}
        else:
            v,f=read_ascii_mesh(local/f"{label}.ply")
            surfaces[label]=surface_score(v,f,cameras,depths,target,cfg) | {"exported_faces":len(f)}
    return dict(wall_s=perf_counter()-start,export_s=extraction,views=render_metrics,surface=surfaces,
                rgb_visibility="first_hit" if first_hit else "checkpoint")


def score_external(data, reference_mesh, candidate_mesh, config, output, renders=None, masks=None):
    """Same observed mask for a registered externally trained mesh and RGB renders."""
    from .data import load_views
    from .experiment import save_json
    from .room import reference_views
    images,cameras=load_views(Path(data)/"final.npz")
    v,f,ref=reference_views(reference_mesh,cameras,masks)
    candidate_v,candidate_f,pred=reference_views(candidate_mesh,cameras)
    depth=[view[1] for view in ref]
    target=surface_context(v,f,cameras,depth,config)
    metrics=surface_score(candidate_v,candidate_f,cameras,depth,target,config)
    views=[]
    files=sorted(Path(renders).glob("*.png")) if renders else []
    if files and len(files)!=len(cameras):
        raise ValueError("The final render folder must contain one ordered PNG per final camera.")
    for i,(gt,p,im) in enumerate(zip(ref,pred,images)):
        kind=gt[2]
        item=dict(view=i,floor=region_depth(p[1],gt[1],kind==0),
                  wall=region_depth(p[1],gt[1],kind==1),
                  doorway=region_depth(p[1],gt[1],kind==2))
        if files:
            other=np.asarray(Image.open(files[i]).convert("RGB"),dtype=float)/255
            if other.shape!=im.shape:
                raise ValueError(f"Final image size mismatch at {files[i]}")
            item.update(image=files[i].name,psnr_db=float(-10*np.log10(max(np.mean((im-other)**2),1e-14))),
                        ssim=ssim(other,im))
        views.append(item)
    save_json(output,dict(external_geometry=str(candidate_mesh),surface=metrics,views=views))


def fitting_diagnostics(checkpoint, fit_file, config, output):
    """Image-only diagnostics for the extent of weak evidence during fitting."""
    from scipy import ndimage
    from .data import load_views
    from .experiment import save_json
    mesh=Mesh.read(checkpoint)
    images,cameras=load_views(fit_file)
    reports=[]
    for i,(image,cam) in enumerate(zip(images,cameras)):
        gray=image@[.25,.50,.25]
        box=lambda x:ndimage.uniform_filter(x,size=3,mode="nearest")
        variance=box(gray*gray)-box(gray)**2
        p=pixels(cam).reshape(-1,2)
        visibility=np.concatenate([1-render(mesh,cam,b,config["tau_px"])["transmittance"]
                                   for b in chunks(p)]).reshape(gray.shape)
        low=variance<=.0011
        missing=visibility<.45
        reports.append(dict(view=i,total=int(low.size),low_variance=int(low.sum()),
                            weakly_covered=int(missing.sum()),low_variance_and_weakly_covered=int((low&missing).sum())))
    save_json(output,dict(variance_threshold=.0011,coverage_threshold=.45,views=reports))
