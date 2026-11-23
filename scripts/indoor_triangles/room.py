"""ScanNet++ handoff using fixed supplied PINHOLE poses and fitting-only SfM."""

import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .experiment import save_json
from .render import pixels, unit


def colmap_loader():
    file=Path(__file__).resolve().parents[2]/"scene"/"colmap_loader.py"
    spec=importlib.util.spec_from_file_location("indoor_colmap_loader",file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_cameras(model):
    col=colmap_loader()
    binary=all((model/name).is_file() for name in ("cameras.bin","images.bin"))
    if binary:
        return col,col.read_intrinsics_binary(model/"cameras.bin"),col.read_extrinsics_binary(model/"images.bin")
    return col,col.read_intrinsics_text(model/"cameras.txt"),col.read_extrinsics_text(model/"images.txt")


HOLDOUT_GAP_M = {"edit": .08, "final": .15}


def write_pinhole(path, records, points=()):
    path.mkdir(parents=True,exist_ok=True)
    with (path/"cameras.txt").open("w") as cams,(path/"images.txt").open("w") as images:
        for ident,item in enumerate(records,1):
            calib=item["intrinsics"]+np.array([0.,0.,.5,.5])
            h,w=item["rgb"].shape[:2]
            cams.write(f"{ident} PINHOLE {w} {h} "+" ".join(map(str,calib))+"\n")
            pose=np.r_[item["qvec"],item["tvec"]]
            images.write(f"{item.get('db_id',ident)} "+" ".join(map(str,pose))+f" {ident} {item['local']}\n\n")
    with (path/"points3D.txt").open("w") as file:
        for i,(point,color,error) in enumerate(points,1):
            file.write(f"{i} "+" ".join(map(str,np.r_[point]))+" "+" ".join(map(str,color))+
                       f" {error}\n")


def fit_only_colmap(fit_dir,fit,staging,use_gpu):
    import pycolmap as p

    db_file=str(staging/"fit.db")
    device=p.Device.cuda if use_gpu else p.Device.cpu
    p.extract_features(db_file,str(fit_dir),camera_mode=p.CameraMode.PER_IMAGE,camera_model="PINHOLE",
                       sift_options=p.SiftExtractionOptions(num_threads=4),device=device)
    db=p.Database(db_file)
    lookup={x.name:x for x in db.read_all_images()}
    for item in fit:
        image=lookup[item["local"]]
        known=db.read_camera(image.camera_id)
        known.params=item["intrinsics"]+[0.,0.,.5,.5]
        known.has_prior_focal_length=True
        db.update_camera(known)
        item["db_id"]=image.image_id
        item["db_camera"]=image.camera_id
    db.close()
    p.match_exhaustive(db_file,sift_options=p.SiftMatchingOptions(num_threads=4,guided_matching=True),
                       device=device)
    known_dir=staging/"known"
    write_pinhole(known_dir,fit)
    # The database can assign camera IDs in an order independent of the records.
    lines=(known_dir/"cameras.txt").read_text().splitlines()
    corrected=[str(item["db_camera"])+" "+line.split(" ",1)[1] for item,line in zip(fit,lines)]
    (known_dir/"cameras.txt").write_text("\n".join(corrected)+"\n")
    lines=(known_dir/"images.txt").read_text().splitlines()
    for i,item in enumerate(fit):
        row=lines[2*i].split()
        row[-2]=str(item["db_camera"])
        lines[2*i]=" ".join(row)
    (known_dir/"images.txt").write_text("\n".join(lines)+"\n")
    tri_dir=staging/"triangulated"
    tri_dir.mkdir()
    options=p.IncrementalPipelineOptions(num_threads=4,fix_existing_images=True,
           ba_refine_focal_length=False,ba_refine_principal_point=False,ba_refine_extra_params=False)
    result=p.triangulate_points(p.Reconstruction(str(known_dir)),db_file,str(fit_dir),str(tri_dir),
                                options=options,refine_intrinsics=False)
    for item in fit:
        image=result.images[item["db_id"]]
        input_pose=np.column_stack((item["rotation"],item["tvec"]))
        if not np.allclose(image.cam_from_world.matrix(),input_pose,atol=1e-8,rtol=0):
            raise ValueError(f"Fixed poses changed after triangulation: {item['name']}")
        expected=item["intrinsics"]+[0.,0.,.5,.5]
        if not np.allclose(result.cameras[image.camera_id].params,expected,atol=1e-8,rtol=0):
            raise ValueError(f"Fixed intrinsics changed after triangulation: {item['name']}")
    return result,p.__version__


def candidate_tracks(model,fit,ref,cell):
    reference=fit[ref]
    image_index={item["db_id"]:i for i,item in enumerate(fit)}
    seen={}
    for _,point in sorted(model.points3D.items(),key=lambda obj:obj[1].error):
        if point.error>.90:
            continue
        ids={image_index[el.image_id] for el in point.track.elements}
        if ref not in ids or len(ids)<3:
            continue
        if max(np.linalg.norm(fit[i]["center"]-reference["center"]) for i in ids)<.18:
            continue
        local=(point.xyz-reference["center"])@reference["rotation"].T
        if local[2]<=.20 or local[2]>15:
            continue
        calib=reference["intrinsics"]
        xy=local[:2]/local[2]*calib[:2]+calib[2:]
        h,w=reference["rgb"].shape[:2]
        if not np.all((xy>=3)&(xy<[w-3,h-3])):
            continue
        key=tuple(np.floor(xy/cell).astype(int))
        seen.setdefault(key,(point.xyz.copy(),xy,np.array([i in ids for i in range(len(fit))]),
                             point.color.tolist(),float(point.error)))
    if len(seen)<10:
        raise ValueError(f"Only {len(seen)} reliable fitting tracks in the locked reference view.")
    records=list(seen.values())
    xyz,xy,obs,colors,error=map(np.asarray,zip(*records))
    return dict(points=xyz,xy=xy,ref=ref,obs=obs,cameras=np.arange(len(fit)),
                tolerance=np.clip(.04+.04*error,.04,.09),score=1/(1+error)),list(zip(xyz,colors.tolist(),error))


def prepare(args, calibration_data=None):
    import json

    start=perf_counter()
    col,calibration,frames=calibration_data if calibration_data is not None else model_cameras(args.model)
    if not args.pose_provenance.strip():
        raise ValueError("Record a concrete calibration origin.")
    by_name={im.name:im for im in frames.values()}
    split=json.loads(args.split.read_text())
    roles=("fit","edit","final")
    all_names=[n for key in roles for n in split[key]]
    if len(set(all_names))!=len(all_names) or len(split["fit"])<3 or min(map(len,split.values()))<1:
        raise ValueError("Use disjoint fitting, edit-validation and final camera names.")
    if args.reference_view not in split["fit"]:
        raise ValueError("The locked reference camera must be in the fitting split.")
    loaded={}
    for idx,name in enumerate(all_names):
        frame=by_name[name]
        intr=calibration[frame.camera_id]
        if intr.model=="PINHOLE":
            fx,fy,cx,cy=intr.params
        elif intr.model=="SIMPLE_PINHOLE":
            fx,cx,cy=intr.params
            fy=fx
        else:
            raise ValueError(f"Undistort cameras first: {name} is {intr.model}")
        raw=Image.open(args.images/name).convert("RGB")
        if raw.size!=(intr.width,intr.height):
            raise ValueError(f"Image and calibration size differ for {name}.")
        dims=(args.width,round(args.width*intr.height/intr.width))
        sx,sy=np.array(dims)/raw.size
        k=np.array([fx*sx,fy*sy,cx*sx-.5,cy*sy-.5])
        R=col.qvec2rotmat(frame.qvec)
        loaded[name]=dict(name=name,local=f"{idx:06}.png",qvec=frame.qvec,tvec=frame.tvec,
                          center=-R.T@frame.tvec,rotation=R,intrinsics=k,
                          rgb=np.array(raw.resize(dims,Image.Resampling.LANCZOS)))
    hws={x["rgb"].shape for x in loaded.values()}
    if len(hws)!=1:
        raise ValueError("Use a common image aspect ratio or a registered common crop.")
    fit=[loaded[name] for name in split["fit"]]
    centers=np.array([item["center"] for item in fit])
    distances={n:float(np.linalg.norm(centers-loaded[n]["center"],axis=1).min())
               for role in ("edit","final") for n in split[role]}
    if any(min(distances[n] for n in split[role])<gap for role,gap in HOLDOUT_GAP_M.items()):
        raise ValueError(f"Retain translated blocks. Minimum nearest-fitting distances are 0.15/0.08 m "
                         f"for final/edit views. Got {distances}.")
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    stock_ok=True
    for role in roles:
        subset=[loaded[name] for name in split[role]]
        for item in subset:
            h,w=item["rgb"].shape[:2]
            stock_ok &= bool(np.allclose(item["intrinsics"][2:],[(w-1)/2,(h-1)/2],atol=1e-4,rtol=0))
        rgb=[item["rgb"] for item in subset]
        np.savez_compressed(out/f"{role}.npz",rgb=rgb,centers=[item["center"] for item in subset],
                            rotations=[item["rotation"] for item in subset],
                            intrinsics=[item["intrinsics"] for item in subset])
        if role in ("fit","final"):
            rendered=out/f"stock_{role}"/"images"
            rendered.mkdir(parents=True)
            for item in subset:
                Image.fromarray(item["rgb"]).save(rendered/item["local"])
    staging=out/"fit_sfm"
    staging.mkdir()
    fitted,version=fit_only_colmap(out/"stock_fit"/"images",fit,staging,args.gpu_features)
    track_pack,xyz=candidate_tracks(fitted,fit,split["fit"].index(args.reference_view),args.cell)
    for role in ("fit","final"):
        subset=[loaded[name] | {"db_id":j+1} for j,name in enumerate(split[role])]
        write_pinhole(out/f"stock_{role}"/"sparse"/"0",subset,xyz)
    np.savez_compressed(out/"fitting_tracks.npz",**track_pack)
    manifest={"split":split,"poses":args.pose_provenance,"reference_view":args.reference_view,
              "nearest_fit_distance_m":distances,"pycolmap":version,"width":args.width,"height":h,
              "stock_centered_camera_compatible":stock_ok,"fit_track_count":len(xyz),
              "wall_s":perf_counter()-start,"fitting_match_images":[item["local"] for item in fit],
              "images":[{"source":n,"copy":loaded[n]["local"],"center_m":loaded[n]["center"].tolist()}
                        for n in all_names]}
    save_json(out/"split_manifest.json",manifest)
    print(json.dumps(manifest,indent=2))


def reference_views(mesh, cameras, masks_path=None):
    """Read the registered scan only in evaluation and cast fixed-view rays."""
    import open3d as o3d

    source=o3d.io.read_triangle_mesh(str(mesh),enable_post_processing=False)
    vertices=np.asarray(source.vertices)
    faces=np.asarray(source.triangles)
    if not len(faces):
        raise ValueError("Registered reference scan has no faces.")
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(source))
    mask=np.load(masks_path)["kinds"] if masks_path else None
    views=[]
    for i,cam in enumerate(cameras):
        dirs=cam.rays(pixels(cam))
        rays=np.concatenate((np.broadcast_to(cam.center,dirs.shape),dirs),axis=-1).astype(np.float32)
        result=scene.cast_rays(o3d.core.Tensor(rays))
        depth=result["t_hit"].numpy()
        normals=unit(result["primitive_normals"].numpy())
        normals*=np.where((normals*dirs).sum(-1)>0,-1.,1.)[...,None]
        labels=np.full(depth.shape,-1) if mask is None else mask[i]
        if labels.shape!=depth.shape:
            raise ValueError("Evaluator surface masks must match the fixed rendered image shapes.")
        views.append((None,depth,labels,normals))
    return vertices,faces,views
