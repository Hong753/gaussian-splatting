#!/usr/bin/env python3
"""Prepare/audit a train-only COLMAP rerun without changing the original region."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.keyframe_pipeline.common import read_model, file_stamp, atomic_json
from utils.region_loss import mask_path
from utils.checkpoint_snapshots import digest


def load_job(workspace):
    job=json.loads((workspace/'job.json').read_text())
    if job['workspace']!=str(workspace.resolve()):
        raise ValueError('COLMAP workspace moved; create a new workspace')
    for path,stamp in job['input_stamps'].items():
        if file_stamp(path)!=stamp: raise ValueError(f'Input changed: {path}')
    return job


def prepare(region,workspace,alignment_error=None,frontend='sift',overlap=20):
    region,workspace=region.resolve(),workspace.resolve()
    meta=json.loads((region/'selection.json').read_text())
    _,cameras,images,stamps=read_model(region)
    manifest=json.loads((region/'keyframes.json').read_text())
    names=manifest['selected_images'];wanted=set(names)
    if len(wanted)!=len(names) or len(names)<3 or not wanted <= {im.name for im in images}:
        raise ValueError('Need at least three distinct, registered training images')
    if wanted!={r['name'] for r in meta['selection'] if r['split']=='train'}:
        raise ValueError('Regional training membership differs from its export')
    chosen=[im for im in images if im.name in wanted]
    centers=np.array([im.C for im in chosen])
    spread=np.linalg.svd(centers-centers.mean(0),compute_uv=False)
    if spread[0]<1e-10 or spread[1]<spread[0]*1e-3:
        raise ValueError('Training camera centers are too nearly collinear for reliable similarity alignment')
    extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max())
    error=float(alignment_error) if alignment_error is not None else .05*extent
    if not np.isfinite(error) or error<=0: raise ValueError('Alignment error must be finite and positive')
    paths=[region/'selection.json',region/'keyframes.json',region/'input.ply',
           *[Path(s[0]) for s in stamps],*[region/'images'/im.name for im in chosen],
           *[mask_path(region/'masks',r['name']) for r in meta['selection']]]
    groups={}; calibrated={}
    for im in chosen:
        if any(c.isspace() for c in im.name):
            raise ValueError('COLMAP location files require image names without whitespace')
        cam=cameras[im.camera_id]
        with Image.open(region/'images'/im.name) as rgb:
            width,height=rgb.size
        if not np.isclose(width/height,cam.width/cam.height,rtol=.001):
            raise ValueError(f'Image/calibration aspect mismatch: {im.name}')
        intr=dict(id=cam.id,width=width,height=height,
                  params=(cam.params*np.array([width/cam.width,height/cam.height]*2)).tolist())
        if cam.id in calibrated and calibrated[cam.id]!=intr:
            raise ValueError('Images sharing a camera have different pixel dimensions')
        calibrated[cam.id]=intr
        groups.setdefault(str(cam.id),[]).append(im.name)
    job=dict(schema=1,region=str(region),workspace=str(workspace),selected=sorted(names),
             input_stamps={str(p.resolve()):file_stamp(p) for p in paths},
             cameras=calibrated,groups=groups,alignment_max_error=error,
             policy='Fresh SIFT/exhaustive matching/incremental SfM/BA on selected train images only; fixed intrinsics; similarity alignment to original train centers; unchanged LiDAR and held-out poses')
    if frontend=='superglue':
        if overlap<1: raise ValueError('Sequential overlap must be positive')
        job.update(frontend='superpoint-superglue',sequential_overlap=overlap,
                   policy='SuperPoint/SuperGlue sequential matching/incremental SfM/BA on selected train images only; fixed intrinsics; similarity alignment to original train centers; unchanged LiDAR and held-out poses')
    # Normalize JSON integer keys before comparing a resumed job.
    job=json.loads(json.dumps(job))
    if (workspace/'job.json').is_file():
        if load_job(workspace)!=job: raise ValueError('Different COLMAP inputs/settings; choose a new workspace')
        print(f'Reusing COLMAP workspace: {workspace}',flush=True)
        return job
    if workspace.exists(): raise FileExistsError(f'Choose a new workspace: {workspace}')
    workspace.mkdir(parents=True)
    (workspace/'images').mkdir();(workspace/'lists').mkdir();(workspace/'logs').mkdir()
    for im in chosen:
        target=workspace/'images'/im.name;target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to((region/'images'/im.name).resolve())
    for cid,group in groups.items():
        (workspace/'lists'/f'camera_{cid}.txt').write_text(''.join(n+'\n' for n in group))
    (workspace/'reference_centers.txt').write_text(''.join(
        im.name+' '+' '.join(format(float(x),'.17g') for x in im.C)+'\n' for im in chosen))
    atomic_json(workspace/'job.json',job)
    print(f'COLMAP: {len(names)} training images only; {len(images)-len(names)} held-out images excluded.\n'
          f'Workspace: {workspace}\nAlignment tolerance: {error:.6g} original scene units.',flush=True)
    return job


def seed_cameras(workspace):
    job=load_job(workspace)
    with sqlite3.connect(workspace/'database.db') as db:
        existing={row[0]:row for row in db.execute('SELECT camera_id,model,width,height,params,prior_focal_length FROM cameras')}
        if set(existing)-{int(cid) for cid in job['cameras']}: raise ValueError('Unexpected cameras in COLMAP database')
        for cid,cam in job['cameras'].items():
            row=(int(cid),1,cam['width'],cam['height'],np.array(cam['params'],dtype='<f8').tobytes(),1)
            if int(cid) in existing:
                if existing[int(cid)]!=row: raise ValueError(f'Camera calibration changed in database: {cid}')
            else:
                db.execute('INSERT INTO cameras(camera_id,model,width,height,params,prior_focal_length) VALUES(?,?,?,?,?,?)',row)


def choose_model(workspace,root):
    job=load_job(workspace);wanted=set(job['selected']);candidates=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        if not any((folder/('images'+ext)).is_file() for ext in ('.bin','.txt')): continue
        _,_,images,_=read_model(folder,model_dir=folder)
        names={im.name for im in images}
        if names-wanted: raise ValueError('COLMAP model contains non-training images')
        candidates.append((len(names),str(folder.resolve()),sorted(wanted-names)))
    if not candidates: raise ValueError(f'COLMAP produced no sparse reconstruction: {root}')
    candidates.sort(reverse=True)
    count,path,missing=candidates[0]
    atomic_json(workspace/'registration_report.json',dict(expected=len(wanted),registered=count,
                selected_model=path,missing=missing,components=[dict(images=n,path=p) for n,p,_ in candidates]))
    if missing:
        raise ValueError(f'Best COLMAP component registered {count}/{len(wanted)} training images; '
                         'training was not started with a silently reduced subset. See registration_report.json.')
    (workspace/'selected_model.txt').write_text(path+'\n')
    print(f'All {count} selected training images registered: {path}',flush=True)


def pose_audit(original,aligned,job):
    wanted=set(job['selected'])
    refined={im.name:im for im in aligned}
    if set(refined)!=wanted: raise ValueError('Aligned model must contain exactly the selected training images')
    reference={im.name:im for im in original}
    centers=[];angles=[];rows=[]
    for name in sorted(wanted):
        before,after=reference[name],refined[name]
        distance=float(np.linalg.norm(after.C-before.C))
        angle=float(np.degrees(Rotation.from_matrix(after.R@before.R.T).magnitude()))
        centers.append(distance);angles.append(angle)
        rows.append(dict(name=name,center_change=distance,rotation_change_degrees=angle))
    fraction=float(np.mean(np.array(centers)<=job['alignment_max_error']))
    report=dict(registered=len(wanted),alignment_max_error=job['alignment_max_error'],
                alignment_inlier_fraction=fraction,
                center_change=dict(zip(('median','p90','max'),map(float,np.quantile(centers,[.5,.9,1])))),
                rotation_change_degrees=dict(zip(('median','p90','max'),map(float,np.quantile(angles,[.5,.9,1])))),
                frames=rows,units='Original scene units for camera-center changes; degrees for rotations',
                caveat='Agreement with old poses is a coordinate-frame sanity check, not proof of improved accuracy.')
    if fraction<.7 or report['rotation_change_degrees']['median']>15:
        report['accepted']=False
    else: report['accepted']=True
    return report


def export(workspace):
    job=load_job(workspace);region=Path(job['region']);out=workspace/'dataset'
    if out.exists():
        report=json.loads((out/'colmap_rerun.json').read_text())
        if not report.get('accepted') or report['workspace']!=str(workspace.resolve()):
            raise ValueError('Existing dataset lacks an accepted COLMAP audit')
        for path,expected in report['colmap_files'].items():
            if digest(path)!=expected: raise ValueError('COLMAP result changed after dataset export')
        for relative,expected in report['exported_files'].items():
            if digest(out/relative)!=expected: raise ValueError(f'Exported COLMAP dataset changed: {relative}')
        print(f'Reusing aligned dataset: {out}',flush=True);return out
    _,cameras,images,_=read_model(region)
    _,calibrated,aligned,aligned_stamps=read_model(workspace/'aligned',model_dir=workspace/'aligned')
    refined={im.name:im for im in aligned};wanted=set(job['selected'])
    originals={im.name:im for im in images}
    audit=pose_audit(images,aligned,job)
    # Mapping may assign its own image IDs; names are the correspondence key.
    for im in aligned:
        cam=calibrated[im.camera_id]
        source_image=originals[im.name]
        expected=job['cameras'][str(source_image.camera_id)]
        if cam.width!=expected['width'] or cam.height!=expected['height'] or not np.allclose(cam.params,expected['params'],rtol=1e-8,atol=1e-6):
            raise ValueError(f'COLMAP changed fixed intrinsics: {im.name}')
    audit.update(workspace=str(workspace.resolve()),parent_region=str(region),
                 colmap_files={s[0]:digest(s[0]) for s in aligned_stamps},
                 camera_pose_protocol='Rerun COLMAP train poses aligned to original LiDAR coordinates; original held-out poses',
                 gaussian_initialization='Same original regional input.ply; new COLMAP sparse points are not used for GS initialization',
                 heldout_used_in_colmap=False,intrinsics_fixed=True,
                 evaluation='Identical images and original ROI masks; training render poses differ from the control')
    if job.get('frontend')=='superpoint-superglue':
        audit['feature_matching']=json.loads((workspace/'superglue.json').read_text())
    atomic_json(workspace/'alignment_report.json',audit)
    if not audit['accepted']:
        raise ValueError('COLMAP-to-LiDAR alignment failed sanity checks; inspect alignment_report.json before training')
    stage=workspace/('.dataset.'+uuid.uuid4().hex+'.tmp');sparse=stage/'sparse/0';sparse.mkdir(parents=True)
    try:
        for name in ('images','masks','input.ply','original_lidar_indices.npy','overlays'):
            if (region/name).exists(): (stage/name).symlink_to((region/name).resolve(),target_is_directory=(region/name).is_dir())
        camera_lines=[f'{cid} PINHOLE {cam.width} {cam.height} '+' '.join(format(x,'.17g') for x in cam.params)+'\n'
                      for cid,cam in sorted(cameras.items())]
        (sparse/'cameras.txt').write_text(''.join(camera_lines))
        lines=[];poses=[]
        for old in images:
            im=refined[old.name] if old.name in wanted else old
            q=Rotation.from_matrix(im.R).as_quat()[[3,0,1,2]];t=-im.R@im.C
            lines.append(f'{old.id} '+' '.join(format(float(x),'.17g') for x in [*q,*t])+f' {old.camera_id} {old.name}\n\n')
            poses.append(dict(name=old.name,R_world_to_camera=im.R.tolist(),t_world_to_camera=t.tolist(),center_world=im.C.tolist()))
        (sparse/'images.txt').write_text(''.join(lines))
        (sparse/'points3D.txt').write_text('# GS initialization uses the unchanged regional LiDAR input.ply.\n')
        (sparse/'points3D.ply').symlink_to('../../input.ply')
        meta=copy.deepcopy(json.loads((region/'selection.json').read_text()))
        meta.update(region=str(out.resolve()),parent_region=str(region),colmap_rerun=str((out/'colmap_rerun.json').resolve()),
                    camera_pose_protocol=audit['camera_pose_protocol'])
        manifest=json.loads((region/'keyframes.json').read_text());manifest['source_path']=str(out.resolve())
        for name,value in [('selection.json',meta),('keyframes.json',manifest),('poses.json',poses)]:
            atomic_json(stage/name,value)
        audit['exported_files']={name:digest(stage/name) for name in (
            'selection.json','keyframes.json','poses.json','sparse/0/cameras.txt','sparse/0/images.txt')}
        atomic_json(stage/'colmap_rerun.json',audit)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage);raise
    print(f'Exported {len(wanted)} new train poses + {len(images)-len(wanted)} unchanged held-out poses: {out}',flush=True)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('frames','colmap_files')},indent=2),flush=True)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','seed-cameras','choose-model','export'))
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--region',type=Path)
    parser.add_argument('--mapping-root',type=Path)
    parser.add_argument('--alignment-max-error',type=float)
    parser.add_argument('--frontend',choices=('sift','superglue'),default='sift')
    parser.add_argument('--sequential-overlap',type=int,default=20)
    args=parser.parse_args(argv);workspace=args.workspace.expanduser().resolve()
    if args.stage=='prepare':
        if args.region is None: parser.error('prepare requires --region')
        prepare(args.region,workspace,args.alignment_max_error,args.frontend,args.sequential_overlap)
    elif args.stage=='seed-cameras': seed_cameras(workspace)
    elif args.stage=='choose-model':
        if args.mapping_root is None: parser.error('choose-model requires --mapping-root')
        choose_model(workspace,args.mapping_root)
    else: export(workspace)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
