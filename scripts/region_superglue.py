#!/usr/bin/env python3
"""Disk-cached SuperPoint/SuperGlue sequential matches for regional COLMAP."""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.region_colmap import load_job
from scripts.keyframe_pipeline.common import natural_key
from utils.checkpoint_snapshots import atomic_json, atomic_write, digest

REVISION='ddcf11f42e7e0732a0c4607648f9448ea8d73590'
MODEL_FILES=('models/superpoint.py','models/superglue.py',
             'models/weights/superpoint_v1.pth','models/weights/superglue_indoor.pth')


def sequential_pairs(names,overlap):
    if overlap<1 or len(set(names))!=len(names): raise ValueError('Need positive overlap and unique image names')
    ordered=sorted(names,key=natural_key)
    return [(a,b) for i,a in enumerate(ordered) for b in ordered[i+1:i+1+overlap]]


def cache_path(workspace,kind,key):
    identity=hashlib.sha256(json.dumps(key).encode()).hexdigest()
    return workspace/kind/(identity+'.npz')


def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    def write(temp):
        with temp.open('wb') as f: np.savez(f,**arrays)
    atomic_write(path,write)


def prepare(workspace,repo,max_size=1600,max_keypoints=4096):
    job=load_job(workspace)
    if job.get('frontend')!='superpoint-superglue': raise ValueError('Need a fresh SuperGlue workspace, not a SIFT cache')
    if max_size<32 or max_keypoints<16: raise ValueError('Image size / keypoint cap too small')
    repo=repo.resolve()
    config=dict(schema=1,job_sha256=digest(workspace/'job.json'),repository=str(repo),revision=REVISION,
                model_sha256={name:digest(repo/name) for name in MODEL_FILES},
                max_size=max_size,max_keypoints=max_keypoints,nms_radius=4,keypoint_threshold=.005,
                weights='indoor',sinkhorn_iterations=100,match_threshold=.2,
                overlap=job['sequential_overlap'],names=sorted(job['selected'],key=natural_key),
                pixel_convention='SuperPoint pixel centers at integers; COLMAP centers at half-integers',
                pairing='Each selected frame with next overlap selected frames, in natural filename order; no loop retrieval')
    path=workspace/'superglue.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('SuperGlue inputs/settings changed; use a new VERSION/workspace')
    atomic_json(path,config)
    pairs=sequential_pairs(config['names'],config['overlap'])
    atomic_write(workspace/'pairs.txt',lambda p:p.write_text(''.join(f'{a} {b}\n' for a,b in pairs)))
    # The native importer creates version-correct image/rig/frame records. These
    # empty placeholders are replaced by learned keypoints before verification.
    for name in config['names']:
        path=workspace/'image_registration'/f'{name}.txt';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('0 128\n')
    print(f'SuperPoint + indoor SuperGlue: {len(config["names"])} images, {len(pairs):,} sequential pairs; overlap={config["overlap"]}.',flush=True)
    return config


def load_config(workspace):
    load_job(workspace)
    config=json.loads((workspace/'superglue.json').read_text())
    if config['job_sha256']!=digest(workspace/'job.json'): raise ValueError('SuperGlue job changed')
    for name,expected in config['model_sha256'].items():
        if digest(Path(config['repository'])/name)!=expected: raise ValueError(f'SuperGlue model cache changed: {name}')
    return config


def model_class(repo,name):
    spec=importlib.util.spec_from_file_location('region_'+name,Path(repo)/'models'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return getattr(module,'SuperPoint' if name=='superpoint' else 'SuperGlue')


def extract_one(model,path,device,max_size):
    import cv2
    import torch
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None: raise ValueError(f'Cannot read image: {path}')
    height,width=image.shape
    ratio=min(1.,max_size/max(width,height));size=(round(width*ratio),round(height*ratio))
    if min(size)<8: raise ValueError(f'Image too small: {path}')
    if size!=(width,height): image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    tensor=torch.from_numpy(image.astype(np.float32)/255.)[None,None].to(device)
    with torch.inference_mode(): result=model({'image':tensor})
    data={key:result[key][0].detach().cpu().numpy().astype(np.float32) for key in ('keypoints','scores','descriptors')}
    # Resize maps (p + .5) to the source pixel-center system. COLMAP's .5 offset
    # then cancels the -.5 needed for integer-centered source coordinates.
    data['keypoints_colmap']=((data['keypoints']+.5)*np.array([width/size[0],height/size[1]])).astype(np.float32)
    data['image_size']=np.array(size,dtype=np.int32)
    data['original_size']=np.array([width,height],dtype=np.int32)
    validate_feature(data)
    return data


def validate_feature(data):
    n=len(data['keypoints'])
    if data['keypoints'].shape!=(n,2) or data['keypoints_colmap'].shape!=(n,2) or data['descriptors'].shape!=(256,n) or data['scores'].shape!=(n,):
        raise ValueError('Invalid SuperPoint cache shapes')
    if not all(np.isfinite(v).all() for v in data.values()): raise ValueError('Nonfinite SuperPoint output')
    for key in ('image_size','original_size'):
        if data[key].shape!=(2,) or (data[key]<8).any(): raise ValueError('Invalid image dimensions')
    return data


def read_feature(workspace,name):
    with np.load(cache_path(workspace,'superpoint',name),allow_pickle=False) as data:
        return validate_feature({k:data[k] for k in data.files})


def extract(workspace,config,device):
    model=model_class(config['repository'],'superpoint')({k:config[k] for k in ('max_keypoints','nms_radius','keypoint_threshold')}).eval().to(device)
    for i,name in enumerate(config['names']):
        path=cache_path(workspace,'superpoint',name)
        if path.exists(): data=read_feature(workspace,name)
        else:
            data=extract_one(model,workspace/'images'/name,device,config['max_size'])
            save_npz(path,**data)
        if len(data['keypoints'])<8: raise ValueError(f'{name}: fewer than 8 SuperPoint keypoints; cannot keep this frame in the pose ablation')
        if i%10==0 or i+1==len(config['names']):
            print(f'SuperPoint {i+1}/{len(config["names"])}: {name}, {len(data["keypoints"])} points',flush=True)


def match_one(model,a,b,device):
    import torch
    inputs={}
    for i,data in enumerate((a,b)):
        for key in ('keypoints','scores','descriptors'):
            inputs[f'{key}{i}']=torch.from_numpy(data[key])[None].to(device)
        w,h=map(int,data['image_size'])
        # Pinned SuperGlue uses image.shape only, to normalize keypoint positions.
        inputs[f'image{i}']=SimpleNamespace(shape=(1,1,h,w))
    with torch.inference_mode(): result=model(inputs)
    indices=result['matches0'][0].cpu().numpy()
    confidence=result['matching_scores0'][0].cpu().numpy()
    valid=indices>=0
    matches=np.column_stack([np.flatnonzero(valid),indices[valid]]).astype(np.uint32)
    if len(matches) and (matches[:,1].max()>=len(b['keypoints']) or len(np.unique(matches[:,1]))!=len(matches)):
        raise ValueError('Invalid or non-mutual SuperGlue matches')
    if not np.isfinite(confidence).all(): raise ValueError('Nonfinite SuperGlue confidence')
    return matches,confidence[valid].astype(np.float32)


def match(workspace,config,device):
    model=model_class(config['repository'],'superglue')({k:config[k] for k in ('weights','sinkhorn_iterations','match_threshold')}).eval().to(device)
    pairs=sequential_pairs(config['names'],config['overlap'])
    # At most one temporal window in CPU memory; only the current pair on GPU.
    feature=lru_cache(maxsize=min(32,config['overlap']+1))(lambda name:read_feature(workspace,name))
    for i,(a,b) in enumerate(pairs):
        path=cache_path(workspace,'superglue_matches',[a,b])
        if not path.exists():
            matches,scores=match_one(model,feature(a),feature(b),device)
            save_npz(path,matches=matches,scores=scores)
        if i%50==0 or i+1==len(pairs):
            print(f'SuperGlue {i+1}/{len(pairs)}: {a} <-> {b}',flush=True)
    counts=[]
    def write_raw(path):
        with path.open('w') as stream:
            for a,b in pairs:
                with np.load(cache_path(workspace,'superglue_matches',[a,b]),allow_pickle=False) as data:
                    matches=data['matches'];scores=data['scores']
                if matches.ndim!=2 or matches.shape[1]!=2 or scores.shape!=(len(matches),): raise ValueError('Invalid pair cache')
                counts.append(len(matches));stream.write(f'{a} {b}\n')
                np.savetxt(stream,matches,fmt='%d');stream.write('\n')
    atomic_write(workspace/'superglue_raw_matches.txt',write_raw)
    atomic_json(workspace/'superglue_matching_report.json',dict(images=len(config['names']),pairs=len(pairs),
        overlap=config['overlap'],matches=sum(counts),median_matches=float(np.median(counts)),
        zero_match_pairs=counts.count(0),geometrically_verified=False))


def import_keypoints(workspace,config):
    job=load_job(workspace)
    expected_cameras={name:int(cid) for cid,names in job['groups'].items() for name in names}
    with sqlite3.connect(workspace/'database.db') as db:
        images={name:(iid,cid) for iid,name,cid in db.execute('SELECT image_id,name,camera_id FROM images')}
        if set(images)!=set(config['names']): raise ValueError('COLMAP image import did not preserve the selected training set')
        for name in config['names']:
            iid,cid=images[name]
            if cid!=expected_cameras[name]: raise ValueError(f'Wrong shared camera for {name}')
            keypoints=read_feature(workspace,name)['keypoints_colmap'].astype('<f4')
            if len(keypoints)<8: raise ValueError(f'Too few keypoints: {name}')
            descriptor=db.execute('SELECT rows FROM descriptors WHERE image_id=?',(iid,)).fetchone()
            if descriptor and descriptor[0]!=0: raise ValueError('Refusing to mix learned features with an existing SIFT database')
            existing=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(iid,)).fetchone()
            row=(len(keypoints),2,keypoints.tobytes())
            if existing and existing[0] and existing!=row: raise ValueError(f'Keypoint ordering changed: {name}')
            db.execute('INSERT OR REPLACE INTO keypoints(image_id,rows,cols,data) VALUES(?,?,?,?)',(iid,*row))
            # SuperGlue descriptors stay in their float32 disk cache. No dummy
            # SIFT descriptors are passed to matching or geometric verification.
            db.execute('DELETE FROM descriptors WHERE image_id=?',(iid,))
    print(f'Imported original-resolution SuperPoint coordinates for {len(images)} images.',flush=True)


def check_verified(workspace,config):
    with sqlite3.connect(workspace/'database.db') as db:
        ids={name:iid for iid,name in db.execute('SELECT image_id,name FROM images')}
        rows=list(db.execute('SELECT pair_id,rows FROM two_view_geometries'))
    if set(ids)!=set(config['names']): raise ValueError('Unexpected training membership in database')
    graph={name:set() for name in ids};lookup={}
    for a,b in sequential_pairs(config['names'],config['overlap']):
        lo,hi=sorted((ids[a],ids[b]));lookup[lo*2147483647+hi]=(a,b)
    verified=0
    for pair_id,n in rows:
        if pair_id not in lookup: raise ValueError('Database contains non-sequential image pairs')
        if n>=15:
            a,b=lookup[pair_id];graph[a].add(b);graph[b].add(a);verified+=1
    remaining=set(graph);components=[]
    while remaining:
        stack=[min(remaining)];component=[];remaining.remove(stack[0])
        while stack:
            name=stack.pop();component.append(name)
            for neighbor in graph[name]&remaining: remaining.remove(neighbor);stack.append(neighbor)
        components.append(sorted(component,key=natural_key))
    report=dict(expected_pairs=len(lookup),stored_pairs=len(rows),verified_pairs=verified,
                inlier_threshold=15,components=components,isolated=[n for n,edges in graph.items() if not edges])
    atomic_json(workspace/'verification_report.json',report)
    print(f'Geometric verification: {verified}/{len(lookup)} pairs have >=15 inliers; {len(components)} connected components.',flush=True)
    if len(components)!=1:
        raise ValueError('Sequential match graph is disconnected. See verification_report.json; increase overlap in a new workspace before training. No frames were silently dropped.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','extract','match','import-keypoints','check-verified'))
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--repository',type=Path)
    parser.add_argument('--max-size',type=int,default=1600)
    parser.add_argument('--max-keypoints',type=int,default=4096)
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda',help='CPU is for small smoke checks; normal runs use CUDA')
    args=parser.parse_args(argv);workspace=args.workspace.expanduser().resolve()
    if args.stage=='prepare':
        if args.repository is None: parser.error('prepare requires --repository')
        prepare(workspace,args.repository,args.max_size,args.max_keypoints);return 0
    config=load_config(workspace)
    if args.stage in ('extract','match'):
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
        import torch
        if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('SuperPoint/SuperGlue requires CUDA in the active 3dgs environment')
        torch.set_num_threads(8)
        print(f'{args.stage}: {torch.cuda.get_device_name(0) if args.device=="cuda" else "CPU"}; physical GPU {args.gpu}',flush=True)
        (extract if args.stage=='extract' else match)(workspace,config,args.device)
    elif args.stage=='import-keypoints': import_keypoints(workspace,config)
    else: check_verified(workspace,config)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
