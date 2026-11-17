"""LiDAR adaptation of GSPrior's distance-conditioned opacity and surface pull.

The reference is an immutable, confidence-gated local plane field from measured
LiDAR, not a rendered-depth TSDF. It never occludes RGB rays. Unknown geometry
receives no prior. There is no extra scale cap or hard prior-based pruning.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np

from utils.checkpoint_snapshots import atomic_json, atomic_write, digest


DEFAULTS=dict(start=5000,ramp_steps=2000,shrink_until=20000,
              band_initial_cells=2.,band_final_cells=1.,on_surface_fraction=.3,
              opacity_weight=.01,opacity_samples=16384,pull_every=100,
              pull_samples=262144,pull_fraction=.25,max_pull_band_fraction=.25,
              confidence=.6,query_chunk=8192)


def policy(value=None):
    result=dict(DEFAULTS)
    if value:
        supplied=json.loads(value) if isinstance(value,str) else value
        if set(supplied)-set(result): raise ValueError('Unknown LiDAR SCP policy field')
        result.update(supplied)
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in result.values()):
        raise ValueError('LiDAR SCP policy must contain finite numbers')
    for key in ('start','ramp_steps','shrink_until','opacity_samples','pull_every','pull_samples','query_chunk'):
        if int(result[key])!=result[key] or result[key]<(0 if key in ('start','ramp_steps') else 1):
            raise ValueError(f'Invalid LiDAR SCP {key}')
        result[key]=int(result[key])
    if (result['shrink_until']<=result['start'] or
        not 0<result['band_final_cells']<=result['band_initial_cells']<=2. or
        not 0<result['on_surface_fraction']<1 or result['opacity_weight']<0 or
        not 0<result['pull_fraction']<=1 or not 0<result['max_pull_band_fraction']<=1 or
        not 0<result['confidence']<=1):
        raise ValueError('Invalid LiDAR SCP band, confidence, schedule, or weight')
    return result


def read_points(path):
    from plyfile import PlyData
    vertices=PlyData.read(str(path))['vertex'].data
    points=np.column_stack([vertices[key] for key in ('x','y','z')]).astype(np.float32)
    if len(points)<24 or not np.isfinite(points).all():
        raise ValueError('LiDAR prior requires at least 24 finite initialization points')
    return points


def prepare_cache(point_cloud,output,voxel_size=None,workers=8):
    """Bounded-neighborhood CPU preprocessing; training lookup stays on CUDA."""
    from scipy.spatial import cKDTree
    from utils.surface_geometry import estimate_patches,points_digest
    source=Path(point_cloud).expanduser().resolve();output=Path(output).expanduser().resolve()
    if workers<1 or voxel_size is not None and (not math.isfinite(voxel_size) or voxel_size<=0):
        raise ValueError('Need positive worker count and voxel size')
    source_hash=digest(source)
    if (output/'metadata.json').is_file():
        meta=json.loads((output/'metadata.json').read_text())
        if (meta['source_sha256']!=source_hash or meta['requested_voxel_size']!=voxel_size or
            meta['schema']!=1 or meta['field_sha256']!=digest(output/'field.npz')):
            raise ValueError('LiDAR prior cache inputs changed; use a fresh cache directory')
        print(f'Reusing LiDAR prior: {output}',flush=True);return meta
    points=read_points(source)
    print(f'LiDAR prior: {len(points):,} measured points; estimating spacing...',flush=True)
    tree=cKDTree(points)
    sample=points[np.linspace(0,len(points)-1,min(8192,len(points)),dtype=int)]
    distances,_=tree.query(sample,k=4,workers=workers)
    spacing=np.sqrt(np.mean(distances[:,1:]**2,axis=1))
    positive=spacing[np.isfinite(spacing)&(spacing>1e-9)]
    if not len(positive): raise ValueError('LiDAR has no nonzero local spacing')
    median=float(np.median(positive));cell=float(voxel_size or 2*median)
    del tree
    origin=points.min(0).astype(np.float64)
    cells=np.floor((points.astype(np.float64)-origin)/cell).astype(np.int64)
    dims=cells.max(0)+1
    if math.prod(map(int,dims))>=np.iinfo(np.int64).max:
        raise ValueError('LiDAR voxel key range overflows int64')
    keys=cells[:,0]+dims[0]*(cells[:,1]+dims[1]*cells[:,2])
    unique,indices=np.unique(keys,return_index=True)
    anchors=points[indices]  # Select measured points, never average wall sheets.
    del cells,keys,indices
    if len(anchors)<24: raise ValueError('Voxel size left fewer than 24 reference points')
    normals=np.zeros_like(anchors);confidence=np.zeros(len(anchors),np.float32)
    for start,normal,conf,_ in estimate_patches(anchors,neighbors=24,batch=8192,workers=workers,max_support=4*cell):
        end=start+len(normal);normals[start:end]=normal;confidence[start:end]=conf
        print(f'LiDAR prior normals: {end:,}/{len(anchors):,}',flush=True)
    trusted=float((confidence>=DEFAULTS['confidence']).mean())
    if trusted<.01:
        raise ValueError(f'Only {trusted:.1%} reliable planar references; inspect LiDAR before training')
    output.mkdir(parents=True,exist_ok=True)
    def write(path):
        with path.open('wb') as f:
            np.savez(f,keys=unique,anchors=anchors,normals=normals,confidence=confidence,
                     origin=origin,dims=dims,cell=np.array(cell))
    atomic_write(output/'field.npz',write)
    meta=dict(schema=1,method='measured_LiDAR_local_plane_field',source=str(source),source_sha256=source_hash,
        points_sha256=points_digest(points),requested_voxel_size=voxel_size,voxel_size=cell,
        median_initial_knn_sigma=median,input_points=len(points),representatives=len(anchors),
        trusted_representative_fraction=trusted,field_sha256=digest(output/'field.npz'),
        distance_units='Original LiDAR coordinate units; no scene-extent scaling',
        paper='https://arxiv.org/abs/2603.19682',
        adaptation='Fixed local LiDAR planes replace self-generated TSDF; sampled opacity/pull; '
                   'no hard outlier deletion, prior scale cap, normal loss, NCC, or depth distortion loss')
    atomic_json(output/'metadata.json',meta)
    print(json.dumps(meta,indent=2),flush=True)
    return meta


class SparseLiDARField:
    def __init__(self,cache,device='cuda',confidence=.6):
        import torch
        self.cache=Path(cache);self.meta=json.loads((self.cache/'metadata.json').read_text())
        if self.meta['schema']!=1 or digest(self.cache/'field.npz')!=self.meta['field_sha256']:
            raise ValueError('Invalid or changed LiDAR prior cache')
        with np.load(self.cache/'field.npz',allow_pickle=False) as data:
            for key in ('keys','anchors','normals','confidence','origin','dims'):
                value=data[key]
                dtype=torch.long if key in ('keys','dims') else torch.float32
                setattr(self,key,torch.as_tensor(value.copy(),dtype=dtype,device=device))
            self.cell=float(data['cell'])
        if (len(self.keys)==0 or not bool((self.keys[1:]>self.keys[:-1]).all()) or
            self.anchors.shape!=(len(self.keys),3) or self.normals.shape!=self.anchors.shape or
            self.confidence.shape!=(len(self.keys),) or not math.isfinite(self.cell) or self.cell<=0):
            raise ValueError('Invalid LiDAR prior shapes/keys')
        self.threshold=confidence
        self.offsets=torch.tensor(list(itertools.product((-1,0,1),repeat=3)),device=device,dtype=torch.long)

    def query(self,xyz):
        """Nearest measured representative in 27 occupied neighboring cells.

        Invalid/unknown cells and unreliable normals remain unconstrained. The
        correspondence is recomputed from current centers, including children.
        """
        import torch
        with torch.no_grad():
            finite=torch.isfinite(xyz).all(-1)
            base=torch.floor((torch.nan_to_num(xyz)-self.origin)/self.cell).long()
            cells=base[:,None]+self.offsets[None]
            inside=((cells>=0)&(cells<self.dims)).all(-1)&finite[:,None]
            key=cells[...,0]+self.dims[0]*(cells[...,1]+self.dims[1]*cells[...,2])
            hit=torch.searchsorted(self.keys,key).clamp(max=len(self.keys)-1)
            found=inside&(self.keys[hit]==key)
            delta=xyz[:,None]-self.anchors[hit]
            distance2=delta.square().sum(-1).masked_fill(~found,float('inf'))
            closest=distance2.argmin(-1)
            row=torch.arange(len(xyz),device=xyz.device)
            index=hit[row,closest]
            normal=self.normals[index];anchor=self.anchors[index]
            supported=found[row,closest]&(distance2[row,closest] <= (2*self.cell)**2)
            valid=supported&(self.confidence[index]>=self.threshold)
            signed=((xyz-anchor)*normal).sum(-1)
            return dict(valid=valid,normal=normal,signed=torch.where(valid,signed,0.),
                        confidence=self.confidence[index],distance2=distance2[row,closest])


def schedule(cfg,cell,iteration):
    weight=cfg['opacity_weight']*min(1.,max(0.,(iteration-cfg['start'])/max(1,cfg['ramp_steps'])))
    progress=min(1.,max(0.,(iteration-cfg['start'])/(cfg['shrink_until']-cfg['start'])))
    band=cell*cfg['band_initial_cells']*(cfg['band_final_cells']/cfg['band_initial_cells'])**progress
    return weight,band


def opacity_loss(opacity,reference,band,on_surface_fraction):
    """Eq. 3 inspired; invalid samples get zero weight, never an off-surface label."""
    import torch
    distance=(reference['signed'].detach().abs()/band).clamp(max=1.)
    target=(distance<=on_surface_fraction).to(opacity.dtype)
    importance=reference['valid'].to(opacity.dtype)*reference['confidence']/(1+distance).square()
    return (importance*(opacity.reshape(-1)-target).square()).mean()


class LiDARSCP:
    def __init__(self,cache,cfg=None,device='cuda',point_cloud=None):
        self.cfg=policy(cfg);self.field=SparseLiDARField(cache,device,self.cfg['confidence'])
        if point_cloud is not None and digest(point_cloud)!=self.field.meta['source_sha256']:
            from utils.surface_geometry import points_digest
            if points_digest(read_points(point_cloud))!=self.field.meta['points_sha256']:
                raise ValueError('LiDAR prior and Gaussian initialization points differ')

    def loss(self,gaussians,iteration):
        import torch
        cfg=self.cfg;xyz=gaussians.get_xyz
        weight,band=schedule(cfg,self.field.cell,iteration)
        stats=dict(prior_weight=weight,band=band)
        self.gradient_sample=None
        if weight==0: return xyz.new_zeros(()),stats
        generator=torch.Generator(device=xyz.device).manual_seed(9713+iteration)
        ids=torch.randint(len(xyz),(min(len(xyz),cfg['opacity_samples']),),device=xyz.device,generator=generator)
        references=[]
        for chunk in ids.split(cfg['query_chunk']): references.append(self.field.query(xyz[chunk].detach()))
        ref={k:torch.cat([r[k] for r in references]) for k in references[0]}
        opacity=gaussians._opacity[ids].sigmoid()
        if iteration%100==0:
            sample=ids[:2048]
            before=gaussians._opacity.grad[sample].detach().clone() if gaussians._opacity.grad is not None else torch.zeros_like(gaussians._opacity[sample])
            self.gradient_sample=(sample,before)
        loss=weight*opacity_loss(opacity,ref,band,cfg['on_surface_fraction'])
        valid=ref['valid'];near=valid&(ref['signed'].abs()<=cfg['on_surface_fraction']*band)
        stats.update(samples=len(ids),supported_fraction=valid.float().mean(),
                     on_surface_fraction=near.float().mean(),weighted_loss=loss.detach(),
                     normal_distance_mean=(ref['signed'].abs()*valid).sum()/valid.sum().clamp(min=1))
        return loss,stats

    def gradient_stats(self,gaussians):
        if self.gradient_sample is None: return {}
        ids,before=self.gradient_sample
        added=gaussians._opacity.grad[ids].detach()-before
        rgb=float(before.norm());prior=float(added.norm())
        return dict(rgb_opacity_gradient_norm=rgb,prior_opacity_gradient_norm=prior,
                    prior_to_rgb_opacity_gradient_ratio=prior/rgb if rgb else None)

    def pull(self,gaussians,iteration):
        import torch
        cfg=self.cfg
        if iteration<=cfg['start'] or iteration%cfg['pull_every']: return None
        weight,band=schedule(cfg,self.field.cell,iteration)
        strength=min(1.,max(0.,(iteration-cfg['start'])/max(1,cfg['ramp_steps'])))
        xyz=gaussians._xyz;n=len(xyz);count=min(n,cfg['pull_samples'])
        # A deterministic cyclic subset avoids allocating a scene-sized randperm.
        start=((iteration//cfg['pull_every'])*cfg['pull_samples'])%n
        ids=(torch.arange(count,device=xyz.device)+start)%n
        moved=0;known=0;max_move=0.
        with torch.no_grad():
            for chunk in ids.split(cfg['query_chunk']):
                ref=self.field.query(xyz[chunk])
                valid=ref['valid']&(ref['signed'].abs()<=band)
                distance=(ref['signed']*cfg['pull_fraction']*strength).clamp(
                    -cfg['max_pull_band_fraction']*band,cfg['max_pull_band_fraction']*band)
                distance=torch.where(valid,distance,0.)
                changed=valid&(distance.abs()>1e-9)
                selected=chunk[changed]
                xyz[selected]-=distance[changed,None]*ref['normal'][changed]
                # A direct projection must not be undone by old Adam momentum.
                state=gaussians.optimizer.state.get(xyz,{})
                for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                    if key in state: state[key][selected]=0
                moved+=int(changed.sum());known+=int(ref['valid'].sum())
                max_move=max(max_move,float(distance.abs().max()))
        return dict(pull_samples=count,pulled=moved,pull_supported=known,max_pull=max_move,band=band)
