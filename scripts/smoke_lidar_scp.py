#!/usr/bin/env python3
"""Check LiDAR SCP opacity gradients, tilted-plane pulls and unknown regions."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.checkpoint_snapshots import atomic_json,digest


def analytic_field(folder):
    normal=np.array([.3,.4,np.sqrt(.75)],np.float32)
    tangent=np.cross(normal,[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
    other=np.cross(normal,tangent)
    x,y=np.meshgrid(np.linspace(-.5,.5,31),np.linspace(-.5,.5,31))
    anchors=(x.ravel()[:,None]*tangent+y.ravel()[:,None]*other).astype(np.float32)
    origin=anchors.min(0).astype(np.float64);cell=.08
    cells=np.floor((anchors.astype(np.float64)-origin)/cell).astype(np.int64);dims=cells.max(0)+1
    keys=cells[:,0]+dims[0]*(cells[:,1]+dims[1]*cells[:,2]);keys,indices=np.unique(keys,return_index=True)
    folder.mkdir(parents=True)
    np.savez(folder/'field.npz',keys=keys,anchors=anchors[indices],normals=np.tile(normal,(len(keys),1)),
             confidence=np.ones(len(keys),np.float32),origin=origin,dims=dims,cell=np.array(cell))
    atomic_json(folder/'metadata.json',dict(schema=1,field_sha256=digest(folder/'field.npz')))
    return normal


def verify(device='cpu'):
    import torch
    from utils.lidar_scp import SparseLiDARField,LiDARSCP,opacity_loss
    with tempfile.TemporaryDirectory() as tmp:
        cache=Path(tmp)/'field';normal=analytic_field(cache)
        xyz=torch.tensor(np.array([normal*0,normal*.04,normal*-.06,[10,10,10]],np.float32),device=device)
        field=SparseLiDARField(cache,device)
        ref=field.query(xyz)
        assert ref['valid'].tolist()==[True,True,True,False]
        cpu=SparseLiDARField(cache,'cpu').query(xyz.cpu())
        torch.testing.assert_close(ref['signed'].cpu(),cpu['signed'],atol=1e-6,rtol=1e-5)
        opacity=torch.nn.Parameter(torch.full((4,1),-1.3862944,device=device))
        loss=opacity_loss(opacity.sigmoid(),ref,.08,.3);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(opacity.grad).all()
        assert opacity.grad[0]<0 and opacity.grad[1]>0 and opacity.grad[2]>0 and opacity.grad[3]==0
        class Gaussians:
            @property
            def get_xyz(self): return self._xyz
        gs=Gaussians();gs._xyz=torch.nn.Parameter(xyz.clone());gs._opacity=opacity
        gs.optimizer=torch.optim.Adam([gs._xyz,opacity],lr=.001)
        gs.optimizer.state[gs._xyz]={'step':torch.tensor(1.),'exp_avg':torch.ones_like(xyz),'exp_avg_sq':torch.ones_like(xyz)}
        prior=LiDARSCP(cache,dict(start=0,ramp_steps=0,shrink_until=1,pull_every=1,pull_samples=100,
                                 band_initial_cells=1.,band_final_cells=1.),device)
        before=gs._xyz.detach().clone();stats=prior.pull(gs,1)
        normal_t=torch.tensor(normal,device=device)
        delta=gs._xyz.detach()-before
        tangent=delta-(delta@normal_t)[:,None]*normal_t
        torch.testing.assert_close(tangent,torch.zeros_like(tangent),atol=1e-6,rtol=0)
        assert torch.all((gs._xyz.detach()[:3]@normal_t).abs()<=(before[:3]@normal_t).abs()+1e-6)
        assert float(delta[1].norm())>0 and float(delta.norm(dim=1).max())<=.02+1e-6
        torch.testing.assert_close(gs._xyz[3],before[3],atol=0,rtol=0)
        assert gs.optimizer.state[gs._xyz]['exp_avg'][1].abs().sum()==0
        assert gs.optimizer.state[gs._xyz]['exp_avg'][3].sum()==3
        # A newly appended child is classified using its current position.
        child=torch.tensor(normal*.05,device=device)[None]
        gs._xyz=torch.nn.Parameter(torch.cat([gs._xyz.detach(),child]));gs._opacity=torch.nn.Parameter(torch.cat([opacity.detach(),opacity.detach()[:1]]))
        gs.optimizer=torch.optim.Adam([gs._xyz,gs._opacity],lr=.001)
        prior.pull(gs,2)
        assert float((gs._xyz[-1]@normal_t).abs())<.05
        penalty,_=prior.loss(gs,3);penalty.backward()
        assert torch.isfinite(penalty) and gs._xyz.grad is None
        assert torch.isfinite(gs._opacity.grad).all()
    return dict(passed=True,device=str(device),lookup_cpu_parity=True,opacity_gradient_directions=True,
                unknown_geometry_unchanged=True,normal_only_bounded_pull=True,momentum_cleared=True,
                new_children_reclassified=True,opacity_loss_no_position_gradient=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='4');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import torch
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this training smoke check')
    result=verify(args.device)
    if args.device=='cuda': result['cuda_device']=torch.cuda.get_device_name(0)
    if args.output: atomic_json(args.output,result)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
