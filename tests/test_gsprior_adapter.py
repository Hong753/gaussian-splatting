"""TSDF coordinates plus actual pinned-upstream optimizer/population integration."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.gsprior_runtime import sample_sdf, apply_pull
from scripts.setup_gsprior import COMMIT, patch_sources


def test_xyz_volume_interpolation_fractional_non_cubic_and_outside():
    xyz=torch.stack(torch.meshgrid(torch.arange(3),torch.arange(4),torch.arange(5),indexing='ij'),-1).float()
    volume=(xyz*torch.tensor([1.,2.,3.])).sum(-1)
    origin=torch.tensor([10.,-7.,3.])
    model=SimpleNamespace(_prior_volume=volume,vol_origin=origin,voxel_size=.25)
    coords=torch.tensor([[0,0,0],[2,3,4],[.5,1.5,2.5],[-.01,0,0],[3,0,0]])
    sampled=sample_sdf(model,origin+coords*.25)
    torch.testing.assert_close(sampled.flatten(),torch.tensor([0.,20.,11.,1.,1.]))
    assert sample_sdf(model,torch.empty((0,3))).shape==(0,1)
    points=(origin+coords[2]*.25).repeat(131073,1)
    torch.testing.assert_close(sample_sdf(model,points),torch.full((131073,1),11.))


def test_pull_keeps_parameter_identity_and_fixed_coordinates():
    for fixed in (False,True):
        parameter=torch.nn.Parameter(torch.zeros((2,3)))
        m=SimpleNamespace(_xyz=parameter,_fixed_xyz=fixed)
        opt=torch.optim.Adam([parameter],lr=.01)
        apply_pull(m,torch.ones((2,1)),torch.ones((2,3)))
        assert m._xyz is parameter and opt.param_groups[0]['params'][0] is parameter
        torch.testing.assert_close(parameter,torch.full((2,3),0. if fixed else -1e-7),atol=0,rtol=0)


@pytest.fixture
def upstream(tmp_path):
    origin=os.environ.get('GSPRIOR_TEST_REPO')
    if not origin: pytest.skip('Set GSPRIOR_TEST_REPO to a clone containing the pinned upstream commit')
    archive=subprocess.check_output(['git','-C',origin,'archive',COMMIT])
    repo=tmp_path/'upstream';repo.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as f: f.extractall(repo,filter='data')
    patch_sources(repo)
    return repo


def test_fresh_pin_patches_compile_and_preserve_released_losses(upstream):
    for name in ('train.py','scene/dataset_readers.py','scene/gaussian_model.py',
                 'arguments/__init__.py','utils/fusion.py','experiment_adapter.py',
                 'submodules/diff-plane-rasterization/diff_plane_rasterization/__init__.py'):
        compile((upstream/name).read_text(),name,'exec')
    train=(upstream/'train.py').read_text()
    assert 'loss += 1e-3*loss_surface + 1e-3*loss_free' in train
    assert 'experiment.update_prior(scene' in train and 'experiment.finish_iteration(scene' in train
    assert 'dataset.antialiasing = False' in train and 'dataset.model_type = "gsprior"' in train
    assert 'run_tsdf_fusion(' not in train
    assert 'if not os.path.exists(ply_path):' in (upstream/'scene/dataset_readers.py').read_text()
    assert 'self.preload_img = False' in (upstream/'arguments/__init__.py').read_text()


def test_actual_upstream_model_fixed_and_adaptive_adam_growth_prior_pull_pruning(upstream):
    # Substitute only CUDA allocation and KNN; execute actual upstream model code.
    program=r'''
import sys, types, argparse
import torch, numpy as np
torch.Tensor.cuda=lambda self,*a,**k:self
for name in ('zeros','ones','empty','tensor','randn'):
    original=getattr(torch,name)
    def cpu(*args,_f=original,**kwargs):
        if str(kwargs.get('device','')).startswith('cuda'):kwargs['device']='cpu'
        return _f(*args,**kwargs)
    setattr(torch,name,cpu)
knn=types.ModuleType('simple_knn');ext=types.ModuleType('simple_knn._C')
ext.distCUDA2=lambda x:torch.cdist(x,x).square().sort(dim=1).values[:,1:4].mean(1)
sys.modules['simple_knn']=knn;sys.modules['simple_knn._C']=ext
from scene import GaussianModel
from arguments import OptimizationParams
from utils.graphics_utils import BasicPointCloud
from experiment_adapter import install_model_controls,apply_pull
p=argparse.ArgumentParser();group=OptimizationParams(p);opt=group.extract(p.parse_args([]))
xyz=np.array([[-.1,-.1,2],[.1,-.1,2],[-.1,.1,2],[.1,.1,2]],np.float32)
for fixed in (False,True):
    m=GaussianModel(3);m.create_from_pcd(BasicPointCloud(xyz,np.full_like(xyz,.5),np.zeros_like(xyz)),1.)
    m.training_setup(opt);install_model_controls(m,fixed)
    before=m.get_xyz.detach().clone()
    m.update_learning_rate(100)
    loss=m._features_dc.sum()+m._opacity.sum()+m._scaling.sum()
    if not fixed:loss=loss+m.get_xyz.sum()
    loss.backward();m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
    assert torch.equal(before,m.get_xyz)==fixed
    if fixed:assert m._xyz not in m.optimizer.state and not m._xyz.requires_grad
    with torch.no_grad():
        m.denom[:]=1;m.denom_abs[:]=1;m.xyz_gradient_accum[:]=1;m.xyz_gradient_accum_abs[:]=1
        m._opacity[:]=0
        m.densify_and_prune(.0002,.0008,.005,100.,None,iteration=600)
        if fixed:
            assert torch.equal(before,m.get_xyz)
            assert not m.xyz_gradient_accum.any() and not m.max_radii2D.any()
            m.gs_state=False;m.set_tsdf(np.zeros((4,4,4),np.float32),[-1,-1,1],1.,1.)
            for iteration in (31500,32000):
                m.denom[:]=1;m.denom_abs[:]=1;m.xyz_gradient_accum[:]=1;m.xyz_gradient_accum_abs[:]=1
                m.densify_and_prune(.0002,.0008,.005,100.,None,iteration=iteration)
                assert torch.equal(before,m.get_xyz) and not m._xyz.requires_grad
            m.reset_opacity(1000)
            assert (m.get_opacity()<=.010001).all()
            m.prune_points(torch.tensor([True,False,False,False]))
            assert torch.equal(before[1:],m.get_xyz) and not m._xyz.requires_grad
            try:m.prune_points(torch.ones(3,dtype=torch.bool))
            except RuntimeError:pass
            else:raise AssertionError('Must reject total population deletion')
        else:
            assert len(m.get_xyz)>4
            old=m._xyz;apply_pull(m,torch.ones_like(m._opacity),torch.ones_like(m._xyz))
            assert m._xyz is old and m.optimizer.param_groups[0]['params'][0] is old
            m.max_all_points=1
            m.densify_and_clone(torch.ones((len(m.get_xyz),1)),.0002,100.)
print('actual upstream fixed/adaptive controls passed')
'''
    result=subprocess.run([sys.executable,'-c',program],cwd=upstream,
        env={**os.environ,'PYTHONPATH':str(upstream),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},
        text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
