import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.smoke_native_lidar import check_empty_rendering
from scripts.audit_pruning import audit
from utils.densification_policy import PruningWouldEmptyScene
import test_native_lidar as native
from test_frozen_native_lidar import options as frozen_options


def test_frozen_default_matches_native_radius_reset_without_growth():
    for opt in (native.options(),frozen_options()):
        model=native.model(opt)
        original=model.get_xyz.detach().clone()
        with torch.no_grad(),native.cpu_allocations():
            model._scaling[:]=-7
            model._opacity[:]=0
            model.xyz_gradient_accum.zero_();model.denom[:]=1
            model.max_radii2D[:]=100  # A previously close camera must not become lifetime pruning.
            model.densify_and_prune(.0002,.005,1.,20,torch.ones(2),record_stats=True,protect_nonempty=True)
        assert torch.equal(model.get_xyz,original)
        assert model.last_densification_stats['prune_screen_candidates']==0
        assert not model.max_radii2D.any()


def test_explicit_radius_window_still_prunes_and_clears_afterwards():
    model=native.model(frozen_options());original=model.get_xyz.detach().clone()
    with torch.no_grad(),native.cpu_allocations():
        model._scaling[:]=-7;model._opacity[:]=0
        model.max_radii2D[:]=torch.tensor([100.,2.])
        model.densify_and_prune(.0002,.005,1.,20,torch.ones(2),record_stats=True,
                                preserve_radii=True,protect_nonempty=True)
    assert torch.equal(model.get_xyz,original[1:]) and not model.max_radii2D.any()
    assert model.last_densification_stats['prune_screen_candidates']==1


@pytest.mark.parametrize('reason',['opacity','world','screen'])
def test_last_rows_not_deleted_and_reason_is_reported(reason):
    model=native.model(frozen_options());xyz=model._xyz;opacity=model._opacity
    with torch.no_grad(),native.cpu_allocations():
        model._scaling[:]=-7;model._opacity[:]=0;model.max_radii2D.zero_()
        if reason=='opacity':model._opacity[:]=-20
        if reason=='world':model._scaling[:]=0
        if reason=='screen':model.max_radii2D[:]=100
        with pytest.raises(PruningWouldEmptyScene) as failure:
            model.densify_and_prune(.0002,.005,1.,20,torch.ones(2),record_stats=True,
                preserve_radii=reason=='screen',protect_nonempty=True)
    assert model._xyz is xyz and model._opacity is opacity and len(model.get_xyz)==2
    assert failure.value.report['prune_'+reason+'_candidates']==2


def test_empty_model_bypasses_cuda_and_invisible_nonempty_view_keeps_rows():
    calls=[]
    class Rasterizer:
        def __init__(self,raster_settings):self.settings=raster_settings
        def __call__(self,**kwargs):
            n=len(kwargs['means3D']);assert n>0,'Empty model entered CUDA'
            calls.append(n)
            zero=sum(value.sum()*0 for value in kwargs.values() if torch.is_tensor(value))
            return (self.settings.bg[:,None,None].expand(3,8,8)+zero,
                    torch.zeros(n,dtype=torch.int32),zero.expand(1,8,8))
    stub=SimpleNamespace(GaussianRasterizationSettings=lambda **kw:SimpleNamespace(**kw),GaussianRasterizer=Rasterizer)
    spec=importlib.util.spec_from_file_location('_empty_renderer',Path(__file__).resolve().parents[1]/'gaussian_renderer/__init__.py')
    module=importlib.util.module_from_spec(spec)
    zeros_like=torch.zeros_like
    def cpu_zeros_like(value,**kwargs):
        kwargs.pop('device',None);return zeros_like(value,**kwargs)
    with patch.dict(sys.modules,{'diff_gaussian_rasterization':stub,
                                'scene.gaussian_model':SimpleNamespace(GaussianModel=object)}):
        spec.loader.exec_module(module)
        with patch.object(module.torch,'zeros_like',side_effect=cpu_zeros_like):
            assert check_empty_rendering(module.render,'cpu')['empty_cloud_sh_backward']
    assert calls==[2]


def test_audit_distinguishes_zero_population_from_missing_reason_counts(tmp_path):
    (tmp_path/'densification.jsonl').write_text('\n'.join(json.dumps(row) for row in (
        dict(iteration=6400,points_before=100,points_after=3,pruned=97),
        dict(iteration=6500,points_before=3,points_after=0,pruned=3))))
    result=audit(tmp_path)
    assert result['first_zero_population']['iteration']==6500 and result['final_population']==0
    assert not result['reason_counts_recorded']
