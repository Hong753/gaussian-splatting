import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.smoke_lidar_scp import verify,analytic_field
from scripts.run_region_ablation import main as prepare_run
from scripts.run_native_lidar import build_plan
from utils.lidar_scp import prepare_cache,policy,schedule,SparseLiDARField,opacity_loss
from test_region_colmap import region,region_source


def test_synthetic_tilted_surface_gradient_pull_unknown_and_child_semantics():
    assert verify('cpu')['passed']


def test_band_schedule_is_in_scene_units_not_scene_extent_and_validates():
    cfg=policy()
    assert schedule(cfg,.04,0)==(0.,.08)
    assert schedule(cfg,.04,5000)==(0.,.08)
    assert schedule(cfg,.04,20000)==(.01,.04)
    assert schedule(cfg,.04,90000)==(.01,.04)
    with pytest.raises(ValueError): policy({'band_initial_cells':-1})
    with pytest.raises(ValueError): policy({'opacity_samples':1.2})
    with pytest.raises(ValueError): policy({'shrink_until':4999})


def test_missing_geometry_has_no_opacity_gradient_and_far_opacity_is_suppressed(tmp_path):
    normal=analytic_field(tmp_path/'cache');field=SparseLiDARField(tmp_path/'cache','cpu')
    xyz=torch.tensor(np.array([normal*.07,[4,5,6],[np.nan,0,0]],np.float32))
    ref=field.query(xyz)
    logits=torch.zeros(3,1,requires_grad=True)
    loss=opacity_loss(logits.sigmoid(),ref,.04,.3);loss.backward()
    assert torch.isfinite(loss) and logits.grad[0]>0 and logits.grad[1:].abs().sum()==0


def test_cached_lidar_field_and_training_plan_preserve_initialization_and_replay(region,tmp_path):
    cache=region/'lidar_scp_v1'
    meta=prepare_cache(region/'input.ply',cache,workers=2)
    assert meta['trusted_representative_fraction']>.9
    assert meta['input_points']>meta['representatives']
    stamp=(cache/'field.npz').stat().st_mtime_ns
    assert prepare_cache(region/'input.ply',cache,workers=2)==meta
    assert (cache/'field.npz').stat().st_mtime_ns==stamp
    with pytest.raises(ValueError,match='inputs changed'):
        prepare_cache(region/'input.ply',cache,voxel_size=.123,workers=2)
    model=region/'runs/scp_test'
    prepare_run(['--region',str(region),'--model',str(model),'--geometry','adaptive',
                 '--position-lr-multiplier','.05','--lidar-scp-cache',str(cache),'--gpu','4'])
    data=json.loads((model/'periodic_evaluation/plan.json').read_text())
    assert str(cache/'field.npz') in data['input_fingerprints']
    assert str(cache/'metadata.json') in data['input_fingerprints']
    values=data['training_args']
    for key in ('source','model','keyframes','point_cloud','loss_mask_dir','lidar_scp_cache'):
        values[key]=Path(values[key])
    plan=build_plan(SimpleNamespace(**values))
    assert plan==data['plan'] and plan['position_lr']['divisor']==20
    assert not plan['hard_culling'] and plan['scale_control']=='off' and plan['pruning']
    assert '--lidar_scp_cache' in plan['stages'][0]['argv']
    assert len(data['selected'])==3 and len(data['frames'])==4
    assert values['debug_preview_every']==500
