"""Unmasked regional restart: actual sharpness selection, native LRs and resets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from arguments import ModelParams,OptimizationParams,PipelineParams
from scripts.run_region_full_image import main as prepare
from scripts import run_native_lidar as native,run_soft_surface_long as periodic
from utils.checkpoint_snapshots import digest
from utils.densification_policy import opacity_reset_due
from test_region_colmap import region,region_source
import test_native_lidar as cpu


def parse_options(command):
    parser=argparse.ArgumentParser()
    mp,op,pp=ModelParams(parser),OptimizationParams(parser),PipelineParams(parser)
    parsed,_=parser.parse_known_args(command[1:])
    return mp.extract(parsed),op.extract(parsed),pp.extract(parsed)


def test_resplit_full_selected_pool_with_actual_sharpness_no_roi_and_replay(region):
    # Original split has 00000..00002 train, 00003 held out. The new winners
    # must include the old held-out final group and must use full RGB sharpness.
    pixels=np.full((64,64,3),80,np.uint8)
    pixels[:16,:16]=((np.indices((16,16)).sum(0)%2)*255)[...,None]
    Image.fromarray(pixels).save(region/'images/00001.png')
    # Invalid ROI files must be irrelevant, not silently used for loss/scoring.
    for path in (region/'masks').glob('*.png'): path.write_bytes(b'not an image')
    before={str(p):digest(p) for p in region.rglob('*') if p.is_file()}
    model=region/'runs/new'
    prepare(['--region',str(region),'--model',str(model),'--gpu','0'])
    data=periodic.load_plan(model)
    manifest=json.loads((model/'keyframes.json').read_text())
    assert manifest['candidate_count']==4 and manifest['group_size']==3
    assert manifest['groups'][0]['images']==['00000.png','00001.png','00002.png']
    assert data['selected']==['00001.png','00003.png']
    assert data['frames']==[f'{i:05d}.png' for i in range(4)]
    assert data['iterations']==[10000,20000,30000]
    plan=data['plan'];command=plan['stages'][0]['argv']
    dataset,opt,pipeline=parse_options(command)
    assert dataset.loss_mask_dir=='' and not dataset.train_test_exp
    assert dataset.resolution==1 and pipeline.antialiasing
    assert not any('/masks/' in p for p in data['input_fingerprints'])
    assert opt.freeze_lidar_positions and opt.position_lr_init==opt.position_lr_final==0
    assert not opt.disable_pruning and not opt.disable_opacity_resets
    assert (opt.feature_lr,opt.opacity_lr,opt.scaling_lr,opt.rotation_lr)==(.0025,.05,.005,.001)
    assert opt.densify_from_iter==500 and opt.densify_until_iter==15000 and opt.densification_interval==100
    assert opt.scale_control=='off' and not plan['hard_culling'] and not plan['densification']['enabled']
    assert plan['opacity_reset_schedule']['steps']==[3000,6000,9000,12000]
    assert [i for i in range(1,opt.densify_until_iter) if opacity_reset_due(opt,i)]==[3000,6000,9000,12000]
    assert data['training_args']['debug_preview_every']==500
    with patch.object(native,'execute_plan') as execute:
        periodic.train(model)
        execute.assert_called_once()
        args,replayed=execute.call_args.args
        assert replayed==plan
    with patch.object(native,'invoke') as smoke,patch.object(native.fixed,'execute_plan'):
        native.execute_plan(args,replayed)
        smoke_command=smoke.call_args.args[0]
        assert '--opacity-resets' in smoke_command and '--region-loss' not in smoke_command
        assert smoke_command[smoke_command.index('--opacity-lr')+1]=='0.05'
    assert all(digest(path)==value for path,value in before.items())
    report=json.loads((model/'region_full_image.json').read_text())
    assert report['train_images']==report['heldout_images']==2
    with pytest.raises(FileExistsError): prepare(['--region',str(region),'--model',str(model)])


def test_native_optimizer_reset_and_prune_preserve_surviving_lidar_xyz():
    _,opt,_=parse_options(['train.py','--freeze_lidar_positions','--scale_control','off',
        '--position_lr_init','0','--position_lr_final','0',
        '--feature_lr','.0025','--opacity_lr','.05','--scaling_lr','.005','--rotation_lr','.001'])
    m=cpu.model(opt);original=m.get_xyz.detach().clone()
    # The shared CUDA-free loader stubs activation helpers; use the real inverse
    # sigmoid here so this exercises the actual opacity reset, not that stub.
    from utils.general_utils import inverse_sigmoid
    m.inverse_opacity_activation=inverse_sigmoid
    rates={g['name']:g['lr'] for g in m.optimizer.param_groups}
    assert rates==dict(xyz=0.,f_dc=.0025,f_rest=.000125,opacity=.05,scaling=.005,rotation=.001)
    (m._opacity.sum()+m._features_dc.sum()+m._scaling.sum()).backward();m.optimizer.step()
    m.reset_opacity()
    assert (m.get_opacity <= .010001).all()
    assert torch.equal(original,m.get_xyz) and not m.get_xyz.requires_grad
    # Native pruning remains active; it can remove rows but cannot move or grow.
    with torch.no_grad(),cpu.cpu_allocations():
        m._opacity[0]=-20
        m._scaling[:]=np.log(.001)
        m.densify_and_prune(.0002,.005,1.,None,torch.ones(2),record_stats=True)
    assert len(m.get_xyz)==1 and torch.equal(m.get_xyz,original[1:])
    assert m.last_densification_stats['cloned_parents']==m.last_densification_stats['split_parents']==0
    assert not m._xyz.requires_grad


@pytest.mark.parametrize('previous',[False,True])
def test_shell_prepares_then_stops_only_named_old_run_and_hands_off(tmp_path,region,previous):
    root=Path(__file__).resolve().parents[1]
    launch=tmp_path/'launch/scripts';launch.mkdir(parents=True)
    shutil.copy(root/'scripts/run_region_full_image.sh',launch/'run_region_full_image.sh')
    (launch/'run_region_full_image.py').symlink_to(root/'scripts/run_region_full_image.py')
    (launch/'stop_reconstruction_run.py').write_text(
        'import os,sys\nfrom pathlib import Path\n'
        'assert sys.argv[1:]==["--model",os.environ["EXPECTED_OLD"],"--gpu","0","--stop"]\n'
        'assert Path(os.environ["EXPECTED_NEW"],"periodic_evaluation/plan.json").exists()\n'
        'Path(os.environ["STOPPED"]).write_text("yes")\n')
    (launch/'run_prepared_reconstruction.sh').write_text(
        'set -eu\ntest -f "$1/periodic_evaluation/plan.json"\nprintf "%s" "$1" > "$HANDOFF"\n')
    conda=tmp_path/'conda.sh';conda.write_text('conda() { test "$1" = activate && test "$2" = 3dgs; }\n')
    bindir=tmp_path/'bin';bindir.mkdir()
    (bindir/'python').write_text('#!'+sys.executable+'\nimport os,sys\nos.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n')
    (bindir/'python').chmod(0o755)
    old=region/'runs/crop_v2';new=region/'runs/stride3_fullimage_fixed_lidar_native_lr_resets_v1'
    if previous:
        (old/'periodic_evaluation').mkdir(parents=True)
        (old/'periodic_evaluation/plan.json').write_text('{}')
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
        'RECONSTRUCTION_CONDA_SH':str(conda),'HANDOFF':str(tmp_path/'handoff'),
        'STOPPED':str(tmp_path/'stopped'),'EXPECTED_OLD':str(old),'EXPECTED_NEW':str(new)}
    result=subprocess.run(['bash',str(launch/'run_region_full_image.sh'),'0',str(region)],
        env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'handoff').read_text()==str(new)
    assert (tmp_path/'stopped').exists()==previous
    if previous: assert (old/'periodic_evaluation/plan.json').read_text()=='{}'
