"""Automatic export, immutable cache reuse, and adaptive training launch."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.select_region import main as select
from scripts.run_region_ablation import prepare_args, main as prepare
from scripts.run_native_lidar import build_plan
from test_region_ablation import region_source


def automatic_args(source, output):
    scene, cloud, keyframes=source
    return ['--source',str(scene),'--point-cloud',str(cloud),'--keyframes',str(keyframes),
            '--output',str(output),'--center','0','0','2','--size','1.4','1.4','.4',
            '--halo','.2','--max-camera-distance','5','--sample-points','100000',
            '--automatic','--export','--reuse']


def test_automatic_export_ignores_review_and_reuses_only_matching_export(tmp_path,region_source):
    output=tmp_path/'auto'; argv=automatic_args(region_source,output)
    # Explicit automatic mode does not apply old manual choices or try to parse them.
    output.with_name('auto.review.json').write_text('unrelated manual session')
    assert select(argv)==0
    meta=json.loads((output/'selection.json').read_text())
    assert meta['selection_mode']=='automatic' and 'manual_review' not in meta
    assert len(meta['selection'])==4
    assert [r['name'] for r in meta['selection'] if r['split']=='train']==['00000.png','00002.png']
    assert not (output/'review.json').exists()
    stamp=(output/'input.ply').stat().st_mtime_ns
    assert select(argv)==0
    assert (output/'input.ply').stat().st_mtime_ns==stamp
    with pytest.raises(ValueError,match='differs'):
        select([*argv,'--halo','.3'])
    # Even changing the original train membership invalidates a cached export.
    manifest=json.loads(region_source[2].read_text());manifest['selected_images'].append('00001.png')
    region_source[2].write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='differs'):
        select(argv)


def test_manual_export_still_requires_review_and_cannot_reuse_auto(tmp_path,region_source):
    argv=automatic_args(region_source,tmp_path/'auto')
    assert select(argv)==0
    meta_path=tmp_path/'auto/selection.json'
    meta=json.loads(meta_path.read_text());meta['selection_mode']='manual'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='selection_mode'):
        select(argv)
    manual=automatic_args(region_source,tmp_path/'manual')
    manual.remove('--automatic');manual.remove('--reuse')
    manual+=['--anchor','00000.png']
    with pytest.raises(ValueError,match='No saved manual review'):
        select(manual)


def test_adaptive_plan_enables_lr_growth_pruning_previews_and_reproduces(tmp_path,region_source):
    region=tmp_path/'auto'; select(automatic_args(region_source,region))
    argv=['--region',str(region),'--geometry','adaptive','--position-lr-multiplier','.05']
    args, protocol=prepare_args(argv)
    plan=build_plan(args); cmd=plan['stages'][0]['argv']
    assert float(cmd[cmd.index('--position_lr_init')+1])==pytest.approx(8e-6)
    assert float(cmd[cmd.index('--position_lr_final')+1])==pytest.approx(8e-8)
    assert plan['position_lr']['divisor']==20
    assert plan['densification']==dict(start=500,until=15000,interval=100,gradient_threshold=.0002,percent_dense=.01)
    assert plan['pruning'] and not plan['hard_culling'] and plan['scale_control']=='off'
    for disabled in ('--freeze_lidar_positions','--fixed_lidar','--disable_pruning','--max_scale_world'):
        assert disabled not in cmd
    assert '--disable_opacity_resets' in cmd and '--antialiasing' in cmd
    assert cmd[cmd.index('--debug_preview_every')+1]=='500'
    assert protocol['geometry_mode']=='adaptive' and protocol['position_lr_multiplier']==.05
    prepare(argv)
    saved=json.loads((args.model/'periodic_evaluation/plan.json').read_text())
    values=saved['training_args']
    for key in ('source','model','keyframes','point_cloud','loss_mask_dir'):
        values[key]=Path(values[key])
    assert build_plan(SimpleNamespace(**values))==saved['plan']
    assert len(saved['frames'])==4 and len(saved['selected'])==2


@pytest.mark.parametrize('value',['0','nan','1.1'])
def test_invalid_lr_fails_before_export_lookup(tmp_path,value):
    with pytest.raises(ValueError,match='Position LR multiplier'):
        prepare_args(['--region',str(tmp_path/'missing'),'--position-lr-multiplier',value])


def test_missing_region_explains_preparation(tmp_path):
    with pytest.raises(FileNotFoundError,match='run_region_auto.sh'):
        prepare_args(['--region',str(tmp_path/'missing')])


def test_one_command_exports_missing_region_before_training_handoff(tmp_path,region_source):
    """Run the actual shell, selection and preparation; stub only conda/GPU/training."""
    scene, cloud, keyframes=region_source
    (scene/'images/00000.png').rename(scene/'images/06663.jpg')
    poses=scene/'sparse/0/images.txt'
    poses.write_text(poses.read_text().replace('00000.png','06663.jpg'))
    keyframes.write_text(keyframes.read_text().replace('00000.png','06663.jpg'))
    repo=Path(__file__).resolve().parents[1]
    launch=tmp_path/'launch/scripts';launch.mkdir(parents=True)
    shutil.copy(repo/'scripts/run_region_auto.sh',launch/'run_region_auto.sh')
    for name in ('select_region.py','run_region_ablation.py'):
        (launch/name).symlink_to(repo/'scripts'/name)
    # The handoff marker is written only after a real, validated plan exists.
    (launch/'run_prepared_reconstruction.sh').write_text(
        'set -eu\ntest -f "$1/periodic_evaluation/plan.json"\nprintf "%s" "$1" > "$LAUNCH_MARKER"\n')
    conda=tmp_path/'conda.sh';conda.write_text('conda() { test "$1" = activate && test "$2" = 3dgs; }\n')
    bindir=tmp_path/'bin';bindir.mkdir()
    for name,script in [('python','#!/bin/sh\nexec '+shlex.quote(sys.executable)+' "$@"\n'),
                        ('nvidia-smi','#!/bin/sh\nexit 0\n')]:
        (bindir/name).write_text(script);(bindir/name).chmod(0o755)
    region=tmp_path/'outputs/3dgs_finetuning/regions/wall_06663_auto_v1'
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
         'RECONSTRUCTION_CONDA_SH':str(conda),'REGION_SOURCE':str(scene),
         'REGION_BASELINE':str(cloud.parent),'LAUNCH_MARKER':str(tmp_path/'handoff')}
    result=subprocess.run(['bash',str(launch/'run_region_auto.sh'),'0',str(region)],env=env,
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    model=region/'runs/crop_adaptive_lr05_v1'
    assert (tmp_path/'handoff').read_text()==str(model)
    saved=json.loads((model/'periodic_evaluation/plan.json').read_text())
    assert saved['gpu']=='0' and saved['plan']['position_lr']['divisor']==20
    assert saved['plan']['pruning'] and len(saved['frames'])==4
    assert json.loads((region/'selection.json').read_text())['selection_mode']=='automatic'
    # A failed GPU query must stop before creating another plan.
    (bindir/'nvidia-smi').write_text('#!/bin/sh\nexit 2\n')
    (tmp_path/'handoff').unlink()
    failed=subprocess.run(['bash',str(launch/'run_region_auto.sh'),'0',str(region),'v2'],
                          env=env,capture_output=True,text=True,timeout=15)
    assert failed.returncode==2 and not (tmp_path/'handoff').exists()
    assert not (region/'runs/crop_adaptive_lr05_v2').exists()
