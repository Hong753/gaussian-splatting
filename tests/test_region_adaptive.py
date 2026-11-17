"""Original and rerun-pose geometry controls share initialization and settings."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import region_colmap
from test_region_colmap import region, region_source, changed_model


@pytest.mark.parametrize('poses',['original','colmap'])
def test_adaptive_launcher_uses_matching_dataset_and_shared_lidar(tmp_path,region,poses):
    root=Path(__file__).resolve().parents[1]
    launch=tmp_path/'launch/scripts';launch.mkdir(parents=True)
    shutil.copy(root/'scripts/run_region_adaptive.sh',launch/'run_region_adaptive.sh')
    for name in ('region_colmap.py','run_region_ablation.py'):
        (launch/name).symlink_to(root/'scripts'/name)
    (launch/'run_prepared_reconstruction.sh').write_text(
        'set -eu\ntest -f "$1/periodic_evaluation/plan.json"\nprintf "%s" "$1" > "$HANDOFF"\n')
    conda=tmp_path/'conda.sh';conda.write_text('conda() { test "$1" = activate && test "$2" = 3dgs; }\n')
    bindir=tmp_path/'bin';bindir.mkdir()
    (bindir/'python').write_text('#!'+sys.executable+'\nimport os,sys\nos.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n')
    (bindir/'python').chmod(0o755)
    (bindir/'nvidia-smi').write_text('#!/bin/sh\nexit 0\n');(bindir/'nvidia-smi').chmod(0o755)
    source=region
    if poses=='colmap':
        workspace=region/'colmap_superpoint_superglue_seq20_v1'
        region_colmap.prepare(region,workspace)
        changed_model(region,workspace/'aligned')
        source=region_colmap.export(workspace)
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
         'RECONSTRUCTION_CONDA_SH':str(conda),'HANDOFF':str(tmp_path/'handoff')}
    result=subprocess.run(['bash',str(launch/'run_region_adaptive.sh'),'2' if poses=='original' else '3',
                          str(region),poses],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    model=Path((tmp_path/'handoff').read_text())
    saved=json.loads((model/'periodic_evaluation/plan.json').read_text())
    plan=saved['plan'];args=saved['training_args']
    assert saved['source']==str(source) and Path(args['point_cloud']).resolve()==region/'input.ply'
    # This tiny fixture retains all original training images, so matching the
    # reference exposure extends 90k to 100k (the real 1349/1670 export uses 90k).
    assert args['iterations']==100000 and args['debug_preview_every']==500
    assert args['checkpoint_every']==10000 and plan['pruning']
    assert not plan.get('freeze_lidar_positions') and plan['position_lr']['divisor']==20
    assert plan['densification']['start']==500 and plan['densification']['until']==15000
    assert plan['scale_control']=='off' and not plan['hard_culling']
    command=plan['stages'][0]['argv']
    assert '--disable_opacity_resets' in command and '--disable_pruning' not in command
    assert len(saved['selected'])==3 and len(saved['frames'])==4
    # All original/held-out pose files and the original LiDAR remain source data.
    assert source==region or (source/'input.ply').samefile(region/'input.ply')
