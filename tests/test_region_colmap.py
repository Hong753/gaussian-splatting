"""COLMAP ablation fidelity and shell orchestration (SfM/CUDA stubbed locally)."""
import json
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts import region_colmap as rerun
from scripts.keyframe_pipeline.common import read_model
from scripts.run_region_ablation import main as prepare_training
from scripts.compare_region import prepare_specs
from scripts.view_comparisons import read_colmap_records
from test_region_ablation import region_source, exported
from test_view_comparisons import checkpoint


def write_model(folder,cameras,images):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'cameras.txt').write_text(''.join(
        f'{cam.id} PINHOLE {cam.width} {cam.height} '+' '.join(map(str,cam.params))+'\n'
        for cam in cameras.values()))
    rows=[]
    for im in images:
        q=Rotation.from_matrix(im.R).as_quat()[[3,0,1,2]];t=-im.R@im.C
        rows.append(f'{im.id} '+' '.join(map(str,[*q,*t]))+f' {im.camera_id} {im.name}\n\n')
    (folder/'images.txt').write_text(''.join(rows))
    (folder/'points3D.txt').write_text('')


@pytest.fixture
def region(tmp_path,region_source):
    source,_,manifest=region_source
    folder,cameras,images,_=read_model(source)
    images[2].C[1]=.08
    write_model(folder,cameras,images)
    manifest.write_text(json.dumps(dict(selected_images=[im.name for im in images[:3]])))
    return exported(tmp_path,region_source)[1]


def changed_model(region,target):
    _,cameras,images,_=read_model(region)
    chosen=set(json.loads((region/'keyframes.json').read_text())['selected_images'])
    images=[im for im in images if im.name in chosen]
    for im in images:
        im.id+=100  # No assumption that COLMAP preserves image IDs.
        im.C[2]+=.001
        im.R=Rotation.from_euler('z',.1,degrees=True).as_matrix()@im.R
    write_model(target,cameras,images)
    return images


def test_prepare_excludes_heldout_and_rejects_stale_or_degenerate_inputs(tmp_path,region):
    job=tmp_path/'colmap';identity=rerun.prepare(region,job)
    assert len(identity['selected'])==3
    assert sorted(p.name for p in (job/'images').iterdir())==identity['selected']
    assert not (job/'images/00003.png').exists()
    assert [line.split()[0] for line in (job/'reference_centers.txt').read_text().splitlines()]==identity['selected']
    assert rerun.prepare(region,job)==identity
    mask=next((region/'masks').rglob('*.png'))
    mask.write_bytes(mask.read_bytes()+b'x')
    with pytest.raises(ValueError,match='Input changed'):
        rerun.load_job(job)
    folder,cameras,images,_=read_model(region)
    for im in images: im.C[1:]=0
    write_model(folder,cameras,images)
    with pytest.raises(ValueError,match='collinear'):
        rerun.prepare(region,tmp_path/'bad')


def test_seed_existing_intrinsics_without_editing_image_database(tmp_path,region):
    job=tmp_path/'job';rerun.prepare(region,job)
    with sqlite3.connect(job/'database.db') as db:
        db.execute('CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY,model INTEGER,width INTEGER,height INTEGER,params BLOB,prior_focal_length INTEGER)')
        db.execute('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT)')
    rerun.seed_cameras(job);rerun.seed_cameras(job)
    with sqlite3.connect(job/'database.db') as db:
        row=db.execute('SELECT * FROM cameras').fetchone()
        assert row[:4]==(1,1,64,64)
        np.testing.assert_array_equal(np.frombuffer(row[4],dtype='<f8'),[64,64,32,32])
        assert not db.execute('SELECT * FROM images').fetchall()
        db.execute('UPDATE cameras SET width=65')
    with pytest.raises(ValueError,match='calibration changed'):
        rerun.seed_cameras(job)


def test_registration_requires_same_training_set_in_one_component(tmp_path,region):
    job=tmp_path/'job';rerun.prepare(region,job)
    root=job/'sparse';images=changed_model(region,root/'0')
    _,cameras,_,_=read_model(root/'0',model_dir=root/'0')
    write_model(root/'0',cameras,images[:2])
    with pytest.raises(ValueError,match='registered 2/3'):
        rerun.choose_model(job,root)
    report=json.loads((job/'registration_report.json').read_text())
    assert report['missing']==['00002.png']
    changed_model(region,root/'1');rerun.choose_model(job,root)
    assert (job/'selected_model.txt').read_text().strip()==str(root/'1')
    _,_,original,_=read_model(region)
    write_model(root/'2',cameras,original)
    with pytest.raises(ValueError,match='non-training'):
        rerun.choose_model(job,root)


def test_export_changes_only_train_poses_and_preserves_lidar_masks_membership(tmp_path,region):
    originals={str(p):rerun.digest(p) for p in region.rglob('*') if p.is_file()}
    job=tmp_path/'job';rerun.prepare(region,job)
    changed_model(region,job/'aligned');out=rerun.export(job)
    assert rerun.export(job)==out
    assert (out/'input.ply').resolve()==region/'input.ply'
    assert (out/'sparse/0/points3D.ply').resolve()==region/'input.ply'
    assert (out/'masks').resolve()==region/'masks'
    from scripts.pose_options import dataset_pose_labels, protocol
    labels=dataset_pose_labels(out)
    assert set(labels)=={'00000.png','00001.png','00002.png'}
    assert not dataset_pose_labels(region)
    assert 'rerun COLMAP' in protocol(dict(cfg=dict(source_path=str(out))))
    old,new=read_colmap_records(region),read_colmap_records(out)
    assert old.keys()==new.keys()
    for name,r in new.items():
        for key in ('fx','fy','cx','cy','width','height'):
            np.testing.assert_equal(r[key],old[name][key])
        assert Path(r['image_path']).resolve()==Path(old[name]['image_path']).resolve()
        if name=='00003.png':
            np.testing.assert_allclose(r['R'],old[name]['R'],atol=1e-12)
            np.testing.assert_allclose(r['T'],old[name]['T'],atol=1e-12)
        else:
            assert not np.allclose(r['R'],old[name]['R'])
            assert not np.allclose(r['T'],old[name]['T'])
    assert originals=={path:rerun.digest(path) for path in originals}
    prepare_training(['--region',str(out),'--gpu','1','--iterations','30','--reference-steps','30'])
    saved=json.loads((out/'runs/crop_v1/periodic_evaluation/plan.json').read_text())
    assert saved['gpu']=='1' and saved['iterations']==[30]
    assert saved['selected']==['00000.png','00001.png','00002.png']
    assert len(saved['frames'])==4
    assert not saved['plan']['pruning'] and saved['plan']['freeze_lidar_positions']
    assert saved['plan']['scale_control']=='off' and not saved['plan']['hard_culling']
    command=saved['plan']['stages'][0]['argv']
    assert command[command.index('--debug_preview_every')+1]=='500'
    assert str(out/'sparse/0/images.txt') in saved['input_fingerprints']
    assert str(out/'colmap_rerun.json') in saved['input_fingerprints']
    (job/'aligned/images.txt').write_text((job/'aligned/images.txt').read_text()+'# edited\n')
    with pytest.raises(ValueError,match='changed after dataset export'):
        rerun.export(job)


@pytest.mark.parametrize('bad',['intrinsics','alignment','heldout'])
def test_reject_invalid_result_before_export(tmp_path,region,bad):
    job=tmp_path/'job';rerun.prepare(region,job);changed_model(region,job/'aligned')
    folder,cameras,images,_=read_model(job/'aligned',model_dir=job/'aligned')
    if bad=='intrinsics': cameras[1].params[0]+=1
    elif bad=='alignment':
        for im in images: im.C[2]+=100
    else:
        images.append(read_model(region)[2][-1])
    write_model(folder,cameras,images)
    with pytest.raises(ValueError): rerun.export(job)
    assert not (job/'dataset').exists()


def test_comparison_requires_explicit_pose_difference_and_uses_own_poses(tmp_path,region):
    job=tmp_path/'job';rerun.prepare(region,job);changed_model(region,job/'aligned');out=rerun.export(job)
    models=[]
    for i,dataset in enumerate((region,out)):
        model=tmp_path/f'model_{i}';model.mkdir();checkpoint(model,30000)
        (model/'cfg_args').write_text(f'Namespace(source_path={str(dataset)!r},eval=False,antialiasing=True)')
        models.append(model)
    with pytest.raises(ValueError,match='different R'): prepare_specs(region,models,[30000]*2)
    specs,_=prepare_specs(region,models,[30000]*2,True)
    assert not np.allclose(specs[0]['records']['00000.png']['R'],specs[1]['records']['00000.png']['R'])
    np.testing.assert_allclose(specs[0]['records']['00003.png']['R'],specs[1]['records']['00003.png']['R'])


@pytest.mark.parametrize('modern,legacy_alignment',[(False,True),(False,False),(True,False)])
def test_shell_gpu1_sequence_cache_and_training_handoff(tmp_path,region,modern,legacy_alignment):
    """Exercise the real launcher/helper/plan; replace COLMAP and the CUDA host."""
    repo=Path(__file__).resolve().parents[1];launch=tmp_path/'launch/scripts';launch.mkdir(parents=True)
    shutil.copy(repo/'scripts/run_region_colmap.sh',launch/'run_region_colmap.sh')
    for name in ('region_colmap.py','run_region_ablation.py'):
        (launch/name).symlink_to(repo/'scripts'/name)
    (launch/'run_prepared_reconstruction.sh').write_text('set -eu\ntest -f "$1/periodic_evaluation/plan.json"\nprintf "%s" "$1" > "$LAUNCH_MARKER"\n')
    conda=tmp_path/'conda.sh';conda.write_text('conda() { test "$1" = activate && test "$2" = 3dgs; }\n')
    bindir=tmp_path/'bin';bindir.mkdir()
    for name,script in [('python','#!/bin/sh\nexec '+shlex.quote(sys.executable)+' "$@"\n'),('nvidia-smi','#!/bin/sh\nexit 0\n')]:
        (bindir/name).write_text(script);(bindir/name).chmod(0o755)
    fixture=tmp_path/'aligned';changed_model(region,fixture)
    fake=bindir/'colmap'
    fake.write_text('#!'+sys.executable+'\n'+'''import json,os,sys,sqlite3,shutil
from pathlib import Path
args=sys.argv[1:];command=args[0]
def opt(name): return args[args.index('--'+name)+1]
legacy=os.environ['LEGACY_ALIGNMENT']=='1'
alignment_options=['robust_alignment','robust_alignment_max_error'] if legacy else ['alignment_max_error']
if os.environ.get('MISSING_ALIGNMENT')=='1': alignment_options=[]
if '-h' in args:
    prefix='Feature' if os.environ['MODERN']=='1' else 'Sift'
    groups={'feature_extractor':[prefix+'Extraction.'+v for v in ['use_gpu','gpu_index','max_image_size']]+['image_list_path','ImageReader.existing_camera_id'],
    'exhaustive_matcher':[prefix+'Matching.'+v for v in ['use_gpu','gpu_index']],
    'mapper':['Mapper.ba_refine_'+v for v in ['focal_length','principal_point','extra_params']],
    'bundle_adjuster':['BundleAdjustment.refine_'+v for v in ['focal_length','principal_point','extra_params']],
    'model_aligner':['ref_images_path','ref_is_gps','alignment_type','transform_path']+alignment_options}
    print(' '.join('--'+v for v in groups[command]));sys.exit(0)
assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
with open(os.environ['CALLS'],'a') as f: f.write(json.dumps(args)+'\\n')
if command=='database_creator':
    with sqlite3.connect(opt('database_path')) as db: db.execute('CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY,model INTEGER,width INTEGER,height INTEGER,params BLOB,prior_focal_length INTEGER)')
elif command=='feature_extractor':
    names=Path(opt('image_list_path')).read_text().splitlines();assert names==['00000.png','00001.png','00002.png']
    assert sorted(p.name for p in Path(opt('image_path')).iterdir())==names
elif command in ['mapper','bundle_adjuster','model_aligner']:
    output=Path(opt('output_path'));output=output/'0' if command=='mapper' else output
    shutil.copytree(os.environ['FIXTURE'],output,dirs_exist_ok=True)
    if command=='model_aligner':
        tolerance='robust_alignment_max_error' if legacy else 'alignment_max_error'
        assert float(opt(tolerance))>0
        assert ('--alignment_max_error' not in args) if legacy else ('--robust_alignment_max_error' not in args and '--robust_alignment' not in args)
        if legacy: assert opt('robust_alignment')=='1'
        Path(opt('transform_path')).write_text('test similarity')
''')
    fake.chmod(0o755)
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'RECONSTRUCTION_CONDA_SH':str(conda),
         'LAUNCH_MARKER':str(tmp_path/'handoff'),'CALLS':str(tmp_path/'calls'),'FIXTURE':str(fixture),'MODERN':str(int(modern)),
         'COLMAP_BIN':str(fake),'LEGACY_ALIGNMENT':str(int(legacy_alignment))}
    result=subprocess.run(['bash',str(launch/'run_region_colmap.sh'),'1',str(region)],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    model=region/'runs/crop_colmap_fixed_v1';assert (tmp_path/'handoff').read_text()==str(model)
    calls=[json.loads(line) for line in (tmp_path/'calls').read_text().splitlines()]
    assert [a[0] for a in calls]==['database_creator','feature_extractor','exhaustive_matcher','mapper','bundle_adjuster','model_aligner']
    assert '--input_path' not in calls[3]  # Fresh SfM, not BA initialized from old poses.
    assert calls[-1][calls[-1].index('--ref_is_gps')+1]=='0'
    tolerance_option='--robust_alignment_max_error' if legacy_alignment else '--alignment_max_error'
    assert float(calls[-1][calls[-1].index(tolerance_option)+1])==pytest.approx(json.loads((region/'colmap_train_v1/job.json').read_text())['alignment_max_error'])
    plan=json.loads((model/'periodic_evaluation/plan.json').read_text());assert plan['gpu']=='1'
    assert plan['training_args']['debug_preview_every']==500
    # Remove only the synthetic, unstarted training plan: cached COLMAP is reused.
    shutil.rmtree(model)
    again=subprocess.run(['bash',str(launch/'run_region_colmap.sh'),'1',str(region)],env=env,capture_output=True,text=True,timeout=60)
    assert again.returncode==0,again.stdout+again.stderr
    assert len((tmp_path/'calls').read_text().splitlines())==6
    missing=subprocess.run(['bash',str(launch/'run_region_colmap.sh'),'1',str(region),'missing'],
                           env={**env,'MISSING_ALIGNMENT':'1'},capture_output=True,text=True,timeout=15)
    assert missing.returncode!=0 and 'Stopped before reconstruction' in missing.stderr
    assert not (region/'colmap_train_missing/database.db').exists()
    assert not (region/'runs/crop_colmap_fixed_missing').exists()
    assert len((tmp_path/'calls').read_text().splitlines())==6
    (bindir/'nvidia-smi').write_text('#!/bin/sh\nexit 2\n')
    failed=subprocess.run(['bash',str(launch/'run_region_colmap.sh'),'1',str(region),'v2'],env=env,capture_output=True,text=True,timeout=15)
    assert failed.returncode==2 and not (region/'colmap_train_v2').exists()
