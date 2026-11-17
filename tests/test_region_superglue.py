"""Learned coordinates, temporal pairing, incremental caches and headless import."""
import json
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import region_superglue as sg
from scripts.region_colmap import prepare as prepare_job
from test_region_colmap import region, region_source, changed_model


class FakeSP:
    def __init__(self,config): pass
    def eval(self): return self
    def to(self,device): return self
    def __call__(self,data):
        xy=torch.tensor([[x,y] for y in (8.,16.,24.,32.) for x in (8.,16.,24.,32.)])
        return dict(keypoints=[xy],scores=[torch.ones(16)],descriptors=[torch.ones(256,16)/16])


class FakeSG(FakeSP):
    calls=0
    def __call__(self,data):
        type(self).calls+=1
        assert data['image0'].shape[-2:]==(64,64)
        assert data['descriptors0'].shape==(1,256,16)
        return dict(matches0=torch.arange(16)[None],matching_scores0=torch.ones(1,16))


def model_cache(path):
    for name in sg.MODEL_FILES:
        file=path/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('pinned model test fixture')
    return path


@pytest.fixture
def workspace(tmp_path,region):
    job=tmp_path/'job';prepare_job(region,job,frontend='superglue',overlap=1)
    sg.prepare(job,model_cache(tmp_path/'network'))
    return job


def test_pairs_use_selected_frame_order_not_all_images_or_lexical_numbers():
    assert sg.sequential_pairs(['frame10.png','frame2.png','frame1.png'],1)==[
        ('frame1.png','frame2.png'),('frame2.png','frame10.png')]
    names=[f'{i:05d}.jpg' for i in range(1349)]
    pairs=sg.sequential_pairs(names,20)
    assert len(pairs)==26770 and len(set(pairs))==len(pairs)
    assert ('00000.jpg','00021.jpg') not in pairs
    with pytest.raises(ValueError): sg.sequential_pairs(['a','a'],2)


def test_resize_converts_superpoint_centers_to_colmap_half_pixels(tmp_path):
    path=tmp_path/'image.png';Image.new('RGB',(128,96)).save(path)
    data=sg.extract_one(FakeSP({}),path,'cpu',64)
    np.testing.assert_array_equal(data['image_size'],[64,48])
    np.testing.assert_allclose(data['keypoints_colmap'],(data['keypoints']+.5)*2)
    # Full-size inference still needs the half-pixel COLMAP convention.
    full=sg.extract_one(FakeSP({}),path,'cpu',128)
    np.testing.assert_allclose(full['keypoints_colmap'],full['keypoints']+.5)


def test_feature_and_pair_caches_resume_and_settings_cannot_mix(workspace,monkeypatch):
    config=sg.load_config(workspace)
    monkeypatch.setattr(sg,'model_class',lambda repo,name:FakeSP if name=='superpoint' else FakeSG)
    sg.extract(workspace,config,'cpu');sg.match(workspace,config,'cpu')
    count=FakeSG.calls; assert count>=2
    raw=(workspace/'superglue_raw_matches.txt').read_text()
    assert raw.startswith('00000.png 00001.png\n0 0\n')
    assert '00001.png 00002.png\n' in raw and '00003.png' not in raw
    sg.extract(workspace,config,'cpu');sg.match(workspace,config,'cpu')
    assert FakeSG.calls==count and (workspace/'superglue_raw_matches.txt').read_text()==raw
    with pytest.raises(ValueError,match='settings changed'):
        sg.prepare(workspace,Path(config['repository']),max_keypoints=8192)
    model=Path(config['repository'])/'models/superpoint.py';model.write_text('changed')
    with pytest.raises(ValueError,match='model cache changed'): sg.load_config(workspace)


def test_database_keypoints_preserve_ids_and_verified_graph(workspace,monkeypatch):
    config=sg.load_config(workspace)
    monkeypatch.setattr(sg,'model_class',lambda repo,name:FakeSP)
    sg.extract(workspace,config,'cpu')
    with sqlite3.connect(workspace/'database.db') as db:
        db.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT,camera_id INTEGER,frame_id INTEGER); '
            'CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB); '
            'CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB); '
            'CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER);')
        for iid,name in enumerate(config['names'],start=10):
            db.execute('INSERT INTO images VALUES(?,?,?,?)',(iid,name,1,100+iid))
            db.execute('INSERT INTO keypoints VALUES(?,?,?,?)',(iid,0,4,b''))
            db.execute('INSERT INTO descriptors VALUES(?,?,?,?)',(iid,0,128,b''))
    sg.import_keypoints(workspace,config);sg.import_keypoints(workspace,config)
    with sqlite3.connect(workspace/'database.db') as db:
        for iid,name,frame_id in db.execute('SELECT image_id,name,frame_id FROM images'):
            rows,cols,blob=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(iid,)).fetchone()
            assert (rows,cols)==(16,2) and frame_id==100+iid
            np.testing.assert_array_equal(np.frombuffer(blob,dtype='<f4').reshape(rows,cols),sg.read_feature(workspace,name)['keypoints_colmap'])
        assert db.execute('SELECT COUNT(*) FROM descriptors').fetchone()[0]==0
        db.execute('INSERT INTO two_view_geometries VALUES(?,?)',(10*2147483647+11,16))
    with pytest.raises(ValueError,match='disconnected'): sg.check_verified(workspace,config)
    with sqlite3.connect(workspace/'database.db') as db:
        db.execute('INSERT INTO two_view_geometries VALUES(?,?)',(11*2147483647+12,16))
    sg.check_verified(workspace,config)
    with sqlite3.connect(workspace/'database.db') as db:
        db.execute('INSERT INTO two_view_geometries VALUES(?,?)',(10*2147483647+12,16))
    with pytest.raises(ValueError,match='outside the sequential/bridge manifest'): sg.check_verified(workspace,config)


def test_bridge_recovery_preserves_caches_and_verifies_only_new_pairs(workspace,monkeypatch):
    config=sg.load_config(workspace)
    monkeypatch.setattr(sg,'model_class',lambda repo,name:FakeSP if name=='superpoint' else FakeSG)
    sg.extract(workspace,config,'cpu');sg.match(workspace,config,'cpu')
    files=list((workspace/'superpoint').glob('*.npz'))+list((workspace/'superglue_matches').glob('*.npz'))
    stamps={p:(p.stat().st_mtime_ns,sg.digest(p)) for p in files}
    names=config['names']
    with sqlite3.connect(workspace/'database.db') as db:
        db.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT); '
                        'CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER);')
        db.executemany('INSERT INTO images VALUES(?,?)',enumerate(names,start=1))
        # The rejected second temporal link is a completed attempt, not pending.
        db.executemany('INSERT INTO two_view_geometries VALUES(?,?)',[(2147483647+2,16),(2*2147483647+3,0)])
    report=sg.check_verified(workspace,config,require_connected=False)
    assert report['component_sizes']==[2,1] and report['pending_pairs']==0
    calls=FakeSG.calls
    sg.match(workspace,config,'cpu',pending_only=True)
    assert FakeSG.calls==calls and (workspace/'superglue_raw_matches.txt').read_text()==''
    pairs=sg.plan_bridges(workspace,config,budget=10)
    assert [p['images'] for p in pairs]==[[names[0],names[2]]]
    assert sg.pending_pairs(workspace,config)==[(names[0],names[2])]
    # Resuming prepare must preserve appended bridge pairs and feature settings.
    assert sg.prepare(workspace,Path(config['repository']))==config
    assert len((workspace/'pairs.txt').read_text().splitlines())==3
    sg.match(workspace,config,'cpu',pending_only=True)
    assert FakeSG.calls==calls+1
    raw=(workspace/'superglue_raw_matches.txt').read_text()
    assert raw.startswith(f'{names[0]} {names[2]}\n') and f'{names[1]} {names[2]}\n' not in raw
    # Crash between matching and verification: cached bridge is reused.
    sg.match(workspace,config,'cpu',pending_only=True)
    assert FakeSG.calls==calls+1
    assert not sg.check_verified(workspace,config,require_connected=False)['connected']
    with pytest.raises(ValueError,match='pending'):
        sg.plan_bridges(workspace,config,budget=10)
    with sqlite3.connect(workspace/'database.db') as db:
        db.execute('INSERT INTO two_view_geometries VALUES(?,?)',(2147483647+3,16))
    assert sg.check_verified(workspace,config)['connected']
    assert sg.plan_bridges(workspace,config)==[]
    for path,stamp in stamps.items():
        assert (path.stat().st_mtime_ns,sg.digest(path))==stamp
    # Held-out files cannot enter the bridge manifest or the graph.
    ledger=json.loads((workspace/'bridge_pairs.json').read_text())
    ledger['rounds'][0]['pairs'][0]['images'][1]='00003.png'
    (workspace/'bridge_pairs.json').write_text(json.dumps(ledger))
    with pytest.raises(ValueError,match='non-training'): sg.all_pairs(workspace,config)


def test_bridge_candidates_are_balanced_deterministic_and_cross_component_only():
    names=[f'{i:05d}.jpg' for i in range(14)]
    groups=[names[i:i+2] for i in range(0,14,2)]
    tried=set(sg.sequential_pairs(names,2))
    descriptors=np.random.default_rng(3).normal(size=(len(names),16))
    descriptors/=np.linalg.norm(descriptors,axis=1,keepdims=True)
    result=sg.choose_bridge_pairs(names,groups,tried,descriptors,28)
    assert result==sg.choose_bridge_pairs(names,groups,tried,descriptors,28)
    assert len(result)==28 and len({tuple(p['images']) for p in result})==28
    membership={n:i for i,g in enumerate(groups) for n in g}
    touched=set()
    for p in result:
        a,b=p['images']
        assert a<b and (a,b) not in tried and membership[a]!=membership[b]
        touched.update([membership[a],membership[b]])
    assert touched==set(range(7))
    assert {p['method'] for p in result}=={'temporal','appearance'}
    assert sg.choose_bridge_pairs(names,groups,set(sg.sequential_pairs(names,99)),descriptors,28)==[]


@pytest.mark.parametrize('modern,recover',[(False,False),(True,False),(False,True)])
def test_superglue_launcher_never_invokes_gpu_sift_and_hands_off_gpu1(tmp_path,region,modern,recover):
    repo=Path(__file__).resolve().parents[1];launch=tmp_path/'launch/scripts';launch.mkdir(parents=True)
    for name in ('run_region_superglue.sh','run_region_colmap.sh'):
        shutil.copy(repo/'scripts'/name,launch/name)
    for name in ('region_colmap.py','run_region_ablation.py'):
        (launch/name).symlink_to(repo/'scripts'/name)
    (launch/'region_superglue.py').write_text(f'''import sys
sys.path[:0]=[{str(repo)!r},{str(repo/'tests')!r}]
from scripts import region_superglue as sg
from test_region_superglue import FakeSP,FakeSG
sg.model_class=lambda repo,name:FakeSP if name=='superpoint' else FakeSG
args=sys.argv[1:]
if args[0] in ('extract','match'): args+=['--device','cpu']
raise SystemExit(sg.main(args))
''')
    (launch/'setup_superglue.sh').write_text('set -eu\ntest -d "$1"\n')
    (launch/'run_prepared_reconstruction.sh').write_text('set -eu\ntest -f "$1/periodic_evaluation/plan.json"\nprintf "%s" "$1" > "$LAUNCH_MARKER"\n')
    conda=tmp_path/'conda.sh';conda.write_text('conda() { test "$1" = activate && test "$2" = 3dgs; }\n')
    bindir=tmp_path/'bin';bindir.mkdir()
    (bindir/'python').write_text('#!'+sys.executable+'\nimport os,sys\n'
        'if len(sys.argv)>2 and sys.argv[1]=="-c" and "SuperPoint/SuperGlue needs CUDA" in sys.argv[2]:\n'
        ' assert os.environ["CUDA_VISIBLE_DEVICES"]=="1";sys.exit(0)\n'
        'os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n')
    (bindir/'nvidia-smi').write_text('#!/bin/sh\nexit 0\n')
    fixture=tmp_path/'aligned';changed_model(region,fixture)
    (bindir/'colmap').write_text('#!'+sys.executable+'\n'+'''import os,sys,json,sqlite3,shutil
from pathlib import Path
args=sys.argv[1:];command=args[0]
def opt(name): return args[args.index('--'+name)+1]
prefix='Feature' if os.environ['MODERN']=='1' else 'Sift'
assert command not in ('feature_extractor','exhaustive_matcher','sequential_matcher')
if '-h' in args:
    groups={'feature_importer':['ImageReader.existing_camera_id','image_list_path','import_path'],
    'matches_importer':[prefix+'Matching.use_gpu','match_type','match_list_path'],
    'mapper':['Mapper.ba_refine_'+v for v in ['focal_length','principal_point','extra_params']],
    'bundle_adjuster':['BundleAdjustment.refine_'+v for v in ['focal_length','principal_point','extra_params']],
    'model_aligner':['ref_images_path','ref_is_gps','alignment_type','alignment_max_error','transform_path']}
    print(' '.join('--'+v for v in groups[command]));sys.exit(0)
assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(args)+'\\n')
if command=='database_creator':
    with sqlite3.connect(opt('database_path')) as db:
        db.executescript('CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY,model INTEGER,width INTEGER,height INTEGER,params BLOB,prior_focal_length INTEGER); CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT,camera_id INTEGER); CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB); CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB); CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER);')
elif command=='feature_importer':
    names=Path(opt('image_list_path')).read_text().splitlines()
    assert names==['00000.png','00001.png','00002.png']
    with sqlite3.connect(opt('database_path')) as db:
        for iid,name in enumerate(names,start=1):
            assert (Path(opt('import_path'))/(name+'.txt')).read_text()=='0 128\\n'
            db.execute('INSERT INTO images VALUES(?,?,?)',(iid,name,int(opt('ImageReader.existing_camera_id'))))
            db.execute('INSERT INTO keypoints VALUES(?,?,?,?)',(iid,0,4,b''))
            db.execute('INSERT INTO descriptors VALUES(?,?,?,?)',(iid,0,128,b''))
elif command=='matches_importer':
    assert opt(prefix+'Matching.use_gpu')=='0' and opt('match_type')=='raw'
    with sqlite3.connect(opt('database_path')) as db:
        ids={n:i for i,n in db.execute('SELECT image_id,name FROM images')}
        assert db.execute('SELECT MIN(rows) FROM keypoints').fetchone()[0]==16
        assert db.execute('SELECT COUNT(*) FROM descriptors').fetchone()[0]==0
        for block in Path(opt('match_list_path')).read_text().strip().split('\\n\\n'):
            lines=block.splitlines();a,b=lines[0].split();i,j=sorted((ids[a],ids[b]))
            count=0 if os.environ['RECOVER']=='1' and (i,j)==(2,3) else len(lines)-1
            db.execute('INSERT INTO two_view_geometries VALUES(?,?)',(i*2147483647+j,count))
    # Simulate the old interrupted workspace after all base geometry rows exist.
    interrupted=Path(os.environ['INTERRUPTED'])
    if os.environ['RECOVER']=='1' and not interrupted.exists():
        interrupted.touch();sys.exit(23)
else:
    output=Path(opt('output_path'));output=output/'0' if command=='mapper' else output
    shutil.copytree(os.environ['FIXTURE'],output,dirs_exist_ok=True)
''')
    for path in bindir.iterdir(): path.chmod(0o755)
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'RECONSTRUCTION_CONDA_SH':str(conda),
         'LAUNCH_MARKER':str(tmp_path/'handoff'),'CALLS':str(tmp_path/'calls'),'FIXTURE':str(fixture),'MODERN':str(int(modern)),
         'COLMAP_BIN':str(bindir/'colmap'),'SUPERGLUE_REPO':str(model_cache(tmp_path/'network')),
         'RECOVER':str(int(recover)),'INTERRUPTED':str(tmp_path/'interrupted'),
         'COLMAP_SEQUENTIAL_OVERLAP':'1' if recover else '2'}
    result=subprocess.run(['bash',str(launch/'run_region_superglue.sh'),'1',str(region)],env=env,capture_output=True,text=True,timeout=60)
    overlap=1 if recover else 2
    job=region/f'colmap_superpoint_superglue_seq{overlap}_v1'
    if recover:
        assert result.returncode==23,result.stdout+result.stderr
        assert (job/'features.done').is_file() and not (job/'matching.done').exists()
        cached={p:p.stat().st_mtime_ns for folder in ('superpoint','superglue_matches') for p in (job/folder).glob('*.npz')}
        result=subprocess.run(['bash',str(launch/'run_region_superglue.sh'),'1',str(region)],env=env,capture_output=True,text=True,timeout=60)
        assert all(p.stat().st_mtime_ns==stamp for p,stamp in cached.items())
        assert len(json.loads((job/'bridge_pairs.json').read_text())['rounds'])==1
    assert result.returncode==0,result.stdout+result.stderr
    model=region/f'runs/crop_colmap_superglue_seq{overlap}_fixed_v1'
    assert (tmp_path/'handoff').read_text()==str(model)
    calls=[json.loads(s) for s in (tmp_path/'calls').read_text().splitlines()]
    assert [a[0] for a in calls]==['database_creator','feature_importer','matches_importer']+(['matches_importer'] if recover else [])+['mapper','bundle_adjuster','model_aligner']
    plan=json.loads((model/'periodic_evaluation/plan.json').read_text())
    assert plan['gpu']=='1' and plan['training_args']['debug_preview_every']==500
    assert len(plan['selected'])==3 and len(plan['frames'])==4
    assert plan['plan']['freeze_lidar_positions'] and not plan['plan']['hard_culling']
    assert len((job/'pairs.txt').read_text().splitlines())==3
    assert json.loads((job/'verification_report.json').read_text())['verified_pairs']==(2 if recover else 3)
