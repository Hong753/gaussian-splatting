"""Real shared-selection preparation, no-AA propagation and source preservation."""
import argparse
import json
from pathlib import Path
import struct
from unittest.mock import patch

import numpy as np
from PIL import Image
from plyfile import PlyData
import pytest

from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts.prepare_orange4f import main, canonical_cloud, grid_geometry
from scripts import run_native_lidar as native, run_soft_surface_long as periodic
from utils.checkpoint_snapshots import digest
from test_region_ablation import region_source


def test_four_runs_share_full_resolution_sharp_stride3_and_cloud(region_source):
    source, cloud, _ = region_source
    pixels=np.zeros((64,64,3),np.uint8)
    pixels[:32,:32]=((np.indices((32,32)).sum(0)%2)*255)[...,None]
    Image.fromarray(pixels).save(source/'images/00001.png')
    before={str(p):digest(p) for p in source.rglob('*') if p.is_file()}
    args=['--source',str(source),'--point-cloud',str(cloud)]
    main(args)
    root=source/'3dgs_finetuning/stride3_noaa_v1'
    suite=json.loads((root/'suite.json').read_text())
    selection=json.loads(Path(suite['selection']).read_text())
    assert selection['selected_images']==['00001.png','00003.png']
    assert suite['frames']==6 and suite['train_frames']==2 and suite['heldout_frames']==4
    assert suite['cloud']['points']==20402 and suite['cloud']['sha256']==digest(cloud)
    assert (root/'shared/input.ply').resolve()==cloud
    assert [r['gpu'] for r in suite['runs']]==['0','1','2','3']
    for i,run in enumerate(suite['runs']):
        model=Path(run['model']);fixed=i%2==1
        assert not run['antialiasing'] and run['fixed_positions']==fixed
        assert json.loads((model/'keyframes.json').read_text())==selection
        if i<2:
            data=periodic.load_plan(model);plan=data['plan']
            p=argparse.ArgumentParser();mp=ModelParams(p);op=OptimizationParams(p);pp=PipelineParams(p)
            parsed,_=p.parse_known_args(plan['stages'][0]['argv'][1:])
            ds,opt,pipe=mp.extract(parsed),op.extract(parsed),pp.extract(parsed)
            assert not pipe.antialiasing and not plan['antialiasing']
            assert ds.resolution==1 and ds.loss_mask_dir=='' and not ds.train_test_exp
            assert opt.freeze_lidar_positions==fixed and plan['densification'].get('enabled',True)==(not fixed)
            assert not opt.disable_pruning and not opt.disable_opacity_resets
            assert (opt.feature_lr,opt.opacity_lr,opt.scaling_lr,opt.rotation_lr)==(.0025,.05,.005,.001)
            assert opt.position_lr_init==(0. if fixed else .00016)
            assert opt.position_lr_final==(0. if fixed else .0000016)
            assert opt.percent_dense==.01 and opt.densification_interval==100
            assert opt.scale_control=='off' and not plan['hard_culling'] and not plan['soft_surface']
            assert data['iterations']==[10000,20000,30000]
            with patch.object(native,'invoke') as invoke,patch.object(native.fixed,'execute_plan'):
                from argparse import Namespace
                native.execute_plan(Namespace(**data['training_args']),plan)
                assert '--no-antialiasing' in invoke.call_args.args[0]
        else:
            cfg=json.loads((model/'gsprior_run.json').read_text())
            assert cfg['iterations']==40000 and cfg['tsdf_iterations']==[20000,30000,40000]
            assert cfg['cloud']==suite['cloud'] and cfg['grid']==suite['grid']
            assert cfg['preview_every']==500 and cfg['max_all_points']==6000000
    split=json.loads((root/'shared/gsprior_data/split.json').read_text())
    assert split['train']==selection['selected_images'] and len(split['test'])==4
    assert all(digest(Path(p))==v for p,v in before.items())
    main(args)  # Cached selection is reused, never recomputed differently per run.
    with pytest.raises(ValueError,match='configuration changed'):
        main(args+['--prior-voxel-size','.03'])
    with (source/'sparse/0/cameras.txt').open('a') as f: f.write('# changed\n')
    with pytest.raises(ValueError,match='input changed'): main(args)


@pytest.mark.parametrize('binary',[False,True])
def test_colmap_cloud_stream_preserves_coordinates_and_colors(tmp_path,binary):
    source=tmp_path/'source';sparse=source/'sparse/0';sparse.mkdir(parents=True)
    (sparse/'cameras.txt').write_text('1 PINHOLE 2 2 1 1 1 1\n')
    (sparse/'images.txt').write_text('')
    values=[(1,1.,2.,3.,11,22,33,.1),(2,-1.,-2.,4.,44,55,66,.2)]
    if binary:
        path=sparse/'points3D.bin'
        with path.open('wb') as f:
            f.write(struct.pack('<Q',2))
            for row in values:
                f.write(struct.pack('<QdddBBBd',*row));f.write(struct.pack('<Qii',1,3,8))
    else:
        path=sparse/'points3D.txt';path.write_text('# points\n'+''.join(' '.join(map(str,r))+' 3 8\n' for r in values))
    before=digest(path);out=tmp_path/'input.ply'
    report=canonical_cloud(source,out)
    v=PlyData.read(out)['vertex'].data
    np.testing.assert_array_equal(np.column_stack([v[k] for k in 'xyz']),[[1,2,3],[-1,-2,4]])
    np.testing.assert_array_equal(np.column_stack([v[k] for k in ('red','green','blue')]),[[11,22,33],[44,55,66]])
    assert report['points']==2 and digest(path)==before


def test_tsdf_bounds_include_all_points_and_cameras_with_bounded_grid():
    bounds=[[-20,-5,-1],[70,10,3]]
    cam={'R':np.eye(3),'T':np.array([-90,0,0])}
    grid=grid_geometry(bounds,{'camera':cam},.02,100000)
    assert grid['voxels']<=100000 and grid['voxel_size']>=.02
    b=np.array(grid['bounds'])
    assert (b[:,0]<bounds[0]).all() and (b[:,1]>[90,10,3]).all()
    assert grid['truncation']==[grid['voxel_size']*x for x in (96,48,24)]


def test_noaa_evaluation_never_reenables_aa(tmp_path):
    model=tmp_path/'model';model.mkdir()
    data=dict(model=str(model),source='/source',gpu='0',plan={'antialiasing':False},
              frames=['a','b'],selected=['a'],report_max_views=32,report_seed=0,expected_frames=2)
    snapshot=dict(iteration=10000,snapshot_id='fixture',gaussians=4,files={})
    calls=[]
    def invoke(argv):
        calls.append(argv)
        if argv[0].endswith('eval_all_frames.py'):
            output=model/'psnr_all_frames/iteration_10000';output.mkdir(parents=True)
            (output/'summary.json').write_text(json.dumps(dict(iteration=10000,model=str(model),source='/source',
                model_type='3dgs',antialiasing=False,occlusion={'occlusion_mode':'off'},
                counts={'all':2,'selected':1,'unselected':1},mean_psnr_db={'all':19.5,'selected':20.,'unselected':19.})))
            (output/'per_frame.csv').write_text('image_name,selected_for_training,psnr_db\na,1,20\nb,0,19\n')
        else:
            output=Path(argv[argv.index('--output')+1]);output.mkdir(parents=True)
            (output/'index.html').write_text('comparisons')
    # Full metric verification is covered in periodic tests; here execute the
    # actual evaluation command builder and inspect every invoked render path.
    with patch.object(periodic.run_occlusion,'invoke',side_effect=invoke), \
         patch.object(periodic,'read_snapshot',return_value=snapshot), \
         patch.object(periodic,'experiment_diagnostics',return_value={}):
        periodic.evaluate(data,snapshot,0.)
    assert len(calls)>=2
    assert all('--no-antialiasing' in cmd and '--antialiasing' not in cmd for cmd in calls)
