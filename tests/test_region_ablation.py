"""Regional data fidelity, hidden geometry rejection, and actual loss gradients."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest
import torch
from scipy.spatial.transform import Rotation

from scripts.region_dataset import RegionData, foreground_mask, inside
from scripts.keyframe_pipeline.common import Camera, Image as CameraImage
from scripts.view_comparisons import read_colmap_records
from scripts.run_region_ablation import prepare_args, main as prepare
from scripts.run_native_lidar import build_plan
from scripts.run_fixed_lidar import checkpoint_steps
from utils.region_loss import load_mask, mask_path, photometric_terms, score


@pytest.fixture
def region_source(tmp_path):
    source=tmp_path/'scene'; sparse=source/'sparse/0'; sparse.mkdir(parents=True)
    (source/'images').mkdir()
    (sparse/'cameras.txt').write_text('1 PINHOLE 64 64 64 64 32 32\n')
    rows=[]
    for i in range(6):
        name=f'{i:05d}.png'
        q=Rotation.from_euler('z',i*2,degrees=True).as_quat()[[3,0,1,2]]
        center=np.array([i*.04 if i<4 else 100.,0.,0.])
        R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix(); t=-R@center
        rows.append(f'{i+1} '+' '.join(map(str,[*q,*t]))+f' 1 {name}\n\n')
        Image.new('RGB',(64,64),(30+i,50,80)).save(source/'images'/name)
    (sparse/'images.txt').write_text(''.join(rows))
    (sparse/'points3D.txt').write_text('')
    xx,yy=np.meshgrid(np.linspace(-1,1,101),np.linspace(-1,1,101))
    xyz=np.concatenate([np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,z)]) for z in (2.,4.)]).astype('f4')
    v=np.zeros(len(xyz),dtype=[(n,'f4') for n in ('x','y','z','nx','ny','nz')]+[(n,'u1') for n in ('red','green','blue')])
    for i,n in enumerate('xyz'): v[n]=xyz[:,i]
    v['red']=100;v['green']=150;v['blue']=200
    ply=tmp_path/'input.ply'; PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(ply))
    keyframes=tmp_path/'keyframes.json'; keyframes.write_text(json.dumps(dict(selected_images=['00000.png','00002.png','00004.png'])))
    return source,ply,keyframes


def exported(tmp_path, source):
    data=RegionData(*source,max_points=100000)
    center=[0.,0.,2.]; size=[1.4,1.4,.4]
    settings=dict(halo=.2,min_overlap=.02,max_camera_distance=5.,frame_start='',frame_end='',exclude=[])
    rows=data.find(center,size,**settings)
    region=tmp_path/'region'
    info=data.export(region,center,size,rows,**settings)
    return data,region,info,rows,settings


def test_region_export_preserves_poses_images_membership_and_all_original_crop_rows(tmp_path,region_source):
    data,region,info,rows,_=exported(tmp_path,region_source)
    assert [r['name'] for r in rows]==[f'{i:05d}.png' for i in range(4)]
    assert [r['name'] for r in rows if r['split']=='train']==['00000.png','00002.png']
    old,new=read_colmap_records(region_source[0]),read_colmap_records(region)
    for name,r in new.items():
        for key in ('R','T','fx','fy','cx','cy','width','height'):
            np.testing.assert_allclose(r[key],old[name][key],atol=1e-12,rtol=0)
        assert Path(r['image_path']).resolve()==Path(old[name]['image_path']).resolve()
        assert load_mask(region/'masks',name,(64,64)).any()
    ids=np.load(region/'original_lidar_indices.npy')
    full=PlyData.read(str(region_source[1]))['vertex'].data
    cropped=PlyData.read(str(region/'input.ply'))['vertex'].data
    np.testing.assert_array_equal(cropped,full[ids])
    expected=np.flatnonzero(inside(np.column_stack([full[n] for n in 'xyz']),info['center'],info['size'],.2))
    np.testing.assert_array_equal(ids,expected)
    assert len(cropped)==info['lidar_count']
    assert (region/'sparse/0/points3D.ply').resolve()==region/'input.ply'
    assert (region/'index.html').is_file()
    assert len(read_colmap_records(region_source[0]))==6  # Original model untouched.
    with pytest.raises(FileExistsError):
        data.export(region,info['center'],info['size'],rows,**info['settings'])


def test_zbuffer_rejects_region_hidden_behind_foreground_wall():
    camera=Camera(1,64,64,np.array([64.,64.,32.,32.]))
    image=CameraImage(1,1,'wall.png',np.eye(3),np.zeros(3))
    xx,yy=np.meshgrid(np.linspace(-1,1,129),np.linspace(-1,1,129))
    wall=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,2.)])
    behind=wall*2  # Same rays, twice as far.
    core=[0,0,4]; size=[2,2,.2]
    unoccluded,_=foreground_mask(behind,image,camera,core,size)
    occluded,_=foreground_mask(np.concatenate([behind,wall]),image,camera,core,size)
    assert unoccluded.sum()>200
    assert occluded.sum()==0


def test_temporal_bounds_and_manual_exclusions_and_stale_export(tmp_path,region_source):
    data,region,info,rows,settings=exported(tmp_path,region_source)
    filtered=data.find(info['center'],info['size'],**{**settings,'frame_start':'00001.png','exclude':['00003.png']})
    assert [r['name'] for r in filtered]==['00001.png','00002.png']
    with pytest.raises(ValueError,match='Selection changed'):
        data.export(tmp_path/'other',info['center'],info['size'],rows,**{**settings,'exclude':['00003.png']})
    assert not (tmp_path/'other').exists()


def test_l1_and_ssim_have_exact_zero_gradients_outside_roi_and_psnr_normalization():
    torch.manual_seed(0)
    pred=torch.rand(3,48,56,requires_grad=True); target=torch.rand_like(pred)
    mask=torch.zeros(1,48,56,dtype=torch.bool);mask[:,8:40,8:48]=True
    l1,ssim=photometric_terms(pred,target,mask)
    (.8*l1+.2*(1-ssim)).backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[:,~mask[0]]==0).all()
    assert pred.grad[:,mask[0]].abs().sum()>0
    a=pred.detach().clone(); a[:,~mask[0]]=10
    l2,s2=photometric_terms(a,target,mask)
    torch.testing.assert_close(l1,l2);torch.testing.assert_close(ssim,s2)
    torch.testing.assert_close(score(pred.detach(),target,mask),score(a,target,mask))
    constant=torch.full_like(pred,.1); zeros=torch.zeros_like(pred)
    torch.testing.assert_close(score(constant,zeros,mask),torch.tensor(20.))
    torch.testing.assert_close(score(constant,zeros),torch.tensor(20.))
    with pytest.raises(ValueError,match='nonempty'):
        photometric_terms(pred,target,torch.zeros_like(mask))
    tiny=torch.zeros_like(mask);tiny[:,2:4,2:4]=True
    with pytest.raises(ValueError,match='SSIM'):
        photometric_terms(pred,target,tiny)


def test_crop_and_full_control_share_views_masks_schedules_and_periodic_plan(tmp_path,region_source):
    _,region,info,_,_=exported(tmp_path,region_source)
    plans=[]
    for context in ('crop','full'):
        args,protocol=prepare_args(['--region',str(region),'--context',context,'--iterations','30','--reference-steps','30'])
        plan=build_plan(args);plans.append(plan)
        cmd=plan['stages'][0]['argv']
        assert '--freeze_lidar_positions' in cmd and '--disable_pruning' in cmd and '--disable_opacity_resets' in cmd
        assert cmd[cmd.index('--loss_mask_dir')+1]==str(region/'masks')
        assert cmd[cmd.index('--position_lr_init')+1]=='0'
        assert protocol['matched_expected_updates_step']==20
        assert checkpoint_steps(args)==[20,30]
        assert plan['scale_control']=='off' and not plan['hard_culling']
    assert plans[0]['initial_points']<plans[1]['initial_points']
    with contextlib.redirect_stdout(io.StringIO()):
        prepare(['--region',str(region),'--iterations','30','--reference-steps','30'])
    saved=json.loads((region/'runs/crop_v1/periodic_evaluation/plan.json').read_text())
    assert saved['iterations']==[20,30]
    assert saved['frames']==[r['name'] for r in info['selection']]
    assert saved['selected']==['00000.png','00002.png']
    assert len(saved['input_fingerprints'])==3+len(saved['frames'])
    # Runtime reconstruction must reproduce the saved plan byte-for-byte.
    values=saved['training_args']
    for key in ('source','model','keyframes','point_cloud','loss_mask_dir'):
        values[key]=Path(values[key])
    assert build_plan(SimpleNamespace(**values))==saved['plan']


def test_mask_tampering_rejected_before_preparing_training(tmp_path,region_source):
    _,region,info,_,_=exported(tmp_path,region_source)
    mask_path(region/'masks',info['selection'][0]['name']).write_bytes(b'broken')
    with pytest.raises(ValueError,match='mask changed'):
        prepare_args(['--region',str(region)])


def test_same_region_comparison_preserves_world_coordinates_and_rejects_changed_pose(tmp_path,region_source):
    from scripts.compare_region import prepare_specs
    from test_view_comparisons import checkpoint
    _,region,info,_,_=exported(tmp_path,region_source)
    model=tmp_path/'full_model';model.mkdir();checkpoint(model,30000)
    (model/'cfg_args').write_text(f'Namespace(source_path={str(region_source[0])!r},eval=False,antialiasing=True)')
    specs,_=prepare_specs(region,[model],[30000])
    assert len(specs[0]['records'])==4
    assert specs[0]['cfg']['loss_mask_dir']==str(region/'masks')
    assert specs[0]['cfg']['resolution']==1
    assert specs[0]['camera_poses']=='original'
    assert specs[0]['cfg']['occlusion_mode']=='off'
    image_file=region/'sparse/0/images.txt'
    lines=image_file.read_text().splitlines()
    values=lines[0].split();values[5]=str(float(values[5])+1);lines[0]=' '.join(values)
    image_file.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='different T'):
        prepare_specs(region,[model],[30000])
