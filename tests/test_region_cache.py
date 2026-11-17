"""Projection parity and persistent cache reuse without repeated point scans."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts import region_dataset as module
from scripts.keyframe_pipeline.common import project as reference_project, Camera, Image
from test_region_ablation import region_source


def config():
    return dict(center=[0.,0.,2.],size=[1.4,1.4,.4],halo=.2,min_overlap=.02,
                max_camera_distance=5.,frame_start='',frame_end='',exclude=[])


def test_projection_and_masks_match_existing_matrix_projection(monkeypatch):
    rng=np.random.default_rng(210)
    points=rng.normal(size=(100000,3)).astype(np.float32);points[:,2]+=3
    K=np.array([64.,70.,31.7,32.1]);camera=Camera(1,64,64,K)
    fast_project=module.project
    for angle in (0.,.1,-.3):
        R=Rotation.from_rotvec([.02,angle,.04]).as_matrix();C=np.array([.1,-.3,.4])
        a=fast_project(points,R,C,K);b=reference_project(points,R,C,K)
        for new,old in zip(a,b): np.testing.assert_allclose(new,old,rtol=1e-10,atol=1e-7)
        image=Image(1,1,'image.png',R,C)
        monkeypatch.setattr(module,'project',fast_project)
        mask,info=module.foreground_mask(points,image,camera,[0,0,2.5],[2,2,2])
        monkeypatch.setattr(module,'project',reference_project)
        expected,old_info=module.foreground_mask(points,image,camera,[0,0,2.5],[2,2,2])
        np.testing.assert_array_equal(mask,expected)
        assert info==old_info


def test_search_export_and_restart_reuse_masks_and_preserve_original_crop(tmp_path,region_source,monkeypatch,capsys):
    cache=tmp_path/'projections'
    data=module.RegionData(*region_source,max_points=100000,cache_dir=cache)
    cfg=config();rows=data.find(**cfg)
    assert data.cache_counts==dict(computed=4,disk_hits=0)
    # Returned rows cannot mutate the internally validated candidate selection.
    rows[0]['overlap']=-1
    rows=data.find(**cfg)
    assert rows[0]['overlap']>0
    def unexpected(*a,**kw): raise AssertionError('Cached camera projected again')
    monkeypatch.setattr(module,'foreground_mask',unexpected)
    progress=[]
    info=data.export(tmp_path/'export',rows=rows,progress=progress.append,**cfg)
    assert data.cache_counts==dict(computed=4,disk_hits=4)
    assert any('temporary export' in message for message in progress)
    assert any('Cropping all' in message for message in progress)
    assert (tmp_path/'export/input.ply').is_file() and info['lidar_count']>4
    resumed=module.RegionData(*region_source,max_points=100000,cache_dir=cache)
    assert resumed.find(**cfg)==rows
    assert resumed.cache_counts==dict(computed=0,disk_hits=4)
    # Search filters can change without invalidating geometry masks.
    subset=resumed.find(**{**cfg,'exclude':['00003.png']})
    assert len(subset)==3 and resumed.cache_counts['computed']==0
    text=capsys.readouterr().out
    assert 'Region search: 0/6 cameras' in text and 'Region search: 6/6 cameras' in text
    assert 'no repeated projections' in text


def test_cache_changes_with_geometry_and_inputs_and_recovers_corrupt_file(tmp_path,region_source):
    cache=tmp_path/'cache';data=module.RegionData(*region_source,max_points=100000,cache_dir=cache)
    cfg=config();name='00000.png'
    original=data.view_mask(name,cfg['center'],cfg['size'])[0]
    file=next(cache.rglob('*.npz'));file.write_bytes(b'incomplete npz')
    recovered=data.view_mask(name,cfg['center'],cfg['size'])[0]
    np.testing.assert_array_equal(original,recovered)
    assert data.cache_counts['computed']==2
    data.view_mask(name,[0,0,4],cfg['size'])
    assert data.cache_counts['computed']==3
    assert len(list(cache.rglob('*.npz')))==2
    poses=region_source[0]/'sparse/0/images.txt'
    poses.write_text(poses.read_text()+'\n')
    changed=module.RegionData(*region_source,max_points=100000,cache_dir=cache)
    assert changed.cache_dir!=data.cache_dir
    changed.view_mask(name,cfg['center'],cfg['size'])
    assert changed.cache_counts==dict(computed=1,disk_hits=0)
    sampled=module.RegionData(*region_source,max_points=10000,cache_dir=cache)
    assert sampled.cache_dir!=changed.cache_dir
