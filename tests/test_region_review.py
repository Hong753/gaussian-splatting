"""Manual review persistence, restore/navigation and exact export membership."""
import json

import numpy as np
import pytest

from scripts.region_dataset import RegionData
from scripts.region_review import CameraReview, reviewed_rows
from scripts.view_comparisons import read_colmap_records
from test_region_ablation import region_source


@pytest.fixture
def review_data(region_source):
    data=RegionData(*region_source,max_points=100000)
    cfg=dict(center=[0.,0.,2.],size=[1.4,1.4,.4],halo=.2,min_overlap=.02,
             max_camera_distance=5.,frame_start='',frame_end='',exclude=[])
    return data,cfg,data.find(**cfg)


def test_review_round_trip_and_navigation_keep_reject_restore(tmp_path,review_data):
    _,cfg,rows=review_data
    session=CameraReview({'source':'fixture'},cfg,rows)
    a,b,c,d=session.names
    assert not reviewed_rows(rows,session.decisions)  # Browsing does not imply approval.
    session.choose(a,'keep');session.choose(b,'reject')
    assert session.neighbor(a,1,'unreviewed')==c
    assert session.neighbor(d,1,'unreviewed')==c
    assert session.neighbor(a,-1,'reject')==b
    assert session.filtered('keep')==[a]
    assert session.counts()==dict(candidates=4,kept=1,train=1,heldout=0,rejected=1,unreviewed=2)
    path=tmp_path/'region.review.json';session.current=c;session.save(path)
    resumed=CameraReview.load(path,{'source':'fixture'})
    assert resumed.decisions==session.decisions and resumed.current==c and resumed.config==cfg
    resumed.choose(b,'keep')
    assert resumed.counts()['heldout']==1 and resumed.counts()['rejected']==0
    resumed.choose(b,'unreviewed')
    assert resumed.filtered('reject')==[] and resumed.neighbor(a,1,'reject') is None
    with pytest.raises(ValueError,match='different source'):
        CameraReview.load(path,{'source':'changed'})
    with pytest.raises(ValueError): resumed.choose('unknown','keep')
    with pytest.raises(ValueError): resumed.choose(a,'auto-visible')


def test_export_excludes_rejected_and_unreviewed_from_cameras_masks_and_training(tmp_path,review_data):
    data,cfg,rows=review_data
    session=CameraReview({},cfg,rows)
    a,b,c,d=session.names
    session.choose(a,'keep');session.choose(b,'reject');session.choose(c,'keep')
    output=tmp_path/'reviewed_region'
    info=data.export(output,rows=rows,review=session.decisions,**cfg)
    assert set(read_colmap_records(output))=={a,c}
    assert json.loads((output/'keyframes.json').read_text())['selected_images']==[a,c]
    assert {r['name'] for r in info['selection']}=={a,c}
    assert {r['split'] for r in info['selection']}=={'train'}  # Rejection is NOT a held-out reassignment.
    for rejected in (b,d): assert not (output/'images'/rejected).exists()
    assert len(list((output/'masks').glob('*.png')))==2
    review=json.loads((output/'review.json').read_text())
    assert review['rejected']==[b] and review['unreviewed']==[d] and review['kept']==[a,c]
    assert review==info['manual_review']
    session.choose(b,'keep')
    second=data.export(tmp_path/'restored_region',rows=rows,review=session.decisions,**cfg)
    assert [r['name'] for r in second['selection'] if r['split']=='eval']==[b]
    np.testing.assert_array_equal(np.load(output/'original_lidar_indices.npy'),np.load(tmp_path/'restored_region/original_lidar_indices.npy'))


def test_unreviewed_and_wrong_candidate_reviews_cannot_export(tmp_path,review_data):
    data,cfg,rows=review_data
    session=CameraReview({},cfg,rows)
    with pytest.raises(ValueError,match='two training'):
        data.export(tmp_path/'empty',rows=rows,review=session.decisions,**cfg)
    assert not (tmp_path/'empty').exists()
    with pytest.raises(ValueError,match='exactly'):
        data.export(tmp_path/'wrong',rows=rows,review={'not-a-candidate':'keep'},**cfg)


def test_headless_export_resumes_exact_saved_review_and_box(tmp_path,review_data):
    from scripts.keyframe_pipeline.common import file_stamp
    from scripts.select_region import main
    data,cfg,rows=review_data
    identity=dict(source=str(data.source),point_cloud=file_stamp(data.point_cloud),
                  keyframes=file_stamp(data.keyframes),poses=data.pose_stamps,sample_points=data.max_points)
    session=CameraReview(identity,cfg,rows)
    for row in rows:
        session.choose(row['name'],'keep' if row['split']=='train' else 'reject')
    path=tmp_path/'saved.review.json'; session.save(path)
    output=tmp_path/'cli_export'
    assert main(['--source',str(data.source),'--point-cloud',str(data.point_cloud),
                 '--keyframes',str(data.keyframes),'--output',str(output),
                 '--sample-points','100000','--review-file',str(path),'--export'])==0
    meta=json.loads((output/'selection.json').read_text())
    assert meta['center']==cfg['center'] and meta['size']==cfg['size']
    assert {r['name'] for r in meta['selection']}==set(session.filtered('keep'))


def test_actual_viser_controls_review_autosave_and_export(tmp_path,review_data):
    import viser
    from scripts.region_review_app import create_app
    data,cfg,rows=review_data
    server=viser.ViserServer(host='127.0.0.1',port=0,verbose=False)
    review_file=tmp_path/'ui.review.json'
    output=tmp_path/'ui_export'
    try:
        app=create_app(server,data,output,review_file,{},cfg,anchor=rows[0]['name'])
        assert app['controls']['export'].disabled
        app['select'](None)
        s=app['state']['session']
        assert s.counts()['unreviewed']==4
        assert not app['controls']['overlay'].visible  # Unmasked RGB is the default review.
        a,b,c,d=s.names
        app['decide']('keep')
        assert app['controls']['candidate'].value==b
        assert np.array_equal(app['controls']['rgb'].image,data.rgb(b))
        app['decide']('reject')
        assert app['controls']['candidate'].value==c
        app['decide']('keep')
        assert not app['controls']['export'].disabled
        app['select'](None)  # Repeating unchanged search must not lose decisions.
        assert app['state']['session'].decisions[a]=='keep'
        assert CameraReview.load(review_file,{}).decisions[b]=='reject'
        app['save'](None)
        assert app['state']['completed']
        assert set(read_colmap_records(output))=={a,c}
        assert app['controls']['keep'].disabled
        assert 'Export complete' in app['controls']['status'].content
    finally:
        server.stop()
