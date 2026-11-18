import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from scripts import prepare_orange4f, run_orange4f_scaffold as runner
from scripts.run_soft_surface_long import load_plan
from test_orange4f_scaffold import build, cpu_backend
from test_region_ablation import region_source
from utils.checkpoint_snapshots import digest
from utils.scaffold_observations import Membership, prepare_membership, sensed_step


def test_measured_row_order_voxel_merge_and_missing_views(tmp_path):
    cloud = tmp_path/'cloud.ply'
    v = np.zeros(4, dtype=[(k,'<f4') for k in 'xyz'])
    v['x'] = [.071, -.01, .011, .012]
    PlyData([PlyElement.describe(v,'vertex')]).write(cloud)
    np.save(tmp_path/'ids.npy', [3,2,2])
    np.save(tmp_path/'empty.npy', np.empty(0,dtype=int))
    manifest = tmp_path/'lidar_observations.json'
    meta = dict(format='lidar_rows_v1', point_cloud_sha256=digest(cloud), frames={'a':'ids.npy','b':'empty.npy'})
    manifest.write_text(json.dumps(meta))
    out = tmp_path/'parents'
    prepare_membership(out, manifest, cloud, cloud, ['b','a'], .02)
    centers = np.unique(np.round(np.column_stack([v[k] for k in 'xyz'])/.02),axis=0)*.02
    m = Membership(out, torch.tensor(centers), ['a','b'])
    assert m.mask('a').tolist() == [False,True,False] and not m.mask('b').any()
    with pytest.raises(ValueError,match='positions/order'):
        Membership(out,torch.tensor(centers).flip(0),['a','b'])
    with pytest.raises(ValueError,match='association'):
        prepare_membership(tmp_path/'missing',manifest,cloud,cloud,['x'],.02)
    meta['point_cloud_sha256'] = 'wrong'
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='different point cloud'):
        prepare_membership(tmp_path/'wrong',manifest,cloud,cloud,['a'],.02)


def test_registered_scans_union_and_heldout_split(tmp_path,region_source):
    source,cloud,_ = region_source
    suite = tmp_path/'suite'
    prepare_orange4f.main(['--source',str(source),'--output',str(suite),'--point-cloud',str(cloud)])
    info = json.loads((suite/'suite.json').read_text())
    names = json.loads(Path(info['selection']).read_text())['selected_images']
    vertices = PlyData.read(cloud)['vertex'].data
    xyz = np.column_stack([vertices[k] for k in 'xyz'])
    pose = np.eye(4)
    pose[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    pose[:3,3] = [1,2,3]
    scans = []
    for i in (0,1):
        local = (xyz[[i]]-pose[:3,3]) @ pose[:3,:3]
        np.save(source/f'{i}.npy', local)
        scans.append(dict(xyz=f'{i}.npy',scanner_to_world=pose.tolist(),images=names))
    manifest = source/'lidar_scans.json'
    manifest.write_text(json.dumps(dict(format='scanner_xyz_v1',scans=scans)))
    args = ['--gpu','4','--variant','sensed','--suite',str(suite)]
    runner.main(args); runner.main(args)
    model = suite/'scaffold_sensed_anchors_v1'
    plan = load_plan(model)
    assert not plan['plan']['pruning'] and plan['plan']['freeze_anchors']
    assert '--sensed_membership' in plan['plan']['stages'][0]['argv']
    meta = json.loads((model/'observations/metadata.json').read_text())
    assert meta['names'] == sorted(names) and meta['per_view_parents'] == [2]*len(names)
    runner.main(['--gpu','5','--variant','fixed-static','--suite',str(suite)])
    ctrl = load_plan(suite/'scaffold_fixed-static_anchors_v1')
    assert not ctrl['plan']['pruning'] and ctrl['selected'] == plan['selected']
    assert '--sensed_membership' not in ctrl['plan']['stages'][0]['argv']


def test_mesh_proxy_requires_explicit_selection_and_matching_camera(tmp_path,region_source):
    from scripts.view_comparisons import read_colmap_records
    from utils.occlusion_depth import camera_signature, record_camera
    from utils.sparse_lidar import frame_filename
    source,cloud,_ = region_source
    name,record = next(iter(read_colmap_records(source).items()))
    signature = camera_signature(record_camera(record,1,'cpu'),1)
    (tmp_path/'frames').mkdir()
    np.savez(tmp_path/'frames'/frame_filename(name),visible_ids=np.array([0,1]),image=name,camera_signature=signature)
    meta = dict(format='scene_occlusion_mesh_v1', images=[name],resolution=1,
        camera_signatures={name:signature},point_cloud_sha256=digest(cloud))
    path = tmp_path/'metadata.json'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='separate'):
        prepare_membership(tmp_path/'wrong',path,cloud,cloud,[name],.02)
    prepare_membership(tmp_path/'proxy',path,cloud,cloud,[name],.02,mesh_source=source)
    meta['camera_signatures'][name] = 'other pose'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='pose mismatch'):
        prepare_membership(tmp_path/'other',path,cloud,cloud,[name],.02,mesh_source=source)


def test_full_forward_and_masked_decoders_and_adam_momentum():
    # Actual parent filtering/offset generation/MLPs with a differentiable CPU compositor.
    diff = ModuleType('diff_gaussian_rasterization')
    diff.GaussianRasterizationSettings = lambda **kw: kw
    def rasterizer(**kwargs):
        def composite(means3D, means2D, opacities, colors_precomp, scales, rotations, **other):
            alpha = opacities[:,0]
            trans = (1-alpha).cumprod(0)
            weights = torch.cat([torch.ones_like(alpha[:1]),trans[:-1]])*alpha
            rgb = (colors_precomp*weights[:,None]).sum(0)
            return rgb, torch.ones_like(alpha), None
        return composite
    diff.GaussianRasterizer = rasterizer
    with cpu_backend() as Model:
        mod = SimpleNamespace(ScaffoldGaussianModel=Model)
        with patch.dict(sys.modules,{'diff_gaussian_rasterization':diff,'models.scaffold_gaussian':mod}):
            path = Path(__file__).resolve().parents[1]/'models/scaffold_renderer.py'
            spec = importlib.util.spec_from_file_location('scaffold_render_observed',path)
            render_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(render_mod)
        m = build(Model,True)
        with torch.no_grad():
            m.mlp_opacity[2].weight.zero_()
            m.mlp_opacity[2].bias.copy_(torch.tensor([.6,-.6]))
            m.mlp_color[0].weight.fill_(.1); m.mlp_color[0].bias.fill_(.5)
            m.mlp_color[2].weight.fill_(.1)
        cam = SimpleNamespace(FoVx=1.,FoVy=1.,camera_center=torch.zeros(3),image_width=4,image_height=4,
                              world_view_transform=torch.eye(4),full_proj_transform=torch.eye(4))
        visible = torch.tensor([False,True,False,True,False,True])
        pipe = SimpleNamespace(debug=False,antialiasing=False)
        def render(sel):
            return render_mod.render(cam,m,pipe,torch.zeros(3),visible_mask=visible,update_anchors=sel)
        plain = render(None)['render']
        selected = torch.tensor([False,False,False,True,False,False])
        gated = render(selected)
        torch.testing.assert_close(gated['render'], plain, rtol=0,atol=0)
        assert gated['active_parents'].tolist() == [3]
        # Only the middle child trains the color decoder; the foreground child still attenuates it.
        grads = []
        for mask in (selected, visible, visible & ~selected):
            out = render(mask)
            grads.append(torch.autograd.grad(out['render'].sum(),m.mlp_color[2].bias)[0])
        torch.testing.assert_close(grads[0]+grads[2],grads[1])
        assert grads[0].abs().sum()>0 and grads[2].abs().sum()>0
        for eligible in (visible,selected):
            render(eligible)['render'].sum().backward()
            assert not m._anchor_feat.grad[~eligible].any()
            sensed_step(m.optimizer,eligible)
            m.optimizer.zero_grad(set_to_none=True)
            if eligible is visible:
                feat = m._anchor_feat.detach().clone()
                mom = m.optimizer.state[m._anchor_feat]['exp_avg'].clone()
                assert mom[1].abs().sum()>0
        torch.testing.assert_close(m._anchor_feat[~selected],feat[~selected],rtol=0,atol=0)
        torch.testing.assert_close(m.optimizer.state[m._anchor_feat]['exp_avg'][~selected],mom[~selected],rtol=0,atol=0)
        assert not torch.equal(m._anchor_feat[selected],feat[selected])


@pytest.mark.parametrize('explicit_rows', [False, True])
def test_indexed_launcher_trains_on_exact_red_rows(tmp_path, explicit_rows):
    import shutil
    import struct
    from test_sparse_observation_viewer import fixture
    from scripts.view_sparse_observations import Inspector

    source = tmp_path/'capture'
    sparse = fixture(source)
    images = sparse/'images.bin'
    with images.open('r+b') as binary:
        binary.write(struct.pack('<Q', 6))
        binary.seek(0, 2)
        for image_id, name in enumerate(('b.jpg', 'c.jpg', 'y.jpg', 'z.jpg'), 40):
            binary.write(struct.pack('<i7di', image_id, 1,0,0,0, 0,0,0, 1) + name.encode() + bytes(9))
            shutil.copyfile(source/'images/empty.jpg', source/'images'/name)
    # The suite uses a different cloud. Indexed training must initialize from the viewed PLY.
    path = sparse/'points3D.ply'
    xyz = PlyData.read(path)['vertex'].data
    vertices = np.zeros(len(xyz), dtype=xyz.dtype.descr + [(k,'u1') for k in ('red','green','blue')])
    for k in 'xyz':
        vertices[k] = xyz[k]
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(path)
    PlyData([PlyElement.describe(vertices[:1], 'vertex')]).write(source/'input.ply')
    if explicit_rows:
        # Explicit rows win for this view, including a row with ambiguous XYZ.
        np.save(sparse/'rows.npy', [4,1,4])
        (sparse/'lidar_observations.json').write_text(json.dumps(dict(format='lidar_rows_v1',
            frames={'09882.jpg':'rows.npy'}, point_cloud_sha256=digest(path))))
    suite = tmp_path/'suite'
    prepare_orange4f.main(['--source',str(source),'--output',str(suite)])
    args = ['--suite',str(suite),'--variant','indexed','--gpu','4']
    assert runner.main(args) == 0
    model = suite/'scaffold_indexed_anchors_v1'
    plan = load_plan(model)
    flags = plan['plan']['stages'][0]['argv']
    assert all(v in flags for v in ('--sensed_membership','--freeze_anchors','--disable_anchor_growth','--disable_anchor_pruning'))
    assert plan['selected'] == ['09882.jpg','empty.jpg']
    assert plan['iterations'] == [10000,20000,30000]
    viewer = Inspector(source, limit=1)
    actual = np.column_stack([PlyData.read(model/'input.ply')['vertex'][k] for k in 'xyz'])
    np.testing.assert_array_equal(actual, viewer.ply.coords(slice(None)))
    with cpu_backend() as Model:
        gs = build(Model, True)
        anchors = gs.voxelize_sample(actual, .02)
    membership = Membership(model/'observations', torch.tensor(anchors), plan['selected'])
    data = json.loads((model/'observations/metadata.json').read_text())
    assert data['format'] == 'viewer_indexed_rows_v1' and data['empty_views'] == 1
    assert data['per_view_indexed_rows'] == [2,0]
    assert data['per_view_source'][0] == viewer.frame(31)['indexed_source']
    rgb_data, encoded = viewer.frame_data(31)
    red = np.frombuffer(encoded, '<u4', offset=8, count=rgb_data['indexed_rows'])
    np.testing.assert_array_equal(red, [1,4] if explicit_rows else [0,2])
    expected = np.unique(np.round(actual[red]/.02),axis=0)*.02
    np.testing.assert_array_equal(anchors[membership.mask('09882.jpg').numpy()], expected)
    assert not membership.mask('empty.jpg').any()
    assert runner.main(args) == 0
    changed = sparse/'rows.npy' if explicit_rows else sparse/'points3D.bin'
    with changed.open('ab') as f:
        f.write(b'x')
    with pytest.raises(ValueError,match='input changed'):
        runner.main(args)


def test_same_checkpoint_ablation_measures_actual_occlusion(tmp_path):
    from scripts import audit_scaffold_render as audit

    # Three parents overlap on one pixel. Only the middle parent is indexed.
    diff = ModuleType('diff_gaussian_rasterization')
    diff.GaussianRasterizationSettings = lambda **kw: kw
    def rasterizer(raster_settings):
        def composite(means3D, opacities, colors_precomp, **unused):
            alpha = opacities[:,0]
            trans = (1-alpha).cumprod(0)
            w = torch.cat([torch.ones_like(alpha[:1]), trans[:-1]]) * alpha
            pixel = (colors_precomp*w[:,None]).sum(0) + trans[-1]*raster_settings['bg']
            return pixel[:,None,None].expand(3,4,4), torch.ones_like(alpha), None
        return composite
    diff.GaussianRasterizer = rasterizer
    with cpu_backend() as Model:
        torch.manual_seed(15)
        path = Path(__file__).resolve().parents[1]/'models/scaffold_renderer.py'
        spec = importlib.util.spec_from_file_location('scaffold_audit_backend',path)
        backend = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'diff_gaussian_rasterization':diff,
                    'models.scaffold_gaussian':SimpleNamespace(ScaffoldGaussianModel=Model)}):
            spec.loader.exec_module(backend)
        gs = build(Model,True)
        gs.eval()
        cam = SimpleNamespace(FoVx=1.,FoVy=1.,camera_center=torch.zeros(3),image_width=4,image_height=4,
                              world_view_transform=torch.eye(4),full_proj_transform=torch.eye(4),
                              image_name='test.jpg',original_image=torch.full((3,4,4),.4))
        visible = torch.tensor([False,True,False,True,False,True])
        selected = torch.tensor([False,False,False,True,False,False])
        pipe = SimpleNamespace(debug=False,antialiasing=False)
        bg = torch.zeros(3)
        with torch.no_grad(), patch.dict(sys.modules, {'models.scaffold_renderer':backend}), \
                patch.object(backend, 'prefilter_voxel', return_value=visible):
            gs.mlp_opacity[2].weight.zero_()
            gs.mlp_opacity[2].bias.copy_(torch.tensor([.8,-.8]).atanh())
            images,stats = audit.compare(cam,gs,pipe,bg,selected)
            torch.testing.assert_close(images[3][:,0,0],torch.tensor([.832,.16,.008]))
            assert stats['indexed_coverage'] == 1. and stats['indexed_children'] == 1 and stats['full_children'] == 3
            assert stats['fraction_indexed_weight_blocked'] == pytest.approx(.8)
            assert not torch.equal(images[1],images[2])
            all_images,_ = audit.compare(cam,gs,pipe,bg,visible)
            torch.testing.assert_close(all_images[1],all_images[2],atol=0,rtol=0)
            empty_images,empty_stats = audit.compare(cam,gs,pipe,bg,torch.zeros_like(visible))
            assert not empty_images[2].any() and empty_stats['indexed_coverage'] == 0
            assert empty_stats['mean_indexed_weight'] == 0 and empty_stats['fraction_indexed_weight_blocked'] is None
        audit.write_report(tmp_path,10000,{'test.jpg':dict(stats,images=['a.png']*4)})
        assert '83.2%' in (tmp_path/'index.html').read_text()
