"""Exercise the Orange4F entry point and retain original indexed LiDAR rows."""
import json
from pathlib import Path
import struct
import sys

import cv2
import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.orange4f_pose_ba import main, pose_split
from scripts.keyframe_pipeline.common import read_model
from scripts.keyframe_pipeline.pose_export import replace_text
from scripts.pose_options import dataset_pose_labels


def test_pilot_samples_contiguous_train_windows_and_real_holdouts():
    names = [f'frame{i}.jpg' for i in range(900)]
    training = names[1::3]
    selected, checks = pose_split(names[::-1], training[::-1], False)
    assert len(selected) == 48 and len(checks) == 12
    assert set(selected) <= set(training) and set(checks).isdisjoint(training)
    for start in (0, 16, 32):
        first = training.index(selected[start])
        assert selected[start:start + 16] == training[first:first + 16]
    assert pose_split(names, training, True) == (training, checks)
    with pytest.raises(ValueError, match='No original held-out'):
        pose_split(names, names, False)


def test_text_overlay_keeps_observation_and_heldout_records(tmp_path):
    source, dest = tmp_path/'source.txt', tmp_path/'dest.txt'
    data = '# comment\n1 1 0 0 0 0 0 0 7 a b.jpg\n1.5 2.5 55\n2 1 0 0 0 0 0 0 7 c.jpg\n4 5 8\n'
    source.write_text(data)
    replace_text(source, dest, {1: [1, 0, 0, 0, .125, 0, 0]})
    assert dest.read_text() == data.replace('0 0 0 7 a b', '0.125 0 0 7 a b')
    assert source.read_text() == data


def test_orange_binary_poses_improve_independent_images(tmp_path, capsys):
    source, root = tmp_path/'scene', tmp_path/'out'
    model = source/'sparse/0'
    model.mkdir(parents=True)
    (source/'images').mkdir()
    suite = root/'stride3_noaa_v1'
    suite.mkdir(parents=True)
    width, height = 400, 300
    fx, fy, cx, cy = 350., 348., 198., 145.
    (model/'cameras.bin').write_bytes(struct.pack('<QiiQQ4d', 1, 1, 1, width, height, fx, fy, cx, cy))
    rng = np.random.default_rng(223)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (800, 1200), np.uint8), (3, 3), .7)
    tex_to_plane = np.array([[.005, 0, -3], [0, .005, -2], [0, 0, 1.]])
    K = np.array([[fx, 0, cx-.5], [0, fy, cy-.5], [0, 0, 1.]])
    payload = bytearray(struct.pack('<Q', 15))
    training, pose_offsets = [], []
    for i in range(15):
        name = f'{i:03d}.png'
        C = np.array([i*.035-.25, .02*np.sin(i), 0.])
        R = Rotation.from_rotvec([0., .005*(i-7), 0.]).as_matrix()
        H = K @ np.column_stack((R[:, 0], R[:, 1], R @ (np.array([0, 0, 3.]) - C)))
        assert cv2.imwrite(str(source/'images'/name), cv2.warpPerspective(texture, H @ tex_to_plane, (width, height)))
        if i % 3 == 1:
            training.append(name)
            C += np.array([.006, .003, -.001]) * (1 if i % 2 else -1)
            R = Rotation.from_rotvec([.0004, .002 * (1 if i % 2 else -1), -.0001]).as_matrix() @ R
            pose_offsets.append(len(payload) + 4)
        q = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
        payload.extend(struct.pack('<i7di', i+1, *q, *(-R @ C), 1) + name.encode() + b'\0')
        payload.extend(struct.pack('<Qddq', 1, 23.5+i, 33.5+i, 123456789))
    (model/'images.bin').write_bytes(payload)
    (model/'points3D.bin').write_bytes(struct.pack('<Q', 0))
    (model/'frame_rows').mkdir()
    (model/'frame_rows/001.txt').write_text('17\n')
    x, y = np.meshgrid(np.arange(-2.6, 2.6, .035), np.arange(-1.8, 1.8, .035))
    verts = np.zeros(x.size, dtype=[(n, 'f4') for n in 'xyz'])
    verts['x'], verts['y'], verts['z'] = x.ravel(), y.ravel(), 3.
    PlyData([PlyElement.describe(verts, 'vertex')]).write(model/'points3D.ply')
    split = suite/'selection.json'
    split.write_text(json.dumps(dict(selected_images=training)))
    (suite/'suite.json').write_text(json.dumps(dict(source=str(source), selection=str(split))))
    argv = ['--output-root', str(root)]
    assert main(argv) == 0
    ba = root/'pose_ba_pilot_v1/ba'
    report = json.loads((ba/'refinement_report.json').read_text())
    views = report['independent_views']
    assert report['accepted'] and views['supported'] == 4
    assert views['median_gain_native_px'] >= .1
    exported = ba/'dataset'
    assert set(dataset_pose_labels(exported)) == set(training)
    sparse = exported/'sparse/0'
    assert (sparse/'points3D.bin').samefile(model/'points3D.bin')
    assert (sparse/'frame_rows/001.txt').samefile(model/'frame_rows/001.txt')
    adjusted = bytearray((sparse/'images.bin').read_bytes())
    assert adjusted != payload
    for start in pose_offsets:
        adjusted[start:start+56] = payload[start:start+56]
    assert adjusted == payload == (model/'images.bin').read_bytes()
    assert read_model(exported)[1][1].params.tolist() == [fx, fy, cx, cy]
    assert main(argv) == 0
    assert 'Reusing validated dataset' in capsys.readouterr().out


def test_no_supported_validation_views_are_inconclusive(monkeypatch):
    from types import SimpleNamespace
    from scripts.keyframe_pipeline import validation

    monkeypatch.setattr(validation, 'extract_features', lambda *args: {})
    im = SimpleNamespace(id=1, name="heldout.jpg", camera_id=1)
    result = validation.check_views(None, None, {1: None}, [im], [], [], None, None, None,
                                    [], [], [], SimpleNamespace())
    assert result["supported"] == 0 and result["median_gain_native_px"] is None
    assert "Insufficient independent photo coverage" in result["rejection_reasons"][0]
