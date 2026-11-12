"""Explicit, checkpoint-specific pose policy for image rendering/evaluation."""
import json
from pathlib import Path


def add_arguments(parser):
    parser.add_argument('--camera-poses',choices=('original','refined'),default='original',
        help='Original COLMAP poses, or cached training-view corrections from the selected checkpoint; held-out poses stay original')


def apply_overrides(spec,args):
    mode=getattr(args,'camera_poses','original')
    if mode=='refined' and (spec['type']!='3dgs' or not spec['cfg'].get('pose_refinement')):
        raise ValueError('Refined camera poses require a pose-refinement 3DGS checkpoint')
    spec['camera_poses']=mode
    if mode=='refined':
        spec['render_variant']='_'.join(filter(None,(spec.get('render_variant',''),'poses_refined')))
    return spec


def worker_arguments(args):
    return ['--camera-poses',args.camera_poses] if getattr(args,'camera_poses','original')!='original' else []


def load_cache(spec,iteration):
    if spec.get('camera_poses','original')=='original': return None
    import numpy as np
    from utils.pose_refinement import source_fingerprints
    from utils.checkpoint_snapshots import digest
    path=Path(spec['path'])/f'point_cloud/iteration_{iteration}/poses.json'
    data=json.loads(path.read_text())
    selected=set(json.loads((Path(spec['path'])/'keyframes.json').read_text())['selected_images'])
    source=str(Path(spec['cfg']['source_path']).resolve())
    if (data.get('schema')!=1 or data['iteration']!=iteration or data['source']!=source or
            data.get('policy')!=spec['cfg'].get('pose_refinement') or
            data['source_fingerprints']!=source_fingerprints(source) or
            set(data['training_images'])!=selected or set(data['frames'])!=selected):
        raise ValueError('Pose cache differs from this checkpoint, source calibration, or training split')
    for name,frame in data['frames'].items():
        for key in ('original_world_to_camera','world_to_camera'):
            matrix=np.asarray(frame[key])
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]):
                raise ValueError(f'Invalid pose transform for {name}')
        for key,limit in (('translation_offset',data['policy']['max_translation']),
                          ('rotation_axis_angle',np.deg2rad(data['policy']['max_rotation_degrees']))):
            value=np.asarray(frame[key])
            if value.shape!=(3,) or not np.isfinite(value).all() or np.linalg.norm(value)>limit+1e-7:
                raise ValueError(f'Pose cache exceeds configured offset bounds for {name}')
        from scipy.spatial.transform import Rotation
        base=np.asarray(frame['original_world_to_camera'])
        center=-base[:3,:3].T@base[:3,3]+base[:3,:3].T@np.asarray(frame['translation_offset'])
        R=Rotation.from_rotvec(frame['rotation_axis_angle']).as_matrix()@base[:3,:3]
        expected=np.eye(4);expected[:3,:3]=R;expected[:3,3]=-R@center
        if not np.allclose(expected,frame['world_to_camera'],atol=1e-5,rtol=1e-5):
            raise ValueError(f'Pose matrix does not match its bounded offsets: {name}')
    data['_path']=str(path);data['_sha256']=digest(path)
    return data


def apply_camera(camera,cache):
    if cache is None or camera.image_name not in cache['frames']: return camera
    import torch
    from utils.pose_refinement import camera_with_transform
    frame=cache['frames'][camera.image_name]
    base=camera.world_view_transform.T
    original=torch.as_tensor(frame['original_world_to_camera'],device=base.device,dtype=base.dtype)
    if not torch.allclose(base,original,atol=1e-5,rtol=1e-5):
        raise ValueError(f'Original camera no longer matches pose cache: {camera.image_name}')
    refined=torch.as_tensor(frame['world_to_camera'],device=base.device,dtype=base.dtype)
    return camera_with_transform(camera,refined)


def protocol(spec):
    return ('cached training-view corrections; unselected frames retain original COLMAP poses'
            if spec.get('camera_poses','original')=='refined' else 'original COLMAP poses for all frames')
