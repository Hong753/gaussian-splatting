"""Publish complete training snapshots before a concurrent reader can see them.

The ready marker is the commit point. A PLY on its own is never an evaluation job.
This module is CPU-safe; the saver imports torch only when saving optimizer state.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, lambda temp: temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'))


def publish_snapshot(scene, iteration):
    """Called after the optimizer step, with training paused until serialization ends."""
    import torch
    model = Path(scene.model_path).resolve()
    folder = model / f'point_cloud/iteration_{iteration}'
    marker = folder / 'ready.json'
    # Refuse to mutate an already published checkpoint while a worker may read it.
    if marker.exists():
        raise FileExistsError(f'Published snapshot already exists: {marker}')
    scene.save(iteration)
    checkpoint = model / f'chkpnt{iteration}.pth'
    atomic_write(checkpoint, lambda temp: torch.save((scene.gaussians.capture(), iteration), temp))
    files = {}
    for path in (folder/'point_cloud.ply', model/'cfg_args', model/'keyframes.json', checkpoint):
        files[str(path.relative_to(model))] = dict(size=path.stat().st_size, sha256=digest(path))
    result = dict(schema=1, snapshot_id=uuid.uuid4().hex, model=str(model), iteration=iteration,
                  state_timing='after_optimizer_step', gaussians=len(scene.gaussians.get_xyz), files=files)
    activation=(getattr(scene.gaussians,'fixed_lidar_state',None) or {}).get('lidar_activation')
    if activation is not None:
        result['allocated_gaussians']=result['gaussians']
        result['gaussians']=int(activation['active'].sum())
    if getattr(scene.gaussians,'pose_refiner',None) is not None:
        path=folder/'poses.json'
        relative=str(path.relative_to(model))
        files[relative]=dict(size=path.stat().st_size,sha256=digest(path))
        result['pose_cache']=relative
    atomic_json(marker, result)
    return result


def read_snapshot(model, iteration, *, verify=True):
    model = Path(model).resolve()
    marker = model / f'point_cloud/iteration_{iteration}/ready.json'
    if not marker.is_file():
        return None
    result = json.loads(marker.read_text())
    required = {f'point_cloud/iteration_{iteration}/point_cloud.ply', 'cfg_args',
                'keyframes.json', f'chkpnt{iteration}.pth'}
    if 'pose_cache' in result:
        relative=f'point_cloud/iteration_{iteration}/poses.json'
        if result['pose_cache']!=relative:
            raise ValueError('Pose cache must belong to this exact checkpoint')
        required.add(relative)
    if (result.get('schema') != 1 or result.get('iteration') != iteration or result.get('model') != str(model)
            or result.get('state_timing') != 'after_optimizer_step' or not result.get('snapshot_id')
            or set(result.get('files', {})) != required):
        raise ValueError(f'Invalid checkpoint publication: {marker}')
    for relative, identity in result['files'].items():
        path = model/relative
        if not path.is_file() or path.stat().st_size != identity['size']:
            raise ValueError(f'Incomplete or changed checkpoint file: {path}')
        # The evaluator never loads the optimizer state; size is sufficient here.
        if verify and not relative.endswith('.pth') and digest(path) != identity['sha256']:
            raise ValueError(f'Checkpoint fingerprint changed: {path}')
    return result
