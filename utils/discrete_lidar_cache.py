"""Reusable CPU kNN preprocessing for a finite set of measured coordinates.

Only this preprocessing uses scipy. Training gathers candidate neighbors and
changes the active population with torch on the training device. No interpolated
surface coordinates are introduced. metadata.json is the completion marker.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_metadata(directory, verify=True):
    directory = Path(directory)
    data = json.loads((directory/'metadata.json').read_text())
    if data.get('schema') != 1:
        raise ValueError('Unknown discrete LiDAR cache schema')
    identity = hashlib.sha256(json.dumps(data['files'], sort_keys=True).encode()).hexdigest()
    if data['cache_id'] != identity:
        raise ValueError('Invalid discrete LiDAR cache identity')
    if verify:
        for name, sha in data['files'].items():
            if digest(directory/name) != sha:
                raise ValueError(f'Discrete LiDAR cache changed: {name}')
    return data


def prepare(point_cloud, directory, *, fraction=.5, voxel=.05, neighbors=16,
            seed=0, workers=8, chunk=131072):
    """Keep one representative per voxel, then fill to the requested fraction.

    Exact duplicate coordinates are represented by their first input row. The
    initial population may exceed fraction*N to preserve voxel coverage.
    """
    import fcntl
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    if not (0 < fraction <= 1 and math.isfinite(voxel) and voxel > 0 and
            3 <= neighbors <= 128 and workers >= 1 and chunk >= 1 and seed >= 0):
        raise ValueError('Invalid discrete LiDAR cache parameters')
    point_cloud, directory = Path(point_cloud).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = dict(source_sha256=digest(point_cloud), fraction=fraction, voxel=voxel,
                      neighbors=neighbors, seed=seed)
        if (directory/'metadata.json').is_file():
            old = read_metadata(directory)
            if old['config'] != config:
                raise ValueError('Existing discrete LiDAR cache has different inputs/settings; choose a new cache directory')
            print(f'Reusing discrete LiDAR candidates: {directory}', flush=True)
            return old
        vertices = PlyData.read(str(point_cloud), mmap='r')['vertex']
        original = np.column_stack([vertices[key] for key in ('x', 'y', 'z')]).astype(np.float32)
        if len(original) < 2 or not np.isfinite(original).all():
            raise ValueError('Need at least two finite LiDAR coordinates')
        # Preserve original file order while removing exact positional duplicates.
        source_ids = np.sort(np.unique(original, axis=0, return_index=True)[1]).astype(np.int64)
        xyz = original[source_ids]
        original_count = len(original)
        del original, vertices
        count = len(xyz)
        if count < 2 or count >= np.iinfo(np.int32).max:
            raise ValueError('Discrete pool must contain 2..2^31-2 unique coordinates')
        print(f'LiDAR pool: {count:,} unique / {original_count:,} input coordinates; building {neighbors}-NN cache', flush=True)
        cells = np.floor((xyz.astype(np.float64)-xyz.min(axis=0))/voxel).astype(np.int64)
        representatives = np.unique(cells, axis=0, return_index=True)[1]
        del cells
        chosen = np.zeros(count, dtype=bool)
        chosen[representatives] = True
        target = max(len(representatives), math.ceil(count*fraction))
        if target > len(representatives):
            unused = np.flatnonzero(~chosen)
            chosen[np.random.default_rng(seed).choice(unused, target-len(representatives), replace=False)] = True
        initial_ids = np.flatnonzero(chosen).astype(np.int64)
        del chosen, representatives
        for name, array in (('xyz', xyz), ('source_ids', source_ids), ('initial_ids', initial_ids)):
            np.save(directory/(name+'.npy'), array, allow_pickle=False)
        tree = cKDTree(xyz)
        table = np.lib.format.open_memmap(directory/'neighbors.npy', mode='w+', dtype=np.int32,
                                         shape=(count, neighbors))
        spacing = np.lib.format.open_memmap(directory/'sigma.npy', mode='w+', dtype=np.float32, shape=(count,))
        for start in range(0, count, chunk):
            stop = min(count, start+chunk)
            distance, indices = tree.query(xyz[start:stop], k=min(neighbors+1, count), workers=workers)
            # Unique positions guarantee that column zero is this point itself.
            if not np.array_equal(indices[:, 0], np.arange(start, stop)):
                raise RuntimeError('kNN self-index mismatch')
            table[start:stop] = -1
            table[start:stop, :indices.shape[1]-1] = indices[:, 1:]
            spacing[start:stop] = np.sqrt(np.maximum(np.mean(distance[:, 1:4]**2, axis=1), 1e-7))
            if start == 0 or stop == count or (start//chunk) % 20 == 0:
                print(f'LiDAR neighbors cached: {stop:,}/{count:,}', flush=True)
        table.flush(); spacing.flush()
        del table, spacing, tree
        files = {name: digest(directory/name) for name in
                 ('xyz.npy', 'source_ids.npy', 'initial_ids.npy', 'neighbors.npy', 'sigma.npy')}
        data = dict(schema=1, config=config, original_count=original_count, count=count,
                    initial_count=len(initial_ids), files=files,
                    cache_id=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
        temporary = directory/'metadata.json.tmp'
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(directory/'metadata.json')
        return data


def load(directory, device):
    import torch
    directory = Path(directory).resolve()
    meta = read_metadata(directory)
    # Copy each mmap once to the training device; no second full CPU pool is kept.
    pool = {'metadata': meta, 'directory': str(directory)}
    for key in ('xyz', 'neighbors', 'sigma', 'source_ids', 'initial_ids'):
        array = np.load(directory/(key+'.npy'), mmap_mode='r', allow_pickle=False)
        pool[key] = torch.tensor(array, device=device)
    return pool
