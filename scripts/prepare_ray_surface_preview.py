#!/usr/bin/env python3
"""Validate every view in one cached trajectory group and fuse a diagnostic mesh.

No new DA3 inference. The group contains broad trajectory context and nearby
companions. Original training caches and checkpoints remain unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_da3_occlusion import save_npz
from scripts.view_comparisons import atomic_json, read_colmap_records
from utils.da3_occlusion import reliable_pixels, camera_matrices, training_tiles
from utils.da3_trajectory import calibrate_group, validate_from_donors
from utils.occlusion_depth import record_camera, camera_signature
from utils.sparse_lidar import frame_filename
from utils.lidar_occlusion import file_digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--depth-dir', type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--point-cloud', type=Path, required=True)
    args = p.parse_args(argv)
    root, output = args.depth_dir.resolve(), args.output.resolve()
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata.get('view_grouping') != 'trajectory_v1':
        raise ValueError('Run trajectory preparation for this image first')
    if file_digest(args.point_cloud) != metadata['point_cloud_sha256']:
        raise ValueError('The preview cloud differs from the registered calibration cloud')
    groups = json.loads((root/'groups.json').read_text())['groups']
    group = next((g for g in groups if args.image in g['targets']), None)
    if group is None:
        raise ValueError('Image not in trajectory grouping')
    path = root/'predictions'/f"group_{group['id']:05d}.npz"
    identity = dict(group_id=group['id'], prediction_sha256=file_digest(path),
        source_metadata_sha256=file_digest(root/'metadata.json'), image=args.image)
    # Metadata completion may change as other groups finish; input identity below
    # is instead pinned by the raw group plus calibration settings and cloud hash.
    identity.pop('source_metadata_sha256')
    preview_meta = {**metadata, 'images':group['images'], 'complete':False,
                    'diagnostic_group':identity, 'diagnostic_only':True}
    records = read_colmap_records(Path(metadata['source']))
    cameras = {n:record_camera(records[n],metadata['resolution'],device='cpu') for n in group['images']}
    preview_meta['camera_signatures'] = {n:camera_signature(c,metadata['edge_pixels']) for n,c in cameras.items()}
    if any(metadata['camera_signatures'].get(n)!=signature for n,signature in preview_meta['camera_signatures'].items()):
        raise ValueError('Camera poses/intrinsics changed since the cached group inference')
    depth_output = output/'depth'
    if (depth_output/'metadata.json').is_file():
        previous = json.loads((depth_output/'metadata.json').read_text())
        keys=('diagnostic_group','point_cloud_sha256','calibration_policy','max_relative_error','confidence_percentile',
              'min_anchors','multiview_tolerance','min_support_views','relative_margin','edge_pixels','camera_signatures')
        if any(previous.get(k)!=preview_meta.get(k) for k in keys):
            raise ValueError('Preview identity changed; choose a fresh output directory')
        if (depth_output/'calibration.json').is_file() and all((depth_output/'frames'/frame_filename(n)).is_file() for n in group['images']):
            print(f'Reusing validated group: {depth_output}',flush=True)
            return 0
    with np.load(path,allow_pickle=False) as raw:
        if raw['images'].tolist()!=group['images']:
            raise ValueError('Cached group identity mismatch')
        prepared=[]
        for i,n in enumerate(group['images']):
            camera=cameras[n]
            full_k,e=camera_matrices(records[n],camera.image_width,camera.image_height)
            k=raw['intrinsics'][i]
            with np.load(root/'anchors'/frame_filename(n),allow_pickle=False) as anchors:
                u,v,z=anchors['u'],anchors['v'],anchors['z']
            pu=np.floor(k[0,0]*(u-full_k[0,2])/full_k[0,0]+k[0,2]+.5).astype(int)
            pv=np.floor(k[1,1]*(v-full_k[1,2])/full_k[1,1]+k[1,2]+.5).astype(int)
            depth,confidence=raw['depth'][i],raw['confidence'][i]
            prepared.append(dict(image=n,raw=depth,confidence=confidence,
                confident=reliable_pixels(depth,confidence,metadata['confidence_percentile']),
                intrinsic=k,extrinsic=e,anchors=(pu,pv,z)))
        calibrated=calibrate_group(prepared,metadata['min_anchors'],metadata['max_relative_error'])
    atomic_json(depth_output/'metadata.json',preview_meta)
    rows=[]
    for i,frame in enumerate(calibrated):
        n=frame['image']
        trusted,support,donors=validate_from_donors(frame,[f for j,f in enumerate(calibrated) if i!=j],
            metadata['multiview_tolerance'],metadata['min_support_views'])
        margin=max(metadata['relative_margin'],frame['report'].get('p90_inlier_relative_error') or 0.)
        if (trusted & ~frame['local']).any():
            margin=max(margin,metadata['max_relative_error'])
        tiles=training_tiles(frame['depth'],trusted,frame['intrinsic'],cameras[n],margin,metadata['edge_pixels'])
        report=dict(frame['report'],image=n,accepted=bool(trusted.any()),donor_images=donors,
            relative_margin=margin,trusted_pixel_fraction=float(trusted.mean()),
            reliable_tile_fraction=float(np.isfinite(tiles).mean()))
        save_npz(depth_output/'frames'/frame_filename(n),image=n,depth=frame['depth'],valid=trusted,
            intrinsics=frame['intrinsic'],extrinsics=frame['extrinsic'],tiles=tiles,report=json.dumps(report),
            calibration_valid=frame['local'],confidence_valid=frame['confident'],support_views=support)
        rows.append(report)
        print(f"{n}: trusted surface {trusted.mean():.1%}",flush=True)
    atomic_json(depth_output/'calibration.json',dict(frames=rows))
    print(f"Surface preview inputs: {len(rows)} posed views -> {depth_output}",flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
