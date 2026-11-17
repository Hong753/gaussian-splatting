"""CPU-only, reproducible LiDAR-region selection and COLMAP subset export.

The coarse point z-buffer is for selecting supervision pixels, NOT a training
occluder. Inspect the exported RGB overlays before training. All selected LiDAR
vertices (not the display sample) are retained in input.ply, in original units.
"""
import hashlib
import html
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image as PILImage
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_closing, binary_erosion
from scipy.spatial.transform import Rotation

from scripts.keyframe_pipeline.common import read_model, image_path, project, file_stamp
from utils.region_loss import mask_path


def inside(points, center, size, padding=0):
    return (np.abs(points-np.asarray(center)) <= np.asarray(size)/2+padding).all(axis=1)


def vertices_xyz(vertices):
    return np.column_stack([vertices[a] for a in 'xyz'])


def validate_box(center, size, halo):
    if (np.shape(center) != (3,) or np.shape(size) != (3,) or
            not np.isfinite([*center, *size, halo]).all() or min(size) <= 0 or halo < 0):
        raise ValueError('Need a finite XYZ center, positive XYZ box size, and nonnegative halo')


def foreground_mask(points, image, camera, center, size, max_side=256):
    """Frontmost measured points at low resolution; no convex hull over holes."""
    scale = min(1., max_side/max(camera.width, camera.height))
    w, h = max(1, round(camera.width*scale)), max(1, round(camera.height*scale))
    K = camera.params * np.array([w/camera.width, h/camera.height]*2)
    uv, z = project(points, image.R, image.C, K)
    good = np.isfinite(uv).all(1) & (z > .05) & (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
    ids = np.flatnonzero(good)
    pixels = np.floor(uv[ids]).astype(np.int64)
    flat = pixels[:,1]*w+pixels[:,0]
    front = np.full(h*w, np.inf)
    np.minimum.at(front, flat, z[ids])
    foreground = np.isclose(z[ids], front[flat], rtol=0, atol=1e-7)
    hit = np.isfinite(front).reshape(h,w)
    core = np.zeros(h*w, dtype=bool)
    chosen = foreground & inside(points[ids], center, size)
    core[flat[chosen]] = True
    core = core.reshape(h,w)
    # Fill only isolated sampling gaps, never measured non-core foreground.
    mask = binary_closing(core, structure=np.ones((3,3))) & (~hit | core)
    mask = binary_erosion(mask, iterations=1, border_value=0)
    return mask, {'overlap': float(mask.mean()), 'measured_coverage': float(hit.mean())}


class RegionData:
    def __init__(self, source, point_cloud, keyframes, max_points=1000000):
        self.source, self.point_cloud, self.keyframes = [Path(p).expanduser().resolve() for p in (source, point_cloud, keyframes)]
        _, self.cameras, self.images, self.pose_stamps = read_model(self.source)
        self.manifest = json.loads(self.keyframes.read_text())
        self.folder = self.manifest.get('image_directory', 'images')
        if self.folder != 'images':
            raise ValueError('Regional export currently requires the canonical images/ directory')
        selected = self.manifest['selected_images']
        names = {im.name for im in self.images}
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= names:
            raise ValueError('Invalid original keyframe manifest')
        self.train_names = set(selected)
        self.vertices = PlyData.read(str(self.point_cloud), mmap='r')['vertex'].data
        fields = set(self.vertices.dtype.names)
        if not {'x','y','z','red','green','blue'} <= fields or {'scale_0','f_dc_0'} & fields:
            raise ValueError('Pass original colored LiDAR input.ply, not learned Gaussian PLY')
        if len(self.vertices) < 4 or max_points < 4:
            raise ValueError('Need at least four LiDAR points')
        # Sampling affects the preview/mask only. Export streams ALL input rows.
        rng = np.random.default_rng(0)
        sample = np.sort(rng.choice(len(self.vertices), min(len(self.vertices), max_points), replace=False))
        v = self.vertices[sample]
        self.points = vertices_xyz(v)
        good = np.isfinite(self.points).all(1)
        self.points = self.points[good]
        self.colors = np.column_stack([v[a] for a in ('red','green','blue')])[good].astype(np.uint8)
        self.sample_count = len(self.points)
        self.max_points = max_points

    def anchor_center(self, name):
        im = next(im for im in self.images if im.name == name)
        camera = self.cameras[im.camera_id]
        uv, z = project(self.points, im.R, im.C, camera.params)
        central = (z > .2) & (np.abs(uv[:,0]-camera.width/2) < .06*camera.width) & (np.abs(uv[:,1]-camera.height/2) < .06*camera.height)
        ids = np.flatnonzero(central)
        if not len(ids):
            raise ValueError('Anchor image has no central LiDAR returns; specify --center or choose another anchor')
        # Near depth quantile avoids seeding the box behind the nearest wall.
        near = ids[z[ids] <= np.quantile(z[ids], .15)]
        return np.median(self.points[near], axis=0)

    def find(self, center, size, halo=1., min_overlap=.02, max_camera_distance=8.,
             frame_start='', frame_end='', exclude=(), progress=None):
        validate_box(center, size, halo)
        if not 0 < min_overlap <= 1 or max_camera_distance < 0:
            raise ValueError('Invalid overlap or camera distance')
        if frame_start and frame_end and frame_start > frame_end:
            raise ValueError('Frame start must not follow frame end')
        rows = []
        for i, im in enumerate(self.images):
            if im.name in exclude or (frame_start and im.name < frame_start) or (frame_end and im.name > frame_end):
                continue
            distance = np.linalg.norm(np.maximum(np.abs(im.C-np.asarray(center))-np.asarray(size)/2, 0))
            if max_camera_distance and distance > max_camera_distance:
                continue
            mask, info = foreground_mask(self.points, im, self.cameras[im.camera_id], center, size)
            if info['overlap'] >= min_overlap and binary_erosion(mask, iterations=5).any():
                rows.append(dict(name=im.name, split='train' if im.name in self.train_names else 'eval',
                                 camera_distance=float(distance), **info))
            if progress and i % 25 == 0:
                progress(i+1, len(self.images), len(rows))
        if not any(r['split'] == 'train' for r in rows):
            raise ValueError('No training keyframes overlap this box; enlarge it or reduce overlap threshold')
        return rows

    def overlay(self, name, center, size):
        im = next(im for im in self.images if im.name == name)
        camera = self.cameras[im.camera_id]
        mask, info = foreground_mask(self.points, im, camera, center, size)
        with PILImage.open(image_path(self.source, self.folder, name)) as raw:
            if not np.isclose(raw.width/raw.height,camera.width/camera.height,rtol=.001):
                raise ValueError(f'{name}: RGB aspect ratio {raw.size} differs from calibration {(camera.width,camera.height)}')
            info['rgb_size']=list(raw.size)
            info['calibration_size']=[camera.width,camera.height]
            rgb = np.asarray(raw.convert('RGB').resize((mask.shape[1], mask.shape[0]), PILImage.Resampling.LANCZOS))
        overlay = (rgb*(.2+.8*mask[...,None])).astype(np.uint8)
        return mask, overlay, info

    def export(self, output, center, size, rows, **settings):
        output = Path(output).expanduser().resolve()
        validate_box(center, size, settings.get('halo', 1.))
        if output.exists():
            raise FileExistsError(f'Choose a fresh export directory: {output}')
        # Reject stale UI results if the box/filter changed after selection.
        verified = self.find(center, size, **settings)
        if rows != verified:
            raise ValueError('Selection changed; rerun Find overlapping views before export')
        names = {r['name'] for r in rows}
        selected = [r['name'] for r in rows if r['split'] == 'train']
        if len(selected) < 2:
            raise ValueError('Need at least two training views')
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = output.parent / ('.'+output.name+'.'+uuid.uuid4().hex+'.tmp')
        stage.mkdir()
        try:
            sparse = stage/'sparse/0'
            sparse.mkdir(parents=True)
            (stage/'masks').mkdir()
            (stage/'overlays').mkdir()
            images = [im for im in self.images if im.name in names]
            camera_ids = {im.camera_id for im in images}
            camera_rows, pose_rows, poses = [], [], []
            image_audit = {}
            calibration_audit = []
            for cid in sorted(camera_ids):
                cam = self.cameras[cid]
                camera_rows.append(f'{cid} PINHOLE {cam.width} {cam.height} '+' '.join(format(p,'.17g') for p in cam.params)+'\n')
                calibration_audit.append(dict(camera_id=cid, principal_point_offset_pixels=[float(cam.params[2]-cam.width/2),float(cam.params[3]-cam.height/2)]))
            for image_index, im in enumerate(images):
                q = Rotation.from_matrix(im.R).as_quat()[[3,0,1,2]]
                t = -im.R@im.C
                pose_rows.append(f'{im.id} '+' '.join(format(p,'.17g') for p in [*q,*t])+f' {im.camera_id} {im.name}\n\n')
                poses.append(dict(name=im.name, camera_id=im.camera_id, R_world_to_camera=im.R.tolist(),
                                  t_world_to_camera=t.tolist(), center_world=im.C.tolist()))
                destination = stage/'images'/im.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(image_path(self.source, self.folder, im.name).resolve())
                mask, overlay, image_info = self.overlay(im.name, center, size)
                image_audit[im.name]=image_info
                PILImage.fromarray(mask.astype(np.uint8)*255).save(mask_path(stage/'masks', im.name))
                PILImage.fromarray(overlay).save(mask_path(stage/'overlays', im.name))
                if image_index % 25 == 0:
                    print(f'Exporting images/masks: {image_index+1}/{len(images)}',flush=True)
            (sparse/'cameras.txt').write_text(''.join(camera_rows))
            (sparse/'images.txt').write_text(''.join(pose_rows))
            # LiDAR initialization is independent of COLMAP feature tracks.
            (sparse/'points3D.txt').write_text('# No feature tracks; initialize from the exported input.ply.\n')
            chunks = []
            for first in range(0, len(self.vertices), 500000):
                xyz = vertices_xyz(self.vertices[first:first+500000])
                good = np.isfinite(xyz).all(1) & inside(xyz, center, size, settings.get('halo',1.))
                chunks.append(first+np.flatnonzero(good))
            ids = np.concatenate(chunks)
            if len(ids) < 4:
                raise ValueError('Region contains fewer than four LiDAR points')
            vertices = self.vertices[ids]
            # Original input.ply includes normal fields; raw PLY may not.
            missing = [n for n in ('nx','ny','nz') if n not in vertices.dtype.names]
            if missing:
                extended = np.zeros(len(vertices), dtype=vertices.dtype.descr+[(n,'f4') for n in missing])
                for field in vertices.dtype.names:
                    extended[field] = vertices[field]
                vertices = extended
            PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(stage/'input.ply'))
            (sparse/'points3D.ply').symlink_to('../../input.ply')
            np.save(stage/'original_lidar_indices.npy', ids)
            # Recompute mask hashes at export: shared by both regional controls.
            mask_hashes = {r['name']: hashlib.sha256(mask_path(stage/'masks',r['name']).read_bytes()).hexdigest() for r in rows}
            info = dict(schema=1, source=str(self.source), original_point_cloud=str(self.point_cloud),
                        original_keyframes=str(self.keyframes), original_train_count=len(self.train_names),
                        original_frame_count=len(self.images), original_lidar_count=len(self.vertices),
                        region=str(output), center=list(map(float,center)), size=list(map(float,size)),
                        settings=settings, lidar_count=len(ids), sample_points=self.sample_count,
                        mask_max_side=256, selection=rows, mask_sha256=mask_hashes,
                        inputs=[*self.pose_stamps,file_stamp(self.point_cloud),file_stamp(self.keyframes)],
                        calibration_audit=calibration_audit,
                        image_audit=image_audit,
                        protocol='Original world coordinates and calibration; all crop+halo LiDAR rows; original training manifest intersection; ROI-only loss and PSNR; no culling.',
                        mask_caveat='Approximate nearest measured LiDAR at 256px; inspect overlays for holes/occlusion/alignment. Not dense ground-truth visibility.')
            for filename, data in [('selection.json',info),('poses.json',poses),('keyframes.json',dict(selected_images=selected,source_path=str(output),image_directory='images',eval=False))]:
                (stage/filename).write_text(json.dumps(data,indent=2)+'\n')
            cards = ''.join('<figure><img loading="lazy" src="overlays/'+mask_path('',r['name']).name+'"><figcaption>'+html.escape(r['name'])+f" · {r['split']} · ROI {r['overlap']:.1%}</figcaption></figure>" for r in rows)
            (stage/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{background:#17202b;color:white;font:14px sans-serif}main{display:flex;flex-wrap:wrap}figure{margin:6px}img{width:320px}</style><h1>Regional supervision preview</h1><p>Bright pixels are supervised. Dim pixels are excluded from loss/PSNR. Original camera poses; no ray culling.</p><p>'+f'{len(selected)} train, {len(rows)-len(selected)} held-out; {len(ids):,} LiDAR vertices including halo.</p><main>'+cards+'</main>')
            stage.rename(output)
        except BaseException:
            shutil.rmtree(stage)
            raise
        print(f'Exported {len(rows)} views ({len(selected)} train), {len(ids):,} LiDAR points: {output}',flush=True)
        return info
