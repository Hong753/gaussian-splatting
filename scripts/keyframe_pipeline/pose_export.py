"""Publish camera pose changes while retaining the input's COLMAP/LiDAR associations."""
from pathlib import Path
import shutil
import struct
import tempfile

from scipy.spatial.transform import Rotation

from .common import atomic_json, read_exact


def replace_binary(source, destination, updates):
    # Copy even huge observation arrays in bounded memory. Patch only the seven
    # doubles in an updated image's header, never the source file or point IDs.
    shutil.copyfile(source, destination)
    with destination.open('r+b') as stream:
        for _ in range(read_exact(stream, '<Q')[0]):
            iid = read_exact(stream, '<i')[0]
            if iid in updates:
                stream.write(struct.pack('<7d', *updates[iid]))
            else:
                stream.seek(56, 1)
            stream.seek(4, 1)
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError('Truncated COLMAP image name')
                if char == b'\0':
                    break
            stream.seek(24 * read_exact(stream, '<Q')[0], 1)


def replace_text(source, destination, updates):
    with source.open() as src, destination.open('w') as dst:
        for line in src:
            if not line.strip() or line.lstrip().startswith('#'):
                dst.write(line)
                continue
            fields = line.strip().split(maxsplit=9)
            iid = int(fields[0])
            if iid in updates:
                fields[1:8] = [format(float(v), '.17g') for v in updates[iid]]
                line = ' '.join(fields) + '\n'
            dst.write(line)
            dst.write(next(src))


def export_poses(output, source, model, cloud, selected, rotations, centers, report, manifest):
    updates = {im.id: [*Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]], *(-R @ C)]
               for im, R, C, row in zip(selected, rotations, centers, report["per_image"], strict=True)
               if row["optimized"]}
    binary = (model / 'images.bin').is_file() and (model / 'cameras.bin').is_file()
    ext = 'bin' if binary else 'txt'
    destination = output / 'dataset'
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp) / 'dataset'
        sparse = stage / model.relative_to(source)
        sparse.mkdir(parents=True)
        # Mirror ancestors of the chosen sparse model, keeping all sidecars
        # (including per-view row indices) and all unaffected scene assets.
        parent = source
        while True:
            asset_dir = stage / parent.relative_to(source)
            for item in parent.iterdir():
                link = asset_dir / item.name
                if link.exists() or (parent == source and item.name in ('keyframes.json', 'image_list.txt', 'lidar_pose_ba.json')):
                    continue
                if parent == model and item.name in ('images.txt', 'images.bin', 'points3D.ply'):
                    continue
                link.symlink_to(item.resolve())
            if parent == model:
                break
            parent = parent / model.relative_to(parent).parts[0]
        name = 'images.' + ext
        (replace_binary if binary else replace_text)(model / name, sparse / name, updates)
        original_ply = model / 'points3D.ply'
        (sparse / 'points3D.ply').symlink_to(original_ply.resolve() if original_ply.is_file() else cloud.resolve())
        (stage / 'lidar_pose_ba.json').symlink_to(output / 'refinement_report.json')
        atomic_json(stage / 'keyframes.json', {**manifest, 'source_path': str(destination),
                    'pose_refinement': str(output / 'refinement_report.json')})
        (stage / 'image_list.txt').write_text(''.join(im.name + '\n' for im in selected))
        stage.rename(destination)
    return destination
