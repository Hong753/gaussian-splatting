#!/usr/bin/env python3
"""Inspect capture metadata for measured image-to-LiDAR associations (CPU only)."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path


def inspect(source, selection=None):
    source = Path(source).expanduser().resolve(strict=True)
    groups = defaultdict(list)
    counts = defaultdict(int)
    recognized = []
    for parent, dirs, files in os.walk(source):
        root = Path(parent)
        dirs[:] = [d for d in dirs if d not in {'.git','images','masks','debug_previews'}]
        if len(root.relative_to(source).parts) >= 5:
            dirs.clear()
        for name in sorted(files):
            p = root/name
            suffix = p.suffix.lower()
            if suffix not in {'.npy','.npz','.bag','.mcap','.pcap','.las','.laz','.ply','.pcd','.csv','.json','.bin','.txt','.db3','.e57'}:
                continue
            key = str(p.parent.relative_to(source)/('*'+suffix))
            counts[key] += 1
            if len(groups[key]) < 2:
                groups[key].append(name)
            if name in ('lidar_scans.json','lidar_observations.json'):
                data = json.loads(p.read_text())
                kind = data.get('format')
                if kind == 'lidar_rows_v1':
                    names = set(data.get('frames', {}))
                elif kind == 'scanner_xyz_v1':
                    names = {n for s in data.get('scans',[]) for n in s.get('images', [s.get('image')]) if n}
                else:
                    names = set()
                row = dict(manifest=str(p), format=kind, associated_image_names=len(names))
                if selection is not None:
                    row.update(matched_training_views=len(selection & names), total_training_views=len(selection))
                recognized.append(row)
    print('Association manifests:', json.dumps(recognized, indent=2))
    for k in sorted(counts):
        print(f'{k}: {counts[k]:,} files, e.g. {", ".join(groups[k])}')
        for name in groups[k][:1]:
            p = source/Path(k).parent/name
            if p.suffix == '.ply':
                with p.open('rb') as f:
                    for _ in range(100):
                        line = f.readline(1024).decode('ascii', errors='replace').strip()
                        if line.startswith(('property ', 'element ')):
                            print('  '+line)
                        if line == 'end_header': break
            elif p.suffix == '.npz':
                import numpy as np
                with np.load(p, allow_pickle=False) as data:
                    print('  arrays:', ', '.join(data.files[:25]))
            elif p.suffix == '.csv':
                with p.open(errors='replace') as f:
                    print('  first line:', f.readline(250).strip())
    if not recognized or not any(r['associated_image_names'] for r in recognized):
        print('No explicit image-to-return mapping recognized. Inspect the listed acquisition files for frame IDs, '
              'timestamps and point-row provenance. COLMAP tracks and mesh visible_ids alone do not establish scanner hits.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_suite = Path(os.environ.get('ORANGE4F_OUTPUT_ROOT','/data/3dgs_outputs/orange4f/3dgs_finetuning'))/'stride3_noaa_v1/suite.json'
    p.add_argument('--source', type=Path, help='Original capture or raw-data directory')
    p.add_argument('--suite', type=Path, default=default_suite, help='Optional prepared suite.json')
    args = p.parse_args()
    suite = json.loads(args.suite.read_text()) if args.suite.is_file() else {}
    source = args.source or Path(suite.get('source','/workspace/colmap_scenes/orange4f'))
    selection = set(json.loads(Path(suite['selection']).read_text())['selected_images']) if suite else None
    inspect(source, selection)


if __name__ == '__main__':
    main()
