#!/usr/bin/env python3
"""Pilot or full-scene LiDAR-anchored pose refinement on the Orange4F stride-3 split."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.keyframes import natural_key, registered_names, select_keyframes
from scripts.keyframe_pipeline.common import atomic_json
from scripts.keyframe_pipeline.refinement import main as refine
from utils.colmap_paths import colmap_model_dir


def pose_split(registered, train, full):
    registered = sorted(registered, key=natural_key)
    train_set = set(train)
    if len(train) != len(train_set) or train_set - set(registered):
        raise ValueError("Stride-3 training names do not match this scene")
    ordered = [n for n in registered if n in train_set]
    if len(ordered) < 3:
        raise ValueError("Need at least three training views")
    positions = {n: i for i, n in enumerate(registered)}
    selected, checks = [], []
    # Sample contiguous keyframes inside three trajectory sections. Consecutive
    # views have enough overlap for tracks. Holdouts come from the original split.
    for block in np.array_split(np.arange(len(ordered)), min(3, max(1, len(ordered) // 16))):
        start = max(0, (len(block) - 16) // 2)
        window = [ordered[i] for i in block[start:start + 16]]
        selected.extend(window)
        nearby = [n for n in registered[positions[window[0]]:positions[window[-1]] + 1] if n not in train_set]
        if nearby:
            indices = np.linspace(0, len(nearby)-1, min(4, len(nearby))).round().astype(int)
            checks.extend(nearby[i] for i in indices)
    if not checks:
        raise ValueError("No original held-out photos near the selected keyframes")
    return ordered if full else selected, checks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, help='Default: dataset of the current Orange4F suite')
    p.add_argument('--output-root', type=Path, default=Path(os.environ.get(
        'ORANGE4F_OUTPUT_ROOT', '/data/3dgs_outputs/orange4f/3dgs_finetuning')))
    p.add_argument('--full', action='store_true', help='Fit all stride-3 training poses')
    args = p.parse_args(argv)
    root = args.output_root.expanduser().resolve()
    suite_path = root / 'stride3_noaa_v1/suite.json'
    suite = json.loads(suite_path.read_text()) if suite_path.is_file() else {}
    source = args.source or (Path(suite['source']) if suite else None)
    if source is None:
        source = next((p for p in (Path('/workspace/colmap_scenes/orange4f'),
                                  Path('/data/gh_workspace/data/COLMAP_BA_export/orange_4F/1_full'))
                       if p.is_dir()), None)
    if source is None:
        p.error('Provide --source /path/to/orange4f (no suite or default dataset found)')
    source = source.expanduser().resolve()
    if suite and Path(suite['source']).resolve() != source:
        p.error('Existing suite belongs to a different source. Set --output-root for this dataset.')
    sparse = colmap_model_dir(source)
    cloud = sparse / 'points3D.ply'
    if not cloud.is_file():
        cloud = Path(suite['cloud']['cached']) if suite else source / 'input.ply'
    if not cloud.is_file():
        p.error('Missing LiDAR PLY in the sparse model, prepared suite, or source/input.ply')
    output = root / ('pose_ba_full_v2' if args.full else 'pose_ba_pilot_v2')
    if output.is_relative_to(source):
        p.error('Set --output-root outside the input scene')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.exit(2, f'Pose refinement is already running in {output}\n')
        manifest = (Path(suite['selection']) if suite else
                    select_keyframes(source, 'images', output / 'stride3', 3, False))
        base = json.loads(manifest.read_text())
        registered = registered_names(source)
        selected, checks = pose_split(registered, base['selected_images'], args.full)
        run_manifest = output / 'fit.json'
        split = dict(source_path=str(source), image_directory='images',
                    eval=False, selected_count=len(selected), selected_images=selected,
                    pose_validation_images=checks, stride3_manifest=str(manifest))
        if run_manifest.exists() and json.loads(run_manifest.read_text()) != split:
            p.error(f'The stored pose split differs in {output}. Use a separate --output-root.')
        atomic_json(run_manifest, split)
        print(f"Fitting {len(selected)} stride-3 training cameras; checking {len(checks)} independent images. "
              f"Report: {output / 'ba/refinement_report.json'}", flush=True)
        refine(['--source', str(source), '--keyframes', str(run_manifest), '--output', str(output / 'ba'),
                '--point-cloud', str(cloud), '--keep-lidar-indices', '--image-first', '--max-points', '1500000',
                '--max-tracks', str(max(8000, 24 * len(selected))),
                '--max-translation', '.03', '--max-rotation-deg', '.5'])
        result = json.loads((output / 'ba/refinement_report.json').read_text())
        views = result['independent_views']
        print(f"Accepted: {result['optimized_cameras']}/{len(selected)} training poses fitted. "
              f"Independent photos improved: {views['improved']}/{views['supported']} supported; "
              f"typical gain {views['median_gain_native_px']:.2f} native px.", flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
