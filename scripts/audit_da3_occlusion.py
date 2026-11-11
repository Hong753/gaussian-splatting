#!/usr/bin/env python3
"""Explain saved DA3 rejection stages without inference or changing the cache."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.sparse_lidar import frame_filename


def summarize(metadata, rows, images=(), detail_count=8):
    stages, global_reasons, local_rejections = Counter(), Counter(), Counter()
    details = []
    for row in rows:
        empty = row.get('reliable_tile_fraction', 0) == 0
        fit = row.get('global_fit', row)
        group = row.get('group_scale', {})
        if not empty:
            stage = 'usable_depth'
        elif group and not group.get('accepted'):
            stage = 'no_group_scale'
        elif group:
            stage = 'no_metric_donor_agreement' if not row.get('trusted_pixel_fraction', 0) else 'no_complete_tile'
        elif not row.get('locally_calibrated_pixel_fraction', 0):
            stage = 'no_local_lidar_validation'
        elif not row.get('trusted_pixel_fraction', 0):
            stage = 'no_multiview_agreement'
        else:
            stage = 'no_complete_tile'
        stages[stage] += 1
        if empty:
            global_reasons[fit.get('reason', 'not recorded')] += 1
            local_rejections.update(row.get('rejected_regions', {}))
        details.append(dict(image=row['image'], stage=stage,
            confident=row.get('confident_pixel_fraction'), local=row.get('locally_calibrated_pixel_fraction'),
            trusted=row.get('trusted_pixel_fraction'), tiles=row.get('reliable_tile_fraction'),
            anchors=fit.get('anchors'), spatial_cells=fit.get('spatial_cells'),
            scale=row.get('scale', fit.get('scale')), inlier_fraction=fit.get('inlier_fraction'),
            median_relative_error=fit.get('median_relative_error'), p90_relative_error=fit.get('p90_relative_error'),
            accepted_regions=row.get('accepted_regions'), tested_regions=row.get('tested_regions'),
            local_rejections=row.get('rejected_regions', {}), group_scale=group,
            donors=row.get('donor_images', []), borrowed=row.get('borrowed_pixel_fraction')))
    selected = [r for r in details if r['image'] in images] if images else sorted(details, key=lambda r: (r['tiles'] or 0, r['image']))[:detail_count]
    return dict(prepared_frames=len(rows), expected_frames=len(metadata.get('images', [])),
        complete=metadata.get('complete', False), policy=metadata.get('calibration_policy'),
        empty_frames=len(rows)-stages['usable_depth'], empty_fraction=(len(rows)-stages['usable_depth'])/max(len(rows), 1),
        mean_usable_tiles=sum(r.get('reliable_tile_fraction', 0) for r in rows)/max(len(rows), 1),
        stages=dict(stages), empty_frame_global_reasons=dict(global_reasons),
        rejected_local_regions=dict(local_rejections), details=selected,
        note='Confidence coverage is a percentile selection, not measured depth accuracy. Rejection reasons do not establish whether LiDAR, depth, or registration is wrong.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--depth-dir', type=Path, required=True)
    p.add_argument('--images', nargs='+', default=[])
    p.add_argument('--details', type=int, default=8)
    p.add_argument('--collect', action='store_true', help='Save selected numeric frames/anchors plus report as alignment_debug.zip')
    args = p.parse_args(argv)
    root = args.depth_dir.resolve()
    metadata = json.loads((root/'metadata.json').read_text())
    rows = json.loads((root/'calibration.json').read_text())['frames']
    report = summarize(metadata, rows, args.images, max(0, args.details))
    print(json.dumps(report, indent=2))
    if args.collect:
        path, temporary = root/'alignment_debug.zip', root/'alignment_debug.tmp.zip'
        try:
            with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
                archive.writestr('audit.json', json.dumps(report, indent=2))
                archive.writestr('metadata.json', json.dumps(metadata, indent=2))
                for name in ('groups.json', 'calibration.json'):
                    if (root/name).is_file():
                        archive.write(root/name, name)
                for item in report['details']:
                    for directory in ('frames', 'anchors'):
                        relative = Path(directory)/frame_filename(item['image'])
                        if (root/relative).is_file():
                            archive.write(root/relative, str(relative))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Alignment debug archive: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
