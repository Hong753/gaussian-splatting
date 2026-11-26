#!/usr/bin/env python3
"""Inspect and normalize relative exposure from shared COLMAP observations.

CPU only. Originals, cameras, and existing training runs are never modified.
The mean exposure is fixed in EV within each connected calibration component.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.colmap_paths import colmap_model_dir
from utils.exposure_balance import apply_exposure, evaluate_holdout, fit_exposures, srgb_to_linear
from utils.exposure_image_io import (load_frames, ordered_map, read_rgb, sample_frame,
                                     save_rgb, save_thumbnail)
from utils.exposure_report import write_report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/data/colmap_scenes/orange4f'))
    parser.add_argument('--images', type=Path, default=Path('images'),
                        help='Image directory, relative to --source or absolute')
    parser.add_argument('--output', type=Path,
                        default=Path('/data/3dgs_outputs/orange4f/exposure_check_v1'))
    parser.add_argument('--samples-per-frame', type=int, default=1500)
    parser.add_argument('--min-observations', type=int, default=40)
    parser.add_argument('--patch-radius', type=int, default=1)
    parser.add_argument('--holdout-fraction', type=float, default=.2)
    parser.add_argument('--max-correction-ev', type=float, default=2.,
                        help='Maximum applied magnitude; uncapped estimates are also saved')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preview-count', type=int, default=24)
    parser.add_argument('--allow-resized', action='store_true',
                        help='Allow known full-frame resizes relative to COLMAP; never crops')
    args = parser.parse_args(argv)
    if (args.samples_per_frame < 10 or args.min_observations < 2 or args.workers < 1
            or not 0 <= args.patch_radius <= 8 or args.seed < 0 or args.preview_count < 1
            or not .05 <= args.holdout_fraction <= .5
            or not math.isfinite(args.max_correction_ev) or args.max_correction_ev <= 0):
        parser.error('Invalid sampling, holdout, worker, or correction options')
    return args


def progress(stage, done, total, started):
    if done == 1 or done == total or done % 25 == 0:
        print(f'{stage}: {done}/{total} | elapsed {time.monotonic()-started:.1f}s', flush=True)


def clean_number(number):
    return float(number) if np.isfinite(number) else None


def main(argv=None):
    args = arguments(argv)
    source = args.source.expanduser().resolve()
    image_root = (source / args.images).resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; use a new --output directory')
    if output == source or source in output.parents or output == image_root or image_root in output.parents:
        raise ValueError('Output must be outside the source dataset/image directory')
    model = colmap_model_dir(source)
    frames = load_frames(model, args.samples_per_frame, args.seed)
    if len(frames) < 2:
        raise ValueError('At least two registered images are required')
    for frame in frames:
        path = image_root / frame.name
        if not path.is_file():
            raise FileNotFoundError(f'Registered image is missing: {path}')
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
            raise ValueError(f'Unsupported image type: {path}')
    print(f'Source: {source}\nRegistered images: {len(frames):,}\nCOLMAP model: {model}\n'
          f'Output: {output}\nCPU workers: {args.workers}', flush=True)
    print('Sampling shared scene points; no whole-image brightness matching.', flush=True)
    started = time.monotonic()
    samples = []
    def sample(frame):
        return sample_frame(frame, image_root, args.patch_radius, args.allow_resized)
    for index, sample_result in enumerate(ordered_map(sample, frames, args.workers)):
        samples.append(sample_result)
        progress('Sample', index+1, len(frames), started)
    frame_ids = np.concatenate([np.full(len(item['tracks']), index, dtype=np.int32)
                                for index, item in enumerate(samples)])
    track_ids = np.concatenate([item['tracks'] for item in samples])
    luminance = np.concatenate([item['log_luminance'] for item in samples])
    if len(track_ids) == 0:
        raise ValueError('No usable shared-point image samples. The COLMAP images model must '
                         'contain POINTS2D with valid POINT3D_IDs; pose-only exports are insufficient.')
    print(f'Fitting exposure from {len(track_ids):,} valid observations; '
          f'holding out {args.holdout_fraction:.0%} of entire tracks.', flush=True)
    fit = fit_exposures(frame_ids, track_ids, luminance, len(frames),
                        min_observations=args.min_observations,
                        holdout_fraction=args.holdout_fraction, seed=args.seed)
    supported = np.asarray(fit['supported'], dtype=bool)
    if not supported.any():
        raise ValueError('No frames have sufficient connected, shared-track support. '
                         'Nothing was exported; whole-image brightness is not used as a fallback.')
    exposure = np.asarray(fit['exposure_ev'])
    corrections = np.where(supported, np.clip(-exposure, -args.max_correction_ev,
                                              args.max_correction_ev), 0.)
    capped = supported & (np.abs(exposure) > args.max_correction_ev)
    applied_validation = evaluate_holdout(frame_ids, track_ids, luminance,
        -corrections, supported, fit['component'], fit['heldout_mask'])
    summary = dict(fit['summary'])
    summary['heldout_estimated'] = summary.pop('heldout', None)
    summary.update(source=str(source), images=str(image_root), colmap_model=str(model),
        output=str(output), registered_images=len(frames), supported_images=int(supported.sum()),
        unsupported_images=int((~supported).sum()), capped_images=int(capped.sum()),
        estimated_exposure_ev_range=[float(exposure[supported].min()), float(exposure[supported].max())],
        applied_correction_ev_range=[float(corrections.min()), float(corrections.max())],
        heldout_applied=applied_validation,
        options={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        notes=[
            'One scalar gain per image in approximate linear sRGB; no local tone mapping or white-balance adjustment.',
            'Target is mean EV (geometric-mean exposure) within each connected component, not equal image brightness.',
            'Disconnected components have no identifiable common exposure and are normalized independently.',
            'Exposure variation is inferred photometric variation; automatic exposure is not proven by this fit.',
            'The correction cannot recover already clipped highlights, motion blur, or missing image detail.',
            'JPEG outputs retain original names and dimensions, re-encoded at quality 100 with 4:4:4 sampling.',
            'PNG inputs remain lossless PNG. Original image files and the COLMAP reconstruction are unchanged.',
            'Held-out validation uses observed point luminance before output clipping/quantization; clipping is reported separately.',
        ])
    warnings = list(summary.get('warnings', []))
    if not summary.get('linear_solver_converged', False):
        warnings.append('The sparse exposure solve hit its iteration limit; estimated gains are provisional.')
    elif not summary.get('converged', False):
        warnings.append('Robust exposure fitting reached its iteration limit; inspect validation and large corrections.')
    components = np.unique(np.asarray(fit['component'])[supported])
    if len(components) > 1:
        warnings.append(f'{len(components)} disconnected calibration components: their relative exposure is unknown.')
    if (~supported).any():
        warnings.append(f'{int((~supported).sum())} unsupported images receive zero exposure correction.')
    if capped.any():
        warnings.append(f'{int(capped.sum())} estimated corrections exceed ±{args.max_correction_ev:g} EV and were capped.')
    if not applied_validation.get('sufficient_evidence', False):
        warnings.append('Too few held-out tracks for a reliable independent consistency check.')
    before = applied_validation.get('median_abs_deviation_before_ev')
    after = applied_validation.get('median_abs_deviation_after_ev')
    if before is not None and after is not None and after > before:
        warnings.append('Held-out median photometric disagreement increased: do not assume normalization is beneficial.')
    if any(item['resized'] for item in samples):
        warnings.append('Known image resizing was explicitly allowed; observation coordinates were rescaled.')
    summary['warnings'] = warnings
    output.mkdir(parents=True, exist_ok=False)
    (output / 'original').symlink_to(image_root, target_is_directory=True)
    (output / 'status.json').write_text(json.dumps({'status': 'exporting', 'source': str(source)}, indent=2)+'\n')
    records = []

    def export(index):
        frame, sample_result = frames[index], samples[index]
        rgb, _ = read_rgb(image_root / frame.name)
        corrected = apply_exposure(rgb, float(corrections[index]))
        corrected_rel = (Path('corrected') / frame.name).as_posix()
        before_rel = f'thumbnails/{index:06d}_before.jpg'
        after_rel = f'thumbnails/{index:06d}_after.jpg'
        save_rgb(corrected, output / corrected_rel)
        save_thumbnail(rgb, output / before_rel)
        save_thumbnail(corrected, output / after_rel)
        original_clipped = (rgb.max(axis=2) == 255)
        corrected_clipped = (corrected.max(axis=2) == 255)
        # The linear threshold identifies newly saturated pixels independently
        # of JPEG re-encoding and of the final uint8 rounding near white.
        linear_max = srgb_to_linear(rgb.max(axis=2).astype(np.float64) / 255.)
        newly_clipped = (linear_max * 2.**corrections[index] > 1.) & ~original_clipped
        status = 'unsupported: unchanged exposure' if not supported[index] else (
            'supported: correction capped' if capped[index] else 'supported')
        return dict(name=frame.name, image_id=frame.image_id, width=sample_result['width'],
            height=sample_result['height'], original_rel=(Path('original') / frame.name).as_posix(),
            corrected_rel=corrected_rel, thumbnail_before=before_rel, thumbnail_after=after_rel,
            estimated_exposure_ev=float(exposure[index]), correction_ev=float(corrections[index]),
            gain=float(2.**corrections[index]), status=status,
            fit_observations=int(fit['fit_observations'][index]),
            valid_observations=sample_result['valid_observations'],
            component=int(fit['component'][index]),
            residual_median_ev=clean_number(fit['residual_median_ev'][index]),
            clip_before=float(original_clipped.mean()), clip_after=float(corrected_clipped.mean()),
            newly_clipped_fraction=float(newly_clipped.mean()), exif=sample_result['exif'])

    for index, record in enumerate(ordered_map(export, range(len(frames)), args.workers)):
        records.append(record)
        progress('Export', index+1, len(frames), started)
    summary['frames_with_new_clipping'] = sum(row['newly_clipped_fraction'] > 0 for row in records)
    summary['mean_newly_clipped_fraction'] = float(np.mean([row['newly_clipped_fraction'] for row in records]))
    if summary['mean_newly_clipped_fraction'] > .001:
        summary['warnings'].append('Exposure brightening clipped additional highlights; inspect clipping statistics and previews.')
    summary['elapsed_seconds'] = time.monotonic() - started
    with (output / 'exposure.csv').open('w', newline='') as stream:
        fields = ['name', 'image_id', 'estimated_exposure_ev', 'correction_ev', 'gain', 'status',
                  'fit_observations', 'valid_observations', 'component', 'residual_median_ev',
                  'clip_before', 'clip_after', 'newly_clipped_fraction', 'width', 'height', 'exif']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow(dict(record, exif=json.dumps(record['exif'], sort_keys=True)))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output / 'frames.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    write_report(output, records, summary, preview_count=args.preview_count)
    (output / 'status.json').write_text(json.dumps({'status': 'complete', 'registered_images': len(frames)}, indent=2)+'\n')
    print(f'\nComplete. Corrected images: {output / "corrected"}\n'
          f'All-frame gallery: {output / "index.html"}\n'
          f'Portable preview: {output / "preview.html"}\n'
          f'Exposure table: {output / "exposure.csv"}', flush=True)
    print(json.dumps({'heldout_applied': applied_validation, 'warnings': summary['warnings']}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
