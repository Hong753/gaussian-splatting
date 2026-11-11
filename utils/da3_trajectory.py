"""Trajectory-spanning multi-view groups with a shared LiDAR metric scale."""
import math

import numpy as np

from utils.da3_occlusion import CALIBRATION_POLICY, fit_scale, calibrate_regions, cross_view_support

POLICY = dict(name='trajectory_shared_metric_v1', local_policy=CALIBRATION_POLICY,
              min_scale_views=3, min_scale_consensus=.5, max_donors=8,
              validation='two independent locally LiDAR-validated donor views; no recursive propagation')


def trajectory_groups(names, group_size=40, shared_views=8, local_neighbors=2):
    """Each target is owned once; groups span the sequence and retain local matches.

    The budget includes globally distributed shared views and adjacent companion
    frames for every target. Ordering follows the registered trajectory manifest.
    Shared/companion predictions are context only; each output has one owner.
    """
    if (len(set(names)) != len(names) or len(names) < 3 or group_size < 3 or
            shared_views < 0 or local_neighbors < 0 or shared_views >= group_size or
            group_size-shared_views < local_neighbors+1):
        raise ValueError('Invalid trajectory group budget/context or duplicate image names')
    count = len(names)
    if count <= group_size:
        yield list(names), list(names)
        return
    shared = set(np.linspace(0, count-1, shared_views).round().astype(int)) if shared_views else set()
    targets_per_group = (group_size-shared_views)//(local_neighbors+1)
    groups = math.ceil(count/targets_per_group)
    for offset in range(groups):
        targets = list(range(offset, count, groups))
        batch = shared | set(targets)
        for target in targets:
            neighbors = sorted((i for i in range(max(0, target-local_neighbors), min(count, target+local_neighbors+1))
                                if i != target), key=lambda i: (abs(i-target), i))
            batch.update(neighbors[:local_neighbors])
        if len(batch) > group_size:
            raise AssertionError('Trajectory group exceeded its image budget')
        yield [names[i] for i in sorted(batch)], [names[i] for i in targets]


def shared_scale(frames, min_anchors=200, max_relative_error=.10):
    """Equal-view robust consensus from non-held-out LiDAR scale proposals.

    This estimates a proposal, not a trusted mask. The spatial holdout and local
    residual tests still determine which donor regions may validate other views.
    """
    proposals, fits = [], []
    for frame in frames:
        _, fit = fit_scale(frame['raw'], frame['confident'], *frame['anchors'], min_anchors,
                           max_relative_error, holdout_grid=CALIBRATION_POLICY['holdout_grid'])
        fits.append(fit)
        value = fit.get('scale')
        if value is not None and np.isfinite(value) and value > 0:
            proposals.append(value)
    report = dict(accepted=False, proposal_views=len(proposals), total_views=len(frames),
                  reason='insufficient views with scale proposals')
    if len(proposals) < POLICY['min_scale_views']:
        return None, report, fits
    logs = np.log(proposals)
    center = np.median(logs)
    consistent = np.abs(np.exp(logs-center)-1) <= max_relative_error
    if consistent.sum() >= POLICY['min_scale_views']:
        center = np.median(logs[consistent])
        consistent = np.abs(np.exp(logs-center)-1) <= max_relative_error
    fraction = float(consistent.mean())
    accepted = consistent.sum() >= POLICY['min_scale_views'] and fraction >= POLICY['min_scale_consensus']
    scale = float(np.exp(center))
    report.update(accepted=bool(accepted), scale=scale, supporting_views=int(consistent.sum()),
        consensus_fraction=fraction, proposal_min=float(min(proposals)), proposal_max=float(max(proposals)),
        reason='shared metric scale proposal' if accepted else 'group scale proposals disagree')
    return scale if accepted else None, report, fits


def calibrate_group(frames, min_anchors=200, max_relative_error=.10):
    scale, group_report, fits = shared_scale(frames, min_anchors, max_relative_error)
    result = []
    for frame, fit in zip(frames, fits):
        if scale is None:
            local = np.zeros_like(frame['confident'])
            report = dict(accepted=False, reason=group_report['reason'], global_fit=fit,
                          accepted_regions=0, tested_regions=0)
        else:
            _, local, report = calibrate_regions(frame['raw'], frame['confident'], *frame['anchors'],
                min_anchors, max_relative_error, scale_override=scale, global_fit=fit)
        report.update(policy=POLICY['name'], group_scale=group_report,
                      local_accepted=bool(local.any()), scale=scale)
        result.append(dict(frame, depth=np.where(np.isfinite(frame['raw']), frame['raw']*(scale or 1), 0).astype(np.float32),
                           local=local, report=report))
    return result


def validate_from_donors(target, others, tolerance=.05, min_support=2, max_donors=8):
    """Allow missing target anchors only when independent metric surfaces agree.

    Donors always use their own local LiDAR validation. Borrowed target pixels
    never become donors, preventing a loop of mutually unsupported predictions.
    """
    if min_support < 2 or max_donors < min_support:
        raise ValueError('Trajectory validation requires at least two independent donor views')
    depth, valid, k, e = (target[key] for key in ('depth', 'confident', 'intrinsic', 'extrinsic'))
    support = np.zeros(depth.shape, np.uint16)
    if not target['report']['group_scale']['accepted']:
        return np.zeros_like(valid), support, []
    # Rank overlap using a coarse ray sample, then validate at full resolution.
    # Distant rooms cannot validate a target simply by sharing the same group.
    step = max(1, int(np.ceil(max(depth.shape)/32)))
    coarse_k = k.copy()
    coarse_k[:2] /= step
    ranked = []
    for index, other in enumerate(others):
        if not other['local'].any():
            continue
        donor = (other['depth'], other['local'], other['intrinsic'], other['extrinsic'])
        score = int(cross_view_support(depth[::step, ::step], valid[::step, ::step], coarse_k, e, [donor], tolerance).sum())
        if score:
            ranked.append((score, index, donor))
    chosen = sorted(ranked, key=lambda item: (-item[0], item[1]))[:max_donors]
    if chosen:
        support = cross_view_support(depth, valid, k, e, [entry[2] for entry in chosen], tolerance)
    trusted = valid & (support >= min_support)
    return trusted, support, [others[entry[1]]['image'] for entry in chosen]
