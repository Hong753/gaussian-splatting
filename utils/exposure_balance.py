"""CPU-only, track-based scalar exposure estimation.

The model is log2(luminance_ij) = exposure_i + radiance_j.  Exposure is
identifiable only up to one additive constant per connected component.
No whole-image brightness statistics are used as an alignment fallback.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg


def _floating_array(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "uif":
        raise ValueError(f"{name} must be numeric")
    value = value.astype(np.result_type(value.dtype, np.float32), copy=False)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def srgb_to_linear(rgb):
    """Convert normalized sRGB values in [0, 1] to approximate linear RGB."""
    rgb = _floating_array(rgb, "sRGB")
    if np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("sRGB values must be normalized to [0, 1]")
    return np.where(rgb <= 0.04045, rgb / 12.92,
                    ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    """Encode finite, nonnegative linear RGB; values above one are not clipped."""
    rgb = _floating_array(rgb, "Linear RGB")
    if np.any(rgb < 0):
        raise ValueError("Linear RGB must be nonnegative")
    return np.where(rgb <= 0.0031308, 12.92 * rgb,
                    1.055 * rgb ** (1.0 / 2.4) - 0.055)


def apply_exposure(rgb, correction_ev):
    """Apply one linear-light gain to an RGB uint8 array and return uint8.

    Positive correction_ev brightens the image.  No tone curve, white-balance
    adjustment, or spatial operation is applied; out-of-range highlights clip.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be a uint8 array with a final RGB dimension")
    if not np.isscalar(correction_ev) or not math.isfinite(float(correction_ev)):
        raise ValueError("correction_ev must be a finite scalar")
    try:
        gain = math.pow(2.0, float(correction_ev))
    except OverflowError as exc:
        raise ValueError("correction_ev is too large") from exc
    linear = srgb_to_linear(rgb.astype(np.float64) / 255.0)
    linear *= gain
    np.clip(linear, 0.0, 1.0, out=linear)
    return np.rint(linear_to_srgb(linear) * 255.0).astype(np.uint8)


def _group_median(values, groups, n_groups):
    """Vectorized medians without a Python loop over potentially many tracks."""
    result = np.full(n_groups, np.nan, dtype=np.float64)
    if not len(values):
        return result
    order = np.lexsort((values, groups))
    counts = np.bincount(groups, minlength=n_groups)
    starts = np.cumsum(counts) - counts
    present = counts > 0
    lo = starts[present] + (counts[present] - 1) // 2
    hi = starts[present] + counts[present] // 2
    sorted_values = values[order]
    result[present] = (sorted_values[lo] + sorted_values[hi]) * 0.5
    return result


def _heldout_diagnostics(frame, track, luminance, heldout_mask, exposure, component,
                         disabled=False):
    report = dict(status="insufficient", sufficient_evidence=False, tracks=0,
                  track_component_groups=0, observations=0,
                  metric="Within-track log-luminance dispersion before output clipping or encoding.",
                  median_abs_deviation_before_ev=None,
                  median_abs_deviation_after_ev=None,
                  p90_abs_deviation_before_ev=None,
                  p90_abs_deviation_after_ev=None, fractional_reduction=None)
    if disabled:
        report.update(status="disabled", reason="Track holdout was disabled.")
        return report
    selected = heldout_mask & (component[frame] >= 0)
    if not np.any(selected):
        report["reason"] = "No held-out observations belong to supported frames."
        return report
    f, t, z = frame[selected], track[selected], luminance[selected]
    # A held-out track must never bridge the independently fitted gauges.
    n_components = int(component.max()) + 1
    keys = t * n_components + component[f]
    _, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    keep = counts[groups] >= 2
    f, t, z, groups = f[keep], t[keep], z[keep], groups[keep]
    if not len(z):
        report["reason"] = "No held-out track has two supported views in one component."
        return report
    _, groups = np.unique(groups, return_inverse=True)
    n_groups = int(groups.max()) + 1
    corrected = z - exposure[f]
    before = np.abs(z - _group_median(z, groups, n_groups)[groups])
    after = np.abs(corrected - _group_median(corrected, groups, n_groups)[groups])
    before_median, after_median = float(np.median(before)), float(np.median(after))
    sufficient = n_groups >= 10
    report.update(
        status="ok" if sufficient else "insufficient",
        sufficient_evidence=sufficient,
        tracks=int(len(np.unique(t))), track_component_groups=n_groups,
        observations=int(len(z)), median_abs_deviation_before_ev=before_median,
        median_abs_deviation_after_ev=after_median,
        p90_abs_deviation_before_ev=float(np.quantile(before, 0.9)),
        p90_abs_deviation_after_ev=float(np.quantile(after, 0.9)),
        fractional_reduction=(1.0 - after_median / before_median)
        if before_median > 1e-12 else None)
    if not sufficient:
        report["reason"] = "Fewer than ten independent held-out track/component groups."
    return report


def evaluate_holdout(frame_ids, track_ids, log_luminance, exposure_ev, supported,
                     component, heldout_mask):
    """Evaluate an estimated OR actually applied exposure on reserved tracks.

    exposure_ev follows the capture-offset sign convention: the corresponding
    image correction is -exposure_ev.  This log-luminance diagnostic applies
    gains before output clipping/quantization; report clipped pixels separately.
    It never compares offsets across disconnected fitted components.
    """
    frame, raw_tracks = np.asarray(frame_ids), np.asarray(track_ids)
    z, exposure = np.asarray(log_luminance, dtype=np.float64), np.asarray(exposure_ev, dtype=np.float64)
    supported, component, mask = np.asarray(supported), np.asarray(component), np.asarray(heldout_mask)
    arrays = (frame, raw_tracks, z, exposure, supported, component, mask)
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("Holdout inputs must be one-dimensional arrays")
    if not (len(frame) == len(raw_tracks) == len(z) == len(mask)):
        raise ValueError("Holdout observation arrays must have matching sizes")
    if not (len(exposure) == len(supported) == len(component)) or not len(exposure):
        raise ValueError("Holdout frame arrays must have matching nonzero sizes")
    if frame.dtype.kind not in "iu" or raw_tracks.dtype.kind not in "iu" or component.dtype.kind not in "iu":
        raise ValueError("Holdout frame, track and component IDs must be integers")
    if supported.dtype != np.bool_ or mask.dtype != np.bool_:
        raise ValueError("supported and heldout_mask must be boolean arrays")
    if not np.isfinite(z).all() or not np.isfinite(exposure).all():
        raise ValueError("Holdout luminance and exposure must be finite")
    if np.any((frame < 0) | (frame >= len(exposure))) or np.any(raw_tracks < 0):
        raise ValueError("Holdout frame or track IDs are invalid")
    if not np.array_equal(supported, component >= 0):
        raise ValueError("Supported frames must have nonnegative component IDs")
    _, tracks = np.unique(raw_tracks, return_inverse=True)
    if len(tracks):
        total = np.bincount(tracks)
        reserved = np.bincount(tracks, weights=mask, minlength=len(total))
        if np.any((reserved != 0) & (reserved != total)):
            raise ValueError("Holdout must reserve entire tracks")
    # Compact component IDs to bound group keys even for externally supplied labels.
    compact = np.full(len(component), -1, dtype=np.int64)
    if supported.any():
        _, compact[supported] = np.unique(component[supported], return_inverse=True)
    return _heldout_diagnostics(frame, tracks, z, mask, exposure, compact)


def fit_exposures(frame_ids, track_ids, log_luminance, n_frames,
                  min_observations=40, holdout_fraction=0.2, seed=0,
                  max_iterations=8):
    """Fit scalar exposures from shared-track log2 luminance observations.

    Inputs contain one observation per (frame, track); frame IDs are zero-based
    rows in [0, n_frames), whereas track IDs may be arbitrary nonnegative ints.
    An entire deterministic random subset of tracks is withheld from fitting.
    The remaining bipartite graph is peeled until every fitted track has two
    supported frames and every supported frame has min_observations samples.

    Huber IRLS solves the sparse two-way additive model.  Track radiances are
    analytically eliminated, leaving a matrix-free, preconditioned sparse CG
    solve in the much smaller frame space.  Memory is linear in observations.

    Returned exposure_ev is the estimated capture offset; apply its NEGATIVE
    as correction_ev.  Supported components are independently centered at mean
    exposure zero.  Unsupported frames have exposure zero and component -1.
    Per-frame residual_median_ev is NaN when unsupported; summary is JSON-safe.
    """
    frame = np.asarray(frame_ids)
    tracks = np.asarray(track_ids)
    z = np.asarray(log_luminance, dtype=np.float64)
    for name, value in (("frame_ids", frame), ("track_ids", tracks)):
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
            raise ValueError(f"{name} exceeds int64 range")
    if z.ndim != 1 or len(frame) != len(tracks) or len(frame) != len(z):
        raise ValueError("Observation arrays must be one-dimensional and equally sized")
    for name, value in (("n_frames", n_frames),
                        ("min_observations", min_observations),
                        ("max_iterations", max_iterations)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(float(holdout_fraction)) or not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be finite and in [0, 1)")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    frame, tracks = frame.astype(np.int64, copy=False), tracks.astype(np.int64, copy=False)
    if not np.isfinite(z).all() or np.any(tracks < 0):
        raise ValueError("Track IDs must be nonnegative and log luminance finite")
    if np.any((frame < 0) | (frame >= n_frames)):
        raise ValueError("Frame IDs are outside [0, n_frames)")
    _, track = np.unique(tracks, return_inverse=True)
    n_tracks = int(track.max()) + 1 if len(track) else 0
    if len(frame):
        order = np.lexsort((frame, track))
        duplicate = ((frame[order[1:]] == frame[order[:-1]]) &
                     (track[order[1:]] == track[order[:-1]]))
        if np.any(duplicate):
            raise ValueError("Duplicate observation for the same frame and track")
    heldout = np.random.default_rng(seed).random(n_tracks) < holdout_fraction
    fit = ~heldout[track]
    while np.any(fit):
        supported = np.bincount(frame[fit], minlength=n_frames) >= min_observations
        next_fit = fit & supported[frame]
        counts = np.bincount(track[next_fit], minlength=n_tracks)
        next_fit &= counts[track] >= 2
        if np.array_equal(next_fit, fit):
            break
        fit = next_fit
    fit_counts = np.bincount(frame[fit], minlength=n_frames)
    supported = fit_counts >= min_observations
    exposure = np.zeros(n_frames, dtype=np.float64)
    component = np.full(n_frames, -1, dtype=np.int32)
    residual_median = np.full(n_frames, np.nan, dtype=np.float64)
    summary = dict(
        model="log2_luminance = frame_exposure_ev + shared_track_radiance_ev",
        gauge="Mean frame exposure is zero independently in each fitted component.",
        component_offsets_identifiable=False,
        component_offset_note="Relative offsets between disconnected components are unidentifiable.",
        observations=int(len(z)), tracks=n_tracks, frames=int(n_frames),
        supported_frames=int(supported.sum()), unsupported_frames=int((~supported).sum()),
        fit_observations=int(fit.sum()), heldout_tracks=int(heldout.sum()),
        holdout_fraction=float(holdout_fraction), seed=int(seed),
        components=0, component_frame_counts=[], converged=False,
        convergence_tolerance_ev=0.001,
        linear_solver_converged=True, iterations=0, history=[])
    if np.any(fit):
        active_frames, f = np.unique(frame[fit], return_inverse=True)
        _, t = np.unique(track[fit], return_inverse=True)
        y = z[fit]
        nf, nt = len(active_frames), int(t.max()) + 1
        # One spanning star per track is enough to find the image components.
        order = np.argsort(t, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(t[order])) + 1]
        anchor = np.empty(nt, dtype=np.int64)
        anchor[t[order[starts]]] = f[order[starts]]
        graph = sparse.coo_matrix((np.ones(len(f), dtype=np.uint8),
                                   (f, anchor[t])), shape=(nf, nf)).tocsr()
        graph.data[:] = 1
        nc, labels = connected_components(graph, directed=False)
        component[active_frames] = labels
        component_counts = np.bincount(labels, minlength=nc)
        summary.update(components=int(nc),
                       component_frame_counts=component_counts.astype(int).tolist())
        weights = np.ones(len(y), dtype=np.float64)
        e = np.zeros(nf, dtype=np.float64)
        residual = np.zeros_like(y)
        for iteration in range(max_iterations):
            track_weight = np.bincount(t, weights=weights, minlength=nt)
            track_y = np.bincount(t, weights=weights * y, minlength=nt) / track_weight
            rhs = np.bincount(f, weights=weights * (y - track_y[t]), minlength=nf)
            # Project away roundoff in each independent gauge direction.
            rhs -= (np.bincount(labels, weights=rhs, minlength=nc) / component_counts)[labels]

            def matvec(value):
                track_mean = np.bincount(t, weights=weights * value[f], minlength=nt) / track_weight
                result = np.bincount(f, weights=weights * (value[f] - track_mean[t]), minlength=nf)
                result += (np.bincount(labels, weights=value, minlength=nc) / component_counts)[labels]
                return result

            diagonal = np.bincount(f, weights=weights * (1.0 - weights / track_weight[t]), minlength=nf)
            diagonal += 1.0 / component_counts[labels]
            operator = LinearOperator((nf, nf), matvec=matvec, dtype=np.float64)
            preconditioner = LinearOperator((nf, nf),
                matvec=lambda value: value / np.maximum(diagonal, 1e-12), dtype=np.float64)
            cg_iterations = [0]

            def count_iteration(_):
                cg_iterations[0] += 1

            previous = e.copy()
            tolerance = {'rtol' if 'rtol' in inspect.signature(cg).parameters else 'tol': 1e-8}
            e, info = cg(operator, rhs, x0=e, **tolerance, atol=1e-10,
                         maxiter=max(200, min(2000, 2 * nf)),
                         M=preconditioner, callback=count_iteration)
            if info < 0 or not np.isfinite(e).all():
                raise RuntimeError("Exposure sparse least-squares solve failed")
            e -= (np.bincount(labels, weights=e, minlength=nc) / component_counts)[labels]
            radiance = np.bincount(t, weights=weights * (y - e[f]), minlength=nt) / track_weight
            residual = y - e[f] - radiance[t]
            mad = float(np.median(np.abs(residual - np.median(residual))))
            delta = max(0.03, 1.345 * 1.4826 * mad)
            change = float(np.max(np.abs(e - previous)))
            summary["linear_solver_converged"] &= info == 0
            summary["history"].append(dict(iteration=iteration + 1,
                cg_iterations=cg_iterations[0], cg_info=int(info),
                max_exposure_change_ev=change, huber_delta_ev=delta,
                median_abs_residual_ev=float(np.median(np.abs(residual)))))
            summary["iterations"] = iteration + 1
            weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-12))
            if iteration >= 1 and change < summary["convergence_tolerance_ev"] and info == 0:
                summary["converged"] = True
                break
        exposure[active_frames] = e
        residual_median[active_frames] = _group_median(np.abs(residual), f, nf)
        summary["fit_median_abs_residual_ev"] = float(np.median(np.abs(residual)))
        summary["status"] = "ok" if summary["converged"] else "iteration_limit"
    else:
        summary.update(status="insufficient_observations", fit_median_abs_residual_ev=None)
    heldout_mask = heldout[track]
    summary["heldout"] = _heldout_diagnostics(
        frame, track, z, heldout_mask, exposure, component,
        disabled=holdout_fraction == 0)
    return dict(exposure_ev=exposure, component=component, supported=supported,
                fit_observations=fit_counts, residual_median_ev=residual_median,
                heldout_mask=heldout_mask, summary=summary)
