"""Fit relative inverse-depth maps to projected reference points."""
import numpy as np


def apply_alignment(depth, params):
    """Align a loaded inverse-depth map and return a positive-value mask."""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        depth = depth[..., 0]
    depth = depth.astype(np.float32, copy=True)
    invalid = np.zeros(depth.shape, dtype=bool)
    if params is None:
        return np.zeros_like(depth), invalid, False
    scale, offset = params["scale"], params["offset"]
    legacy_outlier = ("reliable" not in params and
                      (scale < 0.2 * params["med_scale"] or scale > 5 * params["med_scale"]))
    if (not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0 or
            params.get("reliable") is False or legacy_outlier):
        return np.zeros_like(depth), invalid, False
    aligned = np.maximum(depth, 0) * scale + offset
    valid = np.isfinite(aligned) & (aligned > 0)
    aligned[~valid] = 0
    return aligned, valid, bool(np.any(valid))


def fit_alignment(predicted, reference, seed=42, min_points=50,
                  min_correlation=0.25, max_relative_error=0.35):
    """Robust affine inverse-depth fit with a held-out point diagnostic.

    Inputs are values sampled from the saved PNG divided by 65536, and
    corresponding inverse camera-space z values. The held-out points are not
    spatially independent; diagnostics identify some bad fits, not all of them.
    """
    x = np.asarray(predicted, dtype=np.float64).reshape(-1)
    y = np.asarray(reference, dtype=np.float64).reshape(-1)
    if len(x) != len(y):
        raise ValueError("Depth and reference sample counts differ")
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[valid], y[valid]
    result = {"scale": 0.0, "offset": 0.0, "reliable": False,
              "samples": int(len(x)), "reason": ""}
    if len(x) < min_points:
        result["reason"] = "too_few_points"
        return result
    order = np.random.default_rng(seed).permutation(len(x))
    count = max(1, int(len(x) * 0.8))
    train, test = order[:count], order[count:]
    if len(test) < 2 or np.std(x[train]) < 1e-6:
        result["reason"] = "insufficient_depth_variation"
        return result
    design = np.column_stack((x[train], np.ones(len(train))))
    weights = np.ones(len(train))
    params = np.zeros(2)
    for _ in range(8):
        root = np.sqrt(weights)
        params = np.linalg.lstsq(design * root[:, None], y[train] * root, rcond=None)[0]
        residual = y[train] - design @ params
        deviation = np.median(np.abs(residual - np.median(residual)))
        sigma = max(1.4826 * deviation, 1e-8)
        weights = np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
    scale, offset = map(float, params)
    if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
        result["reason"] = "nonpositive_or_nonfinite_scale"
        return result
    prediction = x[test] * scale + offset
    rel_error = float(np.median(np.abs(prediction - y[test])) / max(float(np.median(y[test])), 1e-12))
    if np.std(prediction) < 1e-12 or np.std(y[test]) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(prediction, y[test])[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    reliable = correlation >= min_correlation and rel_error <= max_relative_error
    result.update(scale=scale, offset=offset, reliable=bool(reliable),
                  validation_samples=int(len(test)), validation_correlation=correlation,
                  validation_median_relative_error=rel_error,
                  reason="ok" if reliable else "poor_heldout_fit")
    return result
