"""Independent photo-only check when the LiDAR warp has too little support."""
import cv2
import numpy as np

from scripts.diagnose_floor_blur import load_rgb, save_image
from scripts.keyframe_pipeline.common import image_path
from scripts.keyframe_pipeline.epipolar import line_errors
from scripts.keyframe_pipeline.features import mutual_descriptor_matches


def descriptors(gray):
    keys, desc = cv2.SIFT_create(contrastThreshold=.018).detectAndCompute(gray, None)
    if desc is None:
        return np.empty((0, 2)), np.empty((0, 128), np.float32)
    xy = np.array([key.pt for key in keys])
    # SIFT may assign multiple orientations to one physical keypoint. Keep these
    # out of independent fit/check sets and avoid counting one speckle twice.
    _, idx = np.unique(np.rint(xy).astype(int), axis=0, return_index=True)
    desc = desc[idx]
    return xy[idx], np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-8))


def compare_pair(gray, ref_xy, ref_desc, photo, roi, ref, view, camera, other_camera, output, index):
    target = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    xy, desc = descriptors(target)
    matches = mutual_descriptor_matches(ref_desc, desc, .75)
    row = dict(name=view.name, matches=len(matches), status="too few distinct matches")
    if len(matches) < 24:
        return row
    left = ref_xy[matches[:, 0]]
    right = xy[matches[:, 1]]
    # Split inside each quadrant before estimating or filtering by geometry.
    quad = (left[:, 0] >= gray.shape[1]/2).astype(int) + 2*(left[:, 1] >= gray.shape[0]/2)
    rng = np.random.default_rng(0)
    fit = np.zeros(len(left), bool)
    for q in range(4):
        indices = rng.permutation(np.flatnonzero(quad == q))
        fit[indices[::2]] = True
    check = ~fit
    row.update(check_matches=int(check.sum()), check_quadrants=len(np.unique(quad[check])))
    if check.sum() < 12 or row["check_quadrants"] < 3:
        return dict(row, status="too few distributed check matches")
    # Directly map neighbor pixels into native reference-crop pixels.
    H, inliers = cv2.findHomography(right[fit], left[fit], cv2.RANSAC, 2.)
    if H is None or inliers is None or not np.isfinite(H).all():
        return dict(row, status="unstable photo warp")
    good = inliers.ravel().astype(bool)
    if good.sum() < 8 or len(np.unique(quad[fit][good])) < 3:
        return dict(row, status="weak spatial support for photo warp")
    hp = np.column_stack((right[check], np.ones(int(check.sum())))) @ H.T
    if (np.abs(hp[:, 2]) < 1e-8).any():
        return dict(row, status="unstable photo warp")
    errors = np.linalg.norm(hp[:, :2]/hp[:, 2:] - left[check], axis=1)
    row.update(photo_median_px=float(np.median(errors)), photo_p90_px=float(np.percentile(errors, 90)),
               status="photo warp checked")
    # Pose-independent matching and selection. Use unfiltered check pairs for this
    # metric so a bad original pose can't remove the very residual we are testing.
    left_frame = dict(uv=left + roi[:2] + .5, K=camera.params, scale=np.ones(2))
    right_frame = dict(uv=right + .5, K=other_camera.params)
    pairs = np.column_stack([np.flatnonzero(check)]*2)
    epi = line_errors(left_frame, right_frame, pairs, ref, view.R, view.C)
    if np.isfinite(epi).all():
        row.update(epipolar_median_px=float(np.median(epi)), epipolar_p90_px=float(np.percentile(epi, 90)))
    warped = cv2.warpPerspective(target, H, gray.shape[::-1])
    row["overlay"] = save_image(output, f"photo_overlay_{index}.png", cv2.merge((warped, gray, warped)))
    return row


def photo_check(source, folder, images, cameras, ref_index, selected, roi, gray, window, limit, output):
    ref = images[ref_index]
    ref_xy, ref_desc = descriptors(gray)
    neighbors = [v for v in images[max(0, ref_index-window):ref_index+window+1]
                 if v.name != ref.name and v.name in selected and np.linalg.norm(v.C-ref.C) >= .01]
    neighbors.sort(key=lambda v: np.linalg.norm(v.C-ref.C))
    rows = []
    for index, view in enumerate(neighbors[:limit]):
        if len(ref_desc) < 24:
            break
        cam = cameras[view.camera_id]
        photo = load_rgb(image_path(source, folder, view.name), cam)
        row = compare_pair(gray, ref_xy, ref_desc, photo, roi, ref, view, cameras[ref.camera_id], cam, output, index)
        rows.append(row)
        print(f"RGB check {index+1}/{min(len(neighbors),limit)}: {view.name}, {row['matches']} mutual matches, "
              f"{row['status']}", flush=True)
    repeatable = [r for r in rows if r.get("photo_p90_px", np.inf) <= 2.]
    if len(repeatable) >= 3:
        message = (f"The photos reproduce the crop in {len(repeatable)} nearby views within 2 px "
                   "(p90 on features held out from the image warp). LiDAR alignment remains unverified.")
    else:
        message = (f"Only {len(repeatable)} nearby views passed the 2 px photo repeatability check. "
                   "This crop provides insufficient evidence to attribute the blur to poses.")
    return dict(reference_features=len(ref_xy), views=rows, repeatable_views=len(repeatable), verdict=message)
