"""Camera egomotion estimation from background optical flow.

The video pipeline only needs scene-consistent pixel coordinates to recover
horizontal velocity from panned phone footage.  This module estimates the
frame-to-frame camera motion from background corners while masking out the
athlete, then accumulates transforms into the first frame's reference system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraMotion:
    """Per-frame camera motion estimated from background optical flow."""

    pairwise_affines: np.ndarray
    cumulative_affines: np.ndarray
    confidence: np.ndarray
    inlier_counts: np.ndarray
    feature_counts: np.ndarray
    frames_without_mask: int = 0
    min_inlier_count: int = 20


def _identity_affines(n_frames: int) -> np.ndarray:
    affines = np.zeros((n_frames, 2, 3), dtype=np.float64)
    affines[:, 0, 0] = 1.0
    affines[:, 1, 1] = 1.0
    return affines


def _affine_to_homogeneous(affine: np.ndarray) -> np.ndarray:
    mat = np.eye(3, dtype=np.float64)
    mat[:2, :] = np.asarray(affine, dtype=np.float64)
    return mat


def _homogeneous_to_affine(matrix: np.ndarray) -> np.ndarray:
    if abs(matrix[2, 2]) > 1e-12:
        matrix = matrix / matrix[2, 2]
    return np.asarray(matrix[:2, :], dtype=np.float64)


def _invert_affine(affine: np.ndarray) -> np.ndarray:
    try:
        return _homogeneous_to_affine(np.linalg.inv(_affine_to_homogeneous(affine)))
    except np.linalg.LinAlgError:
        return _identity_affines(1)[0]


def _athlete_mask(
    landmarks_frame: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    athlete_padding_pct: float,
) -> np.ndarray | None:
    visible = landmarks_frame[:, 2] > 0.5
    if int(np.count_nonzero(visible)) < 4:
        return None

    xy = landmarks_frame[visible, :2].astype(np.float64)
    x_min = float(np.nanmin(xy[:, 0]) * image_width)
    x_max = float(np.nanmax(xy[:, 0]) * image_width)
    y_min = float(np.nanmin(xy[:, 1]) * image_height)
    y_max = float(np.nanmax(xy[:, 1]) * image_height)

    pad_x = athlete_padding_pct * image_width
    pad_y = athlete_padding_pct * image_height
    left = max(0, int(np.floor(x_min - pad_x)))
    right = min(image_width - 1, int(np.ceil(x_max + pad_x)))
    top = max(0, int(np.floor(y_min - pad_y)))
    bottom = min(image_height - 1, int(np.ceil(y_max + pad_y)))

    if right <= left or bottom <= top:
        return None

    mask = np.full((image_height, image_width), 255, dtype=np.uint8)
    mask[top : bottom + 1, left : right + 1] = 0
    return mask


def _read_gray(frame_bgr: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[:2] != (image_height, image_width):
        gray = cv2.resize(gray, (image_width, image_height), interpolation=cv2.INTER_AREA)
    return gray


def estimate_camera_motion(
    video_path: Path,
    landmarks_2d_normalised: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    athlete_padding_pct: float = 0.15,
    max_features: int = 300,
    ransac_threshold_px: float = 3.0,
    min_inlier_count: int = 20,
) -> CameraMotion:
    """Estimate camera motion from masked background KLT tracks.

    Args:
        video_path: Source video.
        landmarks_2d_normalised: ``(T, 33, 3)`` normalised x/y/visibility.
        image_width: Video frame width in pixels.
        image_height: Video frame height in pixels.
        athlete_padding_pct: Extra mask padding as a fraction of frame size.
        max_features: Maximum Shi-Tomasi corners per frame.
        ransac_threshold_px: RANSAC reprojection threshold for partial affine.
        min_inlier_count: Minimum accepted RANSAC inliers.

    Returns:
        ``CameraMotion`` with pairwise transforms mapping frame ``t-1`` to
        frame ``t`` and cumulative transforms mapping frame ``t`` into frame 0.
    """
    import cv2

    n_frames = int(landmarks_2d_normalised.shape[0])
    pairwise = _identity_affines(n_frames)
    cumulative = _identity_affines(n_frames)
    confidence = np.zeros(n_frames, dtype=np.float64)
    inlier_counts = np.zeros(n_frames, dtype=np.int32)
    feature_counts = np.zeros(n_frames, dtype=np.int32)
    frames_without_mask = 0
    if n_frames == 0:
        return CameraMotion(
            pairwise,
            cumulative,
            confidence,
            inlier_counts,
            feature_counts,
            frames_without_mask=frames_without_mask,
            min_inlier_count=min_inlier_count,
        )
    confidence[0] = 1.0

    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, prev_bgr = cap.read()
        if not ok or prev_bgr is None:
            return CameraMotion(
                pairwise,
                cumulative,
                confidence,
                inlier_counts,
                feature_counts,
                frames_without_mask=frames_without_mask,
                min_inlier_count=min_inlier_count,
            )
        prev_gray = _read_gray(prev_bgr, image_width, image_height)

        for t in range(1, n_frames):
            ok, curr_bgr = cap.read()
            if not ok or curr_bgr is None:
                cumulative[t:] = cumulative[t - 1]
                break
            curr_gray = _read_gray(curr_bgr, image_width, image_height)

            mask = _athlete_mask(
                landmarks_2d_normalised[t - 1],
                image_width=image_width,
                image_height=image_height,
                athlete_padding_pct=athlete_padding_pct,
            )
            if mask is None:
                frames_without_mask += 1
                cumulative[t] = cumulative[t - 1]
                prev_gray = curr_gray
                continue

            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=max_features,
                qualityLevel=0.01,
                minDistance=8,
                mask=mask,
            )
            if prev_pts is None or len(prev_pts) < min_inlier_count:
                feature_counts[t] = 0 if prev_pts is None else int(len(prev_pts))
                cumulative[t] = cumulative[t - 1]
                prev_gray = curr_gray
                continue

            feature_counts[t] = int(len(prev_pts))
            curr_pts, status_fwd, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                prev_pts,
                None,
            )
            if curr_pts is None or status_fwd is None:
                cumulative[t] = cumulative[t - 1]
                prev_gray = curr_gray
                continue

            back_pts, status_back, _back_err = cv2.calcOpticalFlowPyrLK(
                curr_gray,
                prev_gray,
                curr_pts,
                None,
            )
            if back_pts is None or status_back is None:
                cumulative[t] = cumulative[t - 1]
                prev_gray = curr_gray
                continue

            prev_xy = prev_pts.reshape(-1, 2)
            curr_xy = curr_pts.reshape(-1, 2)
            back_xy = back_pts.reshape(-1, 2)
            roundtrip_error = np.linalg.norm(prev_xy - back_xy, axis=1)
            good = (
                (status_fwd.reshape(-1) == 1)
                & (status_back.reshape(-1) == 1)
                & (roundtrip_error < 1.0)
            )
            prev_good = prev_xy[good]
            curr_good = curr_xy[good]
            feature_counts[t] = int(len(prev_good))
            if len(prev_good) < min_inlier_count:
                cumulative[t] = cumulative[t - 1]
                prev_gray = curr_gray
                continue

            affine, inliers = cv2.estimateAffinePartial2D(
                prev_good,
                curr_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold_px,
            )
            if affine is None or inliers is None or not np.isfinite(affine).all():
                cumulative[t] = cumulative[t - 1]
                prev_gray = curr_gray
                continue

            inlier_mask = inliers.reshape(-1).astype(bool)
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < min_inlier_count:
                cumulative[t] = cumulative[t - 1]
                prev_gray = curr_gray
                continue

            pairwise[t] = affine.astype(np.float64)
            confidence[t] = inlier_count / max(1, len(prev_good))
            inlier_counts[t] = inlier_count

            prev_to_current_inv = _affine_to_homogeneous(_invert_affine(pairwise[t]))
            cumulative[t] = _homogeneous_to_affine(
                _affine_to_homogeneous(cumulative[t - 1]) @ prev_to_current_inv
            )
            prev_gray = curr_gray
    finally:
        cap.release()

    return CameraMotion(
        pairwise,
        cumulative,
        confidence,
        inlier_counts,
        feature_counts,
        frames_without_mask=frames_without_mask,
        min_inlier_count=min_inlier_count,
    )


def warp_landmarks_remove_camera_motion(
    landmarks_2d_pixel: np.ndarray,
    camera_motion: CameraMotion,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Map valid pixel landmarks into frame 0's reference frame.

    Invalid frames return ``NaN`` x/y coordinates while preserving visibility.
    """
    landmarks = np.asarray(landmarks_2d_pixel, dtype=np.float64)
    if landmarks.ndim != 3 or landmarks.shape[2] < 3:
        raise ValueError("landmarks_2d_pixel must have shape (T, J, 3+)")
    if camera_motion.cumulative_affines.shape[:1] != landmarks.shape[:1]:
        raise ValueError("camera_motion and landmarks must have the same frame count")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (landmarks.shape[0],):
        raise ValueError("valid_mask must have shape (T,)")

    output = np.full((landmarks.shape[0], landmarks.shape[1], 3), np.nan, dtype=np.float64)
    output[:, :, 2] = landmarks[:, :, 2]
    for t in np.flatnonzero(valid):
        xy1 = np.column_stack(
            [
                landmarks[t, :, 0],
                landmarks[t, :, 1],
                np.ones(landmarks.shape[1], dtype=np.float64),
            ]
        )
        warped = (camera_motion.cumulative_affines[t] @ xy1.T).T
        finite = np.isfinite(warped).all(axis=1)
        output[t, finite, 0] = warped[finite, 0]
        output[t, finite, 1] = warped[finite, 1]
    return output


def camera_motion_valid_mask(
    camera_motion: CameraMotion,
    min_confidence: float = 0.5,
    min_inliers: int = 20,
) -> np.ndarray:
    """Return frames with enough background-flow support for egomotion use."""
    valid = (camera_motion.confidence >= min_confidence) & (
        camera_motion.inlier_counts >= min_inliers
    )
    if len(valid):
        valid[0] = True
    return valid.astype(bool)
