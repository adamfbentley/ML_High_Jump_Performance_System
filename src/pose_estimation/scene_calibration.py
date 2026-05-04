"""Scene-fixed calibration from high-jump apparatus anchors.

Phase 9c addresses a specific failure mode in panned phone footage: anatomical
scale can convert pixels to metres, but it cannot remove camera pan.  The
crossbar/uprights provide ground-fixed scene anchors with known metric
separation, allowing image coordinates to be mapped into a bar-centred scene
frame when the apparatus is visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneAnchors:
    """Per-frame high-jump apparatus anchor detections.

    Arrays are shaped ``(T, 2)`` in image pixels as ``(x, y)`` with ``NaN`` for
    missing anchors.  Pixel Y points down in the source image.  Scene Y points
    up after homography fitting.
    """

    upright_left_base_px: np.ndarray | None
    upright_right_base_px: np.ndarray | None
    upright_left_top_px: np.ndarray | None
    upright_right_top_px: np.ndarray | None
    confidence: np.ndarray
    upright_separation_m: float = 4.02
    bar_height_m: float | None = None
    crossbar_confirmed: np.ndarray | None = None


def _infer_frame_count(anchors: SceneAnchors) -> int:
    if anchors.confidence is not None:
        return int(len(anchors.confidence))
    for values in (
        anchors.upright_left_base_px,
        anchors.upright_right_base_px,
        anchors.upright_left_top_px,
        anchors.upright_right_top_px,
    ):
        if values is not None:
            return int(len(values))
    return 0


def _anchor_array(values: np.ndarray | None, n_frames: int) -> np.ndarray:
    if values is None:
        return np.full((n_frames, 2), np.nan, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n_frames, 2):
        raise ValueError(f"anchor arrays must have shape ({n_frames}, 2), got {arr.shape}")
    return arr


def _is_valid_point(point: np.ndarray) -> bool:
    return bool(point.shape == (2,) and np.isfinite(point).all())


def _scene_points(anchors: SceneAnchors) -> dict[str, np.ndarray] | None:
    if anchors.bar_height_m is None or anchors.bar_height_m <= 0:
        return None
    half_width = anchors.upright_separation_m / 2.0
    return {
        "left_base": np.array([-half_width, 0.0], dtype=np.float64),
        "right_base": np.array([half_width, 0.0], dtype=np.float64),
        "left_top": np.array([-half_width, anchors.bar_height_m], dtype=np.float64),
        "right_top": np.array([half_width, anchors.bar_height_m], dtype=np.float64),
    }


def estimate_scene_scale_mpp(anchors: SceneAnchors) -> np.ndarray:
    """Estimate horizontal metres-per-pixel from upright separation.

    Returns:
        ``(T,)`` array with ``NaN`` where neither base nor top separation is
        available.
    """
    n_frames = _infer_frame_count(anchors)
    left_base = _anchor_array(anchors.upright_left_base_px, n_frames)
    right_base = _anchor_array(anchors.upright_right_base_px, n_frames)
    left_top = _anchor_array(anchors.upright_left_top_px, n_frames)
    right_top = _anchor_array(anchors.upright_right_top_px, n_frames)

    scale = np.full(n_frames, np.nan, dtype=np.float64)
    for t in range(n_frames):
        candidates = []
        for left, right in ((left_base[t], right_base[t]), (left_top[t], right_top[t])):
            if _is_valid_point(left) and _is_valid_point(right):
                pixels = float(np.linalg.norm(right - left))
                if pixels > 1.0:
                    candidates.append(anchors.upright_separation_m / pixels)
        if candidates:
            scale[t] = float(np.median(candidates))
    return scale


def scene_anchor_diagnostics(anchors: SceneAnchors) -> dict:
    """Summarise per-frame anchor completeness for reports and validation."""
    n_frames = _infer_frame_count(anchors)
    arrays = (
        _anchor_array(anchors.upright_left_base_px, n_frames),
        _anchor_array(anchors.upright_right_base_px, n_frames),
        _anchor_array(anchors.upright_left_top_px, n_frames),
        _anchor_array(anchors.upright_right_top_px, n_frames),
    )
    valid_counts = np.zeros(n_frames, dtype=np.int32)
    for arr in arrays:
        valid_counts += np.isfinite(arr).all(axis=1)

    histogram = {
        str(n): int(np.count_nonzero(valid_counts == n))
        for n in range(5)
    }
    four_anchor_pct = float(np.mean(valid_counts >= 4) * 100.0) if n_frames else 0.0

    crossbar = anchors.crossbar_confirmed
    if crossbar is not None:
        crossbar_arr = np.asarray(crossbar, dtype=bool)
        n = min(n_frames, len(crossbar_arr))
        crossbar_pct = float(np.mean(crossbar_arr[:n]) * 100.0) if n else 0.0
    else:
        crossbar_pct = 0.0

    return {
        "anchor_valid_counts_histogram": histogram,
        "anchor_four_point_valid_pct": round(four_anchor_pct, 2),
        "crossbar_confirmed_pct": round(crossbar_pct, 2),
    }


def fit_per_frame_homography(anchors: SceneAnchors) -> np.ndarray:
    """Fit image-pixel to scene-metre homographies.

    Scene origin is the midpoint of the upright bases.  Scene X runs along the
    crossbar, positive toward the right upright.  Scene Y is vertical, positive
    upward.  Frames with insufficient anchors are filled with ``NaN``.

    Args:
        anchors: Per-frame apparatus anchor positions.

    Returns:
        ``(T, 3, 3)`` homography matrices mapping image ``(x_px, y_px, 1)`` to
        scene ``(x_m, y_m, 1)``.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for scene homography fitting") from exc

    n_frames = _infer_frame_count(anchors)
    homographies = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
    dst_points = _scene_points(anchors)
    if dst_points is None:
        logger.info("Scene homography skipped: bar_height_m is required for vertical scale.")
        return homographies

    source_arrays = {
        "left_base": _anchor_array(anchors.upright_left_base_px, n_frames),
        "right_base": _anchor_array(anchors.upright_right_base_px, n_frames),
        "left_top": _anchor_array(anchors.upright_left_top_px, n_frames),
        "right_top": _anchor_array(anchors.upright_right_top_px, n_frames),
    }

    for t in range(n_frames):
        src: list[np.ndarray] = []
        dst: list[np.ndarray] = []
        for name, points in source_arrays.items():
            point = points[t]
            if _is_valid_point(point):
                src.append(point)
                dst.append(dst_points[name])

        # A full projective homography needs four point correspondences.  With
        # three anchors we deliberately mark the frame invalid so the caller can
        # use anatomical fallback rather than inventing geometry.
        if len(src) < 4:
            continue

        h_mat, _status = cv2.findHomography(
            np.asarray(src, dtype=np.float64),
            np.asarray(dst, dtype=np.float64),
            method=0,
        )
        if h_mat is None or not np.isfinite(h_mat).all() or abs(h_mat[2, 2]) < 1e-12:
            continue
        homographies[t] = h_mat / h_mat[2, 2]

    return homographies


def homography_valid_mask(
    homographies: np.ndarray,
    confidence: np.ndarray | None = None,
    min_confidence: float = 0.5,
) -> np.ndarray:
    """Return frames whose homography and confidence are usable."""
    if homographies.ndim != 3 or homographies.shape[1:] != (3, 3):
        raise ValueError("homographies must have shape (T, 3, 3)")
    valid = np.isfinite(homographies).all(axis=(1, 2))
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64)
        if conf.shape != (homographies.shape[0],):
            raise ValueError("confidence must have shape (T,)")
        valid &= conf >= min_confidence
    return valid


def warp_landmarks_to_scene(
    landmarks_image_px: np.ndarray,
    homographies: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Warp image-space landmarks into the scene frame.

    Args:
        landmarks_image_px: ``(T, 33, 3)`` as pixel ``x, y, visibility``.
        homographies: ``(T, 3, 3)`` image-pixel to scene-metre transforms.
        valid_mask: ``(T,)`` boolean mask selecting reliable homographies.

    Returns:
        ``(T, 33, 3)`` as scene ``x_m, y_m, visibility``.  Invalid frames have
        ``NaN`` x/y coordinates and preserved visibility.
    """
    landmarks = np.asarray(landmarks_image_px, dtype=np.float64)
    if landmarks.ndim != 3 or landmarks.shape[2] < 3:
        raise ValueError("landmarks_image_px must have shape (T, J, 3+)")
    if homographies.shape[:1] != landmarks.shape[:1]:
        raise ValueError("landmarks and homographies must have the same frame count")
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
        warped = (homographies[t] @ xy1.T).T
        denom = warped[:, 2]
        finite = np.isfinite(warped).all(axis=1) & (np.abs(denom) > 1e-12)
        output[t, finite, 0] = warped[finite, 0] / denom[finite]
        output[t, finite, 1] = warped[finite, 1] / denom[finite]

    return output


def _frames_from_video(video_path: Path) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for scene-anchor detection") from exc

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def _to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.shape[2] >= 3:
        return cv2.cvtColor(arr[:, :, :3].astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)
    raise ValueError("frames must be grayscale or RGB arrays")


def _detect_anchors_in_frame(
    frame: np.ndarray,
    *,
    max_detection_width: int = 960,
) -> tuple[dict[str, np.ndarray], float, bool]:
    """Detect upright/base/top anchors in one frame using edges and Hough lines."""
    import cv2

    gray = _to_gray(frame)
    original_height, original_width = gray.shape[:2]
    scale_back = 1.0
    if max_detection_width > 0 and original_width > max_detection_width:
        scale_back = original_width / max_detection_width
        resized_height = max(1, int(round(original_height / scale_back)))
        gray = cv2.resize(gray, (max_detection_width, resized_height), interpolation=cv2.INTER_AREA)

    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(30, width // 60),
        minLineLength=max(30, int(height * 0.12)),
        maxLineGap=max(8, int(height * 0.025)),
    )

    empty = {
        "left_base": np.array([np.nan, np.nan], dtype=np.float64),
        "right_base": np.array([np.nan, np.nan], dtype=np.float64),
        "left_top": np.array([np.nan, np.nan], dtype=np.float64),
        "right_top": np.array([np.nan, np.nan], dtype=np.float64),
    }
    if lines is None:
        return empty, 0.0, False

    verticals = []
    horizontals = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1.0:
            continue
        angle_deg = abs(float(np.degrees(np.arctan2(dy, dx))))
        angle_from_horizontal = min(angle_deg, abs(180.0 - angle_deg))
        angle_from_vertical = abs(90.0 - angle_deg)

        if angle_from_vertical <= 10.0 and length >= height * 0.12:
            endpoints = np.array([[x1, y1], [x2, y2]], dtype=np.float64)
            top = endpoints[np.argmin(endpoints[:, 1])]
            base = endpoints[np.argmax(endpoints[:, 1])]
            verticals.append(
                {
                    "x": float(np.mean(endpoints[:, 0])),
                    "top": top,
                    "base": base,
                    "length": length,
                }
            )
        elif angle_from_horizontal <= 8.0 and length >= width * 0.15:
            endpoints = np.array([[x1, y1], [x2, y2]], dtype=np.float64)
            horizontals.append(
                {
                    "x_min": float(endpoints[:, 0].min()),
                    "x_max": float(endpoints[:, 0].max()),
                    "y": float(endpoints[:, 1].mean()),
                    "length": length,
                }
            )

    if len(verticals) < 2:
        return empty, 0.0, False

    best_pair = None
    best_score = -np.inf
    for i, left in enumerate(verticals):
        for right in verticals[i + 1:]:
            a, b = sorted((left, right), key=lambda item: item["x"])
            separation = b["x"] - a["x"]
            if separation < width * 0.12:
                continue
            base_mismatch = abs(float(a["base"][1] - b["base"][1]))
            top_mismatch = abs(float(a["top"][1] - b["top"][1]))
            if base_mismatch > height * 0.25:
                continue
            score = separation + 0.5 * (a["length"] + b["length"]) - 2.0 * top_mismatch
            score -= base_mismatch
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        return empty, 0.0, False

    left, right = best_pair
    left_top = left["top"].copy()
    right_top = right["top"].copy()
    expected_top_y = float(np.mean([left_top[1], right_top[1]]))

    crossbar = None
    for horizontal in horizontals:
        spans_uprights = (
            horizontal["x_min"] <= left["x"] + width * 0.05
            and horizontal["x_max"] >= right["x"] - width * 0.05
        )
        near_tops = abs(horizontal["y"] - expected_top_y) <= height * 0.12
        if spans_uprights and near_tops:
            if crossbar is None or horizontal["length"] > crossbar["length"]:
                crossbar = horizontal

    if crossbar is not None:
        left_top[1] = crossbar["y"]
        right_top[1] = crossbar["y"]

    anchors = {
        "left_base": left["base"].astype(np.float64),
        "right_base": right["base"].astype(np.float64),
        "left_top": left_top.astype(np.float64),
        "right_top": right_top.astype(np.float64),
    }
    confidence = 0.65
    if crossbar is not None:
        confidence += 0.25
    confidence += min(0.10, max(0.0, best_score) / max(width + height, 1.0) * 0.05)
    if scale_back != 1.0:
        for point in anchors.values():
            point *= scale_back
    return anchors, float(min(confidence, 1.0)), crossbar is not None


def _track_missing_anchors(
    frames: Sequence[np.ndarray],
    anchors_by_name: dict[str, np.ndarray],
    confidence: np.ndarray,
) -> None:
    """Fill short detection gaps with sparse Lucas-Kanade optical flow."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - dependency is declared
        return

    if len(frames) < 2:
        return

    grays = [_to_gray(frame) for frame in frames]
    names = ("left_base", "right_base", "left_top", "right_top")
    for t in range(1, len(frames)):
        if confidence[t] >= 0.5 or confidence[t - 1] < 0.5:
            continue
        prev_points = np.asarray([anchors_by_name[name][t - 1] for name in names], dtype=np.float32)
        if not np.isfinite(prev_points).all():
            continue
        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            grays[t - 1],
            grays[t],
            prev_points.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None or int(status.sum()) != len(names):
            continue
        for idx, name in enumerate(names):
            anchors_by_name[name][t] = next_points.reshape(-1, 2)[idx]
        confidence[t] = max(0.5, confidence[t - 1] * 0.8)


def detect_scene_anchors(
    frames: Sequence[np.ndarray] | Path,
    *,
    bar_height_m: float | None = None,
    upright_separation_m: float = 4.02,
    detection_stride: int = 15,
    max_detection_width: int = 640,
) -> SceneAnchors:
    """Detect crossbar/upright scene anchors in video frames.

    This is a classical-CV detector: Canny edges, probabilistic Hough lines,
    vertical-upright pairing, horizontal-crossbar confirmation, then sparse
    optical-flow gap filling.  It is intentionally deterministic and local.
    """
    frame_list = _frames_from_video(frames) if isinstance(frames, Path) else list(frames)
    n_frames = len(frame_list)
    anchors_by_name = {
        "left_base": np.full((n_frames, 2), np.nan, dtype=np.float64),
        "right_base": np.full((n_frames, 2), np.nan, dtype=np.float64),
        "left_top": np.full((n_frames, 2), np.nan, dtype=np.float64),
        "right_top": np.full((n_frames, 2), np.nan, dtype=np.float64),
    }
    confidence = np.zeros(n_frames, dtype=np.float64)
    crossbar_confirmed = np.zeros(n_frames, dtype=bool)

    stride = max(1, int(detection_stride))
    for t, frame in enumerate(frame_list):
        if t % stride != 0:
            continue
        detected, conf, has_crossbar = _detect_anchors_in_frame(
            frame,
            max_detection_width=max_detection_width,
        )
        for name, point in detected.items():
            anchors_by_name[name][t] = point
        confidence[t] = conf
        crossbar_confirmed[t] = has_crossbar

    _track_missing_anchors(frame_list, anchors_by_name, confidence)

    return SceneAnchors(
        upright_left_base_px=anchors_by_name["left_base"],
        upright_right_base_px=anchors_by_name["right_base"],
        upright_left_top_px=anchors_by_name["left_top"],
        upright_right_top_px=anchors_by_name["right_top"],
        confidence=confidence,
        upright_separation_m=upright_separation_m,
        bar_height_m=bar_height_m,
        crossbar_confirmed=crossbar_confirmed,
    )
