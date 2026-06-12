"""Detect stable-window high-jump apparatus anchors from labelled base posts.

This helper is for the takeoff-stationary clips where the camera is stable over
the final plant/takeoff window.  The user still provides the two visible upright
base posts; the script then searches upward for the red crossbar using the
base-post line as the expected horizontal direction.

It writes the same one-frame anchor JSON consumed by
``scripts/analyze_stable_takeoff_window.py`` plus a debug image for review.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_UPRIGHT_SEPARATION_M = 4.02


@dataclass(frozen=True)
class BarScanConfig:
    """Tuning parameters for the upward bar scan."""

    scan_step_px: int = 3
    band_radius_px: int = 4
    sweep_samples: int = 140
    sweep_bins: int = 14
    min_bin_coverage: float = 0.35
    min_red_bin_coverage: float = 0.18
    min_pad_to_bar_gap_px: float = 70.0
    min_offset_from_base_px: float = 45.0
    max_offset_from_base_px: float | None = None
    refine_bar_window_px: float = 70.0
    max_refined_bar_abs_slope: float = 0.12
    min_refined_bar_length_px: float = 80.0


@dataclass(frozen=True)
class LineCandidate:
    """A horizontal-ish line candidate parallel to the post-base line."""

    y_center_px: float
    left_px: np.ndarray
    right_px: np.ndarray
    score: float
    coverage: float
    red_coverage: float


@dataclass(frozen=True)
class AnchorDetection:
    """Detected apparatus anchors and diagnostics."""

    points_px: dict[str, list[float]]
    selected_bar: LineCandidate
    pad_top: LineCandidate | None
    candidates: list[LineCandidate]


@dataclass(frozen=True)
class RefinedApparatusGeometry:
    """Refined top and bottom anchor geometry."""

    selected_bar: LineCandidate
    left_base_px: np.ndarray
    right_base_px: np.ndarray
    left_post_line: np.ndarray | None
    right_post_line: np.ndarray | None


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for apparatus anchor detection") from exc
    return cv2


def _parse_point(text: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("points must be formatted as x,y")
    try:
        point = np.asarray([float(parts[0]), float(parts[1])], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("points must contain numeric x,y values") from exc
    if not np.isfinite(point).all():
        raise argparse.ArgumentTypeError("points must be finite")
    return point


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def _read_frame(args: argparse.Namespace) -> np.ndarray:
    cv2 = _require_cv2()
    if args.frame_image is not None:
        frame = cv2.imread(str(args.frame_image))
        if frame is None:
            raise RuntimeError(f"Could not read {args.frame_image}")
        return frame
    if args.video is None:
        raise ValueError("provide --frame-image or --video")
    return _read_video_frame(args.video, args.frame_index)


def _normalise_map(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(values[mask], dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    scale = float(np.percentile(valid, 95))
    if scale <= 1e-6:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values / scale, 0.0, 2.0) / 2.0


def _sample_nearest(values: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = values.shape[:2]
    xi = np.rint(xy[:, 0]).astype(int)
    yi = np.rint(xy[:, 1]).astype(int)
    inside = (0 <= xi) & (xi < w) & (0 <= yi) & (yi < h)
    sampled = np.zeros(len(xy), dtype=np.float64)
    sampled[inside] = values[yi[inside], xi[inside]]
    return sampled


def _support_along_line(
    response: np.ndarray,
    red_response: np.ndarray,
    xy: np.ndarray,
    normal: np.ndarray,
    band_radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = range(-int(band_radius_px), int(band_radius_px) + 1)
    response_stack = []
    red_stack = []
    for offset in offsets:
        shifted = xy + normal.reshape(1, 2) * float(offset)
        response_stack.append(_sample_nearest(response, shifted))
        red_stack.append(_sample_nearest(red_response, shifted))
    return np.max(response_stack, axis=0), np.max(red_stack, axis=0)


def _bin_coverages(
    support: np.ndarray,
    red_support: np.ndarray,
    *,
    bins: int,
    threshold: float,
    red_threshold: float,
) -> tuple[float, float, np.ndarray]:
    chunks = np.array_split(np.arange(len(support)), max(1, int(bins)))
    bin_scores = []
    bin_red_scores = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        bin_scores.append(float(np.percentile(support[chunk], 70)))
        bin_red_scores.append(float(np.percentile(red_support[chunk], 70)))
    scores = np.asarray(bin_scores, dtype=np.float64)
    red_scores = np.asarray(bin_red_scores, dtype=np.float64)
    if scores.size == 0:
        return 0.0, 0.0, scores
    return (
        float(np.mean(scores >= threshold)),
        float(np.mean(red_scores >= red_threshold)),
        scores,
    )


def _candidate_clusters(
    candidates: list[LineCandidate],
    *,
    config: BarScanConfig,
) -> list[LineCandidate]:
    usable = [
        candidate
        for candidate in candidates
        if candidate.coverage >= config.min_bin_coverage
        and candidate.red_coverage >= config.min_red_bin_coverage
    ]
    if not usable:
        return []

    usable.sort(key=lambda candidate: candidate.y_center_px, reverse=True)
    clusters: list[list[LineCandidate]] = []
    for candidate in usable:
        if (
            not clusters
            or abs(candidate.y_center_px - clusters[-1][-1].y_center_px)
            > config.scan_step_px * 2.1
        ):
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    return [max(cluster, key=lambda candidate: candidate.score) for cluster in clusters]


def _red_response(frame_bgr: np.ndarray) -> np.ndarray:
    frame = frame_bgr[:, :, :3].astype(np.float64)
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return np.maximum(0.0, r - 0.70 * g - 0.45 * b)


def _line_from_points(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Return homogeneous image line through two points."""
    a = np.asarray([point_a[0], point_a[1], 1.0], dtype=np.float64)
    b = np.asarray([point_b[0], point_b[1], 1.0], dtype=np.float64)
    line = np.cross(a, b)
    norm = float(np.linalg.norm(line[:2]))
    if norm <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return line / norm


def _bar_line_from_slope(slope: float, intercept: float) -> np.ndarray:
    """Return homogeneous line for y = slope * x + intercept."""
    line = np.asarray([slope, -1.0, intercept], dtype=np.float64)
    return line / float(np.linalg.norm(line[:2]))


def _line_intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    point = np.cross(line_a, line_b)
    if not np.isfinite(point).all() or abs(float(point[2])) <= 1e-9:
        return None
    return np.asarray([point[0] / point[2], point[1] / point[2]], dtype=np.float64)


def _x_at_y(line: np.ndarray, y_px: float) -> float | None:
    """Evaluate a near-vertical line at image y."""
    a, b, c = [float(value) for value in line]
    if abs(a) <= 1e-9:
        return None
    x = -(b * float(y_px) + c) / a
    return float(x) if np.isfinite(x) else None


def _fit_local_upright_line(
    frame_bgr: np.ndarray,
    *,
    base_px: np.ndarray,
    bar_line: np.ndarray,
    approximate_bar_y_px: float,
) -> np.ndarray | None:
    """Fit the visible upright line near a base post.

    The top anchor is the intersection of this post line with the fitted bar
    line.  We score candidate vertical lines by proximity to the labelled base
    post, intersection closeness to the already-detected bar band, vertical
    length, and near-vertical angle.
    """
    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    base = np.asarray(base_px, dtype=np.float64)
    x0 = int(max(0, np.floor(base[0] - 85)))
    x1 = int(min(width, np.ceil(base[0] + 85)))
    y0 = int(max(0, np.floor(min(base[1], approximate_bar_y_px) - 95)))
    y1 = int(min(height, np.ceil(max(base[1], approximate_bar_y_px) + 45)))
    if x1 <= x0 + 5 or y1 <= y0 + 20:
        return None

    roi = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 135)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=22,
        minLineLength=max(55, int((y1 - y0) * 0.24)),
        maxLineGap=26,
    )
    if lines is None:
        return None

    best_line: np.ndarray | None = None
    best_score = -np.inf
    line_candidates: list[dict[str, object]] = []
    for raw in lines[:, 0, :]:
        x1_l, y1_l, x2_l, y2_l = [float(value) for value in raw]
        point_a = np.asarray([x1_l + x0, y1_l + y0], dtype=np.float64)
        point_b = np.asarray([x2_l + x0, y2_l + y0], dtype=np.float64)
        dx = float(point_b[0] - point_a[0])
        dy = float(point_b[1] - point_a[1])
        length = float(np.hypot(dx, dy))
        if length < 1.0:
            continue
        angle_from_vertical = abs(90.0 - abs(float(np.degrees(np.arctan2(dy, dx)))))
        if angle_from_vertical > 14.0:
            continue
        candidate_line = _line_from_points(point_a, point_b)
        if not np.isfinite(candidate_line).all():
            continue
        x_at_base = _x_at_y(candidate_line, float(base[1]))
        if x_at_base is None:
            continue
        intersection = _line_intersection(candidate_line, bar_line)
        if intersection is None:
            continue
        if not (x0 - 20 <= intersection[0] <= x1 + 20 and y0 - 20 <= intersection[1] <= y1 + 20):
            continue

        y_span = abs(dy)
        base_error = abs(float(x_at_base - base[0]))
        bar_band_error = abs(float(intersection[1] - approximate_bar_y_px))
        score = (
            length
            + 0.45 * y_span
            - 3.0 * base_error
            - 1.4 * bar_band_error
            - 7.0 * angle_from_vertical
        )
        if score > best_score:
            best_score = float(score)
            best_line = candidate_line
        line_candidates.append(
            {
                "line": candidate_line,
                "length": float(length),
                "angle_from_vertical": float(angle_from_vertical),
                "score": float(score),
            }
        )

    if best_line is None:
        return None
    paired_centerline = _centerline_from_upright_edge_pair(
        line_candidates,
        base_px=base,
        approximate_bar_y_px=approximate_bar_y_px,
    )
    if paired_centerline is not None:
        return paired_centerline
    refined_line = _refine_upright_centerline(
        frame_bgr,
        base_px=base,
        rough_line=best_line,
        approximate_bar_y_px=approximate_bar_y_px,
    )
    return refined_line if refined_line is not None else best_line


def _refine_upright_centerline(
    frame_bgr: np.ndarray,
    *,
    base_px: np.ndarray,
    rough_line: np.ndarray,
    approximate_bar_y_px: float,
) -> np.ndarray | None:
    """Refine a post edge/axis estimate to the bright upright centreline."""
    height, width = frame_bgr.shape[:2]
    base = np.asarray(base_px, dtype=np.float64)
    if not np.isfinite(rough_line).all():
        return None

    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    y0 = int(max(0, np.floor(min(base[1], approximate_bar_y_px) - 65)))
    y1 = int(min(height, np.ceil(max(base[1], approximate_bar_y_px) + 45)))
    row_centres: list[tuple[float, float, float]] = []

    for y_global in range(y0, y1):
        x_pred = _x_at_y(rough_line, float(y_global))
        if x_pred is None:
            continue
        x_left = int(max(0, np.floor(x_pred - 30)))
        x_right = int(min(width, np.ceil(x_pred + 30)))
        if x_right <= x_left + 4:
            continue
        xs = np.arange(x_left, x_right, dtype=np.float64)
        row = gray[y_global, x_left:x_right]
        threshold = max(float(np.percentile(row, 70)), float(np.mean(row) + 0.25 * np.std(row)))
        active = row >= threshold
        if not np.any(active):
            continue
        pred_idx = int(np.clip(round(float(x_pred - x_left)), 0, len(row) - 1))
        active_indices = np.flatnonzero(active)
        nearest = int(active_indices[np.argmin(np.abs(active_indices - pred_idx))])
        if abs(nearest - pred_idx) > 16:
            continue
        start = nearest
        while start > 0 and active[start - 1]:
            start -= 1
        end = nearest + 1
        while end < len(active) and active[end]:
            end += 1
        if end - start < 2 or end - start > 34:
            continue
        component_xs = xs[start:end]
        component_row = row[start:end]
        weights = np.maximum(component_row - threshold, 0.0)
        if float(np.sum(weights)) <= 8.0:
            continue
        x_center = float(np.sum(component_xs * weights) / np.sum(weights))
        if abs(x_center - x_pred) > 22.0:
            continue
        row_centres.append((float(y_global), x_center, float(np.sum(weights))))

    if len(row_centres) < 12:
        return None

    ys = np.asarray([item[0] for item in row_centres], dtype=np.float64)
    xs = np.asarray([item[1] for item in row_centres], dtype=np.float64)
    weights = np.asarray([item[2] for item in row_centres], dtype=np.float64)
    for _ in range(3):
        design = np.column_stack([ys, np.ones_like(ys)])
        slope_xy, intercept_x = np.linalg.lstsq(
            design * weights[:, None],
            xs * weights,
            rcond=None,
        )[0]
        residual = np.abs(xs - (slope_xy * ys + intercept_x))
        keep = residual <= max(2.5, float(np.percentile(residual, 75)))
        if int(np.count_nonzero(keep)) < 12 or np.all(keep):
            break
        ys, xs, weights = ys[keep], xs[keep], weights[keep]

    top_y = float(np.min(ys))
    bottom_y = float(np.max(ys))
    point_top = np.asarray([slope_xy * top_y + intercept_x, top_y], dtype=np.float64)
    point_bottom = np.asarray([slope_xy * bottom_y + intercept_x, bottom_y], dtype=np.float64)
    return _line_from_points(point_top, point_bottom)


def _centerline_from_upright_edge_pair(
    line_candidates: list[dict[str, object]],
    *,
    base_px: np.ndarray,
    approximate_bar_y_px: float,
) -> np.ndarray | None:
    """Use paired vertical post edges to estimate the upright centreline."""
    base = np.asarray(base_px, dtype=np.float64)
    best_line: np.ndarray | None = None
    best_score = -np.inf
    for idx, first in enumerate(line_candidates):
        for second in line_candidates[idx + 1:]:
            line_a = np.asarray(first["line"], dtype=np.float64)
            line_b = np.asarray(second["line"], dtype=np.float64)
            x_a_base = _x_at_y(line_a, float(base[1]))
            x_b_base = _x_at_y(line_b, float(base[1]))
            x_a_bar = _x_at_y(line_a, float(approximate_bar_y_px))
            x_b_bar = _x_at_y(line_b, float(approximate_bar_y_px))
            if None in (x_a_base, x_b_base, x_a_bar, x_b_bar):
                continue

            width_base = abs(float(x_a_base - x_b_base))
            width_bar = abs(float(x_a_bar - x_b_bar))
            median_width = float(np.median([width_base, width_bar]))
            if not (8.0 <= median_width <= 48.0):
                continue
            if abs(width_base - width_bar) > 22.0:
                continue

            center_base_x = float((x_a_base + x_b_base) / 2.0)
            center_bar_x = float((x_a_bar + x_b_bar) / 2.0)
            base_error = abs(center_base_x - float(base[0]))
            if base_error > 55.0:
                continue

            centerline = _line_from_points(
                np.asarray([center_base_x, float(base[1])], dtype=np.float64),
                np.asarray([center_bar_x, float(approximate_bar_y_px)], dtype=np.float64),
            )
            if not np.isfinite(centerline).all():
                continue

            length_sum = float(first["length"]) + float(second["length"])
            angle_penalty = float(first["angle_from_vertical"]) + float(
                second["angle_from_vertical"]
            )
            width_penalty = abs(width_base - width_bar)
            score = length_sum - 4.0 * base_error - 2.0 * width_penalty - 5.0 * angle_penalty
            if score > best_score:
                best_score = score
                best_line = centerline
    return best_line


def _refine_base_on_post_ground(
    frame_bgr: np.ndarray,
    *,
    rough_base_px: np.ndarray,
    post_line: np.ndarray | None,
) -> np.ndarray:
    """Place the base anchor at post-centre/ground contact.

    The rough user point tells us the local neighbourhood.  When the post line
    is available, the x-coordinate comes from the centreline of the upright; the
    y-coordinate is refined to the strongest local lower edge near the rough
    ground contact.
    """
    cv2 = _require_cv2()
    base = np.asarray(rough_base_px, dtype=np.float64)
    if post_line is None or not np.isfinite(post_line).all():
        return base.copy()

    height, width = frame_bgr.shape[:2]
    x0 = int(max(0, np.floor(base[0] - 90)))
    x1 = int(min(width, np.ceil(base[0] + 90)))
    y0 = int(max(0, np.floor(base[1] - 85)))
    y1 = int(min(height, np.ceil(base[1] + 65)))
    if x1 <= x0 + 5 or y1 <= y0 + 5:
        return base.copy()

    roi = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))

    row_scores: list[tuple[float, float]] = []
    for y_global in np.arange(y0, y1, dtype=np.float64):
        x_center = _x_at_y(post_line, y_global)
        if x_center is None:
            continue
        x_left = int(max(x0, np.floor(x_center - 42)))
        x_right = int(min(x1, np.ceil(x_center + 42)))
        if x_right <= x_left + 3:
            continue
        yy = int(round(y_global)) - y0
        cols = slice(x_left - x0, x_right - x0)
        horizontal_edge = float(np.percentile(gy[yy, cols], 75))
        vertical_edge = float(np.percentile(gx[yy, cols], 60))
        closeness = abs(float(y_global - base[1]))
        lower_prior = max(0.0, float(y_global - base[1])) * 0.08
        score = horizontal_edge + 0.22 * vertical_edge + lower_prior - 0.35 * closeness
        row_scores.append((score, float(y_global)))

    if not row_scores:
        y_contact = float(base[1])
    else:
        _score, y_contact = max(row_scores, key=lambda item: item[0])

    x_contact = _x_at_y(post_line, y_contact)
    if x_contact is None:
        return base.copy()
    contact = np.asarray([x_contact, y_contact], dtype=np.float64)
    if not np.isfinite(contact).all():
        return base.copy()
    return contact


def _fit_local_red_bar_line(
    frame_bgr: np.ndarray,
    *,
    left_base_px: np.ndarray,
    right_base_px: np.ndarray,
    initial_bar: LineCandidate,
    config: BarScanConfig,
) -> RefinedApparatusGeometry | None:
    """Refine the bar angle from local red pixels, then intersect post axes.

    The upward scan deliberately finds the approximate bar band.  This function
    then stops using the base-post slope: it fits the actual red line in a local
    window and evaluates that line at the two upright axes.  Those line/post
    intersections are the top anchors used by the camera solve.
    """
    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    left = np.asarray(left_base_px, dtype=np.float64)
    right = np.asarray(right_base_px, dtype=np.float64)
    if right[0] <= left[0]:
        left, right = right, left

    x0 = int(max(0, np.floor(left[0] - 45)))
    x1 = int(min(width, np.ceil(right[0] + 45)))
    y0 = int(max(0, np.floor(initial_bar.y_center_px - config.refine_bar_window_px)))
    y1 = int(min(height, np.ceil(initial_bar.y_center_px + config.refine_bar_window_px)))
    if x1 <= x0 + 5 or y1 <= y0 + 5:
        return None

    roi = frame_bgr[y0:y1, x0:x1]
    red = _red_response(roi)
    if not np.isfinite(red).any() or float(np.max(red)) <= 1e-6:
        return None

    threshold = max(18.0, float(np.percentile(red, 90)))
    mask = (red >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=24,
        minLineLength=max(30, int(config.min_refined_bar_length_px)),
        maxLineGap=28,
    )

    candidates: list[tuple[float, float, float, float]] = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1_l, y1_l, x2_l, y2_l = [float(value) for value in line]
            x1_g, x2_g = x1_l + x0, x2_l + x0
            y1_g, y2_g = y1_l + y0, y2_l + y0
            dx = x2_g - x1_g
            if abs(dx) < 1.0:
                continue
            slope = float((y2_g - y1_g) / dx)
            if abs(slope) > config.max_refined_bar_abs_slope:
                continue
            length = float(np.hypot(dx, y2_g - y1_g))
            if length < config.min_refined_bar_length_px:
                continue
            intercept = float(y1_g - slope * x1_g)
            x_mid = float((left[0] + right[0]) / 2.0)
            y_mid = float(slope * x_mid + intercept)
            closeness = abs(y_mid - initial_bar.y_center_px)
            # Prefer long, local, red-supported lines.  The Hough line can be a
            # visible central segment of the bar; we intentionally extrapolate
            # it to the upright axes below.
            score = length - 2.5 * closeness - 350.0 * abs(slope)
            candidates.append((score, slope, intercept, length))

    if candidates:
        score, slope, intercept, length = max(candidates, key=lambda item: item[0])
    else:
        yy, xx = np.nonzero(mask)
        if len(xx) < 20:
            return None
        x_global = xx.astype(np.float64) + float(x0)
        y_global = yy.astype(np.float64) + float(y0)
        weights = red[yy, xx].astype(np.float64)
        design = np.column_stack([x_global, np.ones_like(x_global)])
        slope, intercept = np.linalg.lstsq(
            design * weights[:, None],
            y_global * weights,
            rcond=None,
        )[0]
        if abs(float(slope)) > config.max_refined_bar_abs_slope:
            return None
        length = float(right[0] - left[0])
        score = float(np.mean(weights))

    bar_line = _bar_line_from_slope(float(slope), float(intercept))
    left_post_line = _fit_local_upright_line(
        frame_bgr,
        base_px=left,
        bar_line=bar_line,
        approximate_bar_y_px=initial_bar.y_center_px,
    )
    right_post_line = _fit_local_upright_line(
        frame_bgr,
        base_px=right,
        bar_line=bar_line,
        approximate_bar_y_px=initial_bar.y_center_px,
    )
    left_top = (
        _line_intersection(left_post_line, bar_line)
        if left_post_line is not None
        else None
    )
    right_top = (
        _line_intersection(right_post_line, bar_line)
        if right_post_line is not None
        else None
    )
    if left_top is None:
        left_top = np.asarray([left[0], slope * left[0] + intercept], dtype=np.float64)
    if right_top is None:
        right_top = np.asarray([right[0], slope * right[0] + intercept], dtype=np.float64)

    y_center = float(slope * ((left[0] + right[0]) / 2.0) + intercept)
    if abs(y_center - initial_bar.y_center_px) > config.refine_bar_window_px:
        return None

    refined_bar = LineCandidate(
        y_center_px=y_center,
        left_px=left_top,
        right_px=right_top,
        score=float(score),
        coverage=initial_bar.coverage,
        red_coverage=initial_bar.red_coverage,
    )
    refined_left_base = _refine_base_on_post_ground(
        frame_bgr,
        rough_base_px=left,
        post_line=left_post_line,
    )
    refined_right_base = _refine_base_on_post_ground(
        frame_bgr,
        rough_base_px=right,
        post_line=right_post_line,
    )
    return RefinedApparatusGeometry(
        selected_bar=refined_bar,
        left_base_px=refined_left_base,
        right_base_px=refined_right_base,
        left_post_line=left_post_line,
        right_post_line=right_post_line,
    )


def detect_bar_from_base_posts(
    frame_bgr: np.ndarray,
    *,
    left_base_px: np.ndarray,
    right_base_px: np.ndarray,
    config: BarScanConfig = BarScanConfig(),
) -> AnchorDetection:
    """Detect the crossbar line from labelled upright base posts.

    The scan moves upward from the midpoint of the post bases.  At each candidate
    height it samples a narrow band whose direction is parallel to the base-post
    line, then sweeps across the space between the posts and requires line
    support in multiple bins.  The lower red/horizontal structure is treated as
    the landing-pad top; the selected bar is the next red-supported horizontal
    structure above it after a minimum vertical gap.
    """
    cv2 = _require_cv2()
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame_bgr must be a BGR image")

    left = np.asarray(left_base_px, dtype=np.float64)
    right = np.asarray(right_base_px, dtype=np.float64)
    if left.shape != (2,) or right.shape != (2,) or not np.isfinite([left, right]).all():
        raise ValueError("base post points must be finite (2,) arrays")
    if right[0] <= left[0]:
        left, right = right, left

    dx = float(right[0] - left[0])
    if abs(dx) < 1.0:
        raise ValueError("base post x coordinates must be separated")
    slope = float((right[1] - left[1]) / dx)
    x_mid = float((left[0] + right[0]) / 2.0)
    base_y_mid = float((left[1] + right[1]) / 2.0)
    direction = np.asarray([1.0, slope], dtype=np.float64)
    direction /= float(np.linalg.norm(direction))
    normal = np.asarray([-slope, 1.0], dtype=np.float64)
    normal /= float(np.linalg.norm(normal))

    height, width = frame.shape[:2]
    x_min = int(max(0, np.floor(min(left[0], right[0]) - 30)))
    x_max = int(min(width, np.ceil(max(left[0], right[0]) + 30)))
    max_offset = (
        float(config.max_offset_from_base_px)
        if config.max_offset_from_base_px is not None
        else min(base_y_mid - 20.0, height * 0.75)
    )
    y_top = max(0, int(np.floor(base_y_mid - max_offset - 30)))
    y_bottom = min(height, int(np.ceil(base_y_mid - config.min_offset_from_base_px + 30)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y_top:y_bottom, x_min:x_max] = True

    gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.abs(gx * normal[0] + gy * normal[1])

    bgr = frame[:, :, :3].astype(np.float64)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    red = np.maximum(0.0, r - 0.70 * g - 0.45 * b)
    red = cv2.GaussianBlur(red, (3, 3), 0)

    red_norm = _normalise_map(red, mask)
    edge_norm = _normalise_map(edge, mask)
    response = 0.62 * red_norm + 0.38 * edge_norm

    response_threshold = max(0.22, float(np.percentile(response[mask], 82)))
    red_threshold = max(0.18, float(np.percentile(red_norm[mask], 78)))
    x_values = np.linspace(left[0], right[0], int(config.sweep_samples))
    candidates: list[LineCandidate] = []

    start_y = base_y_mid - float(config.min_offset_from_base_px)
    end_y = max(0.0, base_y_mid - max_offset)
    y_centers = np.arange(start_y, end_y, -float(config.scan_step_px))
    for y_center in y_centers:
        y_values = y_center + slope * (x_values - x_mid)
        xy = np.column_stack([x_values, y_values])
        support, red_support = _support_along_line(
            response,
            red_norm,
            xy,
            normal,
            config.band_radius_px,
        )
        coverage, red_coverage, bin_scores = _bin_coverages(
            support,
            red_support,
            bins=config.sweep_bins,
            threshold=response_threshold,
            red_threshold=red_threshold,
        )
        if bin_scores.size == 0:
            continue
        score = float(np.mean(bin_scores) + 0.35 * coverage + 0.25 * red_coverage)
        left_top = np.asarray([left[0], y_center + slope * (left[0] - x_mid)])
        right_top = np.asarray([right[0], y_center + slope * (right[0] - x_mid)])
        candidates.append(
            LineCandidate(
                y_center_px=float(y_center),
                left_px=left_top,
                right_px=right_top,
                score=score,
                coverage=coverage,
                red_coverage=red_coverage,
            )
        )

    clustered = _candidate_clusters(candidates, config=config)
    if not clustered:
        raise RuntimeError("No red-supported horizontal line candidates found above base posts")

    clustered.sort(key=lambda candidate: candidate.y_center_px, reverse=True)
    pad_top = clustered[0]
    bar_pool = [
        candidate
        for candidate in clustered[1:]
        if candidate.y_center_px <= pad_top.y_center_px - config.min_pad_to_bar_gap_px
    ]
    if not bar_pool:
        bar_pool = clustered[1:]
    if not bar_pool:
        raise RuntimeError("Detected pad-top line but no separate crossbar candidate above it")

    # Prefer the nearest sufficiently consistent red line above the pad; ties go
    # to the higher combined score.  This encodes the apparatus prior: moving
    # upward from the bases, the pad top appears first and the bar is the next
    # stable red/horizontal structure, not the highest red feature in the frame.
    selected_bar = max(
        bar_pool,
        key=lambda candidate: (
            candidate.red_coverage >= config.min_red_bin_coverage,
            candidate.coverage >= config.min_bin_coverage,
            candidate.y_center_px,
            candidate.score,
        ),
    )
    refined = _fit_local_red_bar_line(
        frame,
        left_base_px=left,
        right_base_px=right,
        initial_bar=selected_bar,
        config=config,
    )
    if refined is None:
        refined_left_base = left
        refined_right_base = right
    else:
        selected_bar = refined.selected_bar
        refined_left_base = refined.left_base_px
        refined_right_base = refined.right_base_px

    points_px = {
        "left_base": [
            round(float(refined_left_base[0]), 3),
            round(float(refined_left_base[1]), 3),
        ],
        "right_base": [
            round(float(refined_right_base[0]), 3),
            round(float(refined_right_base[1]), 3),
        ],
        "left_top": [
            round(float(selected_bar.left_px[0]), 3),
            round(float(selected_bar.left_px[1]), 3),
        ],
        "right_top": [
            round(float(selected_bar.right_px[0]), 3),
            round(float(selected_bar.right_px[1]), 3),
        ],
    }
    return AnchorDetection(
        points_px=points_px,
        selected_bar=selected_bar,
        pad_top=pad_top,
        candidates=clustered,
    )


def draw_detection_debug(
    frame_bgr: np.ndarray,
    detection: AnchorDetection,
    *,
    max_candidates: int = 10,
) -> np.ndarray:
    """Draw selected anchors and candidate scan lines for review."""
    cv2 = _require_cv2()
    output = frame_bgr.copy()
    points = {
        name: np.asarray(point, dtype=np.float64)
        for name, point in detection.points_px.items()
    }

    for candidate in detection.candidates[:max_candidates]:
        colour = (80, 80, 80)
        thickness = 1
        if detection.pad_top is not None and candidate is detection.pad_top:
            colour = (0, 165, 255)
            thickness = 2
        if candidate is detection.selected_bar:
            colour = (0, 0, 255)
            thickness = 3
        cv2.line(
            output,
            tuple(np.rint(candidate.left_px).astype(int)),
            tuple(np.rint(candidate.right_px).astype(int)),
            colour,
            thickness,
            cv2.LINE_AA,
        )

    cv2.line(
        output,
        tuple(np.rint(points["left_base"]).astype(int)),
        tuple(np.rint(points["right_base"]).astype(int)),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        output,
        tuple(np.rint(points["left_top"]).astype(int)),
        tuple(np.rint(points["right_top"]).astype(int)),
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.line(
        output,
        tuple(np.rint(points["left_base"]).astype(int)),
        tuple(np.rint(points["left_top"]).astype(int)),
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        output,
        tuple(np.rint(points["right_base"]).astype(int)),
        tuple(np.rint(points["right_top"]).astype(int)),
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )

    labels = {
        "left_base": (0, 255, 0),
        "right_base": (0, 255, 0),
        "left_top": (0, 0, 255),
        "right_top": (0, 0, 255),
    }
    for name, point in points.items():
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(output, xy, 9, labels[name], -1, cv2.LINE_AA)
        text = f"{name} ({point[0]:.0f},{point[1]:.0f})"
        dy = -16 if "top" in name else 28
        cv2.putText(
            output,
            text,
            (xy[0] - 105, xy[1] + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            (xy[0] - 105, xy[1] + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    bar = detection.selected_bar
    pad = detection.pad_top
    lines = [
        "AUTO BAR DETECTION FROM BASE POSTS",
        f"bar y_center={bar.y_center_px:.1f} score={bar.score:.3f} "
        f"coverage={bar.coverage:.2f} red={bar.red_coverage:.2f}",
    ]
    if pad is not None:
        lines.append(f"pad top y_center={pad.y_center_px:.1f}")
    y = 36
    for line in lines:
        cv2.putText(
            output,
            line,
            (35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 34
    return output


def _candidate_payload(candidate: LineCandidate) -> dict[str, Any]:
    return {
        "y_center_px": round(float(candidate.y_center_px), 3),
        "left_px": [round(float(v), 3) for v in candidate.left_px],
        "right_px": [round(float(v), 3) for v in candidate.right_px],
        "score": round(float(candidate.score), 6),
        "coverage": round(float(candidate.coverage), 6),
        "red_coverage": round(float(candidate.red_coverage), 6),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frame-image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--left-base", required=True, type=_parse_point)
    parser.add_argument("--right-base", required=True, type=_parse_point)
    parser.add_argument("--bar-height", type=float, required=True)
    parser.add_argument("--upright-separation", type=float, default=DEFAULT_UPRIGHT_SEPARATION_M)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--debug-image", required=True, type=Path)
    parser.add_argument("--scan-step-px", type=int, default=BarScanConfig.scan_step_px)
    parser.add_argument("--band-radius-px", type=int, default=BarScanConfig.band_radius_px)
    parser.add_argument("--sweep-samples", type=int, default=BarScanConfig.sweep_samples)
    parser.add_argument("--sweep-bins", type=int, default=BarScanConfig.sweep_bins)
    parser.add_argument(
        "--min-pad-to-bar-gap-px",
        type=float,
        default=BarScanConfig.min_pad_to_bar_gap_px,
    )
    return parser


def main() -> None:
    cv2 = _require_cv2()
    args = build_parser().parse_args()
    frame = _read_frame(args)
    config = BarScanConfig(
        scan_step_px=args.scan_step_px,
        band_radius_px=args.band_radius_px,
        sweep_samples=args.sweep_samples,
        sweep_bins=args.sweep_bins,
        min_pad_to_bar_gap_px=args.min_pad_to_bar_gap_px,
    )
    detection = detect_bar_from_base_posts(
        frame,
        left_base_px=args.left_base,
        right_base_px=args.right_base,
        config=config,
    )

    payload = {
        "frame_index": int(args.frame_index),
        "bar_height_m": float(args.bar_height),
        "upright_separation_m": float(args.upright_separation),
        "points_px": detection.points_px,
        "detection": {
            "method": "base_post_upward_red_line_scan_right_angle_refine_v2",
            "selected_bar": _candidate_payload(detection.selected_bar),
            "pad_top": (
                _candidate_payload(detection.pad_top)
                if detection.pad_top is not None
                else None
            ),
            "candidate_count": int(len(detection.candidates)),
            "top_candidates": [
                _candidate_payload(candidate)
                for candidate in sorted(
                    detection.candidates,
                    key=lambda item: item.score,
                    reverse=True,
                )[:10]
            ],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.debug_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    debug = draw_detection_debug(frame, detection)
    if not cv2.imwrite(str(args.debug_image), debug):
        raise RuntimeError(f"Could not write {args.debug_image}")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.debug_image}")
    print(
        "bar: "
        f"left={payload['points_px']['left_top']} "
        f"right={payload['points_px']['right_top']} "
        f"y_center={detection.selected_bar.y_center_px:.1f}"
    )


if __name__ == "__main__":
    main()
