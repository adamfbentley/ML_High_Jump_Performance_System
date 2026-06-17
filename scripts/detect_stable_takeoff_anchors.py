"""Detect stable-window high-jump apparatus anchors from labelled or auto-seeded posts.

This helper is for the takeoff-stationary clips where the camera is stable over
the final plant/takeoff window.  The manual path takes the two visible upright
base posts; the auto path seeds those posts from the red apparatus region and
near-vertical line groups, optionally masking the athlete with MediaPipe Pose.
Both paths then search upward for the red crossbar using the base-post line as
the expected horizontal direction.

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

    # --- bar response weighting (mode-dependent) ---
    # "red" mode (default, night clips): the crossbar is red, so the response is
    # dominated by the red channel and the selected bar must be strongly
    # red-supported.  "edge" mode (daylight clips): the crossbar is a thin dark
    # line against the sky, so the response uses an edge + dark-horizontal-line
    # cue and drops the red requirement.  Defaults reproduce the original red
    # behaviour exactly (dark_line_weight=0, require_red=True).
    red_weight: float = 0.62
    edge_weight: float = 0.38
    dark_line_weight: float = 0.0
    dark_line_band_px: int = 6
    require_red: bool = True
    # Optional pose-derived bar-height prior (image y, full frame).  In edge mode,
    # the crossbar is selected as the coverage-meeting candidate closest to this
    # y, rather than the first strong horizontal above the pad (which on daylight
    # is the landing-mat top edge, not the faint crossbar).
    bar_y_prior_px: float | None = None
    bar_y_prior_tol_px: float = 90.0

    @classmethod
    def edge_mode(cls, **overrides) -> "BarScanConfig":
        """Daylight preset: edge + dark-horizontal-line cue, no red requirement."""
        params = dict(
            red_weight=0.0,
            edge_weight=0.5,
            dark_line_weight=0.5,
            require_red=False,
        )
        params.update(overrides)
        return cls(**params)


@dataclass(frozen=True)
class LineCandidate:
    """A horizontal-ish line candidate in image coordinates."""

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


@dataclass(frozen=True)
class AutoBasePostSeed:
    """Automatically seeded upright base posts and diagnostics."""

    left_base_px: np.ndarray
    right_base_px: np.ndarray
    person_bbox_px: list[float] | None
    red_roi_px: list[float]
    method: str
    candidate_count: int


@dataclass(frozen=True)
class VerticalLineGroup:
    """Grouped near-vertical line evidence for an apparatus upright."""

    x_px: float
    y_top_px: float
    y_base_px: float
    length_sum_px: float
    segment_count: int
    mean_angle_from_vertical_deg: float


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
        and (not config.require_red or candidate.red_coverage >= config.min_red_bin_coverage)
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


def _expanded_bbox(
    bbox_px: list[float],
    *,
    width: int,
    height: int,
    pad_fraction: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in bbox_px]
    pad_x = (x1 - x0) * pad_fraction
    pad_y = (y1 - y0) * pad_fraction
    return (
        int(max(0, np.floor(x0 - pad_x))),
        int(max(0, np.floor(y0 - pad_y))),
        int(min(width, np.ceil(x1 + pad_x))),
        int(min(height, np.ceil(y1 + pad_y))),
    )


def _mediapipe_person_bbox(frame_bgr: np.ndarray) -> list[float] | None:
    """Return an athlete bbox from MediaPipe Pose, or None if unavailable.

    MediaPipe Pose detects people, not high-jump apparatus.  We use it only to
    keep athlete pixels from dominating the red/edge search that seeds the
    apparatus geometry.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None

    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    try:
        pose_module = mp.solutions.pose
    except AttributeError:
        return None
    with pose_module.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.35,
    ) as pose:
        result = pose.process(cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2RGB))
    if result.pose_landmarks is None:
        return None

    points: list[tuple[float, float]] = []
    for landmark in result.pose_landmarks.landmark:
        visibility = float(getattr(landmark, "visibility", 1.0))
        if visibility < 0.45:
            continue
        x_px = float(landmark.x) * width
        y_px = float(landmark.y) * height
        if np.isfinite([x_px, y_px]).all():
            points.append((x_px, y_px))
    if len(points) < 6:
        return None

    xy = np.asarray(points, dtype=np.float64)
    x0, y0 = np.percentile(xy, 2, axis=0)
    x1, y1 = np.percentile(xy, 98, axis=0)
    x0_i, y0_i, x1_i, y1_i = _expanded_bbox(
        [float(x0), float(y0), float(x1), float(y1)],
        width=width,
        height=height,
        pad_fraction=0.12,
    )
    if x1_i - x0_i < 12 or y1_i - y0_i < 12:
        return None
    return [float(x0_i), float(y0_i), float(x1_i), float(y1_i)]


def _spans_from_active_mask(active: np.ndarray, *, min_width_px: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_active in enumerate(np.asarray(active, dtype=bool)):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            if idx - start >= min_width_px:
                spans.append((start, idx))
            start = None
    if start is not None and len(active) - start >= min_width_px:
        spans.append((start, len(active)))
    return spans


def _red_apparatus_roi(
    frame_bgr: np.ndarray,
    *,
    person_bbox_px: list[float] | None,
) -> tuple[int, int, int, int]:
    """Find the dominant red apparatus span in the stable takeoff frame."""
    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    red = cv2.GaussianBlur(_red_response(frame_bgr), (5, 5), 0)

    y0 = int(round(0.28 * height))
    y1 = int(round(0.84 * height))
    search_mask = np.zeros((height, width), dtype=bool)
    search_mask[y0:y1, :] = True
    if person_bbox_px is not None:
        bx0, by0, bx1, by1 = _expanded_bbox(
            person_bbox_px,
            width=width,
            height=height,
            pad_fraction=0.35,
        )
        search_mask[by0:by1, bx0:bx1] = False

    valid_values = red[search_mask]
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size == 0:
        raise RuntimeError("No valid pixels available for red apparatus search")

    min_span_width = max(80, int(round(0.075 * width)))
    smooth_kernel = max(21, int(round(0.012 * width)) | 1)
    smoothing = np.ones(smooth_kernel, dtype=np.float64) / float(smooth_kernel)
    best_span: tuple[int, int] | None = None
    best_score = -np.inf

    for percentile in (92, 90, 88, 85, 80):
        threshold = max(15.0, float(np.percentile(valid_values, percentile)))
        red_mask = (red >= threshold) & search_mask
        column_score = np.mean(red_mask[y0:y1, :], axis=0)
        smooth_score = np.convolve(column_score, smoothing, mode="same")
        nonzero_score = smooth_score[smooth_score > 0.0]
        if nonzero_score.size == 0:
            continue
        active_threshold = max(0.002, float(np.percentile(nonzero_score, 80)) * 0.45)
        active = smooth_score >= active_threshold
        for span_start, span_end in _spans_from_active_mask(
            active,
            min_width_px=min_span_width,
        ):
            span_width = span_end - span_start
            if span_width > int(round(0.70 * width)):
                continue
            span_center = (span_start + span_end) / 2.0
            if span_center < 0.38 * width:
                continue
            support = float(np.sum(smooth_score[span_start:span_end]))
            # Prefer a compact, right-half red region.  This rejects red text,
            # track features, and background spectators before post fitting.
            score = support + 0.012 * span_width - 0.0008 * abs(span_center - 0.74 * width)
            if score > best_score:
                best_score = score
                best_span = (span_start, span_end)
        if best_span is not None:
            break

    if best_span is None:
        raise RuntimeError("Could not find a red apparatus span for auto base seeding")

    x0, x1 = best_span
    return (int(x0), int(y0), int(x1), int(y1))


def _vertical_line_groups(
    frame_bgr: np.ndarray,
    *,
    red_roi_px: tuple[int, int, int, int],
    person_bbox_px: list[float] | None,
) -> list[VerticalLineGroup]:
    """Group near-vertical line segments around the red apparatus region."""
    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    red_x0, _red_y0, red_x1, _red_y1 = red_roi_px
    red_width = max(1, red_x1 - red_x0)
    margin = max(170, int(round(0.38 * red_width)))
    x0 = int(max(0, red_x0 - margin))
    x1 = int(min(width, red_x1 + margin))
    y0 = int(max(0, round(0.25 * height)))
    y1 = int(min(height, round(0.87 * height)))
    if x1 <= x0 + 20 or y1 <= y0 + 40:
        return []

    gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1, :3], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 20, 80)
    if person_bbox_px is not None:
        bx0, by0, bx1, by1 = _expanded_bbox(
            person_bbox_px,
            width=width,
            height=height,
            pad_fraction=0.18,
        )
        local_x0 = max(0, bx0 - x0)
        local_x1 = min(edges.shape[1], bx1 - x0)
        local_y0 = max(0, by0 - y0)
        local_y1 = min(edges.shape[0], by1 - y0)
        if local_x1 > local_x0 and local_y1 > local_y0:
            edges[local_y0:local_y1, local_x0:local_x1] = 0

    min_line_length = max(60, int(round(0.055 * height)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=20,
        minLineLength=min_line_length,
        maxLineGap=max(38, int(round(0.05 * height))),
    )
    if lines is None:
        return []

    segments: list[dict[str, float]] = []
    for raw_line in lines[:, 0, :]:
        x1_l, y1_l, x2_l, y2_l = [float(value) for value in raw_line]
        dx = x2_l - x1_l
        dy = y2_l - y1_l
        length = float(np.hypot(dx, dy))
        if length < min_line_length:
            continue
        angle_from_vertical = abs(90.0 - abs(float(np.degrees(np.arctan2(dy, dx)))))
        if angle_from_vertical > 16.0:
            continue
        x_mid = float((x1_l + x2_l) / 2.0 + x0)
        y_top = float(min(y1_l, y2_l) + y0)
        y_base = float(max(y1_l, y2_l) + y0)
        if y_base < 0.45 * height:
            continue
        if not (red_x0 - 0.45 * red_width <= x_mid <= red_x1 + 0.45 * red_width):
            continue
        segments.append(
            {
                "x": x_mid,
                "y_top": y_top,
                "y_base": y_base,
                "length": length,
                "angle": angle_from_vertical,
            }
        )
    if not segments:
        return []

    segments.sort(key=lambda item: item["x"])
    clusters: list[list[dict[str, float]]] = []
    for segment in segments:
        if not clusters:
            clusters.append([segment])
            continue
        cluster_x = float(
            np.average(
                [item["x"] for item in clusters[-1]],
                weights=[item["length"] for item in clusters[-1]],
            )
        )
        if abs(segment["x"] - cluster_x) <= 28.0:
            clusters[-1].append(segment)
        else:
            clusters.append([segment])

    groups: list[VerticalLineGroup] = []
    for cluster in clusters:
        lengths = np.asarray([item["length"] for item in cluster], dtype=np.float64)
        if float(np.sum(lengths)) < min_line_length:
            continue
        x_values = np.asarray([item["x"] for item in cluster], dtype=np.float64)
        y_top_values = np.asarray([item["y_top"] for item in cluster], dtype=np.float64)
        y_base_values = np.asarray([item["y_base"] for item in cluster], dtype=np.float64)
        angle_values = np.asarray([item["angle"] for item in cluster], dtype=np.float64)
        groups.append(
            VerticalLineGroup(
                x_px=float(np.average(x_values, weights=lengths)),
                y_top_px=float(np.percentile(y_top_values, 15)),
                y_base_px=float(np.percentile(y_base_values, 80)),
                length_sum_px=float(np.sum(lengths)),
                segment_count=len(cluster),
                mean_angle_from_vertical_deg=float(np.average(angle_values, weights=lengths)),
            )
        )

    groups.sort(key=lambda item: item.x_px)
    return groups


def _select_upright_pair(
    groups: list[VerticalLineGroup],
    *,
    red_roi_px: tuple[int, int, int, int],
    frame_width: int,
) -> tuple[VerticalLineGroup, VerticalLineGroup]:
    red_x0, _red_y0, red_x1, _red_y1 = red_roi_px
    red_width = max(1.0, float(red_x1 - red_x0))
    red_center = (red_x0 + red_x1) / 2.0
    min_separation = max(210.0, 0.72 * red_width, 0.16 * frame_width)
    max_separation = min(0.70 * frame_width, 1.55 * red_width + 90.0)
    best_pair: tuple[VerticalLineGroup, VerticalLineGroup] | None = None
    best_score = -np.inf

    for left_index, left_group in enumerate(groups):
        for right_group in groups[left_index + 1 :]:
            separation = right_group.x_px - left_group.x_px
            if separation < min_separation or separation > max_separation:
                continue
            if left_group.x_px > red_x0 + 0.27 * red_width:
                continue
            if right_group.x_px < red_x1 - 0.22 * red_width:
                continue
            if right_group.x_px > red_x1 + 0.34 * red_width:
                continue

            left_edge_error = abs(left_group.x_px - red_x0)
            right_edge_error = abs(right_group.x_px - red_x1)
            separation_error = abs(separation - red_width)
            center_error = abs((left_group.x_px + right_group.x_px) / 2.0 - red_center)
            length_bonus = 0.014 * (left_group.length_sum_px + right_group.length_sum_px)
            base_bonus = 0.035 * min(left_group.y_base_px, right_group.y_base_px)
            segment_bonus = 3.0 * (left_group.segment_count + right_group.segment_count)
            angle_penalty = 1.8 * (
                left_group.mean_angle_from_vertical_deg
                + right_group.mean_angle_from_vertical_deg
            )
            score = (
                length_bonus
                + base_bonus
                + segment_bonus
                - left_edge_error
                - right_edge_error
                - 0.45 * separation_error
                - 0.35 * center_error
                - angle_penalty
            )
            if score > best_score:
                best_score = float(score)
                best_pair = (left_group, right_group)

    if best_pair is None:
        raise RuntimeError("Could not select a left/right upright pair from line groups")
    return best_pair


def auto_seed_base_posts(
    frame_bgr: np.ndarray,
    *,
    use_mediapipe_person_mask: bool = True,
) -> AutoBasePostSeed:
    """Seed high-jump upright bases from image evidence before precise fitting."""
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame_bgr must be a BGR image")

    height, width = frame.shape[:2]
    person_bbox = (
        _mediapipe_person_bbox(frame)
        if use_mediapipe_person_mask
        else None
    )
    red_roi = _red_apparatus_roi(frame, person_bbox_px=person_bbox)
    groups = _vertical_line_groups(
        frame,
        red_roi_px=red_roi,
        person_bbox_px=person_bbox,
    )
    if len(groups) < 2:
        raise RuntimeError("Auto base seeding found fewer than two vertical line groups")
    left_group, right_group = _select_upright_pair(
        groups,
        red_roi_px=red_roi,
        frame_width=width,
    )

    left_base = np.asarray([left_group.x_px, left_group.y_base_px], dtype=np.float64)
    right_base = np.asarray([right_group.x_px, right_group.y_base_px], dtype=np.float64)
    return AutoBasePostSeed(
        left_base_px=left_base,
        right_base_px=right_base,
        person_bbox_px=person_bbox,
        red_roi_px=[float(value) for value in red_roi],
        method="red_apparatus_roi_vertical_hough_with_optional_mediapipe_person_mask",
        candidate_count=int(len(groups)),
    )


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
        nudged_line = _nudge_upright_line_to_post_profile(
            frame_bgr,
            rough_line=paired_centerline,
            base_px=base,
            approximate_bar_y_px=approximate_bar_y_px,
        )
        return nudged_line if nudged_line is not None else paired_centerline
    refined_line = _refine_upright_centerline(
        frame_bgr,
        base_px=base,
        rough_line=best_line,
        approximate_bar_y_px=approximate_bar_y_px,
    )
    chosen_line = refined_line if refined_line is not None else best_line
    nudged_line = _nudge_upright_line_to_post_profile(
        frame_bgr,
        rough_line=chosen_line,
        base_px=base,
        approximate_bar_y_px=approximate_bar_y_px,
    )
    return nudged_line if nudged_line is not None else chosen_line


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


def _nudge_upright_line_to_post_profile(
    frame_bgr: np.ndarray,
    *,
    rough_line: np.ndarray,
    base_px: np.ndarray,
    approximate_bar_y_px: float,
) -> np.ndarray | None:
    """Correct small edge bias using the aligned brightness profile of the post."""
    if not np.isfinite(rough_line).all():
        return None

    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    base = np.asarray(base_px, dtype=np.float64)
    y_start = float(min(base[1], approximate_bar_y_px) + 18.0)
    y_end = float(max(base[1], approximate_bar_y_px) - 18.0)
    if y_end <= y_start + 40.0:
        return None

    gray = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    offsets = np.arange(-45, 46, dtype=np.int32)
    brightness_sum = np.zeros(len(offsets), dtype=np.float64)
    sample_count = 0
    row_count = min(180, max(40, int(round(y_end - y_start))))
    for y_global in np.linspace(y_start, y_end, row_count):
        x_pred = _x_at_y(rough_line, float(y_global))
        if x_pred is None:
            continue
        yi = int(round(y_global))
        xs = np.rint(float(x_pred) + offsets).astype(int)
        inside = (0 <= xs) & (xs < width) & (0 <= yi) & (yi < height)
        if int(np.count_nonzero(inside)) < 20:
            continue
        values = np.zeros(len(offsets), dtype=np.float64)
        values[inside] = gray[yi, xs[inside]]
        brightness_sum += values
        sample_count += 1
    if sample_count < 20:
        return None

    brightness = brightness_sum / float(sample_count)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(brightness, kernel, mode="same")
    threshold = max(
        float(np.percentile(smooth, 65)),
        float(np.mean(smooth) + 0.10 * np.std(smooth)),
    )
    active = smooth >= threshold
    runs: list[tuple[int, int, float]] = []
    start: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            if idx - start >= 4:
                runs.append(
                    (
                        int(offsets[start]),
                        int(offsets[idx - 1]),
                        float(np.mean(smooth[start:idx])),
                    )
                )
            start = None
    if start is not None and len(active) - start >= 4:
        runs.append((int(offsets[start]), int(offsets[-1]), float(np.mean(smooth[start:]))))
    if not runs:
        return None

    nearby_post_runs = [
        run
        for run in runs
        if 5 <= run[1] - run[0] + 1 <= 34
        and (
            run[0] <= 3 <= run[1]
            or run[0] <= -3 <= run[1]
            or 0 < run[0] <= 12
            or -12 <= run[1] < 0
        )
    ]
    if not nearby_post_runs:
        return None
    run_start, run_end, _mean_brightness = max(
        nearby_post_runs,
        key=lambda run: (run[2], -(run[1] - run[0])),
    )
    shift_px = (float(run_start) + float(run_end)) / 2.0
    if abs(shift_px) < 1.5 or abs(shift_px) > 20.0:
        return None

    top_y = y_start
    bottom_y = y_end
    top_x = _x_at_y(rough_line, top_y)
    bottom_x = _x_at_y(rough_line, bottom_y)
    if top_x is None or bottom_x is None:
        return None
    return _line_from_points(
        np.asarray([top_x + shift_px, top_y], dtype=np.float64),
        np.asarray([bottom_x + shift_px, bottom_y], dtype=np.float64),
    )


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
            # When the rough seed is on one edge of the upright, two close
            # edge/seam detections can otherwise beat the true outer-edge pair.
            # A modest width bonus makes the full post envelope win while the
            # existing width and angle gates reject unrelated background lines.
            width_bonus = 2.8 * median_width
            score = (
                length_sum
                + width_bonus
                - 4.0 * base_error
                - 2.0 * width_penalty
                - 5.0 * angle_penalty
            )
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


def _fit_landing_pad_top_edge(
    frame_bgr: np.ndarray,
    *,
    initial_pad_top: LineCandidate,
    left_x_px: float,
    right_x_px: float,
) -> LineCandidate | None:
    """Fit the visible red-over-blue landing-pad top edge between the uprights.

    This line is a separate perspective cue from the crossbar and from the base
    posts.  The upward scan gives the approximate vertical band; the final angle
    comes from the image evidence at the pad boundary itself.
    """
    cv2 = _require_cv2()
    frame = np.asarray(frame_bgr)
    height, width = frame.shape[:2]
    left_x = float(min(left_x_px, right_x_px))
    right_x = float(max(left_x_px, right_x_px))
    if right_x <= left_x + 40.0:
        return None

    y_center_guess = float(initial_pad_top.y_center_px)
    x0 = int(max(0, np.floor(left_x - 20.0)))
    x1 = int(min(width, np.ceil(right_x + 20.0)))
    y0 = int(max(0, np.floor(y_center_guess - 80.0)))
    y1 = int(min(height, np.ceil(y_center_guess + 80.0)))
    if x1 <= x0 + 50 or y1 <= y0 + 35:
        return None

    roi = frame[y0:y1, x0:x1, :3].astype(np.float64)
    b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    red_top = np.maximum(0.0, r - 0.65 * g - 0.35 * b)
    blue_face = np.maximum(0.0, b - 0.35 * r - 0.20 * g)

    band_px = 7
    red_above = np.zeros_like(red_top)
    red_below = np.zeros_like(red_top)
    blue_above = np.zeros_like(blue_face)
    blue_below = np.zeros_like(blue_face)
    for dy in range(1, band_px + 1):
        red_above[dy:] += red_top[:-dy]
        red_below[:-dy] += red_top[dy:]
        blue_above[dy:] += blue_face[:-dy]
        blue_below[:-dy] += blue_face[dy:]
    red_above /= float(band_px)
    red_below /= float(band_px)
    blue_above /= float(band_px)
    blue_below /= float(band_px)

    boundary_score = (
        (red_above - red_below)
        + (blue_below - blue_above)
        + 0.25 * (red_above + blue_below)
    )
    boundary_score[:band_px, :] = 0.0
    boundary_score[-band_px:, :] = 0.0
    boundary_score = cv2.GaussianBlur(boundary_score.astype(np.float32), (5, 3), 0)
    valid = boundary_score[np.isfinite(boundary_score)]
    if valid.size == 0 or float(np.max(valid)) <= 1e-6:
        return None

    threshold = float(np.percentile(valid, 97))
    mask = (boundary_score >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=25,
        minLineLength=max(90, int(0.22 * (right_x - left_x))),
        maxLineGap=35,
    )
    if lines is None:
        return None

    best: tuple[float, float, float, float, float] | None = None
    for line in lines[:, 0, :]:
        x1_l, y1_l, x2_l, y2_l = [float(value) for value in line]
        dx = x2_l - x1_l
        if abs(dx) < 1.0:
            continue
        slope = float((y2_l - y1_l) / dx)
        if abs(slope) > 0.25:
            continue
        length = float(np.hypot(dx, y2_l - y1_l))
        if length < max(90.0, 0.20 * (right_x - left_x)):
            continue
        intercept = float(y1_l - slope * x1_l)
        local_mid_x = ((left_x + right_x) / 2.0) - float(x0)
        y_mid_global = float(slope * local_mid_x + intercept + y0)
        closeness = abs(y_mid_global - y_center_guess)

        sample_count = int(max(20, length))
        xs = np.linspace(x1_l, x2_l, sample_count)
        ys = np.linspace(y1_l, y2_l, sample_count)
        xi = np.clip(np.rint(xs).astype(int), 0, boundary_score.shape[1] - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, boundary_score.shape[0] - 1)
        support = float(np.mean(boundary_score[yi, xi]))
        score = length + 0.4 * support - 1.5 * closeness
        if best is None or score > best[0]:
            best = (score, slope, intercept, support, length)

    if best is None:
        return None
    score, slope, intercept, support, _length = best
    left_y = float(slope * (left_x - x0) + intercept + y0)
    right_y = float(slope * (right_x - x0) + intercept + y0)
    y_center = float(slope * (((left_x + right_x) / 2.0) - x0) + intercept + y0)
    return LineCandidate(
        y_center_px=y_center,
        left_px=np.asarray([left_x, left_y], dtype=np.float64),
        right_px=np.asarray([right_x, right_y], dtype=np.float64),
        score=float(score),
        coverage=float(min(1.0, max(0.0, support / max(1.0, float(np.percentile(valid, 99)))))),
        red_coverage=initial_pad_top.red_coverage,
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

    # Dark horizontal-line cue (for daylight): a thin dark bar sits between two
    # brighter rows (sky), so it is darker than its vertical neighbours.
    d = max(1, int(config.dark_line_band_px))
    above = np.roll(gray, d, axis=0)
    below = np.roll(gray, -d, axis=0)
    dark_line = np.maximum(0.0, 0.5 * (above + below) - gray)

    red_norm = _normalise_map(red, mask)
    edge_norm = _normalise_map(edge, mask)
    dark_norm = _normalise_map(dark_line, mask)
    response = (
        config.red_weight * red_norm
        + config.edge_weight * edge_norm
        + config.dark_line_weight * dark_norm
    )

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
        raise RuntimeError(
            "No red-supported horizontal line candidates found above base posts"
            if config.require_red
            else "No horizontal line candidates found above base posts"
        )

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
    strong_min_coverage = max(config.min_bin_coverage, 0.65)
    strong_min_red_coverage = max(config.min_red_bin_coverage, 0.65)

    if config.require_red:
        # Night/red apparatus: the bar must be strongly red-supported.
        def _meets_strong(c: LineCandidate) -> bool:
            return c.red_coverage >= strong_min_red_coverage and c.coverage >= strong_min_coverage

        selected_bar = max(
            bar_pool,
            key=lambda c: (_meets_strong(c), c.y_center_px, c.red_coverage, c.coverage, c.score),
        )
        if not _meets_strong(selected_bar):
            raise RuntimeError(
                "No strongly red-supported crossbar candidate found above the pad; "
                "try a clearer stable frame"
            )
    else:
        # Daylight/edge apparatus: select on combined-response coverage only.
        meeting = [c for c in bar_pool if c.coverage >= strong_min_coverage]
        if not meeting:
            # Relax once: take the best-covered candidates so a faint bar is not lost.
            meeting = sorted(bar_pool, key=lambda c: c.coverage, reverse=True)[:5]
        if not meeting:
            raise RuntimeError(
                "No supported crossbar candidate found above the pad; "
                "try a clearer stable frame"
            )
        if config.bar_y_prior_px is not None:
            # Prefer the candidate nearest the pose-predicted bar height; the
            # landing-mat top edge is a stronger but lower (wrong) horizontal.
            near = [c for c in meeting
                    if abs(c.y_center_px - config.bar_y_prior_px) <= config.bar_y_prior_tol_px]
            pool_for_prior = near if near else meeting
            selected_bar = min(
                pool_for_prior, key=lambda c: abs(c.y_center_px - config.bar_y_prior_px)
            )
        else:
            selected_bar = max(meeting, key=lambda c: (c.y_center_px, c.coverage, c.score))

    refined = (
        _fit_local_red_bar_line(
            frame,
            left_base_px=left,
            right_base_px=right,
            initial_bar=selected_bar,
            config=config,
        )
        if config.require_red
        else None
    )
    if refined is None:
        refined_left_base = left
        refined_right_base = right
    else:
        selected_bar = refined.selected_bar
        refined_left_base = refined.left_base_px
        refined_right_base = refined.right_base_px

    display_pad_top = _fit_landing_pad_top_edge(
        frame,
        initial_pad_top=pad_top,
        left_x_px=float(refined_left_base[0]),
        right_x_px=float(refined_right_base[0]),
    )
    if display_pad_top is None:
        display_pad_top = pad_top
    display_candidates = [
        display_pad_top if candidate is pad_top else candidate
        for candidate in clustered
    ]

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
        pad_top=display_pad_top,
        candidates=display_candidates,
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
    parser.add_argument("--left-base", type=_parse_point)
    parser.add_argument("--right-base", type=_parse_point)
    parser.add_argument(
        "--auto-base-posts",
        action="store_true",
        help="Seed base posts from red apparatus geometry instead of labelled points.",
    )
    parser.add_argument(
        "--mediapipe-person-mask",
        choices=("on", "off"),
        default="on",
        help="Use MediaPipe Pose to mask athlete pixels during auto base seeding.",
    )
    parser.add_argument("--bar-height", type=float, required=True)
    parser.add_argument(
        "--bar-mode",
        choices=("red", "edge"),
        default="red",
        help="red: night red-crossbar response (default). edge: daylight "
        "edge/dark-line response with no red requirement.",
    )
    parser.add_argument(
        "--bar-y-prior",
        type=float,
        default=None,
        help="edge mode: image-y of the expected crossbar (e.g. pose-derived); "
        "selects the candidate nearest this height over the stronger mat-top edge.",
    )
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
    if args.auto_base_posts:
        base_seed = auto_seed_base_posts(
            frame,
            use_mediapipe_person_mask=args.mediapipe_person_mask == "on",
        )
        left_base = base_seed.left_base_px
        right_base = base_seed.right_base_px
        base_seed_payload = {
            "method": base_seed.method,
            "left_base_px": [round(float(value), 3) for value in left_base],
            "right_base_px": [round(float(value), 3) for value in right_base],
            "person_bbox_px": (
                [round(float(value), 3) for value in base_seed.person_bbox_px]
                if base_seed.person_bbox_px is not None
                else None
            ),
            "red_roi_px": [round(float(value), 3) for value in base_seed.red_roi_px],
            "candidate_count": int(base_seed.candidate_count),
            "mediapipe_person_mask": args.mediapipe_person_mask,
        }
    else:
        if args.left_base is None or args.right_base is None:
            raise SystemExit("provide --left-base/--right-base or use --auto-base-posts")
        left_base = args.left_base
        right_base = args.right_base
        base_seed_payload = {
            "method": "manual",
            "left_base_px": [round(float(value), 3) for value in left_base],
            "right_base_px": [round(float(value), 3) for value in right_base],
        }

    common = dict(
        scan_step_px=args.scan_step_px,
        band_radius_px=args.band_radius_px,
        sweep_samples=args.sweep_samples,
        sweep_bins=args.sweep_bins,
        min_pad_to_bar_gap_px=args.min_pad_to_bar_gap_px,
    )
    if args.bar_mode == "edge":
        config = BarScanConfig.edge_mode(bar_y_prior_px=args.bar_y_prior, **common)
    else:
        config = BarScanConfig(**common)
    detection = detect_bar_from_base_posts(
        frame,
        left_base_px=left_base,
        right_base_px=right_base,
        config=config,
    )

    payload = {
        "frame_index": int(args.frame_index),
        "bar_height_m": float(args.bar_height),
        "upright_separation_m": float(args.upright_separation),
        "points_px": detection.points_px,
        "base_seed": base_seed_payload,
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
