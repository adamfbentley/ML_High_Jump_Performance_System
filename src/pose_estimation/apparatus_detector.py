"""Colour-agnostic, geometry-first high-jump apparatus detection.

The earlier stable-window detector (``scripts/detect_stable_takeoff_anchors.py``)
seeds the upright posts from the *red* apparatus region and then scans upward for
the *red* crossbar.  That works on the floodlit night clips (red standards, red
bar) but fails on daylight stationary footage where the standards are pale/white
and the crossbar is a thin dark line against a bright sky, while background
floodlight poles are stronger vertical features than the apparatus itself.

This module detects the apparatus from *geometry* rather than colour:

1.  Extract near-vertical and near-horizontal line segments from a grayscale edge
    map (no colour channel is required).
2.  Cluster the verticals into candidate upright *posts*.
3.  Form candidate upright *pairs* constrained by plausible separation, similar
    height, and roughly level bases.
4.  Admit a pair only when it has an apparatus *relationship*: a near-horizontal
    crossbar that connects the two posts near their tops, and/or a landing-pad
    top edge spanning the gap lower down.  This single requirement is what
    rejects an isolated background floodlight pole — a pole has no paired post
    joined by a crossbar.
5.  Bias selection toward the foreground takeoff zone (large base ``y`` in the
    image) and, when supplied, toward the athlete's horizontal position.

Colour is used only as *optional* additive evidence (a red-apparatus bonus and a
bar/background contrast bonus); it is never required, so the same code path
admits both the night-red and daylight-pale apparatus.

Image coordinates are pixels ``(x, y)`` with ``y`` pointing down, matching the
rest of the pose pipeline.  The detector does not itself convert to the Y-up,
X-forward, Z-lateral scene frame; that conversion happens downstream in
``src/pose_estimation/scene_calibration.py`` once these anchors are solved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "ApparatusConfig",
    "PostCandidate",
    "HorizontalSegment",
    "ApparatusDetection",
    "StableApparatusDetection",
    "detect_apparatus_geometry",
    "detect_apparatus_geometry_stable",
]


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for apparatus geometry detection") from exc
    return cv2


@dataclass(frozen=True)
class ApparatusConfig:
    """Tuning parameters for geometry-first apparatus detection.

    Lengths are in pixels unless noted.  Pixel thresholds that scale with the
    frame are expressed as fractions of the frame height/width and resolved at
    detection time, so the same config works across 960- and 1456-wide clips.
    """

    # --- line extraction ---
    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 30
    max_vertical_angle_deg: float = 16.0
    max_horizontal_angle_deg: float = 12.0
    min_post_segment_len_frac: float = 0.045  # of frame height
    min_horizontal_len_frac: float = 0.030  # of frame width
    hough_max_gap_frac: float = 0.030  # of frame height

    # --- post clustering ---
    post_cluster_px_frac: float = 0.018  # of frame width
    min_post_length_frac: float = 0.060  # of frame height
    min_post_support_frac: float = 0.075  # total vertical segment length / height

    # --- pair geometry ---
    min_separation_frac: float = 0.060  # of frame width
    max_separation_frac: float = 0.420  # of frame width
    min_height_ratio: float = 0.40  # shorter / taller post length
    base_level_tol_frac: float = 0.30  # |base_y diff| / separation

    # --- crossbar relationship ---
    bar_band_top_margin_frac: float = 0.18  # bar may sit this far above post tops (of post len)
    bar_band_depth_frac: float = 0.55  # bar must be in the upper this fraction of the posts
    min_bar_span_frac: float = 0.45  # crossbar x-extent / separation
    max_bar_overhang_frac: float = 0.45  # crossbar may extend this far beyond a post
    bar_post_attach_tol_frac: float = 0.06  # |bar-line x at post - post x| / separation

    # --- landing-pad relationship ---
    min_pad_gap_frac: float = 0.10  # pad must be this far below bar (of post len)
    min_pad_span_frac: float = 0.45  # pad x-extent / separation
    min_pad_bottom_gap_frac: float = 0.10  # pad bottom below pad top by this * post len

    # --- pad/bar vertical-ratio prior ---
    # The image ratio (bar->pad-top) / (bar->pad-bottom) is ~scale-free and
    # perspective-tolerant, and equals 1 - H_pad/H_bar in world heights.  A real
    # apparatus has this triple at the expected ratio; a background building with
    # an apparatus-shaped roofline does not, so this discriminates them.
    expected_pad_height_m: float = 0.70  # competition landing-area top height
    pad_ratio_deadzone: float = 0.22  # no penalty within +/- this of the expected ratio
    pad_ratio_sigma: float = 0.14  # ramp width of the penalty beyond the dead-zone
    w_pad_ratio: float = 5.0

    # --- scoring weights ---
    w_crossbar: float = 6.0
    w_pad: float = 2.0
    w_length: float = 0.004
    w_symmetry: float = 2.0
    w_foreground: float = 1.0
    w_bottom_edge_penalty: float = 4.0  # discourage bases on the bottom track edge
    bottom_edge_margin_frac: float = 0.10  # of frame height
    w_athlete: float = 5.0
    w_angle: float = 0.6
    w_colour_red: float = 1.0
    w_colour_contrast: float = 1.0
    athlete_sigma_frac: float = 0.12  # of frame width

    # --- admission ---
    require_crossbar: bool = False  # if False, a strong pad relationship can admit
    min_confidence: float = 0.30

    # --- temporal aggregation ---
    stable_min_frames: int = 3
    stable_separation_tol_frac: float = 0.06  # of frame width, for clustering detections
    stable_max_anchor_std_frac: float = 0.020  # of frame width


@dataclass(frozen=True)
class PostCandidate:
    """A clustered near-vertical upright candidate in image pixels."""

    x_base_px: float
    x_top_px: float
    y_base_px: float
    y_top_px: float
    length_px: float
    support_px: float
    n_segments: int
    line: np.ndarray  # homogeneous image line (3,)
    mean_angle_from_vertical_deg: float

    def x_at(self, y_px: float) -> float:
        x = _x_at_y(self.line, float(y_px))
        return float(self.x_base_px) if x is None else float(x)


@dataclass(frozen=True)
class HorizontalSegment:
    """A near-horizontal line segment in image pixels."""

    x_left_px: float
    x_right_px: float
    y_left_px: float
    y_right_px: float
    length_px: float
    line: np.ndarray  # homogeneous image line (3,)

    @property
    def y_center_px(self) -> float:
        return float((self.y_left_px + self.y_right_px) / 2.0)


@dataclass(frozen=True)
class ApparatusDetection:
    """Detected apparatus anchors and diagnostics for one frame."""

    left_base_px: np.ndarray
    right_base_px: np.ndarray
    left_top_px: np.ndarray
    right_top_px: np.ndarray
    separation_px: float
    has_crossbar: bool
    has_pad: bool
    crossbar: HorizontalSegment | None
    pad_top: HorizontalSegment | None
    score: float
    confidence: float
    left_post: PostCandidate
    right_post: PostCandidate
    pad_bottom: HorizontalSegment | None = None
    pad_ratio: float | None = None
    pad_ratio_expected: float | None = None

    def points_px(self) -> dict[str, list[float]]:
        return {
            "left_base": [round(float(self.left_base_px[0]), 3), round(float(self.left_base_px[1]), 3)],
            "right_base": [round(float(self.right_base_px[0]), 3), round(float(self.right_base_px[1]), 3)],
            "left_top": [round(float(self.left_top_px[0]), 3), round(float(self.left_top_px[1]), 3)],
            "right_top": [round(float(self.right_top_px[0]), 3), round(float(self.right_top_px[1]), 3)],
        }


@dataclass(frozen=True)
class StableApparatusDetection:
    """Temporally aggregated apparatus geometry across stable frames."""

    left_base_px: np.ndarray
    right_base_px: np.ndarray
    left_top_px: np.ndarray
    right_top_px: np.ndarray
    n_frames_used: int
    n_frames_total: int
    anchor_std_px: float
    crossbar_frac: float
    pad_frac: float
    is_stable: bool
    per_frame: list[ApparatusDetection | None] = field(default_factory=list)

    def points_px(self) -> dict[str, list[float]]:
        return {
            "left_base": [round(float(self.left_base_px[0]), 3), round(float(self.left_base_px[1]), 3)],
            "right_base": [round(float(self.right_base_px[0]), 3), round(float(self.right_base_px[1]), 3)],
            "left_top": [round(float(self.left_top_px[0]), 3), round(float(self.left_top_px[1]), 3)],
            "right_top": [round(float(self.right_top_px[0]), 3), round(float(self.right_top_px[1]), 3)],
        }


# --------------------------------------------------------------------------- #
# Line geometry helpers (pure)
# --------------------------------------------------------------------------- #
def _line_from_points(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    a = np.asarray([point_a[0], point_a[1], 1.0], dtype=np.float64)
    b = np.asarray([point_b[0], point_b[1], 1.0], dtype=np.float64)
    line = np.cross(a, b)
    norm = float(np.linalg.norm(line[:2]))
    if norm <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return line / norm


def _line_intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    point = np.cross(line_a, line_b)
    if not np.isfinite(point).all() or abs(float(point[2])) <= 1e-9:
        return None
    return np.asarray([point[0] / point[2], point[1] / point[2]], dtype=np.float64)


def _x_at_y(line: np.ndarray, y_px: float) -> float | None:
    a, b, c = [float(v) for v in line]
    if abs(a) <= 1e-9:
        return None
    x = -(b * float(y_px) + c) / a
    return float(x) if np.isfinite(x) else None


def _y_at_x(line: np.ndarray, x_px: float) -> float | None:
    a, b, c = [float(v) for v in line]
    if abs(b) <= 1e-9:
        return None
    y = -(a * float(x_px) + c) / b
    return float(y) if np.isfinite(y) else None


# --------------------------------------------------------------------------- #
# Stage 1: line extraction
# --------------------------------------------------------------------------- #
def _expand_bbox(bbox: Sequence[float], *, width: int, height: int, pad_frac: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    pad_x = (x1 - x0) * pad_frac
    pad_y = (y1 - y0) * pad_frac
    return (
        int(max(0, np.floor(x0 - pad_x))),
        int(max(0, np.floor(y0 - pad_y))),
        int(min(width, np.ceil(x1 + pad_x))),
        int(min(height, np.ceil(y1 + pad_y))),
    )


def _segment_in_bbox(x_mid: float, y_mid: float, bbox: tuple[int, int, int, int]) -> bool:
    bx0, by0, bx1, by1 = bbox
    return bx0 <= x_mid <= bx1 and by0 <= y_mid <= by1


def _extract_segments(
    frame_bgr: np.ndarray,
    *,
    config: ApparatusConfig,
    athlete_bbox_px: Sequence[float] | None,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return (vertical_segments, horizontal_segments) from a grayscale edge map."""
    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    # CLAHE keeps both the dark daylight bar and the pale standards visible under
    # the same edge thresholds regardless of overall scene brightness.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, config.canny_low, config.canny_high)

    excluded_bbox: tuple[int, int, int, int] | None = None
    if athlete_bbox_px is not None:
        excluded_bbox = _expand_bbox(athlete_bbox_px, width=width, height=height, pad_frac=0.10)

    min_post_len = max(24.0, config.min_post_segment_len_frac * height)
    min_h_len = max(24.0, config.min_horizontal_len_frac * width)
    max_gap = max(8, int(round(config.hough_max_gap_frac * height)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=config.hough_threshold,
        minLineLength=int(min(min_post_len, min_h_len)),
        maxLineGap=max_gap,
    )
    verticals: list[dict[str, float]] = []
    horizontals: list[dict[str, float]] = []
    if lines is None:
        return verticals, horizontals

    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in raw]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1.0:
            continue
        x_mid = (x1 + x2) / 2.0
        y_mid = (y1 + y2) / 2.0
        if excluded_bbox is not None and _segment_in_bbox(x_mid, y_mid, excluded_bbox):
            continue
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        angle_from_vertical = abs(90.0 - angle)
        angle_from_horizontal = min(angle, abs(180.0 - angle))

        if angle_from_vertical <= config.max_vertical_angle_deg and length >= min_post_len:
            top = np.array([x1, y1]) if y1 < y2 else np.array([x2, y2])
            base = np.array([x2, y2]) if y1 < y2 else np.array([x1, y1])
            verticals.append(
                {
                    "x_top": float(top[0]),
                    "y_top": float(top[1]),
                    "x_base": float(base[0]),
                    "y_base": float(base[1]),
                    "x_mid": float(x_mid),
                    "length": length,
                    "angle_from_vertical": float(angle_from_vertical),
                }
            )
        elif angle_from_horizontal <= config.max_horizontal_angle_deg and length >= min_h_len:
            left = np.array([x1, y1]) if x1 < x2 else np.array([x2, y2])
            right = np.array([x2, y2]) if x1 < x2 else np.array([x1, y1])
            horizontals.append(
                {
                    "x_left": float(left[0]),
                    "y_left": float(left[1]),
                    "x_right": float(right[0]),
                    "y_right": float(right[1]),
                    "length": length,
                }
            )
    return verticals, horizontals


# --------------------------------------------------------------------------- #
# Stage 2: cluster verticals into posts
# --------------------------------------------------------------------------- #
def _cluster_posts(
    verticals: list[dict[str, float]],
    *,
    config: ApparatusConfig,
    width: int,
    height: int,
) -> list[PostCandidate]:
    if not verticals:
        return []
    cluster_px = max(8.0, config.post_cluster_px_frac * width)
    min_length = max(30.0, config.min_post_length_frac * height)
    min_support = max(30.0, config.min_post_support_frac * height)

    verticals = sorted(verticals, key=lambda s: s["x_mid"])
    clusters: list[list[dict[str, float]]] = []
    for seg in verticals:
        if not clusters:
            clusters.append([seg])
            continue
        members = clusters[-1]
        weights = np.array([m["length"] for m in members], dtype=np.float64)
        cluster_x = float(np.average([m["x_mid"] for m in members], weights=weights))
        if abs(seg["x_mid"] - cluster_x) <= cluster_px:
            members.append(seg)
        else:
            clusters.append([seg])

    posts: list[PostCandidate] = []
    for members in clusters:
        support = float(sum(m["length"] for m in members))
        if support < min_support:
            continue
        # Fit x = m*y + b (near-vertical) from endpoints, weighted by length.
        ys: list[float] = []
        xs: list[float] = []
        ws: list[float] = []
        for m in members:
            for xk, yk in ((m["x_top"], m["y_top"]), (m["x_base"], m["y_base"])):
                ys.append(yk)
                xs.append(xk)
                ws.append(m["length"])
        ys_a = np.asarray(ys, dtype=np.float64)
        xs_a = np.asarray(xs, dtype=np.float64)
        ws_a = np.asarray(ws, dtype=np.float64)
        y_top = float(np.min(ys_a))
        y_base = float(np.max(ys_a))
        if y_base - y_top < min_length:
            continue
        design = np.column_stack([ys_a, np.ones_like(ys_a)])
        try:
            slope, intercept = np.linalg.lstsq(design * ws_a[:, None], xs_a * ws_a, rcond=None)[0]
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate cluster
            continue
        x_top = float(slope * y_top + intercept)
        x_base = float(slope * y_base + intercept)
        line = _line_from_points(
            np.array([x_top, y_top], dtype=np.float64),
            np.array([x_base, y_base], dtype=np.float64),
        )
        if not np.isfinite(line).all():
            continue
        mean_angle = float(
            np.average([m["angle_from_vertical"] for m in members],
                       weights=[m["length"] for m in members])
        )
        posts.append(
            PostCandidate(
                x_base_px=x_base,
                x_top_px=x_top,
                y_base_px=y_base,
                y_top_px=y_top,
                length_px=float(y_base - y_top),
                support_px=support,
                n_segments=len(members),
                line=line,
                mean_angle_from_vertical_deg=mean_angle,
            )
        )
    posts.sort(key=lambda p: p.x_base_px)
    return posts


def _build_horizontals(
    horizontals: list[dict[str, float]],
) -> list[HorizontalSegment]:
    out: list[HorizontalSegment] = []
    for h in horizontals:
        line = _line_from_points(
            np.array([h["x_left"], h["y_left"]], dtype=np.float64),
            np.array([h["x_right"], h["y_right"]], dtype=np.float64),
        )
        if not np.isfinite(line).all():
            continue
        out.append(
            HorizontalSegment(
                x_left_px=float(h["x_left"]),
                x_right_px=float(h["x_right"]),
                y_left_px=float(h["y_left"]),
                y_right_px=float(h["y_right"]),
                length_px=float(h["length"]),
                line=line,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Stage 3: crossbar / pad relationships for a candidate pair
# --------------------------------------------------------------------------- #
def _bar_band_for_pair(left: PostCandidate, right: PostCandidate, config: ApparatusConfig) -> tuple[float, float]:
    """Image-y band in which a crossbar may connect this pair near the tops."""
    length = float(min(left.length_px, right.length_px))
    top = min(left.y_top_px, right.y_top_px)
    y_hi = top - config.bar_band_top_margin_frac * length  # may sit slightly above tops
    y_lo = top + config.bar_band_depth_frac * length
    return y_hi, y_lo


def _find_crossbar(
    left: PostCandidate,
    right: PostCandidate,
    horizontals: list[HorizontalSegment],
    *,
    config: ApparatusConfig,
) -> tuple[HorizontalSegment | None, float, np.ndarray | None, np.ndarray | None]:
    """Find a horizontal that connects the pair near its tops.

    Returns (segment, support, left_top, right_top).  The tops are the
    intersections of each post line with the crossbar line, so a leaning post
    still yields a geometrically correct top anchor.
    """
    separation = abs(right.x_base_px - left.x_base_px)
    if separation <= 1.0:
        return None, 0.0, None, None
    y_hi, y_lo = _bar_band_for_pair(left, right, config)
    min_span = config.min_bar_span_frac * separation
    max_overhang = config.max_bar_overhang_frac * separation

    best: tuple[float, HorizontalSegment, np.ndarray, np.ndarray] | None = None
    for h in horizontals:
        left_hit = _line_intersection(h.line, left.line)
        right_hit = _line_intersection(h.line, right.line)
        if left_hit is None or right_hit is None:
            continue
        bar_y = float((left_hit[1] + right_hit[1]) / 2.0)
        if not (y_hi <= bar_y <= y_lo):
            continue
        # The horizontal segment must physically span most of the inter-post gap
        # near the post x-positions, not just pass through the extended lines.
        lo_x = min(left.x_at(bar_y), right.x_at(bar_y))
        hi_x = max(left.x_at(bar_y), right.x_at(bar_y))
        overlap = min(h.x_right_px, hi_x) - max(h.x_left_px, lo_x)
        if overlap < min_span:
            continue
        # A real crossbar ends close to its standards.  Building rooflines, fence
        # rails, and the bottom track edge run far past both posts, so reject any
        # horizontal that overhangs either upright by more than ``max_overhang``.
        left_overhang = max(0.0, lo_x - h.x_left_px)
        right_overhang = max(0.0, h.x_right_px - hi_x)
        if left_overhang > max_overhang or right_overhang > max_overhang:
            continue
        span_frac = float(overlap / separation)
        support = float(h.length_px * span_frac)
        if best is None or support > best[0]:
            best = (support, h, left_hit.astype(np.float64), right_hit.astype(np.float64))

    if best is None:
        return None, 0.0, None, None
    support, seg, left_top, right_top = best
    return seg, support, left_top, right_top


def _find_pad_lines(
    left: PostCandidate,
    right: PostCandidate,
    horizontals: list[HorizontalSegment],
    *,
    bar_y: float | None,
    config: ApparatusConfig,
) -> tuple[HorizontalSegment | None, float, HorizontalSegment | None, HorizontalSegment | None]:
    """Find the landing-pad edges spanning the uprights below the bar region.

    Returns ``(pad, pad_support, top_line, bottom_line)`` where:

    * ``pad`` / ``pad_support`` are the **best-supported** qualifying horizontal
      (the pad edge used for ``has_pad`` and the pad score term — identical
      semantics to the original detector, so selection is unchanged); and
    * ``top_line`` / ``bottom_line`` are the **highest** and **lowest** qualifying
      horizontals (the latter clearly below the former), used only for the
      bar/pad vertical-ratio prior.  ``bottom_line`` is ``None`` when no distinct
      lower edge exists.
    """
    separation = abs(right.x_base_px - left.x_base_px)
    if separation <= 1.0:
        return None, 0.0, None, None
    length = float(min(left.length_px, right.length_px))
    y_top = min(left.y_top_px, right.y_top_px)
    y_base = max(left.y_base_px, right.y_base_px)
    ref_bar_y = bar_y if bar_y is not None else (y_top + config.bar_band_depth_frac * length)
    pad_min_y = ref_bar_y + config.min_pad_gap_frac * length
    pad_max_y = y_base + 0.35 * length
    lo_x = min(left.x_base_px, right.x_base_px)
    hi_x = max(left.x_base_px, right.x_base_px)
    min_span = config.min_pad_span_frac * separation
    max_overhang = config.max_bar_overhang_frac * separation

    candidates: list[tuple[float, float, HorizontalSegment]] = []  # (y_center, support, seg)
    for h in horizontals:
        y_c = h.y_center_px
        if not (pad_min_y <= y_c <= pad_max_y):
            continue
        overlap = min(h.x_right_px, hi_x) - max(h.x_left_px, lo_x)
        if overlap < min_span:
            continue
        # Same overhang gate as the crossbar: a pad edge sits between the
        # standards, not across the whole frame like the bottom track line.
        if max(0.0, lo_x - h.x_left_px) > max_overhang or max(0.0, h.x_right_px - hi_x) > max_overhang:
            continue
        support = float(h.length_px * (overlap / separation))
        candidates.append((y_c, support, h))
    if not candidates:
        return None, 0.0, None, None

    # Best-support pad edge (unchanged behaviour for has_pad / scoring).
    best_support, best_seg = max(((s, seg) for _, s, seg in candidates), key=lambda c: c[0])
    # Highest and lowest qualifying lines for the ratio triple.
    candidates.sort(key=lambda c: c[0])
    top_y, _ts, top_seg = candidates[0]
    min_bottom_gap = config.min_pad_bottom_gap_frac * length
    bottom_seg: HorizontalSegment | None = None
    for y_c, _support, seg in reversed(candidates):
        if y_c - top_y >= min_bottom_gap:
            bottom_seg = seg
            break
    return best_seg, best_support, top_seg, bottom_seg


# --------------------------------------------------------------------------- #
# Optional colour evidence
# --------------------------------------------------------------------------- #
def _red_response(frame_bgr: np.ndarray) -> np.ndarray:
    frame = frame_bgr[:, :, :3].astype(np.float64)
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return np.maximum(0.0, r - 0.70 * g - 0.45 * b)


def _bar_colour_bonus(
    frame_bgr: np.ndarray,
    left_top: np.ndarray,
    right_top: np.ndarray,
    config: ApparatusConfig,
) -> float:
    """Additive-only colour evidence along the crossbar.

    Rewards either a red bar (night apparatus) or a bar whose brightness clearly
    differs from the band just above it (dark daylight bar on bright sky).  Never
    a requirement; absence simply yields 0.
    """
    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    n = 40
    xs = np.linspace(float(left_top[0]), float(right_top[0]), n)
    ys = np.linspace(float(left_top[1]), float(right_top[1]), n)
    xi = np.clip(np.rint(xs).astype(int), 0, width - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, height - 1)

    red = _red_response(frame_bgr)
    red_on_bar = float(np.mean(red[yi, xi]))
    red_bonus = config.w_colour_red * float(np.tanh(red_on_bar / 40.0))

    gray = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)
    above_yi = np.clip(yi - max(6, int(0.012 * height)), 0, height - 1)
    bar_lum = gray[yi, xi]
    above_lum = gray[above_yi, xi]
    contrast = float(np.mean(np.abs(bar_lum - above_lum)))
    contrast_bonus = config.w_colour_contrast * float(np.tanh(contrast / 50.0))
    return red_bonus + contrast_bonus


# --------------------------------------------------------------------------- #
# Stage 4: pair scoring and selection
# --------------------------------------------------------------------------- #
def _score_pair(
    frame_bgr: np.ndarray,
    left: PostCandidate,
    right: PostCandidate,
    horizontals: list[HorizontalSegment],
    *,
    config: ApparatusConfig,
    width: int,
    height: int,
    athlete_center_x: float | None,
    bar_height_m: float | None = None,
) -> ApparatusDetection | None:
    separation = abs(right.x_base_px - left.x_base_px)
    min_sep = config.min_separation_frac * width
    max_sep = config.max_separation_frac * width
    if separation < min_sep or separation > max_sep:
        return None

    height_ratio = min(left.length_px, right.length_px) / max(left.length_px, right.length_px)
    if height_ratio < config.min_height_ratio:
        return None
    base_level_diff = abs(left.y_base_px - right.y_base_px)
    if base_level_diff > config.base_level_tol_frac * separation:
        return None

    crossbar, bar_support, bar_left_top, bar_right_top = _find_crossbar(
        left, right, horizontals, config=config
    )
    bar_y = None
    if crossbar is not None and bar_left_top is not None and bar_right_top is not None:
        bar_y = float((bar_left_top[1] + bar_right_top[1]) / 2.0)
    pad, pad_support, pad_top_line, pad_bottom_line = _find_pad_lines(
        left, right, horizontals, bar_y=bar_y, config=config
    )

    has_crossbar = crossbar is not None
    has_pad = pad is not None
    if config.require_crossbar and not has_crossbar:
        return None
    if not has_crossbar and not has_pad:
        # An isolated pair of verticals with no connecting bar and no pad edge is
        # a pair of background poles, not the apparatus.
        return None

    # Tops: prefer crossbar/post intersections; else the post tops.
    if has_crossbar:
        left_top = bar_left_top
        right_top = bar_right_top
    else:
        left_top = np.array([left.x_top_px, left.y_top_px], dtype=np.float64)
        right_top = np.array([right.x_top_px, right.y_top_px], dtype=np.float64)
    left_base = np.array([left.x_base_px, left.y_base_px], dtype=np.float64)
    right_base = np.array([right.x_base_px, right.y_base_px], dtype=np.float64)

    # --- score ---
    score = 0.0
    score += config.w_crossbar * float(np.tanh(bar_support / max(1.0, 0.5 * separation)))
    score += config.w_pad * float(np.tanh(pad_support / max(1.0, 0.5 * separation)))
    score += config.w_length * (left.support_px + right.support_px)
    score += config.w_symmetry * height_ratio
    score -= config.w_symmetry * (base_level_diff / max(1.0, separation))
    # Foreground prior: apparatus bases sit on the near mat, low in the image.
    base_y_mean = float((left.y_base_px + right.y_base_px) / 2.0)
    score += config.w_foreground * (base_y_mean / height)
    # ...but the very bottom of the frame is the track edge, not the mat.  Posts
    # whose bases reach that band are usually mat/track edges merged with a
    # standard, so penalise them rather than rewarding "lowest".
    bottom_y = (1.0 - config.bottom_edge_margin_frac) * height
    if base_y_mean > bottom_y:
        score -= config.w_bottom_edge_penalty * ((base_y_mean - bottom_y) / max(1.0, config.bottom_edge_margin_frac * height))
    score -= config.w_angle * (left.mean_angle_from_vertical_deg + right.mean_angle_from_vertical_deg)
    # Athlete proximity prior: the takeoff apparatus is near the jumper.
    if athlete_center_x is not None:
        pair_center_x = float((left.x_base_px + right.x_base_px) / 2.0)
        sigma = max(1.0, config.athlete_sigma_frac * width)
        score += config.w_athlete * float(np.exp(-((pair_center_x - athlete_center_x) ** 2) / (2.0 * sigma ** 2)))
    if has_crossbar:
        score += _bar_colour_bonus(frame_bgr, left_top, right_top, config)

    # Pad/bar vertical-ratio prior: the image ratio (bar->pad-top)/(bar->pad-bottom)
    # should match 1 - H_pad/H_bar.  Used as a PENALTY only — a triple at the
    # expected ratio costs nothing, a clearly wrong ratio is penalised.  This
    # rejects an apparatus-shaped background building (whose roofline/ground triple
    # has the wrong ratio) WITHOUT rewarding — and thereby promoting — any wide
    # pair that happens to form a near-correct triple.
    pad_ratio: float | None = None
    pad_ratio_expected: float | None = None
    if (
        has_crossbar and bar_y is not None
        and pad_top_line is not None and pad_bottom_line is not None
    ):
        top_gap = pad_top_line.y_center_px - bar_y
        full_gap = pad_bottom_line.y_center_px - bar_y
        if full_gap > 1.0 and top_gap > 0.0:
            pad_ratio = float(top_gap / full_gap)
            if bar_height_m is not None and bar_height_m > 0.0:
                pad_ratio_expected = float(
                    max(0.05, min(0.95, 1.0 - config.expected_pad_height_m / bar_height_m))
                )
                excess = max(0.0, abs(pad_ratio - pad_ratio_expected) - config.pad_ratio_deadzone)
                match = float(np.exp(-(excess ** 2) / (2.0 * config.pad_ratio_sigma ** 2)))
                score -= config.w_pad_ratio * (1.0 - match)

    # --- confidence in [0, 1] ---
    confidence = 0.0
    confidence += 0.45 if has_crossbar else 0.0
    confidence += 0.20 if has_pad else 0.0
    confidence += 0.15 * height_ratio
    confidence += 0.10 * float(np.tanh(bar_support / max(1.0, 0.5 * separation)))
    confidence += 0.10 * max(0.0, 1.0 - (left.mean_angle_from_vertical_deg + right.mean_angle_from_vertical_deg) / 24.0)
    # A clearly wrong pad/bar ratio is evidence *against* a real apparatus.
    if pad_ratio is not None and pad_ratio_expected is not None:
        excess = max(0.0, abs(pad_ratio - pad_ratio_expected) - config.pad_ratio_deadzone)
        match = float(np.exp(-(excess ** 2) / (2.0 * config.pad_ratio_sigma ** 2)))
        confidence -= 0.25 * (1.0 - match)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return ApparatusDetection(
        left_base_px=left_base,
        right_base_px=right_base,
        left_top_px=np.asarray(left_top, dtype=np.float64),
        right_top_px=np.asarray(right_top, dtype=np.float64),
        separation_px=float(separation),
        has_crossbar=has_crossbar,
        has_pad=has_pad,
        crossbar=crossbar,
        pad_top=pad,
        score=float(score),
        confidence=confidence,
        left_post=left,
        right_post=right,
        pad_bottom=pad_bottom_line,
        pad_ratio=pad_ratio,
        pad_ratio_expected=pad_ratio_expected,
    )


def detect_apparatus_geometry(
    frame_bgr: np.ndarray,
    *,
    athlete_bbox_px: Sequence[float] | None = None,
    athlete_center_x_px: float | None = None,
    bar_height_m: float | None = None,
    config: ApparatusConfig = ApparatusConfig(),
) -> ApparatusDetection | None:
    """Detect high-jump apparatus anchors from frame geometry (colour-agnostic).

    Args:
        frame_bgr: ``(H, W, 3)`` BGR image.
        athlete_bbox_px: Optional ``[x0, y0, x1, y1]`` athlete box.  Segments
            inside it are dropped (so the jumper is never mistaken for a post),
            and its horizontal centre biases pair selection toward the takeoff
            apparatus.  Pass ``None`` to detect without a pose hint.
        athlete_center_x_px: Optional takeoff-zone bias *without* masking, for
            athlete-free inputs such as a synthesized static plate (the jumper is
            already gone, and at apex the bbox would overlap — and wrongly mask —
            the crossbar).  Ignored when ``athlete_bbox_px`` is given.
        bar_height_m: Optional known crossbar height (m).  When given, the
            pad/bar vertical-ratio prior is active and rewards pairs whose
            (bar->pad-top)/(bar->pad-bottom) image ratio matches ``1 - H_pad/H_bar``
            — the main discriminator against apparatus-shaped background buildings.
        config: Tuning parameters.

    Returns:
        The best-scoring admissible apparatus detection, or ``None`` when no
        upright pair with a crossbar/pad relationship clears the gates.
    """
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame_bgr must be a BGR image")
    height, width = frame.shape[:2]

    verticals, horizontal_segments = _extract_segments(
        frame, config=config, athlete_bbox_px=athlete_bbox_px
    )
    posts = _cluster_posts(verticals, config=config, width=width, height=height)
    if len(posts) < 2:
        return None
    horizontals = _build_horizontals(horizontal_segments)

    athlete_center_x: float | None = None
    if athlete_bbox_px is not None:
        athlete_center_x = float((athlete_bbox_px[0] + athlete_bbox_px[2]) / 2.0)
    elif athlete_center_x_px is not None:
        athlete_center_x = float(athlete_center_x_px)

    best: ApparatusDetection | None = None
    for i, left in enumerate(posts):
        for right in posts[i + 1:]:
            detection = _score_pair(
                frame,
                left,
                right,
                horizontals,
                config=config,
                width=width,
                height=height,
                athlete_center_x=athlete_center_x,
                bar_height_m=bar_height_m,
            )
            if detection is None:
                continue
            if best is None or detection.score > best.score:
                best = detection

    if best is None or best.confidence < config.min_confidence:
        return None
    return best


# --------------------------------------------------------------------------- #
# Temporal aggregation
# --------------------------------------------------------------------------- #
def detect_apparatus_geometry_stable(
    frames: Sequence[np.ndarray],
    *,
    athlete_bboxes: Sequence[Sequence[float] | None] | None = None,
    bar_height_m: float | None = None,
    config: ApparatusConfig = ApparatusConfig(),
) -> StableApparatusDetection:
    """Aggregate apparatus geometry across multiple stationary frames.

    A stationary camera should yield a *stable* apparatus geometry.  We detect
    per frame, keep the largest cluster of mutually consistent detections
    (matched by base separation and left-base x), and report the per-anchor
    median.  ``is_stable`` requires at least ``stable_min_frames`` agreeing
    frames and a per-anchor spread below ``stable_max_anchor_std_frac``.
    """
    frame_list = list(frames)
    n_total = len(frame_list)
    if n_total == 0:
        raise ValueError("frames must be non-empty")
    width = int(frame_list[0].shape[1])

    per_frame: list[ApparatusDetection | None] = []
    for idx, frame in enumerate(frame_list):
        bbox = None
        if athlete_bboxes is not None and idx < len(athlete_bboxes):
            bbox = athlete_bboxes[idx]
        per_frame.append(
            detect_apparatus_geometry(
                frame, athlete_bbox_px=bbox, bar_height_m=bar_height_m, config=config
            )
        )

    detected = [d for d in per_frame if d is not None]
    sep_tol = config.stable_separation_tol_frac * width

    def _key(d: ApparatusDetection) -> np.ndarray:
        return np.array([d.left_base_px[0], d.right_base_px[0]], dtype=np.float64)

    # Greedy clustering of consistent detections by (left_base_x, right_base_x).
    best_cluster: list[ApparatusDetection] = []
    for anchor in detected:
        cluster = [d for d in detected if float(np.max(np.abs(_key(d) - _key(anchor)))) <= sep_tol]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if not best_cluster:
        nan_pt = np.full(2, np.nan, dtype=np.float64)
        return StableApparatusDetection(
            left_base_px=nan_pt.copy(),
            right_base_px=nan_pt.copy(),
            left_top_px=nan_pt.copy(),
            right_top_px=nan_pt.copy(),
            n_frames_used=0,
            n_frames_total=n_total,
            anchor_std_px=float("nan"),
            crossbar_frac=0.0,
            pad_frac=0.0,
            is_stable=False,
            per_frame=per_frame,
        )

    def _median_anchor(name: str) -> np.ndarray:
        pts = np.array([getattr(d, name) for d in best_cluster], dtype=np.float64)
        return np.median(pts, axis=0)

    left_base = _median_anchor("left_base_px")
    right_base = _median_anchor("right_base_px")
    left_top = _median_anchor("left_top_px")
    right_top = _median_anchor("right_top_px")

    spreads = []
    for name in ("left_base_px", "right_base_px", "left_top_px", "right_top_px"):
        pts = np.array([getattr(d, name) for d in best_cluster], dtype=np.float64)
        spreads.append(float(np.mean(np.std(pts, axis=0))))
    anchor_std = float(np.mean(spreads))

    crossbar_frac = float(np.mean([d.has_crossbar for d in best_cluster]))
    pad_frac = float(np.mean([d.has_pad for d in best_cluster]))
    is_stable = (
        len(best_cluster) >= config.stable_min_frames
        and anchor_std <= config.stable_max_anchor_std_frac * width
    )

    return StableApparatusDetection(
        left_base_px=left_base,
        right_base_px=right_base,
        left_top_px=left_top,
        right_top_px=right_top,
        n_frames_used=len(best_cluster),
        n_frames_total=n_total,
        anchor_std_px=anchor_std,
        crossbar_frac=crossbar_frac,
        pad_frac=pad_frac,
        is_stable=is_stable,
        per_frame=per_frame,
    )


# --------------------------------------------------------------------------- #
# Debug overlay
# --------------------------------------------------------------------------- #
def draw_apparatus_debug(
    frame_bgr: np.ndarray,
    detection: ApparatusDetection,
    *,
    extra_label: str | None = None,
) -> np.ndarray:
    """Draw labelled anchors, crossbar, posts, and pad-top candidate for review."""
    cv2 = _require_cv2()
    out = frame_bgr.copy()
    pts = {
        "left_base": detection.left_base_px,
        "right_base": detection.right_base_px,
        "left_top": detection.left_top_px,
        "right_top": detection.right_top_px,
    }

    def _pt(p: np.ndarray) -> tuple[int, int]:
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    # Posts (yellow), crossbar (red), pad-top (orange).
    cv2.line(out, _pt(pts["left_base"]), _pt(pts["left_top"]), (0, 220, 255), 2, cv2.LINE_AA)
    cv2.line(out, _pt(pts["right_base"]), _pt(pts["right_top"]), (0, 220, 255), 2, cv2.LINE_AA)
    cv2.line(out, _pt(pts["left_top"]), _pt(pts["right_top"]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(out, _pt(pts["left_base"]), _pt(pts["right_base"]), (0, 255, 0), 2, cv2.LINE_AA)
    if detection.pad_top is not None:
        pad = detection.pad_top
        cv2.line(
            out,
            (int(round(pad.x_left_px)), int(round(pad.y_left_px))),
            (int(round(pad.x_right_px)), int(round(pad.y_right_px))),
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    # Pad-bottom / ground line (magenta) — the third line of the ratio triple.
    if detection.pad_bottom is not None:
        pb = detection.pad_bottom
        cv2.line(
            out,
            (int(round(pb.x_left_px)), int(round(pb.y_left_px))),
            (int(round(pb.x_right_px)), int(round(pb.y_right_px))),
            (255, 0, 200),
            2,
            cv2.LINE_AA,
        )

    colours = {
        "left_base": (0, 255, 0),
        "right_base": (0, 255, 0),
        "left_top": (0, 0, 255),
        "right_top": (0, 0, 255),
    }
    for name, p in pts.items():
        xy = _pt(p)
        cv2.circle(out, xy, 8, colours[name], -1, cv2.LINE_AA)
        text = f"{name} ({p[0]:.0f},{p[1]:.0f})"
        dy = -14 if "top" in name else 24
        for colour, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(out, text, (xy[0] - 95, xy[1] + dy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, colour, thick, cv2.LINE_AA)

    lines = [
        "GEOMETRY-FIRST APPARATUS DETECTION",
        f"sep={detection.separation_px:.0f}px score={detection.score:.2f} "
        f"conf={detection.confidence:.2f} crossbar={detection.has_crossbar} pad={detection.has_pad}",
    ]
    if detection.pad_ratio is not None:
        exp = detection.pad_ratio_expected
        exp_txt = f" exp={exp:.2f}" if exp is not None else ""
        lines.append(f"pad/bar ratio={detection.pad_ratio:.2f}{exp_txt} (bar/pad-top/pad-bottom)")
    if extra_label:
        lines.append(extra_label)
    y = 34
    for line in lines:
        for colour, thick in ((0, 0, 0), 6), ((255, 255, 255), 2):
            cv2.putText(out, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.74, colour, thick, cv2.LINE_AA)
        y += 32
    return out
