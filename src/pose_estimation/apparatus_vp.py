"""Vanishing-point-constrained apparatus line fit inside a pose-localized ROI.

Once the pose prior (:mod:`apparatus_pose_prior`) brackets the apparatus and
gives approximate **bar** and **ground** image-lines plus a metric scale, the
remaining job is to fit the five lines of the apparatus precisely:

* two near-vertical **standards** (left / right uprights); and
* three near-horizontal lines that are parallel in 3D and therefore share one
  **vanishing point** in the image — the **bar**, the **pad top**, and the
  **pad bottom (ground)**.

We seed each line from the pose prior, collect the real edge segments near that
seed, and fit a line by total-least-squares — so the fitted lines follow the
true perspective slope rather than being forced horizontal/vertical.  The four
corners are then the standard-x-bar / standard-x-ground intersections, which is
what the earlier blob/Hough heuristics got wrong.

This is appearance- and colour-agnostic and, because it is seeded by the
pose-derived ROI, it never wanders onto background structure.  Pixel coordinates
are ``(x, y)`` with ``y`` down, expressed in the **ROI-crop** frame; the caller
offsets them back to full-frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pose_estimation.apparatus_detector import _line_intersection

__all__ = ["VpConfig", "ApparatusLineFit", "fit_apparatus_lines", "draw_line_fit"]


@dataclass(frozen=True)
class VpConfig:
    max_vertical_angle_deg: float = 22.0
    max_horizontal_angle_deg: float = 22.0
    min_seg_len_frac: float = 0.05  # of crop diagonal
    # x-band (as a fraction of expected standard separation) around each seeded
    # standard within which vertical segments are gathered.
    standard_x_band_frac: float = 0.28
    # y-band (in metres, via the pose scale) around each seeded horizontal.
    horizontal_y_band_m: float = 0.35
    min_support_segments: int = 1


@dataclass(frozen=True)
class ApparatusLineFit:
    left_base_px: np.ndarray
    right_base_px: np.ndarray
    left_top_px: np.ndarray
    right_top_px: np.ndarray
    left_standard: np.ndarray  # homogeneous line (3,)
    right_standard: np.ndarray
    bar_line: np.ndarray
    pad_top_line: np.ndarray | None
    ground_line: np.ndarray
    horizon_vp_px: np.ndarray | None  # shared horizontal vanishing point, if finite
    n_support: dict

    def points_px(self) -> dict[str, list[float]]:
        def _r(p):
            return [round(float(p[0]), 3), round(float(p[1]), 3)]
        return {
            "left_base": _r(self.left_base_px),
            "right_base": _r(self.right_base_px),
            "left_top": _r(self.left_top_px),
            "right_top": _r(self.right_top_px),
        }


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for the VP line fit") from exc
    return cv2


def _segments(gray: np.ndarray, min_len: float) -> np.ndarray:
    """Return (N, 4) line segments [x1, y1, x2, y2] via LSD, else HoughP."""
    cv2 = _require_cv2()
    segs: list[list[float]] = []
    try:
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(gray)[0]
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                if np.hypot(x2 - x1, y2 - y1) >= min_len:
                    segs.append([float(x1), float(y1), float(x2), float(y2)])
    except Exception:  # pragma: no cover - LSD unavailable in some builds
        segs = []
    if not segs:
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180.0, threshold=30,
            minLineLength=int(min_len), maxLineGap=int(0.5 * min_len) or 1,
        )
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                segs.append([float(x1), float(y1), float(x2), float(y2)])
    return np.asarray(segs, dtype=np.float64).reshape(-1, 4)


def _angle_from_horizontal(seg: np.ndarray) -> float:
    dx, dy = seg[2] - seg[0], seg[3] - seg[1]
    a = abs(np.degrees(np.arctan2(dy, dx)))
    return float(min(a, abs(180.0 - a)))


def _fit_line_tls(points: np.ndarray) -> np.ndarray | None:
    """Total-least-squares homogeneous line through points (N>=2, 2)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    normal /= norm
    c = -float(normal @ centroid)
    return np.array([normal[0], normal[1], c], dtype=np.float64)


def _seg_endpoints(segs: np.ndarray) -> np.ndarray:
    return np.vstack([segs[:, :2], segs[:, 2:4]]) if segs.size else np.empty((0, 2))


def _fit_standard(
    segs: np.ndarray, x_seed: float, *, x_band: float, y_lo: float, y_hi: float,
    max_vert_angle: float,
) -> tuple[np.ndarray | None, int]:
    """Fit a near-vertical standard from segments in an x-band around the seed."""
    if segs.size == 0:
        return None, 0
    keep = []
    for s in segs:
        if _angle_from_horizontal(s) < (90.0 - max_vert_angle):
            continue  # not vertical enough
        xm, ym = (s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0
        if abs(xm - x_seed) <= x_band and (y_lo - 0.15 * (y_hi - y_lo)) <= ym <= (y_hi + 0.15 * (y_hi - y_lo)):
            keep.append(s)
    if len(keep) < 1:
        return None, 0
    pts = _seg_endpoints(np.asarray(keep))
    return _fit_line_tls(pts), len(keep)


def _fit_horizontal(
    segs: np.ndarray, y_seed: float, *, y_band: float, x_lo: float, x_hi: float,
    max_horiz_angle: float,
) -> tuple[np.ndarray | None, int]:
    """Fit a near-horizontal line from segments in a y-band around the seed."""
    if segs.size == 0:
        return None, 0
    keep = []
    for s in segs:
        if _angle_from_horizontal(s) > max_horiz_angle:
            continue
        xm, ym = (s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0
        if abs(ym - y_seed) <= y_band and (x_lo - 0.10 * (x_hi - x_lo)) <= xm <= (x_hi + 0.10 * (x_hi - x_lo)):
            keep.append(s)
    if len(keep) < 1:
        return None, 0
    pts = _seg_endpoints(np.asarray(keep))
    return _fit_line_tls(pts), len(keep)


def _vertical_line_at(x: float) -> np.ndarray:
    return np.array([1.0, 0.0, -float(x)], dtype=np.float64)


def _horizontal_line_at(y: float) -> np.ndarray:
    return np.array([0.0, 1.0, -float(y)], dtype=np.float64)


def fit_apparatus_lines(
    crop_bgr: np.ndarray,
    *,
    bar_x_px: float,
    bar_y_px: float,
    ground_y_px: float,
    scale_px_per_m: float,
    upright_separation_m: float = 4.02,
    config: VpConfig = VpConfig(),
) -> ApparatusLineFit | None:
    """Fit the five apparatus lines inside a ROI crop, seeded by pose priors.

    All ``*_px`` seeds are in the crop frame.  Returns ``None`` if the crop is
    degenerate.  Lines for which no supporting segments are found fall back to the
    seeded vertical/horizontal through the prior, so a result is always returned
    when the geometry is well-posed.
    """
    cv2 = _require_cv2()
    crop = np.asarray(crop_bgr)
    if crop.ndim != 3 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    diag = float(np.hypot(w, h))
    min_len = max(12.0, config.min_seg_len_frac * diag)
    segs = _segments(gray, min_len)

    half_sep_px = (upright_separation_m / 2.0) * scale_px_per_m
    x_left_seed = bar_x_px - half_sep_px
    x_right_seed = bar_x_px + half_sep_px
    x_band = config.standard_x_band_frac * (2.0 * half_sep_px)
    y_band = config.horizontal_y_band_m * scale_px_per_m

    left_std, n_left = _fit_standard(
        segs, x_left_seed, x_band=x_band, y_lo=bar_y_px, y_hi=ground_y_px,
        max_vert_angle=config.max_vertical_angle_deg,
    )
    right_std, n_right = _fit_standard(
        segs, x_right_seed, x_band=x_band, y_lo=bar_y_px, y_hi=ground_y_px,
        max_vert_angle=config.max_vertical_angle_deg,
    )
    if left_std is None:
        left_std = _vertical_line_at(x_left_seed)
    if right_std is None:
        right_std = _vertical_line_at(x_right_seed)

    x_lo, x_hi = min(x_left_seed, x_right_seed), max(x_left_seed, x_right_seed)
    bar_line, n_bar = _fit_horizontal(
        segs, bar_y_px, y_band=y_band, x_lo=x_lo, x_hi=x_hi,
        max_horiz_angle=config.max_horizontal_angle_deg,
    )
    ground_line, n_ground = _fit_horizontal(
        segs, ground_y_px, y_band=y_band, x_lo=x_lo, x_hi=x_hi,
        max_horiz_angle=config.max_horizontal_angle_deg,
    )
    # Pad-top seed: midway between bar and ground, biased toward a ~0.7 m pad.
    pad_top_seed = ground_y_px - 0.70 * scale_px_per_m
    pad_top_line, n_pad = _fit_horizontal(
        segs, pad_top_seed, y_band=y_band, x_lo=x_lo, x_hi=x_hi,
        max_horiz_angle=config.max_horizontal_angle_deg,
    )
    if bar_line is None:
        bar_line = _horizontal_line_at(bar_y_px)
    if ground_line is None:
        ground_line = _horizontal_line_at(ground_y_px)

    def _intersect(a: np.ndarray, b: np.ndarray, fallback_x: float, fallback_y: float) -> np.ndarray:
        p = _line_intersection(a, b)
        if p is None or not np.isfinite(p).all() or not (-0.5 * w <= p[0] <= 1.5 * w) or not (-0.5 * h <= p[1] <= 1.5 * h):
            return np.array([fallback_x, fallback_y], dtype=np.float64)
        return p

    left_top = _intersect(left_std, bar_line, x_left_seed, bar_y_px)
    right_top = _intersect(right_std, bar_line, x_right_seed, bar_y_px)
    left_base = _intersect(left_std, ground_line, x_left_seed, ground_y_px)
    right_base = _intersect(right_std, ground_line, x_right_seed, ground_y_px)

    horizon_vp = _line_intersection(bar_line, ground_line)
    if horizon_vp is not None and not np.isfinite(horizon_vp).all():
        horizon_vp = None

    return ApparatusLineFit(
        left_base_px=left_base,
        right_base_px=right_base,
        left_top_px=left_top,
        right_top_px=right_top,
        left_standard=left_std,
        right_standard=right_std,
        bar_line=bar_line,
        pad_top_line=pad_top_line,
        ground_line=ground_line,
        horizon_vp_px=horizon_vp,
        n_support={
            "left_standard": n_left,
            "right_standard": n_right,
            "bar": n_bar,
            "ground": n_ground,
            "pad_top": n_pad,
        },
    )


def draw_line_fit(crop_bgr: np.ndarray, fit: ApparatusLineFit) -> np.ndarray:
    """Draw the fitted standards, bar, pad-top, ground, and corners on the crop."""
    cv2 = _require_cv2()
    out = crop_bgr.copy()

    def _pt(p) -> tuple[int, int]:
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    lb, rb, lt, rt = fit.left_base_px, fit.right_base_px, fit.left_top_px, fit.right_top_px
    cv2.line(out, _pt(lt), _pt(lb), (0, 220, 255), 2, cv2.LINE_AA)   # left standard
    cv2.line(out, _pt(rt), _pt(rb), (0, 220, 255), 2, cv2.LINE_AA)   # right standard
    cv2.line(out, _pt(lt), _pt(rt), (0, 0, 255), 3, cv2.LINE_AA)     # bar
    cv2.line(out, _pt(lb), _pt(rb), (255, 0, 200), 2, cv2.LINE_AA)   # ground
    if fit.pad_top_line is not None:
        # Draw the pad-top across the standard span at its fitted y.
        a, b, c = fit.pad_top_line
        x0v, x1v = float(min(lb[0], lt[0])), float(max(rb[0], rt[0]))
        def _y_at(x):
            return None if abs(b) < 1e-9 else (-(a * x + c) / b)
        y0v, y1v = _y_at(x0v), _y_at(x1v)
        if y0v is not None and y1v is not None:
            cv2.line(out, (int(x0v), int(y0v)), (int(x1v), int(y1v)), (0, 165, 255), 2, cv2.LINE_AA)
    for p, col in ((lb, (0, 255, 0)), (rb, (0, 255, 0)), (lt, (0, 0, 255)), (rt, (0, 0, 255))):
        cv2.circle(out, _pt(p), 7, col, -1, cv2.LINE_AA)
    ns = fit.n_support
    label = (f"VP FIT  std L/R={ns['left_standard']}/{ns['right_standard']} "
             f"bar={ns['bar']} pad={ns['pad_top']} ground={ns['ground']}")
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
    return out
