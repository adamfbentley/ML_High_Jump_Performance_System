"""Pose-localized apparatus region-of-interest.

The athlete physically interacts with the apparatus, and that interaction is the
one cue that generalises across every clip in the corpus regardless of lighting,
colour, or camera angle:

* her centre-of-mass **apex** sits essentially over the crossbar at ~bar height;
* her **takeoff plant** is just in front of the near standard; and
* her standing stature gives a metric **pixels-per-metre** ruler.

This module turns an already-extracted 2D pose track into a metric-informed
region-of-interest (ROI) that brackets the apparatus, plus pose-derived estimates
of the **bar** and **ground** image-lines.  A line-based detector restricted to
this ROI cannot latch onto background structures (rooflines, masts) the athlete
never touches — which is exactly what fooled the colour/geometry detector.

The core :func:`compute_apparatus_roi` is a pure function over a ``(T, 33, 3)``
BlazePose landmark array (normalised ``x, y, visibility``); MediaPipe extraction
lives in the caller.  Image coordinates are pixels ``(x, y)`` with ``y`` down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ApparatusROI", "compute_apparatus_roi"]

# BlazePose landmark indices.
_HIP_IDX = (23, 24)
_FOOT_IDX = (27, 28, 29, 30, 31, 32)  # ankles, heels, foot indices


@dataclass(frozen=True)
class ApparatusROI:
    """A pose-derived bracket around the apparatus, in pixels.

    ``bar_y_est_px`` / ``ground_y_px`` are the pose-derived horizontal-line
    priors (the crossbar sits ~``bar_height_m`` above the ground line); the line
    detector can seed its bar / pad-bottom search from them.
    """

    x_min_px: float
    x_max_px: float
    y_min_px: float
    y_max_px: float
    bar_x_px: float
    bar_y_est_px: float
    ground_y_px: float
    scale_px_per_m: float
    apex_frame: int
    takeoff_frame: int
    n_valid_frames: int

    def as_bbox(self) -> tuple[float, float, float, float]:
        return (self.x_min_px, self.y_min_px, self.x_max_px, self.y_max_px)


def _visible_xy(landmarks_frame: np.ndarray, idx, *, w: int, h: int, min_vis: float):
    """Return pixel (x, y) of visible landmarks among ``idx``, or empty (0, 2)."""
    pts = landmarks_frame[list(idx)]
    vis = pts[:, 2] >= min_vis
    if not np.any(vis):
        return np.empty((0, 2), dtype=np.float64)
    xy = pts[vis, :2].astype(np.float64)
    xy[:, 0] *= w
    xy[:, 1] *= h
    return xy


def compute_apparatus_roi(
    landmarks_2d: np.ndarray,
    *,
    image_w: int,
    image_h: int,
    athlete_height_m: float = 1.75,
    bar_height_m: float = 1.75,
    upright_separation_m: float = 4.02,
    min_visibility: float = 0.4,
    x_margin_frac: float = 0.35,
    y_margin_frac: float = 0.25,
) -> ApparatusROI | None:
    """Bracket the apparatus from a 2D pose track.

    Args:
        landmarks_2d: ``(T, 33, 3)`` normalised BlazePose landmarks (x, y, vis).
        image_w, image_h: full-frame pixel size the landmarks normalise to.
        athlete_height_m: standing stature, used as the metric ruler.
        bar_height_m: crossbar height above the ground line.
        upright_separation_m: distance between the two standards.
        min_visibility: landmark visibility threshold.
        x_margin_frac: extra horizontal slack beyond the standards, as a fraction
            of the standard separation.
        y_margin_frac: vertical slack above the bar / below the ground, as a
            fraction of the bar height.

    Returns:
        An :class:`ApparatusROI`, or ``None`` when the track lacks usable hips.
    """
    lm = np.asarray(landmarks_2d, dtype=np.float64)
    if lm.ndim != 3 or lm.shape[1] < 33 or lm.shape[2] < 3:
        raise ValueError("landmarks_2d must be (T, 33, 3)")
    n_frames = lm.shape[0]

    hip_xy: list[tuple[int, float, float]] = []  # (frame, x_px, y_px)
    stature_px: list[tuple[int, float]] = []  # (frame, full-body pixel height)
    foot_y_by_frame: dict[int, float] = {}
    for t in range(n_frames):
        hips = _visible_xy(lm[t], _HIP_IDX, w=image_w, h=image_h, min_vis=min_visibility)
        if hips.shape[0] > 0:
            hip_xy.append((t, float(np.mean(hips[:, 0])), float(np.mean(hips[:, 1]))))
        # Full-body vertical extent (any visible landmark) as a stature proxy.
        vis = lm[t, :, 2] >= min_visibility
        if np.count_nonzero(vis) >= 6:
            ys = lm[t, vis, 1] * image_h
            stature_px.append((t, float(ys.max() - ys.min())))
        feet = _visible_xy(lm[t], _FOOT_IDX, w=image_w, h=image_h, min_vis=min_visibility)
        if feet.shape[0] > 0:
            foot_y_by_frame[t] = float(np.max(feet[:, 1]))

    if not hip_xy or not stature_px:
        return None

    # Apex = highest CoM (smallest image y) → over the bar.
    apex_frame, bar_x_px, bar_y_apex = min(hip_xy, key=lambda r: r[2])

    # Metric ruler: the tallest upright projection on the *approach* to apex is the
    # athlete near the apparatus depth, standing tall.  Fall back to the whole
    # clip if nothing precedes the apex.
    pre = [s for s in stature_px if s[0] <= apex_frame]
    pool = pre if pre else stature_px
    takeoff_frame, tallest_px = max(pool, key=lambda r: r[1])
    if tallest_px <= 1.0:
        return None
    scale_px_per_m = tallest_px / float(athlete_height_m)

    # Ground line: the athlete's foot at the takeoff frame (nearest valid foot).
    if takeoff_frame in foot_y_by_frame:
        ground_y_px = foot_y_by_frame[takeoff_frame]
    elif foot_y_by_frame:
        nearest = min(foot_y_by_frame, key=lambda f: abs(f - takeoff_frame))
        ground_y_px = foot_y_by_frame[nearest]
    else:
        # No feet ever visible: infer ground from the apex CoM and bar height.
        ground_y_px = bar_y_apex + bar_height_m * scale_px_per_m

    # Bar line: metric estimate from the ground, cross-checked against the apex.
    bar_y_metric = ground_y_px - bar_height_m * scale_px_per_m
    bar_y_est_px = float(min(bar_y_apex, bar_y_metric))  # the higher (smaller y) of the two

    half_x = (upright_separation_m / 2.0) * scale_px_per_m * (1.0 + x_margin_frac)
    y_pad = y_margin_frac * bar_height_m * scale_px_per_m
    x_min = max(0.0, bar_x_px - half_x)
    x_max = min(float(image_w), bar_x_px + half_x)
    y_min = max(0.0, min(bar_y_est_px, bar_y_apex) - y_pad)
    y_max = min(float(image_h), ground_y_px + y_pad)

    return ApparatusROI(
        x_min_px=float(x_min),
        x_max_px=float(x_max),
        y_min_px=float(y_min),
        y_max_px=float(y_max),
        bar_x_px=float(bar_x_px),
        bar_y_est_px=float(bar_y_est_px),
        ground_y_px=float(ground_y_px),
        scale_px_per_m=float(scale_px_per_m),
        apex_frame=int(apex_frame),
        takeoff_frame=int(takeoff_frame),
        n_valid_frames=len(hip_xy),
    )
