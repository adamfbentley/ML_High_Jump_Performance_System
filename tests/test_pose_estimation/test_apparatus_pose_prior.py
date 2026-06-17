"""Tests for the pose-localized apparatus ROI (pure, no MediaPipe)."""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation.apparatus_pose_prior import compute_apparatus_roi

IMAGE_W, IMAGE_H = 1920, 1080


def _synthetic_jump_track() -> np.ndarray:
    """A synthetic BlazePose track of a jump over a bar at a known scale.

    Ground at y=900 px, scale 200 px/m (stature 1.75 m -> 356 px), bar at 1.75 m
    -> y=550 px, bar centred at x=960.  Approach (upright, getting closer), apex
    (CoM over the bar), then descent.
    """
    n = 30
    lm = np.zeros((n, 33, 3), dtype=np.float32)
    apex = 15
    ground_y = 900.0

    def _set(t, x, nose_y, hip_y, knee_y, foot_y):
        for idx, (px, py) in {
            0: (x, nose_y),
            11: (x - 15, nose_y + 30), 12: (x + 15, nose_y + 30),
            23: (x - 10, hip_y), 24: (x + 10, hip_y),
            25: (x - 8, knee_y), 26: (x + 8, knee_y),
            27: (x - 10, foot_y), 28: (x + 10, foot_y),
            29: (x - 10, foot_y), 30: (x + 10, foot_y),
            31: (x - 12, foot_y), 32: (x + 12, foot_y),
        }.items():
            lm[t, idx] = (px / IMAGE_W, py / IMAGE_H, 1.0)

    for i in range(apex):  # approach 0..14, upright, getting closer/taller
        frac = i / (apex - 1)
        x = 600 + 360 * frac
        extent = 340 + 16 * frac  # 340 -> 356 px stature
        _set(i, x, ground_y - extent, ground_y - 0.53 * extent, ground_y - 0.27 * extent, ground_y)

    # Apex: CoM over the bar, body roughly horizontal (small vertical extent).
    _set(apex, 960, 520, 550, 565, 580)

    for j in range(apex + 1, n):  # descent behind the bar
        frac = (j - apex) / (n - apex - 1)
        x = 960 + 200 * frac
        hip_y = 560 + 240 * frac
        _set(j, x, hip_y - 40, hip_y, hip_y + 60, hip_y + 120)
    return lm


def test_compute_roi_recovers_apex_scale_and_lines():
    lm = _synthetic_jump_track()
    roi = compute_apparatus_roi(lm, image_w=IMAGE_W, image_h=IMAGE_H)
    assert roi is not None
    assert roi.apex_frame == 15
    assert roi.bar_x_px == pytest.approx(960, abs=15)
    assert roi.scale_px_per_m == pytest.approx(200, abs=8)
    assert roi.ground_y_px == pytest.approx(900, abs=10)
    assert roi.bar_y_est_px == pytest.approx(550, abs=20)


def test_compute_roi_brackets_the_apparatus_width():
    lm = _synthetic_jump_track()
    roi = compute_apparatus_roi(lm, image_w=IMAGE_W, image_h=IMAGE_H)
    assert roi is not None
    # The ROI must contain both standards (+/-2.01 m about the bar centre).
    half_span_px = (4.02 / 2.0) * roi.scale_px_per_m
    assert roi.x_min_px <= roi.bar_x_px - half_span_px
    assert roi.x_max_px >= roi.bar_x_px + half_span_px
    # ...but not the whole frame (it is a genuine restriction).
    assert (roi.x_max_px - roi.x_min_px) < IMAGE_W
    # Bar line sits above the ground line.
    assert roi.bar_y_est_px < roi.ground_y_px


def test_compute_roi_returns_none_without_hips():
    lm = np.zeros((10, 33, 3), dtype=np.float32)  # nothing visible
    assert compute_apparatus_roi(lm, image_w=IMAGE_W, image_h=IMAGE_H) is None
