"""Tests for scale_calibration module."""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation.scale_calibration import (
    calibrate_landmarks_to_world,
    compute_scale_factor,
    estimate_standing_pixel_height,
)


def _make_upright_landmarks_2d(
    n_frames: int = 10,
    nose_y: float = 0.1,
    ankle_y: float = 0.7,
) -> np.ndarray:
    """Create fake normalised 2D landmarks with a standing person.

    y increases downward: nose at 0.1, ankles at 0.7 → pixel height 0.6.
    """
    lm = np.zeros((n_frames, 33, 3), dtype=np.float32)
    # visibility = 1.0 for all
    lm[:, :, 2] = 1.0
    # Nose (idx 0)
    lm[:, 0, 0] = 0.5
    lm[:, 0, 1] = nose_y
    # Shoulders
    lm[:, 11, 1] = 0.25
    lm[:, 12, 1] = 0.25
    # Hips
    lm[:, 23, 1] = 0.45
    lm[:, 24, 1] = 0.45
    # Left ankle (27)
    lm[:, 27, 1] = ankle_y
    # Right ankle (28)
    lm[:, 28, 1] = ankle_y
    return lm


def test_standing_pixel_height():
    lm = _make_upright_landmarks_2d(n_frames=10, nose_y=0.1, ankle_y=0.7)
    height, best_frame = estimate_standing_pixel_height(lm)
    assert height == pytest.approx(0.6, abs=0.01)


def test_compute_scale_factor():
    lm = _make_upright_landmarks_2d(n_frames=10, nose_y=0.1, ankle_y=0.7)
    scale = compute_scale_factor(lm, height_m=1.78)
    # nose-to-ankle span = 0.6, effective height = 1.78 * 0.95 = 1.691
    expected = 1.691 / 0.6
    assert scale == pytest.approx(expected, rel=0.01)


def test_calibrate_produces_positive_heights():
    """After calibration, ankle Y should be near 0 and nose Y should be ~1.7 m."""
    lm_2d = _make_upright_landmarks_2d(n_frames=20, nose_y=0.1, ankle_y=0.7)
    # Fake world landmarks with matching shape
    lm_3d = np.zeros((20, 33, 4), dtype=np.float32)
    lm_3d[:, :, :2] = lm_2d[:, :, :2]
    lm_3d[:, :, 3] = 1.0  # visibility

    calibrated = calibrate_landmarks_to_world(lm_2d, lm_3d, height_m=1.78)

    # Ankle Y should be near 0 (ground reference)
    ankle_y = (calibrated[:, 27, 1] + calibrated[:, 28, 1]) / 2
    assert ankle_y.min() >= -0.05, f"Ankle below ground: {ankle_y.min()}"

    # Nose Y should be roughly 1.5–1.8 m
    nose_y = calibrated[:, 0, 1]
    assert 1.2 < nose_y.mean() < 2.0, f"Nose height unrealistic: {nose_y.mean()}"


def test_calibrate_shape():
    lm_2d = _make_upright_landmarks_2d(n_frames=5)
    lm_3d = np.zeros((5, 33, 4), dtype=np.float32)
    calibrated = calibrate_landmarks_to_world(lm_2d, lm_3d, height_m=1.78)
    assert calibrated.shape == (5, 33, 4)
    assert calibrated.dtype == np.float32
