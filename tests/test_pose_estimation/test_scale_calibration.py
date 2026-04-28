"""Tests for scale_calibration module."""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation.scale_calibration import (
    calibrate_landmarks_to_world,
    compute_per_frame_scale_mpp,
    compute_scale_factor,
    estimate_standing_pixel_height,
)


# BlazePose indices used by the segment references
HIP_L, HIP_R = 23, 24
KNEE_L, KNEE_R = 25, 26
ANKLE_L, ANKLE_R = 27, 28


def _place_segment(
    lm: np.ndarray,
    proximal_idx: int,
    distal_idx: int,
    proximal_xy_norm: tuple[float, float],
    pixel_length: float,
    image_width: int,
    image_height: int,
    visibility: float = 1.0,
) -> None:
    """Place a vertical segment of a known pixel length onto a landmarks_2d array.

    Distal lands directly below proximal in the image (downward Y), so the
    segment's pixel length is exactly `pixel_length` regardless of aspect ratio.
    """
    px, py = proximal_xy_norm
    lm[:, proximal_idx, 0] = px
    lm[:, proximal_idx, 1] = py
    lm[:, proximal_idx, 2] = visibility
    lm[:, distal_idx, 0] = px
    lm[:, distal_idx, 1] = py + (pixel_length / image_height)
    lm[:, distal_idx, 2] = visibility


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
    scale = compute_scale_factor(lm, height_m=1.75)
    # nose-to-ankle span = 0.6, effective height = 1.75 * 0.95 = 1.691
    expected = 1.691 / 0.6
    assert scale == pytest.approx(expected, rel=0.01)


def test_calibrate_produces_positive_heights():
    """After calibration, ankle Y should be near 0 and nose Y should be ~1.7 m."""
    lm_2d = _make_upright_landmarks_2d(n_frames=20, nose_y=0.1, ankle_y=0.7)
    # Fake world landmarks with matching shape
    lm_3d = np.zeros((20, 33, 4), dtype=np.float32)
    lm_3d[:, :, :2] = lm_2d[:, :, :2]
    lm_3d[:, :, 3] = 1.0  # visibility

    calibrated = calibrate_landmarks_to_world(lm_2d, lm_3d, height_m=1.75)

    # Ankle Y should be near 0 (ground reference)
    ankle_y = (calibrated[:, 27, 1] + calibrated[:, 28, 1]) / 2
    assert ankle_y.min() >= -0.05, f"Ankle below ground: {ankle_y.min()}"

    # Nose Y should be roughly 1.5–1.8 m
    nose_y = calibrated[:, 0, 1]
    assert 1.2 < nose_y.mean() < 2.0, f"Nose height unrealistic: {nose_y.mean()}"


def test_calibrate_shape():
    lm_2d = _make_upright_landmarks_2d(n_frames=5)
    lm_3d = np.zeros((5, 33, 4), dtype=np.float32)
    calibrated = calibrate_landmarks_to_world(lm_2d, lm_3d, height_m=1.75)
    assert calibrated.shape == (5, 33, 4)
    assert calibrated.dtype == np.float32


# ── Phase 9a: multi-segment per-frame scale calibration ────────────────────


def test_video_scale_recovers_known_mpp_thigh_only():
    """A pure synthetic case: thigh of known metric length and known pixel
    length must produce metres-per-pixel = length_m / pixel_length exactly.
    Output is constant across frames (single video-wide scale)."""
    T, W, H = 12, 1920, 1080
    lm = np.zeros((T, 33, 3), dtype=np.float32)
    _place_segment(lm, HIP_L, KNEE_L, (0.45, 0.40), 200.0, W, H)
    _place_segment(lm, HIP_R, KNEE_R, (0.55, 0.40), 200.0, W, H)

    mpp, info = compute_per_frame_scale_mpp(
        lm, image_width=W, image_height=H, thigh_length_m=0.45,
    )
    assert mpp is not None
    assert mpp.shape == (T,)
    expected = 0.45 / 200.0
    np.testing.assert_allclose(mpp, expected, rtol=1e-6)
    assert (mpp == mpp[0]).all()  # constant across frames
    assert info["scalar_scale_mpp"] == pytest.approx(expected, rel=1e-6)
    assert info["segments_used"]["thigh_left"] == T
    assert info["segments_used"]["thigh_right"] == T


def test_video_scale_median_across_thigh_and_shank():
    """When thigh and shank are both visible with consistent pixel lengths,
    the video scale is the median of their per-segment estimates."""
    T, W, H = 6, 1280, 720
    lm = np.zeros((T, 33, 3), dtype=np.float32)
    _place_segment(lm, HIP_L, KNEE_L, (0.40, 0.30), 200.0, W, H)
    _place_segment(lm, KNEE_L, ANKLE_L, (0.40, 0.30 + 200.0 / H), 220.0, W, H)
    _place_segment(lm, HIP_R, KNEE_R, (0.60, 0.30), 200.0, W, H)
    _place_segment(lm, KNEE_R, ANKLE_R, (0.60, 0.30 + 200.0 / H), 220.0, W, H)

    mpp, info = compute_per_frame_scale_mpp(
        lm, image_width=W, image_height=H,
        thigh_length_m=0.45, shank_length_m=0.45,
    )
    assert mpp is not None
    expected_thigh = 0.45 / 200.0
    expected_shank = 0.45 / 220.0
    lo, hi = sorted([expected_thigh, expected_shank])
    assert lo - 1e-9 <= mpp[0] <= hi + 1e-9


def test_video_scale_rejects_foreshortened_frames():
    """The 95th-percentile-projection approach must select the in-plane
    (longest) projection per segment, ignoring frames where the segment is
    foreshortened.  Without this, the across-frame scale would be biased
    by stride-cycle foreshortening during run-up."""
    T, W, H = 50, 1920, 1080
    lm = np.zeros((T, 33, 3), dtype=np.float32)
    lm[:, :, 2] = 1.0
    # 80% of frames foreshortened (50 px); 20% in-plane (200 px) — the true projection.
    pixel_lengths = np.where(np.arange(T) % 5 == 0, 200.0, 50.0)
    for t in range(T):
        # Hip and knee placed with that frame's projected pixel length.
        lm[t, HIP_L, :2] = (0.40, 0.30)
        lm[t, KNEE_L, :2] = (0.40, 0.30 + pixel_lengths[t] / H)
        lm[t, HIP_R, :2] = (0.60, 0.30)
        lm[t, KNEE_R, :2] = (0.60, 0.30 + pixel_lengths[t] / H)

    mpp, info = compute_per_frame_scale_mpp(
        lm, image_width=W, image_height=H,
        thigh_length_m=0.45, projection_percentile=95.0,
    )
    assert mpp is not None
    # True in-plane scale: 0.45 / 200 ≈ 2.15e-3.  Foreshortened-biased scale
    # would be 0.45 / 50 = 8.6e-3 — 4× too large.
    truth = 0.45 / 200.0
    assert abs(mpp[0] - truth) / truth < 0.05, (
        f"Foreshortened projections should be rejected; got mpp={mpp[0]:.4e}, truth={truth:.4e}"
    )


def test_video_scale_skips_low_visibility_landmarks():
    """Visibility-gated segments contribute nothing; the count reflects this."""
    T, W, H = 20, 1920, 1080
    lm = np.zeros((T, 33, 3), dtype=np.float32)
    _place_segment(lm, HIP_L, KNEE_L, (0.45, 0.40), 200.0, W, H)
    _place_segment(lm, HIP_R, KNEE_R, (0.55, 0.40), 200.0, W, H)
    # Drop visibility on frames 5–10 below the default 0.7 threshold
    lm[5:11, [HIP_L, HIP_R, KNEE_L, KNEE_R], 2] = 0.3

    mpp, info = compute_per_frame_scale_mpp(
        lm, image_width=W, image_height=H, thigh_length_m=0.45,
    )
    assert mpp is not None
    # 14 frames remain visible per segment, 2 segments → 28 contributions total
    assert info["n_valid_frames"] == 28
    # Single video-wide scale is correct
    np.testing.assert_allclose(mpp, 0.45 / 200.0, rtol=1e-6)


def test_video_scale_no_segment_lengths_returns_none():
    """When no anthropometric reference is provided, return None plus a warning."""
    lm = np.zeros((5, 33, 3), dtype=np.float32)
    lm[:, :, 2] = 1.0
    mpp, info = compute_per_frame_scale_mpp(lm, image_width=1920, image_height=1080)
    assert mpp is None
    assert "warning" in info


def test_video_scale_no_visible_segments_returns_none():
    """If no frame has any high-confidence segment, return None gracefully."""
    lm = np.zeros((5, 33, 3), dtype=np.float32)
    # All landmarks at visibility 0 → nothing can be measured
    mpp, info = compute_per_frame_scale_mpp(
        lm, image_width=1920, image_height=1080, thigh_length_m=0.45,
    )
    assert mpp is None
    assert "warning" in info


def test_calibrate_uses_multi_segment_when_lengths_provided():
    """When segment lengths are passed to calibrate_landmarks_to_world, the
    nose Y should match the athlete's actual standing height to high accuracy
    — better than the legacy nose-ankle fallback which assumed nose ≈ 95%
    of full height."""
    T, W, H = 10, 1920, 1080
    lm_2d = np.zeros((T, 33, 3), dtype=np.float32)
    lm_2d[:, :, 2] = 1.0
    # Build a synthetic upright body where every key landmark sits on a
    # column of the image. Ankles at the bottom, nose 940 px above ankles.
    # Athlete A-like proportions: thigh = 200 px, shank = 220 px.
    ankle_y_norm = 0.95
    knee_y_norm = ankle_y_norm - 220.0 / H
    hip_y_norm = knee_y_norm - 200.0 / H
    nose_y_norm = ankle_y_norm - 940.0 / H

    lm_2d[:, 0, :2] = (0.50, nose_y_norm)            # nose
    lm_2d[:, HIP_L, :2] = (0.48, hip_y_norm)
    lm_2d[:, HIP_R, :2] = (0.52, hip_y_norm)
    lm_2d[:, KNEE_L, :2] = (0.48, knee_y_norm)
    lm_2d[:, KNEE_R, :2] = (0.52, knee_y_norm)
    lm_2d[:, ANKLE_L, :2] = (0.48, ankle_y_norm)
    lm_2d[:, ANKLE_R, :2] = (0.52, ankle_y_norm)

    lm_3d = np.zeros((T, 33, 4), dtype=np.float32)
    lm_3d[:, :, 3] = 1.0

    calibrated = calibrate_landmarks_to_world(
        lm_2d, lm_3d, height_m=1.75,
        image_width=W, image_height=H,
        thigh_length_m=0.45, shank_length_m=0.45,
    )
    nose_y = float(calibrated[:, 0, 1].mean())
    # 940 px × (0.45/200 m/px) = 2.021 m ankle-to-nose.
    # Tolerate small slack from per-frame median across 4 segments.
    assert 1.95 < nose_y < 2.10, f"Nose Y {nose_y:.3f} outside expected range"


def test_calibrate_falls_back_when_no_lengths_provided():
    """No segment lengths → use legacy nose-ankle path; existing tests above
    still cover the legacy semantics. Here we just confirm no exception
    and a sensible nose height range."""
    lm_2d = _make_upright_landmarks_2d(n_frames=10, nose_y=0.1, ankle_y=0.7)
    lm_3d = np.zeros((10, 33, 4), dtype=np.float32)
    lm_3d[:, :, 3] = 1.0
    calibrated = calibrate_landmarks_to_world(lm_2d, lm_3d, height_m=1.75)
    nose_y = float(calibrated[:, 0, 1].mean())
    assert 1.2 < nose_y < 2.0
