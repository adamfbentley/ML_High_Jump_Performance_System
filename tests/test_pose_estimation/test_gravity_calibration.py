"""Tests for gravity-anchored metres-per-pixel calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation.gravity_calibration import (
    GravityMppConfig,
    calibrate_landmarks_with_gravity_mpp,
    fit_gravity_mpp_from_com_pixels,
)


def _projectile_com(
    *,
    n_frames: int = 24,
    fps: float = 30.0,
    mpp: float = 0.01,
    x0_px: float = 200.0,
    y0_px: float = 300.0,
    vx_px_s: float = 180.0,
    vy_px_s: float = 340.0,
    accel_x_px_s2: float = 0.0,
) -> np.ndarray:
    t = np.arange(n_frames, dtype=float) / fps
    g_px = 9.81 / mpp
    x = x0_px + vx_px_s * t + 0.5 * accel_x_px_s2 * t**2
    y = y0_px + vy_px_s * t - 0.5 * g_px * t**2
    return np.column_stack([x, y, np.zeros(n_frames, dtype=float)])


def _projectile_landmarks_2d(
    *,
    n_frames: int,
    fps: float,
    image_width: int,
    image_height: int,
    mpp: float,
) -> np.ndarray:
    com = _projectile_com(n_frames=n_frames, fps=fps, mpp=mpp)
    landmarks = np.zeros((n_frames, 33, 3), dtype=np.float32)
    landmarks[:, :, 2] = 1.0
    # Put every landmark at the same moving point so segment-model CoM equals it.
    landmarks[:, :, 0] = (com[:, 0] / image_width)[:, None]
    landmarks[:, :, 1] = ((image_height - com[:, 1]) / image_height)[:, None]
    # Add visible ankles for the ground-reference convention.
    landmarks[:, 27, 1] = landmarks[:, 0, 1] + 40.0 / image_height
    landmarks[:, 28, 1] = landmarks[:, 0, 1] + 40.0 / image_height
    return landmarks


def test_gravity_mpp_recovers_known_projectile_scale():
    fps = 30.0
    truth_mpp = 0.012
    com = _projectile_com(n_frames=30, fps=fps, mpp=truth_mpp)

    mpp, info = fit_gravity_mpp_from_com_pixels(
        com,
        fps,
        start_frame=0,
        end_frame=len(com),
    )

    assert info["accepted"] is True
    assert mpp == pytest.approx(truth_mpp, rel=1e-4)
    assert info["downward_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert info["y_r_squared"] > 0.999


def test_gravity_mpp_uses_acceleration_magnitude_for_rolled_camera():
    fps = 30.0
    truth_mpp = 0.01
    g_px = 9.81 / truth_mpp
    roll_like_x_accel = 0.25 * g_px
    y_accel = -np.sqrt(g_px**2 - roll_like_x_accel**2)
    t = np.arange(30, dtype=float) / fps
    com = np.column_stack(
        [
            100.0 + 20.0 * t + 0.5 * roll_like_x_accel * t**2,
            300.0 + 250.0 * t + 0.5 * y_accel * t**2,
            np.zeros_like(t),
        ]
    )

    mpp, info = fit_gravity_mpp_from_com_pixels(
        com,
        fps,
        start_frame=0,
        end_frame=len(com),
    )

    assert info["accepted"] is True
    assert mpp == pytest.approx(truth_mpp, rel=1e-4)
    assert info["horizontal_accel_fraction"] == pytest.approx(0.25, abs=1e-3)


def test_gravity_mpp_rejects_short_flight_window():
    com = _projectile_com(n_frames=8)

    mpp, info = fit_gravity_mpp_from_com_pixels(
        com,
        30.0,
        start_frame=0,
        end_frame=len(com),
    )

    assert mpp is None
    assert info["accepted"] is False
    assert info["decision_reason"] == "insufficient_flight_frames"


def test_gravity_mpp_flags_large_horizontal_acceleration():
    fps = 30.0
    truth_mpp = 0.01
    g_px = 9.81 / truth_mpp
    com = _projectile_com(
        n_frames=30,
        fps=fps,
        mpp=truth_mpp,
        accel_x_px_s2=0.8 * g_px,
    )

    mpp, info = fit_gravity_mpp_from_com_pixels(
        com,
        fps,
        start_frame=0,
        end_frame=len(com),
    )

    assert mpp is not None
    assert info["accepted"] is False
    assert "horizontal_acceleration_high" in info["decision_reason"]


def test_gravity_calibrated_landmarks_recover_takeoff_velocity_scale():
    fps = 30.0
    truth_mpp = 0.01
    image_width, image_height = 800, 600
    landmarks_2d = _projectile_landmarks_2d(
        n_frames=30,
        fps=fps,
        image_width=image_width,
        image_height=image_height,
        mpp=truth_mpp,
    )
    landmarks_3d = np.zeros((30, 33, 4), dtype=np.float32)
    landmarks_3d[:, :, 3] = 1.0

    calibrated, info = calibrate_landmarks_with_gravity_mpp(
        landmarks_2d,
        landmarks_3d,
        fps=fps,
        image_width=image_width,
        image_height=image_height,
        takeoff_frame=0,
    )
    vx = np.gradient(calibrated[:, 0, 0], 1.0 / fps)

    assert info["accepted"] is True
    assert info["mpp"] == pytest.approx(truth_mpp, rel=1e-4)
    assert np.nanmedian(vx[:10]) == pytest.approx(1.8, rel=0.02)


def test_gravity_low_r2_sets_rejected_quality_flag():
    rng = np.random.default_rng(123)
    com = _projectile_com(n_frames=30, mpp=0.01)
    com[:, 1] += rng.normal(0.0, 45.0, size=len(com))

    _mpp, info = fit_gravity_mpp_from_com_pixels(
        com,
        30.0,
        start_frame=0,
        end_frame=len(com),
        config=GravityMppConfig(min_y_r_squared=0.98),
    )

    assert info["accepted"] is False
    assert "parabola_fit_low_r2" in info["decision_reason"]
