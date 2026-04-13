"""Tests for flight phase analysis.

Covers: arch transition detection (sub-phase timing), FlightMetrics
dataclass new fields, and the parabola fitter.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kinematics.flight_analysis import (
    FlightMetrics,
    detect_arch_transition,
    fit_com_parabola,
    compute_clearance_profile,
)


# ── detect_arch_transition ────────────────────────────────────────────────

def _make_knee_trajectory(fps: float, n: int, reversal_frame: int) -> np.ndarray:
    """Synthetic free-knee trajectory that rises until reversal_frame then falls."""
    t = np.arange(n) / fps
    # Parabolic: knee_y peaks at reversal_frame
    t_peak = reversal_frame / fps
    amplitude = 0.4  # metres
    knee_y = amplitude - amplitude * ((t - t_peak) / t_peak) ** 2
    knee_y = np.clip(knee_y, 0.0, None)
    return np.column_stack([np.zeros(n), knee_y, np.zeros(n)])


def test_detect_arch_transition_near_reversal():
    """The detected transition should occur within ±3 frames of the true reversal."""
    fps = 100.0
    n = 40
    true_reversal = 18  # frame index within flight phase

    knee_traj = _make_knee_trajectory(fps, n, true_reversal)
    detected = detect_arch_transition(knee_traj, fps, min_extension_frames=5)

    assert detected is not None, "Transition should be detected in clear parabolic motion"
    assert abs(detected - true_reversal) <= 3, (
        f"Expected transition near frame {true_reversal}, got {detected}"
    )


def test_detect_arch_transition_monotone_rising_returns_none():
    """Knee that only rises (never drops) → no transition detected."""
    fps = 100.0
    n = 30
    knee_y = np.linspace(0.0, 0.5, n)
    knee_traj = np.column_stack([np.zeros(n), knee_y, np.zeros(n)])
    detected = detect_arch_transition(knee_traj, fps)
    assert detected is None


def test_detect_arch_transition_too_short_returns_none():
    """Trajectory shorter than min_extension_frames → no transition."""
    fps = 100.0
    knee_traj = np.zeros((4, 3))
    detected = detect_arch_transition(knee_traj, fps, min_extension_frames=5)
    assert detected is None


def test_detect_arch_transition_at_start_of_drop():
    """Simple: knee rises for first 10 frames, then drops.  Transition at frame ~10."""
    fps = 100.0
    n = 25
    knee_y = np.concatenate([
        np.linspace(0.0, 0.4, 11),   # rising
        np.linspace(0.4, 0.1, 14),   # falling
    ])
    knee_traj = np.column_stack([np.zeros(n), knee_y, np.zeros(n)])
    detected = detect_arch_transition(knee_traj, fps, min_extension_frames=5)
    assert detected is not None
    # Transition should be at or near frame 10
    assert abs(detected - 10) <= 2


# ── FlightMetrics new fields ──────────────────────────────────────────────

def test_flight_metrics_new_fields_have_defaults():
    """arch_transition_frame and timing fields should default to None."""
    m = FlightMetrics(
        peak_com_height_m=2.10,
        com_height_above_bar_cm=5.0,
        time_to_peak_ms=420.0,
        total_flight_time_ms=900.0,
        bar_height_m=2.05,
        min_body_clearance_cm=3.0,
        hip_angle_at_bar_deg=200.0,
        knee_angle_at_bar_deg=170.0,
        head_clearance_cm=10.0,
        trail_leg_clearance_cm=4.0,
        estimated_angular_momentum_h=None,
        estimated_angular_momentum_l=None,
    )
    assert m.arch_transition_frame is None
    assert m.time_to_arch_transition_ms is None
    assert m.vertical_extension_time_ms is None


def test_flight_metrics_with_transition_fields():
    """Verify transition fields can be set explicitly."""
    m = FlightMetrics(
        peak_com_height_m=2.10,
        com_height_above_bar_cm=5.0,
        time_to_peak_ms=420.0,
        total_flight_time_ms=900.0,
        bar_height_m=2.05,
        min_body_clearance_cm=3.0,
        hip_angle_at_bar_deg=200.0,
        knee_angle_at_bar_deg=170.0,
        head_clearance_cm=10.0,
        trail_leg_clearance_cm=4.0,
        estimated_angular_momentum_h=None,
        estimated_angular_momentum_l=None,
        arch_transition_frame=18,
        time_to_arch_transition_ms=180.0,
        vertical_extension_time_ms=180.0,
    )
    assert m.arch_transition_frame == 18
    assert m.time_to_arch_transition_ms == 180.0
    assert m.vertical_extension_time_ms == 180.0


# ── Integrated: transition timing from trajectory ─────────────────────────

def test_transition_timing_ms_from_synthetic_flight():
    """Confirm that arch transition time is computed correctly from fps."""
    fps = 100.0
    n = 40
    true_reversal = 15  # frame 15 within flight phase

    knee_traj = _make_knee_trajectory(fps, n, true_reversal)
    detected_frame = detect_arch_transition(knee_traj, fps, min_extension_frames=5)

    assert detected_frame is not None
    timing_ms = detected_frame / fps * 1000.0
    expected_ms = true_reversal / fps * 1000.0
    # Should be within ±30 ms of expected (3 frames at 100 Hz)
    assert abs(timing_ms - expected_ms) <= 30.0, (
        f"Transition timing {timing_ms:.1f} ms too far from expected {expected_ms:.1f} ms"
    )


# ── fit_com_parabola ──────────────────────────────────────────────────────

def test_fit_com_parabola_recovers_gravity():
    """Parabolic fit on ideal projectile motion should recover g ≈ 9.81 m/s²."""
    fps = 200.0
    n = 80
    g = 9.81
    t = np.arange(n) / fps
    vy0 = 3.5
    y0 = 1.0
    y = y0 + vy0 * t - 0.5 * g * t ** 2

    com = np.column_stack([np.zeros(n), y, np.zeros(n)])
    result = fit_com_parabola(com, fps)

    assert abs(result["g_estimated_mps2"] - g) < 0.5
    assert abs(result["vy0_mps"] - vy0) < 0.1
    assert result["r_squared"] > 0.999


# ── compute_clearance_profile ─────────────────────────────────────────────

def test_clearance_profile_above_bar():
    """All body parts 0.5 m above bar → positive clearances of ~50 cm."""
    bar_h = 2.0
    T = 5
    landmarks = np.zeros((T, 33, 3))
    landmarks[:, :, 1] = bar_h + 0.5  # all landmarks 0.5 m above bar

    profile = compute_clearance_profile(landmarks, bar_h, bar_position_x=0.0, fps=100.0)
    for name, clearance_arr in profile.items():
        np.testing.assert_allclose(clearance_arr, 0.5, atol=1e-6,
                                   err_msg=f"Landmark {name} clearance wrong")


def test_clearance_profile_below_bar():
    """Landmarks below bar → negative clearance."""
    bar_h = 2.0
    T = 3
    landmarks = np.zeros((T, 33, 3))
    landmarks[:, :, 1] = bar_h - 0.1

    profile = compute_clearance_profile(landmarks, bar_h, bar_position_x=0.0, fps=100.0)
    for name, clearance_arr in profile.items():
        assert np.all(clearance_arr < 0.0), f"{name} should be below bar"
