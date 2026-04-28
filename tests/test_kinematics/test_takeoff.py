"""Tests for takeoff biomechanics analysis.

Covers: TakeoffMetrics new fields, body alignment deviation,
foot-to-ground angle, arm drive metrics, and free-knee drive metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_jump_video import generate_report
from src.data_pipeline.sample import BiomechanicalSample, MovementType, SubjectInfo
from src.kinematics.takeoff_analysis import (
    TakeoffMetrics,
    estimate_grf_from_com,
    compute_takeoff_angle,
    predict_max_com_height,
    compute_impulse,
    compute_body_alignment_deviation,
    compute_foot_to_ground_angle,
    compute_arm_drive_metrics,
    compute_free_knee_drive_metrics,
)


# ── TakeoffMetrics new fields ─────────────────────────────────────────────

def test_takeoff_metrics_new_fields_have_defaults():
    """New Athlete A-priority fields should all default to 0.0."""
    m = TakeoffMetrics(
        ground_contact_time_ms=120.0,
        time_to_peak_force_ms=80.0,
        horizontal_velocity_mps=6.5,
        vertical_velocity_mps=3.2,
        resultant_velocity_mps=7.2,
        takeoff_angle_deg=26.0,
        takeoff_distance_from_bar_cm=60.0,
        com_height_at_takeoff_cm=105.0,
        peak_vertical_grf_bw=3.5,
        average_vertical_grf_bw=2.1,
        braking_impulse_ns=80.0,
        propulsive_impulse_ns=320.0,
        knee_angle_at_takeoff_deg=170.0,
        hip_angle_at_takeoff_deg=175.0,
        ankle_angle_at_takeoff_deg=110.0,
        trunk_lean_deg=5.0,
    )
    assert m.body_alignment_deviation_deg == 0.0
    assert m.foot_to_ground_angle_deg == 0.0
    assert m.arm_drive_peak_speed_mps == 0.0
    assert m.arm_drive_peak_timing_ms == 0.0
    assert m.free_knee_drive_peak_speed_mps == 0.0
    assert m.free_knee_drive_peak_timing_ms == 0.0


# ── compute_body_alignment_deviation ─────────────────────────────────────

def _make_straight_body_landmarks() -> np.ndarray:
    """All joints on the Y-axis (perfectly straight vertical body)."""
    landmarks = np.zeros((33, 3))
    # Right side (takeoff = right): ankle=28, knee=26, hip=24, shoulder=12, nose=0
    landmarks[28] = [0.0, 0.0, 0.0]    # ankle (ground)
    landmarks[26] = [0.0, 0.45, 0.0]   # knee
    landmarks[24] = [0.0, 0.90, 0.0]   # hip
    landmarks[12] = [0.0, 1.40, 0.0]   # shoulder
    landmarks[0]  = [0.0, 1.75, 0.0]   # nose / head
    return landmarks


def test_body_alignment_straight_line_gives_zero():
    """Perfectly straight body → ~0° alignment deviation."""
    landmarks = _make_straight_body_landmarks()
    dev = compute_body_alignment_deviation(landmarks, takeoff_side="right")
    assert dev < 1.0, f"Expected ~0°, got {dev:.3f}°"


def test_body_alignment_bent_knee_gives_positive():
    """Knee displaced forward by 10 cm → non-zero alignment deviation."""
    landmarks = _make_straight_body_landmarks()
    landmarks[26, 0] = 0.10   # push right knee 10 cm forward (X)
    dev = compute_body_alignment_deviation(landmarks, takeoff_side="right")
    assert dev > 0.5, f"Expected positive deviation, got {dev:.3f}°"


def test_body_alignment_left_side():
    """Left-side takeoff should work symmetrically."""
    landmarks = np.zeros((33, 3))
    landmarks[27] = [0.0, 0.0, 0.0]    # left ankle
    landmarks[25] = [0.0, 0.45, 0.0]   # left knee
    landmarks[23] = [0.0, 0.90, 0.0]   # left hip
    landmarks[11] = [0.0, 1.40, 0.0]   # left shoulder
    landmarks[0]  = [0.0, 1.75, 0.0]   # head
    dev = compute_body_alignment_deviation(landmarks, takeoff_side="left")
    assert dev < 1.0, f"Expected ~0°, got {dev:.3f}°"


# ── compute_foot_to_ground_angle ──────────────────────────────────────────

def test_foot_to_ground_angle_horizontal_foot():
    """Foot perfectly horizontal (ankle and toe at same height) → 0°."""
    ankle = np.array([0.0, 0.0, 0.0])
    toe   = np.array([0.2, 0.0, 0.0])
    angle = compute_foot_to_ground_angle(ankle, toe)
    assert abs(angle) < 1e-5


def test_foot_to_ground_angle_positive_plantarflexion():
    """Toe raised above heel → positive angle."""
    ankle = np.array([0.0, 0.0, 0.0])
    toe   = np.array([0.2, 0.1, 0.0])   # toe 10 cm higher than heel
    angle = compute_foot_to_ground_angle(ankle, toe)
    expected = np.degrees(np.arctan2(0.1, 0.2))
    assert abs(angle - expected) < 0.5


def test_foot_to_ground_angle_negative_dorsiflexion():
    """Heel raised above toe → negative angle (dorsiflexed/heel strike)."""
    ankle = np.array([0.0, 0.05, 0.0])  # ankle (heel) 5 cm high
    toe   = np.array([0.2, 0.0, 0.0])   # toe lower
    angle = compute_foot_to_ground_angle(ankle, toe)
    assert angle < 0.0


# ── compute_arm_drive_metrics ─────────────────────────────────────────────

def test_arm_drive_peak_speed_constant_returns_nonzero():
    """Wrist accelerating from rest → peak speed at end."""
    fps = 100.0
    n = 20
    t = np.arange(n) / fps
    # Wrist moves in Y with constant acceleration
    wrist = np.column_stack([np.zeros(n), 0.5 * 10.0 * t ** 2, np.zeros(n)])
    peak_speed, timing_ms = compute_arm_drive_metrics(wrist, fps)
    assert peak_speed > 0.0
    # Peak should be at the end (maximum speed at last frame)
    assert timing_ms >= (n - 2) / fps * 1000.0


def test_arm_drive_single_frame_returns_zero():
    """Single-frame input cannot produce a velocity → return (0, 0)."""
    wrist = np.zeros((1, 3))
    peak_speed, timing_ms = compute_arm_drive_metrics(wrist, fps=100.0)
    assert peak_speed == 0.0
    assert timing_ms == 0.0


def test_arm_drive_timing_in_ms():
    """Peak speed at frame 10 of 100 Hz sequence → timing = 100 ms."""
    fps = 100.0
    n = 20
    # Constant forward wrist motion, then sudden speed at frame 10
    wrist = np.zeros((n, 3))
    wrist[10:, 0] = np.linspace(0, 1, n - 10)  # only X component moves after frame 10
    peak_speed, timing_ms = compute_arm_drive_metrics(wrist, fps)
    # The peak frame should be around 10
    assert 50.0 <= timing_ms <= 200.0, f"Unexpected timing {timing_ms} ms"


# ── compute_free_knee_drive_metrics ───────────────────────────────────────

def test_free_knee_drive_rising_knee():
    """Knee accelerating upward → positive peak speed."""
    fps = 100.0
    n = 20
    t = np.arange(n) / fps
    knee = np.column_stack([np.zeros(n), 5.0 * t, np.zeros(n)])  # constant 5 m/s up
    peak_speed, timing_ms = compute_free_knee_drive_metrics(knee, fps)
    assert peak_speed > 0.0


def test_free_knee_drive_only_upward_component():
    """Only upward (positive Y) velocity is counted; downward is excluded."""
    fps = 100.0
    n = 10
    # Knee moves downward → upward speed should be 0
    knee = np.column_stack([np.zeros(n), -np.linspace(0, 1, n), np.zeros(n)])
    peak_speed, _ = compute_free_knee_drive_metrics(knee, fps)
    assert peak_speed == 0.0


def test_free_knee_drive_single_frame_returns_zero():
    knee = np.zeros((1, 3))
    peak_speed, timing = compute_free_knee_drive_metrics(knee, fps=100.0)
    assert peak_speed == 0.0
    assert timing == 0.0


# ── Legacy functions still work ───────────────────────────────────────────

def test_estimate_grf_static():
    """At rest (zero acceleration), GRF should equal body weight."""
    mass = 70.0
    acc = np.zeros((1, 3))
    grf = estimate_grf_from_com(acc, mass)
    # GRF_y should be m*g = 70 * 9.81
    np.testing.assert_allclose(grf[0, 1], mass * 9.81, rtol=1e-5)


def test_compute_takeoff_angle_45_degrees():
    """Equal horizontal and vertical velocity → 45° takeoff angle."""
    vel = np.array([3.0, 3.0, 0.0])
    angle = compute_takeoff_angle(vel)
    assert abs(angle - 45.0) < 0.01


def test_predict_max_com_height_physics():
    """h_max = h0 + vy²/(2g).  Verify against manual calculation."""
    h0 = 1.05
    vy = 4.0
    expected = h0 + vy ** 2 / (2 * 9.81)
    assert abs(predict_max_com_height(h0, vy) - expected) < 1e-6


def test_compute_impulse_constant_force():
    """Constant vertical force F over T seconds → impulse ≈ F*T.

    The trapezoidal rule integrates over N-1 intervals from N points, so the
    result is F * (N-1)/fps rather than F * N/fps.  The boundary error is
    exactly one time-step: F * dt = 700 * 0.01 = 7 N·s.
    """
    fps = 100.0
    n = 100
    f = np.zeros((n, 3))
    f[:, 1] = 700.0  # 700 N vertical
    impulse = compute_impulse(f, fps, axis=1)
    # Trapezoidal integration: F * (N-1) * dt
    expected = 700.0 * (n - 1) / fps
    assert abs(impulse - expected) < 0.01   # floating-point tolerance only


# ── Video report takeoff-frame selection ─────────────────────────────────

def _make_video_sample_for_takeoff_frame_test(
    *,
    contact_slice: slice | None,
    velocity_spike_frame: int,
    true_takeoff_frame: int,
) -> BiomechanicalSample:
    """Synthetic calibrated video sample with ankle Y in metres."""
    n = 30
    fps = 100.0
    pose_3d = np.zeros((n, 33, 3), dtype=float)
    pose_3d[:, 27, 1] = 0.25
    pose_3d[:, 28, 1] = 0.25
    if contact_slice is not None:
        pose_3d[contact_slice, 28, 1] = 0.02

    com_position = np.zeros((n, 3), dtype=float)
    com_position[:, 1] = np.linspace(1.0, 1.8, n)
    com_velocity = np.zeros((n, 3), dtype=float)
    com_velocity[:, 0] = 3.0
    com_velocity[:, 1] = 0.5
    com_velocity[true_takeoff_frame, 1] = 3.4
    com_velocity[velocity_spike_frame, 1] = 20.0
    grf = np.zeros((n, 3), dtype=float)
    grf[:, 1] = 65.0 * 9.81

    return BiomechanicalSample(
        dataset_name="unit_test",
        trial_id="synthetic_takeoff",
        subject=SubjectInfo(subject_id="athlete", body_mass_kg=65.0, height_m=1.75),
        movement_type=MovementType.HIGH_JUMP,
        fps=fps,
        com_position=com_position,
        com_velocity=com_velocity,
        grf=grf,
        pose_3d=pose_3d,
    )


def test_generate_report_uses_last_ground_contact_for_takeoff_frame():
    """Contact-anchored takeoff should ignore later single-frame vy spikes."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=slice(8, 12),
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )

    report = generate_report(sample, pinn_grf=None)

    assert report["takeoff_frame"] == 11
    assert report["velocity"]["takeoff_vertical_mps"] == 3.4


def test_generate_report_falls_back_to_argmax_vy_without_ground_contacts():
    """Very short or failed pose clips still produce a takeoff estimate."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=None,
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )

    report = generate_report(sample, pinn_grf=None)

    assert report["takeoff_frame"] == 20
    assert report["velocity"]["takeoff_vertical_mps"] == 20.0
