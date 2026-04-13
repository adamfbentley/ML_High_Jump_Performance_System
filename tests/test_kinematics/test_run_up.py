"""Tests for run-up analysis metrics.

Covers: stride ground contact time, foot-strike-under-hip offset,
acceleration rhythm, foot contact classification, and per-stride
curve deviation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kinematics.run_up_analysis import (
    RunUpMetrics,
    detect_ground_contacts,
    compute_horizontal_velocity,
    fit_curve_radius,
    compute_stride_ground_contact_times,
    compute_foot_strike_under_hip,
    compute_acceleration_rhythm,
    classify_foot_contact,
    compute_per_stride_curve_deviation,
)


# ── detect_ground_contacts ────────────────────────────────────────────────

def test_detect_ground_contacts_single_contact():
    """Simple contact: ankle near ground for frames 5–14."""
    n = 30
    positions = np.zeros((n, 3))
    positions[:, 1] = 20.0  # ankle high (not in contact)
    positions[5:15, 1] = 2.0  # ankle near ground
    contacts = detect_ground_contacts(positions, fps=100.0, height_threshold_cm=5.0)
    assert len(contacts) == 1
    start, end = contacts[0]
    assert start == 5
    assert end == 14


def test_detect_ground_contacts_multiple():
    """Two separate contacts."""
    n = 60
    positions = np.zeros((n, 3))
    positions[:, 1] = 20.0
    positions[5:15, 1] = 2.0
    positions[35:45, 1] = 2.0
    contacts = detect_ground_contacts(positions, fps=100.0)
    assert len(contacts) == 2


def test_detect_ground_contacts_none():
    """Ankle always above threshold → no contacts."""
    positions = np.ones((30, 3)) * 20.0
    contacts = detect_ground_contacts(positions, fps=100.0)
    assert contacts == []


# ── compute_stride_ground_contact_times ───────────────────────────────────

def test_stride_ground_contact_times_units():
    """10-frame contact at 100 Hz → 100 ms."""
    contacts = [(5, 14)]  # 10 frames (inclusive)
    times_ms = compute_stride_ground_contact_times(contacts, fps=100.0)
    assert len(times_ms) == 1
    assert abs(times_ms[0] - 100.0) < 1e-6


def test_stride_ground_contact_times_multiple():
    """Check multiple contacts produce correct durations."""
    contacts = [(0, 9), (20, 24)]  # 10 frames and 5 frames at 100 Hz
    times_ms = compute_stride_ground_contact_times(contacts, fps=100.0)
    assert len(times_ms) == 2
    assert abs(times_ms[0] - 100.0) < 1e-6
    assert abs(times_ms[1] - 50.0) < 1e-6


# ── compute_foot_strike_under_hip ────────────────────────────────────────

def test_foot_strike_under_hip_zero():
    """Foot directly under hip at contact → ~0 cm offset."""
    n = 10
    foot = np.zeros((n, 3))
    hip = np.zeros((n, 3))
    contacts = [(0, 5)]
    offsets = compute_foot_strike_under_hip(foot, hip, contacts)
    assert len(offsets) == 1
    assert abs(offsets[0]) < 1e-6


def test_foot_strike_under_hip_ahead():
    """Foot 0.30 m ahead of hip (overstriding) → +30 cm."""
    n = 10
    foot = np.zeros((n, 3))
    hip = np.zeros((n, 3))
    foot[:, 0] = 0.30  # foot 30 cm ahead in X (forward)
    contacts = [(0, 5)]
    offsets = compute_foot_strike_under_hip(foot, hip, contacts)
    assert abs(offsets[0] - 30.0) < 1e-3


# ── compute_acceleration_rhythm ───────────────────────────────────────────

def test_acceleration_rhythm_constant_speed():
    """Constant horizontal speed → acceleration rhythm ≈ 0."""
    n = 60
    speed = np.ones(n) * 5.0  # 5 m/s constant
    contacts = [(5, 14), (25, 34)]
    rhythm = compute_acceleration_rhythm(speed, contacts, fps=100.0)
    assert len(rhythm) == 2
    for a in rhythm:
        assert abs(a) < 0.01


def test_acceleration_rhythm_positive():
    """Linearly increasing speed → positive mean acceleration."""
    n = 60
    t = np.arange(n) / 100.0
    speed = 3.0 + 2.0 * t  # 2 m/s² constant acceleration
    contacts = [(0, 19)]
    rhythm = compute_acceleration_rhythm(speed, contacts, fps=100.0)
    assert len(rhythm) == 1
    assert rhythm[0] > 0.0


# ── classify_foot_contact ─────────────────────────────────────────────────

def test_classify_foot_contact_toe():
    """Heel higher than toe → toe strike."""
    label = classify_foot_contact(ankle_y=5.0, toe_y=2.0, heel_y=8.0)
    assert label == "toe"


def test_classify_foot_contact_heel():
    """Toe higher than heel → heel strike."""
    label = classify_foot_contact(ankle_y=5.0, toe_y=8.0, heel_y=2.0)
    assert label == "heel"


def test_classify_foot_contact_flat():
    """Toe and heel at same height → flat."""
    label = classify_foot_contact(ankle_y=3.0, toe_y=3.0, heel_y=3.0)
    assert label == "flat"


# ── compute_per_stride_curve_deviation ────────────────────────────────────

def test_curve_deviation_on_arc_is_zero():
    """Points exactly on the arc should have zero deviation."""
    center = np.array([0.0, 10.0])
    radius = 10.0
    # Two contact points exactly on the circle
    theta1, theta2 = 0.0, 0.5
    p1 = center + radius * np.array([np.cos(theta1), np.sin(theta1)])
    p2 = center + radius * np.array([np.cos(theta2), np.sin(theta2)])

    foot_positions_xz = np.vstack([p1, p1, p2, p2])  # fake trajectory
    contacts = [(0, 1), (2, 3)]
    deviations = compute_per_stride_curve_deviation(
        foot_positions_xz, contacts, center, radius
    )
    assert len(deviations) == 2
    for d in deviations:
        assert abs(d) < 1e-3


def test_curve_deviation_off_arc():
    """Point 0.50 m off the arc → deviation ≈ 50 cm."""
    center = np.array([0.0, 10.0])
    radius = 10.0
    # Point at distance (radius + 0.5) from center
    p = center + (radius + 0.5) * np.array([1.0, 0.0])

    foot_positions_xz = np.array([p])
    contacts = [(0, 0)]
    deviations = compute_per_stride_curve_deviation(
        foot_positions_xz, contacts, center, radius
    )
    assert abs(deviations[0] - 50.0) < 1e-3


def test_curve_deviation_no_params_returns_empty():
    """If curve parameters are None, returns empty list."""
    foot_positions_xz = np.zeros((5, 2))
    contacts = [(0, 2), (3, 4)]
    deviations = compute_per_stride_curve_deviation(
        foot_positions_xz, contacts, None, None
    )
    assert deviations == []


# ── RunUpMetrics dataclass ────────────────────────────────────────────────

def test_runup_metrics_construction_with_defaults():
    """New optional fields should accept default values."""
    m = RunUpMetrics(
        peak_horizontal_velocity_mps=7.5,
        velocity_at_penultimate_mps=7.2,
        velocity_at_takeoff_mps=6.8,
        velocity_loss_penultimate_pct=5.6,
        step_count=8,
        step_lengths_cm=[150.0] * 8,
        step_frequencies_hz=[2.0] * 8,
        penultimate_step_length_cm=170.0,
        last_step_length_cm=120.0,
        curve_radius_m=9.5,
        lean_angle_deg=15.0,
    )
    assert m.stride_ground_contact_times_ms == []
    assert m.foot_strike_under_hip_offset_cm == []
    assert m.acceleration_rhythm_mps2 == []
    assert m.foot_contact_labels == []
    assert m.curve_start_step is None
    assert m.per_stride_curve_deviation_cm == []


def test_runup_metrics_construction_with_all_fields():
    """Verify all Imogen-priority fields are stored correctly."""
    m = RunUpMetrics(
        peak_horizontal_velocity_mps=7.5,
        velocity_at_penultimate_mps=7.2,
        velocity_at_takeoff_mps=6.8,
        velocity_loss_penultimate_pct=5.6,
        step_count=3,
        step_lengths_cm=[150.0, 160.0, 120.0],
        step_frequencies_hz=[2.0, 2.1, 2.3],
        penultimate_step_length_cm=160.0,
        last_step_length_cm=120.0,
        curve_radius_m=9.5,
        lean_angle_deg=15.0,
        stride_ground_contact_times_ms=[110.0, 100.0, 95.0],
        foot_strike_under_hip_offset_cm=[5.0, 3.0, 2.0],
        acceleration_rhythm_mps2=[0.5, 0.4, 0.1],
        foot_contact_labels=["toe", "toe", "flat"],
        curve_start_step=2,
        per_stride_curve_deviation_cm=[0.0, 1.5, 2.0],
    )
    assert m.stride_ground_contact_times_ms == [110.0, 100.0, 95.0]
    assert m.curve_start_step == 2
    assert m.foot_contact_labels[0] == "toe"


# ── compute_horizontal_velocity ───────────────────────────────────────────

def test_horizontal_velocity_constant_forward_motion():
    """Body moving at 5 m/s in X → horizontal speed ≈ 5 m/s everywhere."""
    n = 50
    fps = 100.0
    t = np.arange(n) / fps
    com = np.column_stack([5.0 * t, np.zeros(n), np.zeros(n)])
    speed = compute_horizontal_velocity(com, fps)
    # Exclude edge frames where gradient may be slightly off
    np.testing.assert_allclose(speed[5:-5], 5.0, atol=0.01)


# ── fit_curve_radius ──────────────────────────────────────────────────────

def test_fit_curve_radius_perfect_circle():
    """Points on a circle of known radius → fitted radius matches."""
    radius = 8.0
    theta = np.linspace(0, np.pi / 2, 30)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    points = np.column_stack([x, z])
    fitted = fit_curve_radius(points)
    assert fitted is not None
    assert abs(fitted - radius) < 0.2


def test_fit_curve_radius_too_few_points():
    points = np.random.randn(3, 2)
    assert fit_curve_radius(points) is None
