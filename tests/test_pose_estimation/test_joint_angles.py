"""Tests for joint angle computation."""

import numpy as np
import pytest

from src.pose_estimation.skeleton.joint_angles import (
    angle_between_vectors,
    compute_joint_angle,
    compute_hip_abduction_angle,
    compute_all_joint_angles,
    compute_joint_angles_sequence,
)


def test_angle_between_parallel_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 0.0])
    assert abs(angle_between_vectors(v1, v2)) < 0.01


def test_angle_between_perpendicular_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert abs(angle_between_vectors(v1, v2) - 90.0) < 1e-5


def test_angle_between_opposite_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert abs(angle_between_vectors(v1, v2) - 180.0) < 0.01


def test_compute_joint_angle_straight_leg():
    # Points in a straight line = 180 degrees
    hip = np.array([0, 1, 0])
    knee = np.array([0, 0, 0])
    ankle = np.array([0, -1, 0])
    assert abs(compute_joint_angle(hip, knee, ankle) - 180.0) < 0.01


def test_compute_joint_angle_right_angle():
    hip = np.array([0, 1, 0])
    knee = np.array([0, 0, 0])
    ankle = np.array([1, 0, 0])
    assert abs(compute_joint_angle(hip, knee, ankle) - 90.0) < 1e-5


def test_compute_all_joint_angles_returns_expected_keys():
    landmarks = np.random.randn(33, 3)
    angles = compute_all_joint_angles(landmarks)
    # All original flexion/extension keys plus the two new abduction keys
    expected_keys = {
        "left_knee", "right_knee", "left_hip", "right_hip",
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_ankle", "right_ankle",
        "left_hip_abduction", "right_hip_abduction",
    }
    assert set(angles.keys()) == expected_keys


def test_compute_joint_angles_sequence_shape():
    seq = np.random.randn(10, 33, 3)
    angles = compute_joint_angles_sequence(seq)
    for name, vals in angles.items():
        assert vals.shape == (10,)


# ── Hip abduction / adduction tests ──────────────────────────────────────

def test_hip_abduction_vertical_thigh_gives_zero_left():
    """A thigh hanging straight down (Y-axis) should give 0° abduction."""
    hip = np.array([0.0, 1.0, 0.0])
    knee = np.array([0.0, 0.0, 0.0])  # thigh vector = [0, -1, 0]
    angle = compute_hip_abduction_angle(hip, knee, side="left")
    assert abs(angle) < 0.5, f"Expected ~0°, got {angle:.3f}°"


def test_hip_abduction_vertical_thigh_gives_zero_right():
    """A thigh hanging straight down should give 0° for the right leg too."""
    hip = np.array([0.0, 1.0, 0.0])
    knee = np.array([0.0, 0.0, 0.0])
    angle = compute_hip_abduction_angle(hip, knee, side="right")
    assert abs(angle) < 0.5, f"Expected ~0°, got {angle:.3f}°"


def test_hip_abduction_lateral_displacement_left():
    """Left thigh displaced 30° laterally (+Z) should give ~30° abduction."""
    # Thigh vector: 30° from downward Y toward +Z
    # [ty, tz] = [-cos(30°), sin(30°)] = [-0.866, 0.5]
    angle_expected = 30.0
    hip = np.array([0.0, 0.866, 0.0])
    knee = np.array([0.0, 0.0, 0.5])   # thigh = [0, -0.866, 0.5]
    angle = compute_hip_abduction_angle(hip, knee, side="left")
    assert abs(angle - angle_expected) < 1.0, (
        f"Expected ~{angle_expected}°, got {angle:.3f}°"
    )


def test_hip_abduction_lateral_displacement_right():
    """Right thigh displaced 30° laterally (-Z) should give ~30° abduction."""
    angle_expected = 30.0
    hip = np.array([0.0, 0.866, 0.0])
    knee = np.array([0.0, 0.0, -0.5])   # thigh = [0, -0.866, -0.5]
    angle = compute_hip_abduction_angle(hip, knee, side="right")
    assert abs(angle - angle_expected) < 1.0, (
        f"Expected ~{angle_expected}°, got {angle:.3f}°"
    )


def test_hip_adduction_negative_left():
    """Left thigh crossing midline (-Z direction) should be negative (adduction)."""
    hip = np.array([0.0, 0.866, 0.0])
    knee = np.array([0.0, 0.0, -0.5])   # thigh toward -Z for left leg = adduction
    angle = compute_hip_abduction_angle(hip, knee, side="left")
    assert angle < 0.0, f"Expected negative (adduction), got {angle:.3f}°"


def test_hip_abduction_in_sequence():
    """Hip abduction keys appear in sequence output for every frame."""
    seq = np.random.randn(5, 33, 3)
    angles = compute_joint_angles_sequence(seq)
    assert "left_hip_abduction" in angles
    assert "right_hip_abduction" in angles
    assert angles["left_hip_abduction"].shape == (5,)
    assert angles["right_hip_abduction"].shape == (5,)
