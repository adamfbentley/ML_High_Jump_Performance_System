"""Tests for joint angle computation."""

import numpy as np
import pytest

from src.pose_estimation.skeleton.joint_angles import (
    angle_between_vectors,
    compute_joint_angle,
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
    expected_keys = {"left_knee", "right_knee", "left_hip", "right_hip",
                     "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
                     "left_ankle", "right_ankle"}
    assert set(angles.keys()) == expected_keys


def test_compute_joint_angles_sequence_shape():
    seq = np.random.randn(10, 33, 3)
    angles = compute_joint_angles_sequence(seq)
    for name, vals in angles.items():
        assert vals.shape == (10,)
