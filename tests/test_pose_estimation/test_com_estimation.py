"""Tests for CoM estimation."""

import numpy as np
import pytest

from src.pose_estimation.skeleton.com_estimation import (
    compute_com_frame,
    compute_com_trajectory,
)


def test_com_frame_returns_3d_point():
    landmarks = np.random.randn(33, 3)
    com = compute_com_frame(landmarks)
    assert com.shape == (3,)


def test_com_is_within_convex_hull_of_landmarks():
    """CoM should be somewhere within the body, not outside it."""
    landmarks = np.random.randn(33, 3)
    com = compute_com_frame(landmarks)
    # Loose check: CoM should be within the bounding box of key landmarks
    key_indices = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulders, hips, knees, ankles
    key_positions = landmarks[key_indices, :]
    for dim in range(3):
        assert com[dim] >= key_positions[:, dim].min() - 1.0
        assert com[dim] <= key_positions[:, dim].max() + 1.0


def test_com_trajectory_shapes():
    seq = np.random.randn(20, 33, 3)
    result = compute_com_trajectory(seq, fps=30.0)
    assert result["position"].shape == (20, 3)
    assert result["velocity"].shape == (20, 3)
    assert result["acceleration"].shape == (20, 3)


def test_com_trajectory_stationary():
    """If all landmarks are stationary, velocity and acceleration should be ~0."""
    frame = np.random.randn(33, 3)
    seq = np.tile(frame, (30, 1, 1))  # 30 identical frames
    result = compute_com_trajectory(seq, fps=30.0)
    assert np.allclose(result["velocity"], 0, atol=1e-10)
    assert np.allclose(result["acceleration"], 0, atol=1e-10)
