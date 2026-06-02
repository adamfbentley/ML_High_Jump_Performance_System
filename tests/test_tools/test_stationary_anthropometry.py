"""Tests for the raw stationary-video anthropometry diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.evaluate_stationary_anthropometry import evaluate_landmark_arrays

W, H = 1000, 1000
KNOWN_LENGTHS_M = {
    "leg": 0.84,
    "shank": 0.45,
    "thigh": 0.39,
    "arm": 0.66,
}


def _make_synthetic_landmarks(n_frames: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Create an upright pose whose 2D and 3D proportions match known lengths."""
    landmarks_2d = np.zeros((n_frames, 33, 3), dtype=np.float32)
    landmarks_3d = np.zeros((n_frames, 33, 4), dtype=np.float32)
    landmarks_2d[:, :, 2] = 1.0
    landmarks_3d[:, :, 3] = 1.0

    for x, indices in ((0.40, (23, 25, 27)), (0.60, (24, 26, 28))):
        landmarks_2d[:, indices[0], :2] = (x, 0.10)
        landmarks_2d[:, indices[1], :2] = (x, 0.10 + KNOWN_LENGTHS_M["thigh"])
        landmarks_2d[:, indices[2], :2] = (x, 0.10 + KNOWN_LENGTHS_M["leg"])
        landmarks_3d[:, indices[0], :3] = (x, 0.0, 0.0)
        landmarks_3d[:, indices[1], :3] = (x, -KNOWN_LENGTHS_M["thigh"], 0.0)
        landmarks_3d[:, indices[2], :3] = (x, -KNOWN_LENGTHS_M["leg"], 0.0)

    for x, indices in ((0.35, (11, 13, 15)), (0.65, (12, 14, 16))):
        landmarks_2d[:, indices[0], :2] = (x, 0.10)
        landmarks_2d[:, indices[1], :2] = (x, 0.10 + KNOWN_LENGTHS_M["arm"] / 2)
        landmarks_2d[:, indices[2], :2] = (x, 0.10 + KNOWN_LENGTHS_M["arm"])
        landmarks_3d[:, indices[0], :3] = (x, 0.0, 0.0)
        landmarks_3d[:, indices[1], :3] = (x, -KNOWN_LENGTHS_M["arm"] / 2, 0.0)
        landmarks_3d[:, indices[2], :3] = (x, -KNOWN_LENGTHS_M["arm"], 0.0)

    return landmarks_2d, landmarks_3d


def test_raw_anthropometry_recovers_synthetic_proportions():
    landmarks_2d, landmarks_3d = _make_synthetic_landmarks()

    report = evaluate_landmark_arrays(
        landmarks_2d,
        landmarks_3d,
        image_width=W,
        image_height=H,
        anchor_segment="leg",
        known_lengths_m=KNOWN_LENGTHS_M,
    )

    for method in ("projected_2d_p95", "raw_world_3d_median"):
        estimates = report[method]["estimates_from_anchor"]
        for segment, known_length in KNOWN_LENGTHS_M.items():
            assert estimates[segment]["estimated_length_m"] == pytest.approx(known_length)
            assert estimates[segment]["error_pct"] == pytest.approx(0.0)


def test_raw_anthropometry_excludes_low_visibility_chain():
    landmarks_2d, landmarks_3d = _make_synthetic_landmarks()
    landmarks_2d[:, [15, 16], 2] = 0.2
    landmarks_3d[:, [15, 16], 3] = 0.2

    report = evaluate_landmark_arrays(
        landmarks_2d,
        landmarks_3d,
        image_width=W,
        image_height=H,
        anchor_segment="leg",
        known_lengths_m=KNOWN_LENGTHS_M,
    )

    for method in ("projected_2d_p95", "raw_world_3d_median"):
        arm = report[method]["segment_summaries"]["arm"]
        estimate = report[method]["estimates_from_anchor"]["arm"]
        assert arm["n_observations"] == 0
        assert arm["representative"] is None
        assert estimate["estimated_length_m"] is None
