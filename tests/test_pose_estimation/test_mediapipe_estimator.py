"""Tests for MediaPipe video timeline helpers and ROI crop remap."""

from __future__ import annotations

import numpy as np
import pytest
from src.pose_estimation.estimators.mediapipe_estimator import (
    PoseFrame,
    PoseSequence,
    _aggregate_smoothed_crop,
    _missing_pose_frame,
    _nominal_fps_from_timestamps,
    remap_normalized_to_full_frame,
)


def test_nominal_fps_prefers_decoded_timestamp_cadence() -> None:
    timestamps_ms = [0.0, 33.333333, 66.666667, 100.0]

    fps = _nominal_fps_from_timestamps(timestamps_ms, fallback_fps=27.2)

    assert fps == pytest.approx(30.0, rel=1e-4)


def test_nominal_fps_falls_back_without_timestamp_deltas() -> None:
    assert _nominal_fps_from_timestamps([0.0], fallback_fps=29.97) == pytest.approx(29.97)


def test_missing_pose_frame_preserves_index_and_zero_visibility() -> None:
    frame = _missing_pose_frame(frame_index=7, timestamp_ms=233.0)

    assert frame.frame_index == 7
    assert frame.timestamp_ms == pytest.approx(233.0)
    assert frame.landmarks_2d.shape == (33, 3)
    assert frame.landmarks_3d is not None
    assert frame.landmarks_3d.shape == (33, 4)
    assert not frame.is_valid
    assert frame.landmarks_2d[:, 2].sum() == pytest.approx(0.0)


# ── bbox → full-frame remap (Change 3) ────────────────────────────────────

def _make_landmarks(x: float, y: float, vis: float = 1.0) -> np.ndarray:
    """Uniform landmark grid at (x, y) for testing remap."""
    lm = np.zeros((33, 3), dtype=np.float32)
    lm[:, 0] = x
    lm[:, 1] = y
    lm[:, 2] = vis
    return lm


def test_remap_identity_bbox_unchanged() -> None:
    """bbox covering the full frame → landmarks pass through unchanged."""
    lm = _make_landmarks(0.5, 0.5)
    remapped = remap_normalized_to_full_frame(lm, bbox_norm=(0.0, 0.0, 1.0, 1.0))
    np.testing.assert_allclose(remapped[:, :2], lm[:, :2], atol=1e-6)


def test_remap_half_width_bbox_maps_x_correctly() -> None:
    """Crop occupies x=[0.25, 0.75] of full frame. Landmark at x=0.5 in crop
    should map to x = 0.25 + 0.5 * 0.5 = 0.50 in full frame."""
    lm = _make_landmarks(0.5, 0.5)
    remapped = remap_normalized_to_full_frame(lm, bbox_norm=(0.25, 0.0, 0.75, 1.0))
    assert remapped[0, 0] == pytest.approx(0.50, abs=1e-6)
    assert remapped[0, 1] == pytest.approx(0.50, abs=1e-6)  # y unaffected


def test_remap_corner_crop_round_trip() -> None:
    """Landmark at (0.0, 0.0) in a top-left quarter crop → (bbox_x1, bbox_y1)."""
    bbox = (0.1, 0.2, 0.6, 0.7)
    lm = _make_landmarks(0.0, 0.0)
    remapped = remap_normalized_to_full_frame(lm, bbox_norm=bbox)
    assert remapped[0, 0] == pytest.approx(bbox[0], abs=1e-6)
    assert remapped[0, 1] == pytest.approx(bbox[1], abs=1e-6)


def test_remap_full_extent_maps_to_bbox_corners() -> None:
    """Landmark at (1.0, 1.0) in crop → (x2, y2) of bbox in full frame."""
    bbox = (0.2, 0.3, 0.8, 0.9)
    lm_bottom_right = _make_landmarks(1.0, 1.0)
    remapped = remap_normalized_to_full_frame(lm_bottom_right, bbox_norm=bbox)
    assert remapped[0, 0] == pytest.approx(bbox[2], abs=1e-6)
    assert remapped[0, 1] == pytest.approx(bbox[3], abs=1e-6)


def test_remap_preserves_visibility_channel() -> None:
    """The visibility column (index 2) must not be altered by the remap."""
    lm = _make_landmarks(0.4, 0.6, vis=0.87)
    remapped = remap_normalized_to_full_frame(lm, bbox_norm=(0.1, 0.1, 0.9, 0.9))
    np.testing.assert_allclose(remapped[:, 2], lm[:, 2], atol=1e-6)


def test_remap_does_not_mutate_input() -> None:
    """remap_normalized_to_full_frame must not modify its input array."""
    lm = _make_landmarks(0.3, 0.7)
    original = lm.copy()
    remap_normalized_to_full_frame(lm, bbox_norm=(0.0, 0.0, 0.5, 0.5))
    np.testing.assert_array_equal(lm, original)


# ── _aggregate_smoothed_crop ───────────────────────────────────────────────

def _make_sequence_with_landmark_at(x: float, y: float, n_frames: int = 5) -> PoseSequence:
    """Build a minimal PoseSequence with a single visible landmark position."""
    seq = PoseSequence(video_path="fake.mp4", fps=30.0)
    for i in range(n_frames):
        lm = np.zeros((33, 3), dtype=np.float32)
        lm[0, 0] = x
        lm[0, 1] = y
        lm[0, 2] = 1.0  # visible
        seq.frames.append(PoseFrame(frame_index=i, timestamp_ms=float(i * 33), landmarks_2d=lm))
    return seq


def test_aggregate_crop_returns_none_when_no_landmarks() -> None:
    seq = PoseSequence(video_path="fake.mp4", fps=30.0)
    for i in range(3):
        seq.frames.append(_missing_pose_frame(i, float(i * 33)))
    assert _aggregate_smoothed_crop(seq) is None


def test_aggregate_crop_includes_landmark_with_margin() -> None:
    """Landmark at centre (0.5, 0.5) should produce a crop smaller than the full frame."""
    seq = _make_sequence_with_landmark_at(0.5, 0.5)
    bbox = _aggregate_smoothed_crop(seq, margin=0.20)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert 0.0 <= x1 < 0.5
    assert 0.0 <= y1 < 0.5
    assert 0.5 < x2 <= 1.0
    assert 0.5 < y2 <= 1.0
