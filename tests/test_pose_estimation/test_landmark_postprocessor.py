"""Tests for the landmark post-processing pipeline.

Validates gap filling, Butterworth filtering, and segment length enforcement
using synthetic landmark data with known properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pose_estimation.skeleton.landmark_postprocessor import (
    PostProcessorConfig,
    butterworth_filter,
    enforce_segment_lengths,
    fill_gaps,
    postprocess_landmarks,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_landmarks(T: int = 100, n_joints: int = 33, with_vis: bool = True) -> np.ndarray:
    """Create synthetic smooth landmark trajectory (sinusoidal motion)."""
    t = np.linspace(0, 2 * np.pi, T)
    C = 4 if with_vis else 3
    landmarks = np.zeros((T, n_joints, C), dtype=np.float32)

    for j in range(n_joints):
        # Smooth sinusoidal motion at ~2 Hz
        landmarks[:, j, 0] = 0.1 * j + 0.02 * np.sin(2 * t + j)  # x
        landmarks[:, j, 1] = 0.5 + 0.05 * np.cos(2 * t + j * 0.5)  # y
        landmarks[:, j, 2] = 0.01 * np.sin(t + j * 0.3)  # z
        if with_vis:
            landmarks[:, j, 3] = 0.95  # high visibility

    return landmarks


# ── Gap Filling Tests ──────────────────────────────────────────────────


class TestGapFilling:
    def test_no_gaps_unchanged(self):
        lm = _make_landmarks(50)
        result = fill_gaps(lm, min_confidence=0.3)
        np.testing.assert_array_almost_equal(result, lm)

    def test_fills_low_confidence_frames(self):
        lm = _make_landmarks(50)
        # Drop visibility for frames 20-25 on joint 5
        lm[20:26, 5, 3] = 0.1  # below threshold
        lm[20:26, 5, :3] = 999.0  # garbage values

        result = fill_gaps(lm, min_confidence=0.3)

        # Interpolated values should NOT be 999.0
        assert not np.any(result[20:26, 5, :3] > 100)
        # Should be close to the smooth trajectory
        expected = _make_landmarks(50)
        np.testing.assert_array_almost_equal(
            result[20:26, 5, :3], expected[20:26, 5, :3], decimal=1
        )

    def test_visibility_updated(self):
        lm = _make_landmarks(50)
        lm[10, 3, 3] = 0.05  # low vis
        result = fill_gaps(lm, min_confidence=0.3)
        assert result[10, 3, 3] == pytest.approx(0.3)

    def test_all_low_confidence_unchanged(self):
        lm = _make_landmarks(50)
        lm[:, 5, 3] = 0.1  # entire sequence below threshold
        original = lm[:, 5, :3].copy()
        result = fill_gaps(lm, min_confidence=0.3)
        # Can't interpolate if all frames are bad — should be unchanged
        np.testing.assert_array_equal(result[:, 5, :3], original)


# ── Butterworth Filter Tests ──────────────────────────────────────────


class TestButterworthFilter:
    def test_removes_high_frequency_noise(self):
        fps = 30.0
        lm = _make_landmarks(100)
        # Add high-frequency noise (15 Hz — above the 10 Hz cutoff)
        t = np.linspace(0, 2 * np.pi * 5, 100)  # 5 cycles in 100/30 = 3.3s → 1.5 Hz
        noise = 0.01 * np.sin(2 * np.pi * 15 * np.linspace(0, 100 / fps, 100))
        lm[:, 0, 0] += noise

        result = butterworth_filter(lm, fps, cutoff_hz=10.0, order=2)

        # The 15 Hz noise should be attenuated
        residual = result[:, 0, 0] - _make_landmarks(100)[:, 0, 0]
        original_noise = lm[:, 0, 0] - _make_landmarks(100)[:, 0, 0]
        assert np.std(residual) < np.std(original_noise) * 0.5

    def test_preserves_low_frequency_signal(self):
        fps = 30.0
        lm = _make_landmarks(100)  # 2 Hz motion
        result = butterworth_filter(lm, fps, cutoff_hz=10.0, order=2)

        # Low-frequency content should be mostly unchanged (within 5%)
        np.testing.assert_array_almost_equal(
            result[:, 10, :3], lm[:, 10, :3], decimal=2
        )

    def test_short_sequence_returned_unchanged(self):
        lm = _make_landmarks(5)
        result = butterworth_filter(lm, 30.0, cutoff_hz=10.0)
        np.testing.assert_array_equal(result, lm)

    def test_visibility_preserved(self):
        lm = _make_landmarks(50)
        lm[10, 5, 3] = 0.42  # specific visibility
        result = butterworth_filter(lm, 30.0)
        assert result[10, 5, 3] == pytest.approx(0.42)

    def test_cutoff_above_nyquist_returned_unchanged(self):
        lm = _make_landmarks(50)
        result = butterworth_filter(lm, 30.0, cutoff_hz=20.0)  # ≥ nyquist (15)
        np.testing.assert_array_equal(result, lm)


# ── Segment Length Enforcement Tests ──────────────────────────────────


class TestSegmentEnforcement:
    def test_thigh_length_matches_anthropometry(self):
        height_m = 1.78
        lm = _make_landmarks(50)
        # Set left hip (23) and left knee (25) with known positions
        lm[:, 23, :3] = [0.0, 1.0, 0.0]
        lm[:, 25, :3] = [0.0, 0.5, 0.0]  # 0.5 m apart (too long for 0.245 * 1.78 = 0.436)

        result = enforce_segment_lengths(lm, height_m, weight=1.0)

        target = 0.245 * height_m
        for t in range(50):
            actual = np.linalg.norm(result[t, 25, :3] - result[t, 23, :3])
            assert actual == pytest.approx(target, abs=0.001)

    def test_weight_zero_no_change(self):
        lm = _make_landmarks(50)
        result = enforce_segment_lengths(lm, 1.78, weight=0.0)
        np.testing.assert_array_equal(result, lm)

    def test_direction_preserved(self):
        """Segment enforcement should not change joint angle directions."""
        lm = _make_landmarks(20)
        lm[:, 23, :3] = [0.0, 1.0, 0.0]  # hip
        lm[:, 25, :3] = [0.1, 0.5, 0.05]  # knee (some angle)

        result = enforce_segment_lengths(lm, 1.78, weight=1.0)

        # Direction from hip to knee should be preserved
        orig_dir = lm[0, 25, :3] - lm[0, 23, :3]
        new_dir = result[0, 25, :3] - result[0, 23, :3]
        cos_angle = np.dot(orig_dir, new_dir) / (
            np.linalg.norm(orig_dir) * np.linalg.norm(new_dir)
        )
        assert cos_angle > 0.999


# ── Full Pipeline Tests ───────────────────────────────────────────────


class TestFullPipeline:
    def test_default_config_runs(self):
        lm = _make_landmarks(60)
        config = PostProcessorConfig(height_m=1.78)
        result = postprocess_landmarks(lm, fps=30.0, config=config)
        assert result.shape == lm.shape

    def test_no_processing(self):
        lm = _make_landmarks(50)
        config = PostProcessorConfig(
            do_gap_fill=False, do_filter=False, do_segment_enforce=False
        )
        result = postprocess_landmarks(lm, fps=30.0, config=config)
        np.testing.assert_array_equal(result, lm)

    def test_derivatives_smoother_after_filtering(self):
        """Key physics test: filtered landmarks should produce smoother
        acceleration (= better GRF estimates from F = m*a)."""
        fps = 30.0
        lm = _make_landmarks(100)
        # Add realistic noise
        rng = np.random.default_rng(42)
        lm[:, :, :3] += rng.normal(0, 0.003, lm[:, :, :3].shape).astype(np.float32)

        filtered = postprocess_landmarks(
            lm, fps,
            PostProcessorConfig(
                do_gap_fill=False, do_filter=True,
                do_segment_enforce=False, filter_cutoff_hz=10.0
            ),
        )

        # Compute CoM-proxy acceleration (just use hip midpoint)
        dt = 1.0 / fps
        for lm_data, label in [(lm, "raw"), (filtered, "filtered")]:
            hip_mid = (lm_data[:, 23, :3] + lm_data[:, 24, :3]) / 2
            vel = np.gradient(hip_mid, dt, axis=0)
            acc = np.gradient(vel, dt, axis=0)
            if label == "raw":
                raw_acc_std = np.std(acc)
            else:
                filt_acc_std = np.std(acc)

        # Filtered acceleration should be noticeably smoother
        assert filt_acc_std < raw_acc_std * 0.9, (
            f"Filtered acc std ({filt_acc_std:.4f}) should be < 90% of "
            f"raw acc std ({raw_acc_std:.4f})"
        )

    def test_segment_enforce_skipped_without_height(self):
        lm = _make_landmarks(50)
        config = PostProcessorConfig(
            do_gap_fill=False, do_filter=False,
            do_segment_enforce=True, height_m=None  # no height provided
        )
        result = postprocess_landmarks(lm, fps=30.0, config=config)
        np.testing.assert_array_equal(result, lm)
