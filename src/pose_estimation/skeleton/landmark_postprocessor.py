"""Post-processing pipeline for MediaPipe 3D landmarks.

Applies standard biomechanics signal processing to raw per-frame landmarks
before joint angle / CoM computation.  Each step is optional and configurable.

Processing order:
    1. Confidence-weighted gap filling (interpolate low-visibility frames)
    2. Butterworth low-pass filter (remove tracking jitter)
    3. Segment length enforcement (project landmarks onto consistent skeleton)

References:
    Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.
    Robertson, D.G.E. et al. (2014). Research Methods in Biomechanics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt


# ── Anthropometric segment length ratios (fraction of standing height) ────
# From Winter (2009) Table 4.1 and Drillis & Contini (1966).
# Each entry: (proximal_landmark_idx, distal_landmark_idx, fraction_of_height)
SEGMENT_LENGTH_RATIOS: dict[str, tuple[int, int, float]] = {
    "left_upper_arm": (11, 13, 0.186),    # shoulder → elbow
    "right_upper_arm": (12, 14, 0.186),
    "left_forearm": (13, 15, 0.146),       # elbow → wrist
    "right_forearm": (14, 16, 0.146),
    "left_thigh": (23, 25, 0.245),         # hip → knee
    "right_thigh": (24, 26, 0.245),
    "left_shank": (25, 27, 0.246),         # knee → ankle
    "right_shank": (26, 28, 0.246),
    "left_foot": (27, 31, 0.055),          # ankle → toe
    "right_foot": (28, 32, 0.055),
    "trunk": (None, None, 0.288),          # shoulder mid → hip mid (special)
}


@dataclass
class PostProcessorConfig:
    """Configuration for the landmark post-processing pipeline.

    Attributes:
        do_gap_fill: Interpolate landmarks with visibility below threshold.
        gap_fill_min_confidence: Visibility threshold; below this is treated as missing.
        do_filter: Apply Butterworth low-pass filter.
        filter_cutoff_hz: Low-pass cutoff frequency (Hz).  Standard biomechanics:
            6 Hz for walking, 10-12 Hz for running/jumping (Winter 2009).
        filter_order: Butterworth filter order (applied zero-phase via filtfilt → 2×order).
        do_segment_enforce: Enforce anthropometric segment lengths.
        height_m: Standing height (metres); required for segment length enforcement.
        segment_enforce_weight: Blend factor 0-1; 1.0 = fully enforce, 0.0 = no change.
    """

    do_gap_fill: bool = True
    gap_fill_min_confidence: float = 0.3

    do_filter: bool = True
    filter_cutoff_hz: float = 10.0  # 10 Hz — good default for jumping
    filter_order: int = 2

    do_segment_enforce: bool = True
    height_m: float | None = None
    segment_enforce_weight: float = 0.8


# ── 1. Gap Filling ────────────────────────────────────────────────────────


def fill_gaps(
    landmarks: np.ndarray,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """Interpolate landmarks where visibility drops below threshold.

    Uses linear interpolation on per-coordinate time series.  Extrapolates
    at sequence edges using nearest-neighbour (constant) fill.

    Args:
        landmarks: (T, 33, 4) array with columns [x, y, z, visibility].
        min_confidence: Visibility threshold — below this is treated as a gap.

    Returns:
        (T, 33, 4) array with gaps interpolated.  Visibility column is updated
        to reflect interpolated frames (set to ``min_confidence``).
    """
    result = landmarks.copy()
    T, N, _ = result.shape

    for j in range(N):
        vis = result[:, j, 3] if result.shape[2] > 3 else np.ones(T)
        low = vis < min_confidence
        if not low.any() or low.all():
            continue

        good = ~low
        good_idx = np.where(good)[0]
        bad_idx = np.where(low)[0]

        for ax in range(3):  # x, y, z
            result[bad_idx, j, ax] = np.interp(
                bad_idx, good_idx, result[good_idx, j, ax]
            )

        # Mark interpolated frames
        if result.shape[2] > 3:
            result[bad_idx, j, 3] = min_confidence

    return result


# ── 2. Butterworth Low-Pass Filter ────────────────────────────────────────


def butterworth_filter(
    landmarks: np.ndarray,
    fps: float,
    cutoff_hz: float = 10.0,
    order: int = 2,
) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter to 3D landmark time series.

    This is the standard step in biomechanics signal processing (Winter 2009,
    Ch. 2).  Removes high-frequency tracking jitter while preserving the
    underlying movement kinematics.

    filtfilt applies the filter forward and backward, so effective order is 2×N
    and there is zero phase distortion.

    Args:
        landmarks: (T, 33, C) where C ≥ 3.  Only x/y/z columns are filtered;
            the visibility column (if present) is preserved unchanged.
        fps: Sampling rate (frames per second).
        cutoff_hz: Low-pass cutoff frequency in Hz.
        order: Filter order (before doubling by filtfilt).

    Returns:
        (T, 33, C) filtered landmarks.
    """
    T = landmarks.shape[0]
    if T < 12:
        # Too short for reliable filtering — return as-is
        return landmarks.copy()

    nyquist = fps / 2.0
    if cutoff_hz >= nyquist:
        # Cutoff above Nyquist — can't filter meaningfully
        return landmarks.copy()

    b, a = butter(order, cutoff_hz / nyquist, btype="low")

    result = landmarks.copy()
    n_joints = landmarks.shape[1]

    # Minimum pad length for filtfilt stability
    padlen = min(3 * max(len(b), len(a)), T - 1)

    for j in range(n_joints):
        for ax in range(3):  # x, y, z only
            signal = landmarks[:, j, ax].astype(np.float64)
            if np.isnan(signal).any():
                continue
            result[:, j, ax] = filtfilt(b, a, signal, padlen=padlen).astype(
                landmarks.dtype
            )

    return result


# ── 3. Segment Length Enforcement ─────────────────────────────────────────


def enforce_segment_lengths(
    landmarks: np.ndarray,
    height_m: float,
    weight: float = 0.8,
) -> np.ndarray:
    """Project landmarks so that limb segment lengths match anthropometry.

    For each segment, the distal landmark is moved along the proximal→distal
    direction to enforce the target length.  A blend weight controls how
    aggressively the correction is applied (1.0 = fully enforce, 0.0 = no change).

    This preserves joint angles while making the skeleton geometrically
    consistent frame-to-frame.

    Args:
        landmarks: (T, 33, C) array, C ≥ 3.
        height_m: Athlete standing height in metres.
        weight: Blend factor in [0, 1].

    Returns:
        (T, 33, C) landmarks with segment lengths enforced.
    """
    result = landmarks.copy()
    T = landmarks.shape[0]

    for _seg_name, (prox_idx, dist_idx, ratio) in SEGMENT_LENGTH_RATIOS.items():
        if prox_idx is None:
            # Trunk: skip — handled by shoulder/hip midpoints, more complex
            continue

        target_length = height_m * ratio

        for t in range(T):
            p = result[t, prox_idx, :3]
            d = result[t, dist_idx, :3]
            direction = d - p
            current_length = np.linalg.norm(direction)

            if current_length < 1e-6:
                continue

            unit_dir = direction / current_length
            corrected_d = p + unit_dir * target_length

            # Blend between original and corrected
            result[t, dist_idx, :3] = (1 - weight) * d + weight * corrected_d

    return result


# ── Full Pipeline ─────────────────────────────────────────────────────────


def postprocess_landmarks(
    landmarks: np.ndarray,
    fps: float,
    config: PostProcessorConfig | None = None,
) -> np.ndarray:
    """Run the full post-processing pipeline on raw MediaPipe landmarks.

    Processing order: gap fill → Butterworth filter → segment enforcement.

    Args:
        landmarks: (T, 33, C) raw landmarks from MediaPipe (C=3 or 4).
        fps: Video frame rate.
        config: Processing options.  Uses sensible defaults if None.

    Returns:
        (T, 33, C) post-processed landmarks.
    """
    if config is None:
        config = PostProcessorConfig()

    result = landmarks

    # 1. Gap fill
    if config.do_gap_fill and result.shape[2] >= 4:
        result = fill_gaps(result, min_confidence=config.gap_fill_min_confidence)

    # 2. Butterworth filter
    if config.do_filter:
        result = butterworth_filter(
            result, fps,
            cutoff_hz=config.filter_cutoff_hz,
            order=config.filter_order,
        )

    # 3. Segment length enforcement
    if config.do_segment_enforce and config.height_m is not None:
        result = enforce_segment_lengths(
            result, config.height_m,
            weight=config.segment_enforce_weight,
        )

    return result
