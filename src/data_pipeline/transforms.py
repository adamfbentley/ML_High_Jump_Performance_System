"""Common transforms for biomechanical data pre-processing.

Applied to BiomechanicalSample before feeding into torch Datasets.
"""

from __future__ import annotations

import numpy as np

from src.data_pipeline.sample import BiomechanicalSample


def normalize_by_body_mass(sample: BiomechanicalSample) -> BiomechanicalSample:
    """Normalize forces and torques by body mass (→ per-kg units).

    This makes dynamics data comparable across subjects of different sizes
    and is standard practice in biomechanics research.
    """
    mass = sample.subject.body_mass_kg
    if mass is None or mass <= 0:
        return sample

    if sample.grf is not None:
        sample.grf = sample.grf / mass
    if sample.joint_torques is not None:
        sample.joint_torques = sample.joint_torques / mass

    return sample


def normalize_by_height(sample: BiomechanicalSample) -> BiomechanicalSample:
    """Normalize positions by body height (→ dimensionless lengths).

    Makes spatial data comparable across subjects of different stature.
    """
    height = sample.subject.height_m
    if height is None or height <= 0:
        return sample

    for attr in ["marker_positions", "com_position", "pose_3d"]:
        val = getattr(sample, attr)
        if val is not None:
            setattr(sample, attr, val / height)

    if sample.com_velocity is not None:
        sample.com_velocity = sample.com_velocity / height
    if sample.com_acceleration is not None:
        sample.com_acceleration = sample.com_acceleration / height

    return sample


def window_sample(
    sample: BiomechanicalSample,
    window_size: int,
    stride: int | None = None,
) -> list[BiomechanicalSample]:
    """Split a sample into fixed-length time windows.

    Args:
        sample: The full trial.
        window_size: Number of frames per window.
        stride: Step between window starts. Defaults to window_size (no overlap).

    Returns:
        List of windowed BiomechanicalSamples.
    """
    if stride is None:
        stride = window_size

    n = sample.n_frames
    if n < window_size:
        return []

    windows = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(sample.get_window(start, end))

    return windows


def lowpass_filter(
    data: np.ndarray,
    fps: float,
    cutoff_hz: float = 12.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-lag Butterworth low-pass filter (standard for biomech data).

    Args:
        data: (T, ...) time-series data along axis 0.
        fps: Sampling rate in Hz.
        cutoff_hz: Filter cutoff frequency.
        order: Butterworth filter order.

    Returns:
        Filtered data, same shape as input.
    """
    from scipy.signal import butter, filtfilt

    nyquist = fps / 2.0
    if cutoff_hz >= nyquist:
        return data

    b, a = butter(order, cutoff_hz / nyquist, btype="low")

    original_shape = data.shape
    if data.ndim == 1:
        return filtfilt(b, a, data).astype(data.dtype)

    # Apply along time axis for each channel
    flat = data.reshape(data.shape[0], -1)
    filtered = np.zeros_like(flat)
    for col in range(flat.shape[1]):
        filtered[:, col] = filtfilt(b, a, flat[:, col])

    return filtered.reshape(original_shape).astype(data.dtype)


def compute_derivatives(
    positions: np.ndarray,
    fps: float,
    filter_cutoff_hz: float | None = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration from positions via numerical differentiation.

    Optionally filters positions first (recommended for noisy pose data).

    Args:
        positions: (T, ...) position data.
        fps: Sampling rate.
        filter_cutoff_hz: Apply low-pass filter before differentiating.
            None = no filtering.

    Returns:
        (velocity, acceleration) arrays of same shape as positions.
    """
    if filter_cutoff_hz is not None and fps > 2 * filter_cutoff_hz:
        positions = lowpass_filter(positions, fps, filter_cutoff_hz)

    dt = 1.0 / fps
    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)

    return velocity, acceleration
