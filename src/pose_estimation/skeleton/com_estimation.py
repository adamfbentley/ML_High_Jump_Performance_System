"""Center of Mass (CoM) estimation from 3D pose data.

Uses a segment-based model with mass proportions from
de Leva (1996) and Winter (2009) to estimate whole-body CoM trajectory.
"""

from __future__ import annotations

import numpy as np


# Segment mass proportions as fraction of total body mass (de Leva, 1996 - male)
# Each entry: (proximal_landmark, distal_landmark, mass_fraction, com_position_from_proximal)
SEGMENT_MODEL_MALE = {
    "head_neck": (0, 0, 0.0694, 0.5),        # approximated at head landmark
    "trunk": (11, 23, 0.4346, 0.44),          # shoulder midpoint → hip midpoint
    "left_upper_arm": (11, 13, 0.0271, 0.5772),
    "right_upper_arm": (12, 14, 0.0271, 0.5772),
    "left_forearm_hand": (13, 15, 0.0228, 0.4574),
    "right_forearm_hand": (14, 16, 0.0228, 0.4574),
    "left_thigh": (23, 25, 0.1416, 0.4095),
    "right_thigh": (24, 26, 0.1416, 0.4095),
    "left_shank": (25, 27, 0.0433, 0.4459),
    "right_shank": (26, 28, 0.0433, 0.4459),
    "left_foot": (27, 31, 0.0137, 0.4415),
    "right_foot": (28, 32, 0.0137, 0.4415),
}

# Use shoulder and hip midpoints as proxy for trunk endpoints
# Landmarks 11, 12 = shoulders; 23, 24 = hips


def compute_com_frame(
    landmarks_3d: np.ndarray,
    segment_model: dict | None = None,
) -> np.ndarray:
    """Compute whole-body center of mass for a single frame.

    Args:
        landmarks_3d: (33, 3+) array of 3D joint positions.
        segment_model: Segment definitions (default: male model).

    Returns:
        (3,) center of mass position.
    """
    if segment_model is None:
        segment_model = SEGMENT_MODEL_MALE

    pos = landmarks_3d[:, :3]
    com = np.zeros(3)
    total_mass = 0.0

    for _name, (prox_idx, dist_idx, mass_frac, com_pos) in segment_model.items():
        # Handle trunk specially: use midpoints
        if _name == "trunk":
            proximal = (pos[11] + pos[12]) / 2  # shoulder midpoint
            distal = (pos[23] + pos[24]) / 2    # hip midpoint
        else:
            proximal = pos[prox_idx]
            distal = pos[dist_idx]

        segment_com = proximal + com_pos * (distal - proximal)
        com += mass_frac * segment_com
        total_mass += mass_frac

    return com / total_mass


def compute_com_trajectory(
    landmarks_3d_seq: np.ndarray,
    fps: float,
) -> dict[str, np.ndarray]:
    """Compute CoM position, velocity, and acceleration over a sequence.

    Args:
        landmarks_3d_seq: (T, 33, 3+) time series of 3D landmarks.
        fps: Frame rate for numerical differentiation.

    Returns:
        Dict with keys:
            'position': (T, 3) CoM positions
            'velocity': (T, 3) CoM velocities (central differences)
            'acceleration': (T, 3) CoM accelerations
    """
    n_frames = landmarks_3d_seq.shape[0]
    positions = np.array([
        compute_com_frame(landmarks_3d_seq[t]) for t in range(n_frames)
    ])

    dt = 1.0 / fps
    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)

    return {
        "position": positions,
        "velocity": velocity,
        "acceleration": acceleration,
    }
