"""Joint angle computation from 3D landmark positions.

Computes anatomically meaningful joint angles from 3D pose sequences,
essential for kinematics and as input features for the PINN.
"""

from __future__ import annotations

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 3D vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def compute_joint_angle(
    proximal: np.ndarray,
    joint: np.ndarray,
    distal: np.ndarray,
) -> float:
    """Compute the flexion/extension angle at a joint.

    Args:
        proximal: 3D position of the proximal segment endpoint.
        joint: 3D position of the joint center.
        distal: 3D position of the distal segment endpoint.

    Returns:
        Joint angle in degrees (180° = full extension).
    """
    v1 = proximal - joint
    v2 = distal - joint
    return angle_between_vectors(v1, v2)


# BlazePose landmark index triplets: (proximal, joint, distal)
JOINT_ANGLE_DEFINITIONS = {
    "left_knee": (23, 25, 27),    # hip → knee → ankle
    "right_knee": (24, 26, 28),
    "left_hip": (25, 23, 11),     # knee → hip → shoulder
    "right_hip": (26, 24, 12),
    "left_elbow": (11, 13, 15),   # shoulder → elbow → wrist
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23), # elbow → shoulder → hip
    "right_shoulder": (14, 12, 24),
    "left_ankle": (25, 27, 31),   # knee → ankle → toe
    "right_ankle": (26, 28, 32),
}


def compute_all_joint_angles(landmarks_3d: np.ndarray) -> dict[str, float]:
    """Compute all defined joint angles from a single frame of 3D landmarks.

    Args:
        landmarks_3d: (33, 3) or (33, 4) array of 3D positions.

    Returns:
        Dict mapping joint name to angle in degrees.
    """
    pos = landmarks_3d[:, :3]
    angles = {}
    for name, (prox_idx, joint_idx, dist_idx) in JOINT_ANGLE_DEFINITIONS.items():
        angles[name] = compute_joint_angle(
            pos[prox_idx], pos[joint_idx], pos[dist_idx]
        )
    return angles


def compute_joint_angles_sequence(
    landmarks_3d_seq: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute joint angles for every frame in a 3D pose sequence.

    Args:
        landmarks_3d_seq: (T, 33, 3+) array of 3D positions over time.

    Returns:
        Dict mapping joint name to (T,) array of angles in degrees.
    """
    n_frames = landmarks_3d_seq.shape[0]
    result: dict[str, list[float]] = {name: [] for name in JOINT_ANGLE_DEFINITIONS}

    for t in range(n_frames):
        angles = compute_all_joint_angles(landmarks_3d_seq[t])
        for name, val in angles.items():
            result[name].append(val)

    return {name: np.array(vals) for name, vals in result.items()}
