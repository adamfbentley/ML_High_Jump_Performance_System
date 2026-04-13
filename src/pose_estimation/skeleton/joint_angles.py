"""Joint angle computation from 3D landmark positions.

Computes anatomically meaningful joint angles from 3D pose sequences,
essential for kinematics and as input features for the PINN.

Coordinate convention (Y-up, right-handed):
  X = forward (direction of run-up)
  Y = up
  Z = lateral

All angles returned in degrees (UI/output convention; convert to radians
before feeding into PINN training).
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


def compute_hip_abduction_angle(
    hip: np.ndarray,
    knee: np.ndarray,
    side: str,
) -> float:
    """Compute signed hip abduction/adduction angle in the frontal (YZ) plane.

    Uses the frontal-plane YZ projection of the thigh vector (hip→knee) relative
    to the downward Y-axis (anatomical neutral / hanging straight).

    Convention (Y-up / X-forward / Z-lateral):
      - Positive angle = abduction  (thigh moves away from midline)
      - Negative angle = adduction  (thigh moves toward midline)

    For the LEFT leg, abduction is toward +Z.
    For the RIGHT leg, abduction is toward -Z.

    Args:
        hip:  (3,) 3D position of the hip joint.
        knee: (3,) 3D position of the knee joint.
        side: "left" or "right".

    Returns:
        Signed hip abduction angle in degrees.
    """
    thigh_vec = knee - hip  # hip → knee
    ty = thigh_vec[1]       # vertical component
    tz = thigh_vec[2]       # lateral component

    if abs(ty) < 1e-8 and abs(tz) < 1e-8:
        return 0.0

    # Signed angle from downward Y-axis (−Y) toward ±Z in the frontal plane.
    # arctan2(tz, −ty):
    #   thigh straight down [ty=−1, tz=0]  → arctan2(0, 1)  =   0° ✓
    #   thigh full left     [ty=0,  tz=+1] → arctan2(1, 0)  =  90° (abduction left)
    #   thigh full right    [ty=0,  tz=−1] → arctan2(−1, 0) = −90° (adduction left)
    angle_deg = float(np.degrees(np.arctan2(tz, -ty)))

    # For the right leg, abduction is toward −Z, so flip sign.
    if side == "right":
        angle_deg = -angle_deg

    return angle_deg


# BlazePose landmark index triplets: (proximal, joint, distal)
JOINT_ANGLE_DEFINITIONS = {
    "left_knee": (23, 25, 27),    # hip → knee → ankle
    "right_knee": (24, 26, 28),
    "left_hip": (25, 23, 11),     # knee → hip → shoulder  (flexion/extension)
    "right_hip": (26, 24, 12),
    "left_elbow": (11, 13, 15),   # shoulder → elbow → wrist
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),  # elbow → shoulder → hip
    "right_shoulder": (14, 12, 24),
    "left_ankle": (25, 27, 31),   # knee → ankle → toe
    "right_ankle": (26, 28, 32),
}

# BlazePose landmark indices used for hip abduction: (hip, knee, side)
HIP_ABDUCTION_DEFINITIONS = {
    "left_hip_abduction":  (23, 25, "left"),   # left hip → left knee
    "right_hip_abduction": (24, 26, "right"),  # right hip → right knee
}


def compute_all_joint_angles(landmarks_3d: np.ndarray) -> dict[str, float]:
    """Compute all defined joint angles from a single frame of 3D landmarks.

    Returns flexion/extension angles for standard joints AND bilateral
    hip abduction/adduction angles in the frontal (YZ) plane.

    Args:
        landmarks_3d: (33, 3) or (33, 4) array of 3D positions (BlazePose order).

    Returns:
        Dict mapping joint name to angle in degrees.
        Keys include: left_knee, right_knee, left_hip, right_hip,
                      left_elbow, right_elbow, left_shoulder, right_shoulder,
                      left_ankle, right_ankle,
                      left_hip_abduction, right_hip_abduction.
    """
    pos = landmarks_3d[:, :3]
    angles: dict[str, float] = {}

    # Flexion / extension angles
    for name, (prox_idx, joint_idx, dist_idx) in JOINT_ANGLE_DEFINITIONS.items():
        angles[name] = compute_joint_angle(
            pos[prox_idx], pos[joint_idx], pos[dist_idx]
        )

    # Frontal-plane hip abduction / adduction angles
    for name, (hip_idx, knee_idx, side) in HIP_ABDUCTION_DEFINITIONS.items():
        angles[name] = compute_hip_abduction_angle(
            pos[hip_idx], pos[knee_idx], side
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
        Includes both flexion/extension and hip abduction angles.
    """
    n_frames = landmarks_3d_seq.shape[0]
    all_keys = list(JOINT_ANGLE_DEFINITIONS.keys()) + list(HIP_ABDUCTION_DEFINITIONS.keys())
    result: dict[str, list[float]] = {name: [] for name in all_keys}

    for t in range(n_frames):
        frame_angles = compute_all_joint_angles(landmarks_3d_seq[t])
        for name, val in frame_angles.items():
            result[name].append(val)

    return {name: np.array(vals) for name, vals in result.items()}
