"""Unified biomechanical data sample for cross-dataset pre-training.

All dataset loaders convert their native format into BiomechanicalSample,
enabling a single training pipeline to consume data from AddBiomechanics,
BioCV, OpenCap, and any future datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class MovementType(Enum):
    """Categories of movement relevant to high jump transfer learning."""

    COUNTERMOVEMENT_JUMP = "cmj"
    DROP_JUMP = "drop_jump"
    SQUAT_JUMP = "squat_jump"
    VERTICAL_JUMP = "vertical_jump"
    RUNNING = "running"
    SPRINTING = "sprinting"
    WALKING = "walking"
    HOPPING = "hopping"
    HIGH_JUMP = "high_jump"
    OTHER = "other"


# How relevant each movement type is for high jump pre-training (0→1)
MOVEMENT_RELEVANCE = {
    MovementType.HIGH_JUMP: 1.0,
    MovementType.COUNTERMOVEMENT_JUMP: 0.9,
    MovementType.DROP_JUMP: 0.85,
    MovementType.SQUAT_JUMP: 0.8,
    MovementType.VERTICAL_JUMP: 0.8,
    MovementType.SPRINTING: 0.7,
    MovementType.RUNNING: 0.6,
    MovementType.HOPPING: 0.5,
    MovementType.WALKING: 0.3,
    MovementType.OTHER: 0.2,
}


@dataclass
class SubjectInfo:
    """Anthropometric and demographic information about a subject."""

    subject_id: str
    body_mass_kg: Optional[float] = None
    height_m: Optional[float] = None
    sex: Optional[str] = None  # "M" or "F"
    age_years: Optional[float] = None
    # Segment lengths (measured or estimated)
    thigh_length_m: Optional[float] = None
    shank_length_m: Optional[float] = None
    foot_length_m: Optional[float] = None
    trunk_length_m: Optional[float] = None
    upper_arm_length_m: Optional[float] = None
    forearm_length_m: Optional[float] = None


@dataclass
class BiomechanicalSample:
    """A single trial or time window of biomechanical data.

    All arrays are time-indexed along axis 0 with shape (T, ...).
    Not all fields are required — loaders populate what's available.
    """

    # ── Metadata ─────────────────────────────────────────────
    dataset_name: str                           # e.g., "addbiomechanics", "biocv"
    trial_id: str                               # unique identifier within dataset
    subject: SubjectInfo
    movement_type: MovementType = MovementType.OTHER
    fps: float = 0.0

    # ── Kinematics ───────────────────────────────────────────
    # Joint angles in radians: (T, n_joints)
    joint_angles: Optional[np.ndarray] = None
    joint_angular_velocities: Optional[np.ndarray] = None
    joint_angular_accelerations: Optional[np.ndarray] = None
    joint_names: list[str] = field(default_factory=list)

    # 3D marker/landmark positions: (T, n_markers, 3)
    marker_positions: Optional[np.ndarray] = None
    marker_names: list[str] = field(default_factory=list)

    # Center of mass: (T, 3)
    com_position: Optional[np.ndarray] = None
    com_velocity: Optional[np.ndarray] = None
    com_acceleration: Optional[np.ndarray] = None

    # ── Dynamics ─────────────────────────────────────────────
    # Ground reaction force: (T, 3)
    grf: Optional[np.ndarray] = None
    # Center of pressure: (T, 3)
    cop: Optional[np.ndarray] = None
    # Joint torques: (T, n_joints)
    joint_torques: Optional[np.ndarray] = None

    # ── Pose (for vision pre-training) ───────────────────────
    # 2D pose landmarks: (T, n_landmarks, 2 or 3) — pixel coords + confidence
    pose_2d: Optional[np.ndarray] = None
    # 3D pose landmarks: (T, n_landmarks, 3)
    pose_3d: Optional[np.ndarray] = None
    pose_landmark_names: list[str] = field(default_factory=list)
    # Video frame paths (if available)
    frame_paths: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        """Number of time frames in the sample."""
        for arr in [
            self.joint_angles, self.marker_positions, self.com_position,
            self.grf, self.pose_2d, self.pose_3d,
        ]:
            if arr is not None:
                return arr.shape[0]
        return 0

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    @property
    def has_dynamics(self) -> bool:
        """Whether this sample includes force/torque data."""
        return self.grf is not None or self.joint_torques is not None

    @property
    def has_kinematics(self) -> bool:
        """Whether this sample includes joint angle data."""
        return self.joint_angles is not None

    @property
    def has_poses(self) -> bool:
        """Whether this sample includes 2D or 3D pose data."""
        return self.pose_2d is not None or self.pose_3d is not None

    @property
    def relevance_score(self) -> float:
        """How relevant this sample is for high jump pre-training."""
        return MOVEMENT_RELEVANCE.get(self.movement_type, 0.2)

    def get_window(self, start: int, end: int) -> "BiomechanicalSample":
        """Extract a time window from this sample."""
        sliced = BiomechanicalSample(
            dataset_name=self.dataset_name,
            trial_id=f"{self.trial_id}_w{start}_{end}",
            subject=self.subject,
            movement_type=self.movement_type,
            fps=self.fps,
            joint_names=self.joint_names,
            marker_names=self.marker_names,
            pose_landmark_names=self.pose_landmark_names,
        )
        for attr in [
            "joint_angles", "joint_angular_velocities", "joint_angular_accelerations",
            "marker_positions", "com_position", "com_velocity", "com_acceleration",
            "grf", "cop", "joint_torques", "pose_2d", "pose_3d",
        ]:
            val = getattr(self, attr)
            if val is not None:
                setattr(sliced, attr, val[start:end])
        if self.frame_paths:
            sliced.frame_paths = self.frame_paths[start:end]
        return sliced

    def validate(self) -> list[str]:
        """Check for common issues. Returns list of warnings."""
        warnings = []
        if self.fps <= 0:
            warnings.append("fps is not set")
        if self.subject.body_mass_kg is None:
            warnings.append("body mass unknown — needed for dynamics")
        n = self.n_frames
        for attr in [
            "joint_angles", "joint_angular_velocities", "joint_angular_accelerations",
            "grf", "joint_torques", "com_position", "com_velocity", "com_acceleration",
            "pose_2d", "pose_3d",
        ]:
            val = getattr(self, attr)
            if val is not None and val.shape[0] != n:
                warnings.append(f"{attr} has {val.shape[0]} frames, expected {n}")
        return warnings
