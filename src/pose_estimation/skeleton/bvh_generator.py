"""BVH (Biovision Hierarchy) file generation from 3D pose landmarks.

Converts 3D joint positions into a standardized BVH skeleton format,
scaled to the athlete's personal anthropometrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AnthropometricProfile:
    """Athlete body measurements for skeleton rig personalization."""

    height_cm: float
    weight_kg: float
    leg_length_cm: float | None = None
    thigh_length_cm: float | None = None
    shank_length_cm: float | None = None
    trunk_length_cm: float | None = None
    arm_length_cm: float | None = None
    forearm_length_cm: float | None = None
    shoulder_width_cm: float | None = None
    hip_width_cm: float | None = None

    def estimate_missing(self) -> None:
        """Fill missing segment lengths from height using regression ratios.

        Based on Winter (2009) anthropometric proportions.
        """
        h = self.height_cm
        if self.leg_length_cm is None:
            self.leg_length_cm = h * 0.530
        if self.thigh_length_cm is None:
            self.thigh_length_cm = h * 0.245
        if self.shank_length_cm is None:
            self.shank_length_cm = h * 0.246
        if self.trunk_length_cm is None:
            self.trunk_length_cm = h * 0.300
        if self.arm_length_cm is None:
            self.arm_length_cm = h * 0.186
        if self.forearm_length_cm is None:
            self.forearm_length_cm = h * 0.146
        if self.shoulder_width_cm is None:
            self.shoulder_width_cm = h * 0.259
        if self.hip_width_cm is None:
            self.hip_width_cm = h * 0.191


# Standard BVH skeleton hierarchy for high jump analysis
# Maps parent → children in the kinematic chain
SKELETON_HIERARCHY = {
    "Hips": ["Spine", "LeftUpLeg", "RightUpLeg"],
    "Spine": ["Spine1"],
    "Spine1": ["Neck", "LeftShoulder", "RightShoulder"],
    "Neck": ["Head"],
    "Head": [],
    "LeftShoulder": ["LeftArm"],
    "LeftArm": ["LeftForeArm"],
    "LeftForeArm": ["LeftHand"],
    "LeftHand": [],
    "RightShoulder": ["RightArm"],
    "RightArm": ["RightForeArm"],
    "RightForeArm": ["RightHand"],
    "RightHand": [],
    "LeftUpLeg": ["LeftLeg"],
    "LeftLeg": ["LeftFoot"],
    "LeftFoot": ["LeftToeBase"],
    "LeftToeBase": [],
    "RightUpLeg": ["RightLeg"],
    "RightLeg": ["RightFoot"],
    "RightFoot": ["RightToeBase"],
    "RightToeBase": [],
}

# Mapping from MediaPipe BlazePose landmark indices to BVH joint names
BLAZEPOSE_TO_BVH = {
    0: "Head",
    11: "LeftShoulder", 12: "RightShoulder",
    13: "LeftArm", 14: "RightArm",
    15: "LeftForeArm", 16: "RightForeArm",
    23: "LeftUpLeg", 24: "RightUpLeg",
    25: "LeftLeg", 26: "RightLeg",
    27: "LeftFoot", 28: "RightFoot",
    31: "LeftToeBase", 32: "RightToeBase",
}


@dataclass
class BVHJoint:
    """Single joint in the BVH hierarchy."""

    name: str
    offset: np.ndarray  # (3,) offset from parent in rest pose (cm)
    channels: list[str] = field(default_factory=lambda: ["Zrotation", "Xrotation", "Yrotation"])
    children: list[BVHJoint] = field(default_factory=list)
    is_end_site: bool = False


def build_skeleton_rig(anthropometrics: AnthropometricProfile) -> BVHJoint:
    """Build a personalized BVH skeleton from athlete measurements.

    Args:
        anthropometrics: Athlete body measurements (missing values auto-estimated).

    Returns:
        Root BVHJoint (Hips) with full hierarchy populated.
    """
    anthropometrics.estimate_missing()
    a = anthropometrics

    # Build joint offset table scaled to athlete (in cm)
    # Offsets are relative to parent joint in rest pose (T-pose)
    offsets = {
        "Hips": np.array([0.0, a.leg_length_cm, 0.0]),
        "Spine": np.array([0.0, a.trunk_length_cm * 0.4, 0.0]),
        "Spine1": np.array([0.0, a.trunk_length_cm * 0.6, 0.0]),
        "Neck": np.array([0.0, a.trunk_length_cm * 0.15, 0.0]),
        "Head": np.array([0.0, a.height_cm * 0.13, 0.0]),
        "LeftShoulder": np.array([-a.shoulder_width_cm / 2, 0.0, 0.0]),
        "RightShoulder": np.array([a.shoulder_width_cm / 2, 0.0, 0.0]),
        "LeftArm": np.array([-a.arm_length_cm, 0.0, 0.0]),
        "RightArm": np.array([a.arm_length_cm, 0.0, 0.0]),
        "LeftForeArm": np.array([-a.forearm_length_cm, 0.0, 0.0]),
        "RightForeArm": np.array([a.forearm_length_cm, 0.0, 0.0]),
        "LeftHand": np.array([-a.height_cm * 0.108, 0.0, 0.0]),
        "RightHand": np.array([a.height_cm * 0.108, 0.0, 0.0]),
        "LeftUpLeg": np.array([-a.hip_width_cm / 2, 0.0, 0.0]),
        "RightUpLeg": np.array([a.hip_width_cm / 2, 0.0, 0.0]),
        "LeftLeg": np.array([0.0, -a.thigh_length_cm, 0.0]),
        "RightLeg": np.array([0.0, -a.thigh_length_cm, 0.0]),
        "LeftFoot": np.array([0.0, -a.shank_length_cm, 0.0]),
        "RightFoot": np.array([0.0, -a.shank_length_cm, 0.0]),
        "LeftToeBase": np.array([0.0, 0.0, a.height_cm * 0.055]),
        "RightToeBase": np.array([0.0, 0.0, a.height_cm * 0.055]),
    }

    def _build(name: str) -> BVHJoint:
        children_names = SKELETON_HIERARCHY.get(name, [])
        children = [_build(cn) for cn in children_names]
        is_end = len(children_names) == 0
        channels = ["Xposition", "Yposition", "Zposition",
                     "Zrotation", "Xrotation", "Yrotation"] if name == "Hips" else \
                   ["Zrotation", "Xrotation", "Yrotation"]
        return BVHJoint(
            name=name,
            offset=offsets.get(name, np.zeros(3)),
            channels=channels if not is_end else [],
            children=children,
            is_end_site=is_end,
        )

    return _build("Hips")


def write_bvh(
    root_joint: BVHJoint,
    motion_data: np.ndarray,
    fps: float,
    output_path: str | Path,
) -> Path:
    """Write a BVH file from skeleton hierarchy and motion data.

    Args:
        root_joint: Root of the BVH skeleton hierarchy.
        motion_data: (T, N_channels) rotation/position data per frame.
        fps: Frames per second.
        output_path: Where to write the .bvh file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["HIERARCHY"]

    def _write_joint(joint: BVHJoint, indent: int = 0) -> None:
        prefix = "\t" * indent
        if joint.is_end_site:
            lines.append(f"{prefix}End Site")
            lines.append(f"{prefix}{{")
            lines.append(f"{prefix}\tOFFSET {joint.offset[0]:.4f} {joint.offset[1]:.4f} {joint.offset[2]:.4f}")
            lines.append(f"{prefix}}}")
            return

        tag = "ROOT" if indent == 0 else "JOINT"
        lines.append(f"{prefix}{tag} {joint.name}")
        lines.append(f"{prefix}{{")
        lines.append(f"{prefix}\tOFFSET {joint.offset[0]:.4f} {joint.offset[1]:.4f} {joint.offset[2]:.4f}")
        lines.append(f"{prefix}\tCHANNELS {len(joint.channels)} {' '.join(joint.channels)}")
        for child in joint.children:
            _write_joint(child, indent + 1)
        lines.append(f"{prefix}}}")

    _write_joint(root_joint)

    n_frames = motion_data.shape[0]
    frame_time = 1.0 / fps if fps > 0 else 0.0333
    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")
    for frame in motion_data:
        lines.append(" ".join(f"{v:.4f}" for v in frame))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
