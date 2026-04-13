"""High jump phase segmentation and run-up kinematics analysis.

Decomposes a jump attempt into phases (approach, curve, penultimate,
takeoff, flight, landing) and extracts run-up metrics.

Athlete A's priority metrics (from athlete brief):
  - Stride length (all strides)
  - Ground contact time per stride  (ms)
  - Foot-strike-under-hip offset    (cm)
  - Acceleration rhythm             (m/s per stride)
  - Curve adherence per stride      (cm deviation from arc)
  - Curve start step                (which stride number)
  - Point of contact                ("toe", "flat", "heel")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class JumpPhase(str, Enum):
    """Phases of a Fosbury Flop high jump attempt."""
    APPROACH = "approach"         # Straight-line run-up
    CURVE = "curve"               # Curved approach (last 3-5 steps)
    PENULTIMATE = "penultimate"   # Second-to-last step (braking/lowering CoM)
    TAKEOFF = "takeoff"           # Plant foot contact → last ground contact
    FLIGHT = "flight"             # Airborne over the bar
    LANDING = "landing"           # Contact with mat


@dataclass
class PhaseSegment:
    """A detected phase with frame boundaries."""
    phase: JumpPhase
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float


@dataclass
class RunUpMetrics:
    """Extracted metrics from the approach and curve phases.

    All list fields are ordered stride-by-stride from first contact to takeoff.
    Stride = one complete step (foot contact to next same-foot contact).
    """

    # ── Velocity profile ──────────────────────────────────────────────────
    peak_horizontal_velocity_mps: float
    velocity_at_penultimate_mps: float
    velocity_at_takeoff_mps: float
    velocity_loss_penultimate_pct: float   # braking between penultimate and takeoff

    # ── Step characteristics ───────────────────────────────────────────────
    step_count: int
    step_lengths_cm: list[float]
    step_frequencies_hz: list[float]
    penultimate_step_length_cm: float
    last_step_length_cm: float

    # ── Curve geometry ─────────────────────────────────────────────────────
    curve_radius_m: float | None
    lean_angle_deg: float | None           # body lean into curve (mean over curve phase)

    # ── Per-stride ground contact time (Athlete A priority) ──────────────────
    stride_ground_contact_times_ms: list[float] = field(default_factory=list)
    # ms per contact; length == step_count

    # ── Foot-strike-under-hip offset (Athlete A priority) ────────────────────
    foot_strike_under_hip_offset_cm: list[float] = field(default_factory=list)
    # Horizontal distance between foot-strike position and hip at same instant.
    # Positive = foot ahead of hip (overstriding), negative = foot behind hip.

    # ── Acceleration rhythm (Athlete A priority) ─────────────────────────────
    acceleration_rhythm_mps2: list[float] = field(default_factory=list)
    # Per-stride mean horizontal acceleration (m/s²).

    # ── Foot contact classification (Athlete A priority) ──────────────────────
    foot_contact_labels: list[str] = field(default_factory=list)
    # "toe", "flat", or "heel" per stride.

    # ── Curve start step (Athlete A priority) ───────────────────────────────
    curve_start_step: int | None = None
    # Step number (0-indexed from first stride) at which the J-curve begins.

    # ── Per-stride curve deviation (Athlete A priority) ─────────────────────
    per_stride_curve_deviation_cm: list[float] = field(default_factory=list)
    # For each stride during the curve phase, the lateral distance (cm) from
    # the ideal arc described by curve_radius_m and curve centre.

    # ── Per-stride arm lateral swing (Athlete A priority) ────────────────────
    per_stride_arm_lateral_swing_cm: list[float] = field(default_factory=list)
    # Maximum lateral (Z-axis) displacement of the wrist from the ipsilateral
    # shoulder's Z position during each stride's ground-contact window.
    # Athlete A: "I have a habit of swinging my arm out to the side which can
    # throw off my body position."  Positive = arm swings outward laterally.


def detect_ground_contacts(
    ankle_positions: np.ndarray,
    fps: float,
    height_threshold_cm: float = 5.0,
) -> list[tuple[int, int]]:
    """Detect foot-ground contact phases from ankle vertical position.

    Args:
        ankle_positions: (T, 3) ankle trajectory [x, y, z] where y is vertical.
        fps: Frame rate.
        height_threshold_cm: Maximum ankle height to count as ground contact.

    Returns:
        List of (start_frame, end_frame) tuples for each contact period.
    """
    is_contact = ankle_positions[:, 1] < height_threshold_cm
    contacts: list[tuple[int, int]] = []
    in_contact = False
    start = 0

    for i, c in enumerate(is_contact):
        if c and not in_contact:
            start = i
            in_contact = True
        elif not c and in_contact:
            contacts.append((start, i - 1))
            in_contact = False
    if in_contact:
        contacts.append((start, len(is_contact) - 1))

    return contacts


def compute_horizontal_velocity(
    com_positions: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Compute horizontal (XZ plane) velocity magnitude over time.

    Args:
        com_positions: (T, 3) center of mass trajectory.
        fps: Frame rate.

    Returns:
        (T,) horizontal speed in m/s.
    """
    dt = 1.0 / fps
    velocity = np.gradient(com_positions, dt, axis=0)
    horizontal_speed = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 2] ** 2)
    return horizontal_speed


def fit_curve_radius(
    com_positions_xz: np.ndarray,
) -> float | None:
    """Estimate the radius of the curved approach from CoM horizontal path.

    Fits a circle to the last portion of the approach trajectory
    using algebraic circle fitting.

    Args:
        com_positions_xz: (N, 2) horizontal positions during curve phase.

    Returns:
        Estimated curve radius in meters, or None if fitting fails.
    """
    if len(com_positions_xz) < 5:
        return None

    x = com_positions_xz[:, 0]
    z = com_positions_xz[:, 1]

    # Algebraic circle fit: (x - a)^2 + (z - b)^2 = r^2
    A = np.column_stack([2 * x, 2 * z, np.ones_like(x)])
    b = x ** 2 + z ** 2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, c, d = result
        radius = np.sqrt(d + a ** 2 + c ** 2)
        return float(radius) if np.isfinite(radius) and radius > 0.5 else None
    except np.linalg.LinAlgError:
        return None


def compute_stride_ground_contact_times(
    contacts: list[tuple[int, int]],
    fps: float,
) -> list[float]:
    """Compute ground contact duration for each stride.

    Args:
        contacts: List of (start_frame, end_frame) from detect_ground_contacts.
        fps: Frame rate.

    Returns:
        List of contact durations in milliseconds, one per detected contact.
    """
    dt_ms = 1000.0 / fps
    return [float((end - start + 1) * dt_ms) for start, end in contacts]


def compute_foot_strike_under_hip(
    foot_positions: np.ndarray,
    hip_positions: np.ndarray,
    contacts: list[tuple[int, int]],
) -> list[float]:
    """Compute foot-strike-under-hip offset for each stride.

    Measures the horizontal (XZ-plane) distance from the foot-strike position
    to the hip at the same instant.  Positive = foot ahead of hip (overstriding).

    Args:
        foot_positions: (T, 3) ankle/foot position trajectory.
        hip_positions:  (T, 3) hip joint position trajectory.
        contacts:       List of (start_frame, end_frame) ground-contact intervals.

    Returns:
        List of offsets in centimetres, one per contact.  Positive = overstriding.
    """
    offsets_cm = []
    for start, _end in contacts:
        foot_xz = foot_positions[start, [0, 2]]  # XZ at contact moment
        hip_xz = hip_positions[start, [0, 2]]
        # Signed projection: positive when foot is further along X (forward direction).
        # Sign is derived solely from the X-component (forward/backward axis).
        diff = foot_xz - hip_xz
        sign = np.sign(diff[0]) if diff[0] != 0.0 else 1.0
        offset_m = sign * float(np.linalg.norm(diff))
        offsets_cm.append(offset_m * 100.0)
    return offsets_cm


def compute_acceleration_rhythm(
    horizontal_speed: np.ndarray,
    contacts: list[tuple[int, int]],
    fps: float,
) -> list[float]:
    """Compute mean horizontal acceleration within each ground contact interval.

    Gives the per-stride 'acceleration rhythm' metric Athlete A identified:
    how consistently the athlete builds speed through the run-up.

    Args:
        horizontal_speed: (T,) horizontal speed in m/s.
        contacts: List of (start_frame, end_frame) from detect_ground_contacts.
        fps: Frame rate.

    Returns:
        List of mean acceleration values in m/s², one per contact.
    """
    dt = 1.0 / fps
    accel = np.gradient(horizontal_speed, dt)
    rhythm = []
    for start, end in contacts:
        if end > start:
            rhythm.append(float(np.mean(accel[start:end + 1])))
        else:
            rhythm.append(0.0)
    return rhythm


def classify_foot_contact(
    ankle_y: float,
    toe_y: float,
    heel_y: float,
    toe_threshold_cm: float = 2.0,
    heel_threshold_cm: float = 2.0,
) -> str:
    """Classify foot-strike as toe, flat, or heel contact.

    At the instant of contact (first frame of the contact interval):
      - heel contact: heel significantly lower than toe
      - toe contact:  toe significantly lower than heel
      - flat:         both at similar height

    Args:
        ankle_y:  Ankle Y position (cm or m — same units for all).
        toe_y:    Toe Y position.
        heel_y:   Heel Y position.
        toe_threshold_cm: Threshold for classifying toe vs flat (same unit as inputs).
        heel_threshold_cm: Threshold for classifying heel vs flat.

    Returns:
        "toe", "heel", or "flat".
    """
    diff = heel_y - toe_y  # positive = heel higher than toe = toe contact
    if diff > toe_threshold_cm:
        return "toe"
    elif diff < -heel_threshold_cm:
        return "heel"
    return "flat"


def compute_per_stride_curve_deviation(
    foot_positions_xz: np.ndarray,
    contacts: list[tuple[int, int]],
    curve_center_xz: np.ndarray | None,
    curve_radius_m: float | None,
) -> list[float]:
    """Compute each stride's lateral deviation from the ideal approach arc.

    For each ground-contact event during the curved portion, computes the
    distance from the foot-strike point to the nearest point on the ideal
    circle.  A large deviation means the athlete stepped off the curve.

    Args:
        foot_positions_xz: (T, 2) foot position in the horizontal XZ plane.
        contacts:          List of (start_frame, end_frame) contact intervals.
        curve_center_xz:   (2,) centre of the ideal arc in XZ plane.
        curve_radius_m:    Radius of the ideal arc in metres.

    Returns:
        List of deviations in centimetres, one per contact.
        Returns empty list if curve parameters are unavailable.
    """
    if curve_center_xz is None or curve_radius_m is None:
        return []

    deviations_cm = []
    for start, _end in contacts:
        foot_xz = foot_positions_xz[start]
        dist_to_center = float(np.linalg.norm(foot_xz - curve_center_xz))
        deviation_m = abs(dist_to_center - curve_radius_m)
        deviations_cm.append(deviation_m * 100.0)
    return deviations_cm


def compute_arm_lateral_swing(
    wrist_positions: np.ndarray,
    shoulder_positions: np.ndarray,
    contacts: list[tuple[int, int]],
) -> list[float]:
    """Compute per-stride maximum lateral arm swing relative to the shoulder.

    Measures how far the wrist deviates laterally (Z-axis) from the ipsilateral
    shoulder's Z position during each ground-contact window.  A large positive
    value means the arm is swinging out to the side rather than driving forward.

    Athlete A: "I have a habit of swinging my arm out to the side which can throw
    off my body position."

    Args:
        wrist_positions:    (T, 3) ipsilateral wrist position trajectory.
        shoulder_positions: (T, 3) ipsilateral shoulder position trajectory.
        contacts:           List of (start_frame, end_frame) contact intervals.

    Returns:
        List of maximum lateral wrist-to-shoulder Z-offsets in centimetres,
        one per contact.  Positive = arm swings outward from the body midline.
    """
    swings_cm = []
    for start, end in contacts:
        wrist_z = wrist_positions[start:end + 1, 2]
        shoulder_z = shoulder_positions[start:end + 1, 2]
        lateral_offset = np.abs(wrist_z - shoulder_z)
        swings_cm.append(float(np.max(lateral_offset)) * 100.0)
    return swings_cm
