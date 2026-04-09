"""High jump phase segmentation and run-up kinematics analysis.

Decomposes a jump attempt into phases (approach, curve, penultimate,
takeoff, flight, landing) and extracts run-up metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Extracted metrics from the approach and curve phases."""

    # Velocity profile
    peak_horizontal_velocity_mps: float
    velocity_at_penultimate_mps: float
    velocity_at_takeoff_mps: float
    velocity_loss_penultimate_pct: float  # braking

    # Step characteristics
    step_count: int
    step_lengths_cm: list[float]
    step_frequencies_hz: list[float]
    penultimate_step_length_cm: float
    last_step_length_cm: float

    # Curve geometry
    curve_radius_m: float | None
    lean_angle_deg: float | None       # body lean into curve


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
    contacts = []
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
