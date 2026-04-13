"""Takeoff biomechanics analysis.

Extracts critical takeoff parameters: plant timing, takeoff angle,
vertical/horizontal velocity, ground contact time, and estimated
ground reaction force from CoM trajectory (inverse dynamics).

Imogen's priority takeoff metrics:
  - Speed at takeoff (penultimate → takeoff stride transition)
  - Ground contact time of the takeoff foot (ms)
  - Body angle / body in straight line at takeoff (deg deviation)
  - Takeoff foot angle to the mat (deg)
  - Speed and timing of arm drive (m/s, ms)
  - Speed and timing of free-knee drive (m/s, ms)

All angles in degrees, times in ms, speeds in m/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# numpy 2.0 renamed trapz → trapezoid; support both.
try:
    _trapz = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]


@dataclass
class TakeoffMetrics:
    """Biomechanical metrics extracted from the takeoff phase."""

    # ── Timing ────────────────────────────────────────────────────────────
    ground_contact_time_ms: float    # plant foot contact duration
    time_to_peak_force_ms: float     # time from contact to max GRF

    # ── Velocities at takeoff instant ─────────────────────────────────────
    horizontal_velocity_mps: float   # forward speed
    vertical_velocity_mps: float     # upward speed
    resultant_velocity_mps: float
    takeoff_angle_deg: float         # angle of CoM velocity vector from horizontal

    # ── Positions ─────────────────────────────────────────────────────────
    takeoff_distance_from_bar_cm: float | None  # horizontal distance to bar at liftoff
    com_height_at_takeoff_cm: float  # CoM height at last ground contact

    # ── Forces (estimated from inverse dynamics) ──────────────────────────
    peak_vertical_grf_bw: float      # peak GRF in bodyweights
    average_vertical_grf_bw: float
    braking_impulse_ns: float        # horizontal braking during plant
    propulsive_impulse_ns: float     # vertical propulsive impulse

    # ── Joint angles at takeoff instant ───────────────────────────────────
    knee_angle_at_takeoff_deg: float
    hip_angle_at_takeoff_deg: float
    ankle_angle_at_takeoff_deg: float
    trunk_lean_deg: float            # trunk angle from vertical

    # ── Body alignment (Imogen priority) ─────────────────────────────────
    # Maximum angular deviation (deg) of any intermediate joint (knee, hip,
    # shoulder) from the ankle-to-head reference line.  0° = perfectly straight.
    body_alignment_deviation_deg: float = 0.0

    # ── Foot-to-ground angle (Imogen priority) ────────────────────────────
    # Angle of the takeoff foot (heel→toe vector) relative to the horizontal
    # plane at the moment of foot strike.  Positive = toe up (dorsiflexed).
    foot_to_ground_angle_deg: float = 0.0

    # ── Arm drive (Imogen priority) ───────────────────────────────────────
    arm_drive_peak_speed_mps: float = 0.0      # peak wrist speed during ground contact
    arm_drive_peak_timing_ms: float = 0.0      # ms from contact start to peak arm speed

    # ── Free-knee drive (Imogen priority) ────────────────────────────────
    free_knee_drive_peak_speed_mps: float = 0.0   # peak upward knee speed
    free_knee_drive_peak_timing_ms: float = 0.0   # ms from contact start to peak


def estimate_grf_from_com(
    com_acceleration: np.ndarray,
    body_mass_kg: float,
) -> np.ndarray:
    """Estimate ground reaction force from CoM acceleration (Newton's 2nd law).

    F_GRF = m * (a_CoM + g)

    Args:
        com_acceleration: (T, 3) CoM acceleration in m/s^2.
        body_mass_kg: Athlete's body mass.

    Returns:
        (T, 3) estimated GRF in Newtons.
    """
    g = np.array([0.0, 9.81, 0.0])  # gravity vector (y-up)
    return body_mass_kg * (com_acceleration + g)


def compute_takeoff_angle(
    com_velocity_at_takeoff: np.ndarray,
) -> float:
    """Compute takeoff angle from the CoM velocity vector at last ground contact.

    Args:
        com_velocity_at_takeoff: (3,) velocity vector [vx, vy, vz].

    Returns:
        Takeoff angle in degrees from horizontal.
    """
    v_horizontal = np.sqrt(
        com_velocity_at_takeoff[0] ** 2 + com_velocity_at_takeoff[2] ** 2
    )
    v_vertical = com_velocity_at_takeoff[1]
    angle_rad = np.arctan2(v_vertical, v_horizontal)
    return float(np.degrees(angle_rad))


def predict_max_com_height(
    com_height_at_takeoff_m: float,
    vertical_velocity_mps: float,
) -> float:
    """Predict maximum CoM height from takeoff conditions (projectile motion).

    h_max = h_takeoff + v_y^2 / (2g)

    Args:
        com_height_at_takeoff_m: CoM height at takeoff in meters.
        vertical_velocity_mps: Vertical velocity at takeoff in m/s.

    Returns:
        Predicted peak CoM height in meters.
    """
    g = 9.81
    return com_height_at_takeoff_m + (vertical_velocity_mps ** 2) / (2 * g)


def compute_impulse(
    force: np.ndarray,
    fps: float,
    axis: int = 1,
) -> float:
    """Compute impulse (integral of force over time) along an axis.

    Args:
        force: (T, 3) force time series in Newtons.
        fps: Frame rate.
        axis: Which component (0=X, 1=Y vertical, 2=Z).

    Returns:
        Impulse in Newton-seconds.
    """
    dt = 1.0 / fps
    return float(_trapz(force[:, axis], dx=dt))


def compute_body_alignment_deviation(
    landmarks: np.ndarray,
    takeoff_side: str = "right",
) -> float:
    """Compute whole-body alignment deviation at the takeoff instant.

    Measures how far the body deviates from a straight ankle-to-head line
    (the ideal Imogen describes as "body in straight line at takeoff").
    Returns the maximum angular deviation (degrees) at any intermediate joint
    (knee, hip, shoulder) from the ankle–head reference line.

    Args:
        landmarks:    (33, 3) BlazePose 3D landmarks for one frame.
        takeoff_side: "left" or "right" — which is the plant/takeoff leg.

    Returns:
        Maximum body-alignment deviation in degrees (0° = perfectly straight).
    """
    pos = landmarks[:, :3]

    if takeoff_side == "left":
        ankle_idx, knee_idx, hip_idx, shoulder_idx = 27, 25, 23, 11
    else:
        ankle_idx, knee_idx, hip_idx, shoulder_idx = 28, 26, 24, 12

    head_idx = 0  # BlazePose nose landmark as head reference

    ankle = pos[ankle_idx]
    head = pos[head_idx]
    line_vec = head - ankle
    line_len_sq = float(np.dot(line_vec, line_vec))

    if line_len_sq < 1e-8:
        return 0.0

    max_deviation_deg = 0.0
    for joint_idx in [knee_idx, hip_idx, shoulder_idx]:
        joint = pos[joint_idx]
        to_joint = joint - ankle
        t = float(np.dot(to_joint, line_vec)) / line_len_sq
        projected = ankle + t * line_vec
        deviation_vec = joint - projected
        deviation_dist = float(np.linalg.norm(deviation_vec))

        # Express deviation as an angle from the reference line.
        projected_dist = abs(t) * float(np.sqrt(line_len_sq))
        angle_deg = float(np.degrees(np.arctan2(deviation_dist, projected_dist + 1e-8)))
        max_deviation_deg = max(max_deviation_deg, angle_deg)

    return max_deviation_deg


def compute_foot_to_ground_angle(
    ankle_pos: np.ndarray,
    toe_pos: np.ndarray,
) -> float:
    """Compute the takeoff foot's angle relative to the ground at foot strike.

    The foot vector runs from the ankle (heel reference) to the toe.
    The angle is measured from the horizontal (XZ) plane.

    Positive = toe elevated above heel (plantarflexed / toe-strike).
    Negative = heel elevated above toe (dorsiflexed / heel-strike).

    Args:
        ankle_pos: (3,) position of the ankle landmark.
        toe_pos:   (3,) position of the toe/ball-of-foot landmark.

    Returns:
        Foot-to-ground angle in degrees.
    """
    foot_vec = toe_pos - ankle_pos
    horizontal_mag = float(np.sqrt(foot_vec[0] ** 2 + foot_vec[2] ** 2))
    vertical = float(foot_vec[1])

    if horizontal_mag < 1e-8 and abs(vertical) < 1e-8:
        return 0.0

    return float(np.degrees(np.arctan2(vertical, horizontal_mag)))


def compute_arm_drive_metrics(
    wrist_positions: np.ndarray,
    fps: float,
) -> tuple[float, float]:
    """Compute peak arm-drive speed and its timing within the ground contact.

    Arm-drive speed is the magnitude of wrist velocity.  Imogen identified
    "speed and timing of arms driving" as a priority takeoff metric.

    Args:
        wrist_positions: (T, 3) wrist position trajectory during ground contact.
        fps:             Frame rate.

    Returns:
        (peak_speed_mps, timing_ms) — peak wrist speed and the time from
        the start of ground contact to that peak.
    """
    if len(wrist_positions) < 2:
        return 0.0, 0.0

    dt = 1.0 / fps
    velocity = np.gradient(wrist_positions, dt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)

    peak_frame = int(np.argmax(speed))
    return float(speed[peak_frame]), float(peak_frame / fps * 1000.0)


def compute_free_knee_drive_metrics(
    free_knee_positions: np.ndarray,
    fps: float,
) -> tuple[float, float]:
    """Compute peak free-knee upward drive speed and its timing.

    The free (non-takeoff) knee drives upward during the takeoff ground contact;
    Imogen identified "speed and timing of knee (non-takeoff leg) drive" as a
    priority metric.

    Only the upward (positive Y) component is used — adduction/lateral
    motion is excluded.

    Args:
        free_knee_positions: (T, 3) free-knee position during ground contact.
        fps:                 Frame rate.

    Returns:
        (peak_speed_mps, timing_ms) — peak upward knee speed and time from
        start of ground contact.
    """
    if len(free_knee_positions) < 2:
        return 0.0, 0.0

    dt = 1.0 / fps
    vy = np.gradient(free_knee_positions[:, 1], dt)  # vertical velocity
    upward_speed = np.maximum(vy, 0.0)               # only upward motion

    peak_frame = int(np.argmax(upward_speed))
    return float(upward_speed[peak_frame]), float(peak_frame / fps * 1000.0)
