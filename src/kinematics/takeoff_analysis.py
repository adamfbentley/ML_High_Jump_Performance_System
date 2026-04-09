"""Takeoff biomechanics analysis.

Extracts critical takeoff parameters: plant timing, takeoff angle,
vertical/horizontal velocity, ground contact time, and estimated
ground reaction force from CoM trajectory (inverse dynamics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TakeoffMetrics:
    """Biomechanical metrics extracted from the takeoff phase."""

    # Timing
    ground_contact_time_ms: float    # plant foot contact duration
    time_to_peak_force_ms: float     # time from contact to max GRF

    # Velocities at takeoff instant
    horizontal_velocity_mps: float   # forward speed
    vertical_velocity_mps: float     # upward speed
    resultant_velocity_mps: float
    takeoff_angle_deg: float         # angle of CoM velocity vector from horizontal

    # Positions
    takeoff_distance_from_bar_cm: float | None  # horizontal distance to bar at liftoff
    com_height_at_takeoff_cm: float  # CoM height at last ground contact

    # Forces (estimated from inverse dynamics)
    peak_vertical_grf_bw: float      # peak GRF in bodyweights
    average_vertical_grf_bw: float
    braking_impulse_ns: float        # horizontal braking during plant
    propulsive_impulse_ns: float     # vertical propulsive impulse

    # Joint angles at takeoff instant
    knee_angle_at_takeoff_deg: float
    hip_angle_at_takeoff_deg: float
    ankle_angle_at_takeoff_deg: float
    trunk_lean_deg: float            # trunk angle from vertical


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
    return float(np.trapz(force[:, axis], dx=dt))
