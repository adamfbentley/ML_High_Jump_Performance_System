"""Flight phase analysis: bar clearance geometry and CoM parabola.

Analyzes the airborne phase — CoM trajectory relative to the bar,
body configuration during clearance, and landing preparation.

Imogen's key flight metric:
  "The main thing I would work on in the flight is the amount of time
   extending vertically off the ground with knee driving up before
   dropping knee and head to go over the bar."

Sub-phases (frame indices are relative to the start of the flight phase):
  1. Vertical extension  — knee/head up; free-knee driving upward.
  2. Arch clearance      — free-knee reverses (starts descending), head drops,
                           hips push upward over the bar.

The transition between the two sub-phases is detected as the free-knee
angular-velocity reversal (see detect_arch_transition).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlightMetrics:
    """Biomechanical metrics from the flight phase."""

    # ── CoM trajectory ────────────────────────────────────────────────────
    peak_com_height_m: float           # maximum CoM height achieved
    com_height_above_bar_cm: float     # clearance margin (positive = cleared)
    time_to_peak_ms: float             # time from takeoff to peak height
    total_flight_time_ms: float

    # ── Bar clearance ─────────────────────────────────────────────────────
    bar_height_m: float
    min_body_clearance_cm: float       # closest body point to bar during clearance

    # ── Body configuration at bar ─────────────────────────────────────────
    hip_angle_at_bar_deg: float        # arch angle (extension = Fosbury arch)
    knee_angle_at_bar_deg: float
    head_clearance_cm: float
    trail_leg_clearance_cm: float      # often the limiting factor

    # ── Angular momentum ─────────────────────────────────────────────────
    estimated_angular_momentum_h: float | None  # about transverse axis
    estimated_angular_momentum_l: float | None  # about longitudinal axis

    # ── Flight sub-phase timing (Imogen priority) ─────────────────────────
    # Frame index within the flight phase (0 = first airborne frame) at which
    # the free knee stops rising and begins to drop.  None if not detected.
    arch_transition_frame: int | None = None

    # Time from takeoff (ms) to the arch transition.  None if not detected.
    time_to_arch_transition_ms: float | None = None

    # Duration of the vertical extension sub-phase (takeoff → arch transition).
    # Equal to time_to_arch_transition_ms when transition is detected.
    vertical_extension_time_ms: float | None = None


def detect_arch_transition(
    free_knee_positions: np.ndarray,
    fps: float,
    min_extension_frames: int = 5,
) -> int | None:
    """Detect the arch-transition frame within the flight phase.

    The transition is the Fosbury Flop "switch" described by Imogen:
    the free knee stops being driven upward (positive vertical velocity)
    and begins to drop (negative vertical velocity) as the athlete
    moves from vertical extension into arch clearance over the bar.

    The detection uses the first zero-crossing of the free-knee vertical
    velocity after a minimum extension period.

    Args:
        free_knee_positions: (T, 3) free-knee (non-takeoff leg) position during
                             the flight phase.  Frame indices are relative to
                             the start of the flight phase.
        fps:                 Frame rate (Hz).
        min_extension_frames: Minimum number of frames that must elapse before
                             a valid transition is accepted (avoids noise at
                             the start of flight).

    Returns:
        Frame index (within the flight phase) of the transition, or None if
        not detected (e.g. too short a flight phase or monotonically rising knee).
    """
    if len(free_knee_positions) < min_extension_frames + 2:
        return None

    knee_y = free_knee_positions[:, 1]   # vertical component (Y-up)
    dt = 1.0 / fps
    knee_vy = np.gradient(knee_y, dt)

    # Find first frame after the minimum extension window where vertical
    # velocity transitions from non-negative to negative.
    for i in range(min_extension_frames, len(knee_vy) - 1):
        if knee_vy[i] >= 0.0 and knee_vy[i + 1] < 0.0:
            return i

    return None


def fit_com_parabola(
    com_trajectory: np.ndarray,
    fps: float,
) -> dict[str, float]:
    """Fit a parabolic trajectory to the flight-phase CoM.

    In ideal projectile motion (no air resistance):
        y(t) = y0 + vy0*t - 0.5*g*t^2

    Args:
        com_trajectory: (T, 3) CoM positions during flight.
        fps: Frame rate.

    Returns:
        Dict with fitted parameters and goodness-of-fit.
    """
    n = len(com_trajectory)
    t = np.arange(n) / fps
    y = com_trajectory[:, 1]  # vertical component

    # Fit quadratic: y = a*t^2 + b*t + c
    coeffs = np.polyfit(t, y, 2)
    a, b, c = coeffs

    # Physics: a ≈ -g/2, b ≈ vy0, c ≈ y0
    g_estimated = -2 * a
    vy0_estimated = b
    y0_estimated = c

    y_fitted = np.polyval(coeffs, t)
    residuals = y - y_fitted
    r_squared = 1.0 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Peak height from fit
    t_peak = -b / (2 * a) if a != 0 else 0
    y_peak = c - b ** 2 / (4 * a) if a != 0 else c

    return {
        "g_estimated_mps2": float(g_estimated),
        "vy0_mps": float(vy0_estimated),
        "y0_m": float(y0_estimated),
        "t_peak_s": float(t_peak),
        "y_peak_m": float(y_peak),
        "r_squared": float(r_squared),
        "residual_std_m": float(np.std(residuals)),
    }


def compute_clearance_profile(
    body_landmarks_3d_seq: np.ndarray,
    bar_height_m: float,
    bar_position_x: float,
    fps: float,
) -> dict[str, np.ndarray]:
    """Compute how each body part clears the bar over time.

    Args:
        body_landmarks_3d_seq: (T, 33, 3) landmarks during flight.
        bar_height_m: Bar height in meters.
        bar_position_x: Bar x-coordinate.
        fps: Frame rate.

    Returns:
        Dict mapping landmark names to (T,) clearance arrays (positive = above bar).
    """
    key_landmarks = {
        "head": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }

    clearances = {}
    for name, idx in key_landmarks.items():
        heights = body_landmarks_3d_seq[:, idx, 1]
        clearances[name] = heights - bar_height_m

    return clearances
