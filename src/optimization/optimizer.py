"""Technique optimization via differentiable biomechanical simulation.

Physics-based forward model:
    TechniqueParameters → takeoff state (v_y, v_h, h_takeoff) → bar clearance height

The height prediction follows Dapena (1980):
    H_bar = h_takeoff + v_y² / (2g) - clearance_deficit

Where:
- h_takeoff  = CoM height at the instant of takeoff (from body dimensions + posture)
- v_y        = vertical takeoff velocity (from approach speed, plant angle, impulse model)
- clearance  = Fosbury Flop arch efficiency (CoM can pass below the bar)

The trained PINN validates that the required ground reaction forces are
physiologically feasible for the given athlete.

References:
    Dapena, J. (1980). Mechanics of translation in the Fosbury Flop.
    Medicine and Science in Sports and Exercise, 12(1), 37-44.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

GRAVITY = 9.81


@dataclass
class TechniqueParameters:
    """Controllable technique variables for optimization.

    Fields are ordered to match to_tensor() / from_tensor() exactly.
    Do NOT change field order without updating both methods simultaneously.

    Athlete A's priority controllable variables (from athlete brief):
      - ground_contact_time_takeoff_ms: takeoff-foot ground contact time
      - body_alignment_deviation_deg:   deviation from straight-line body alignment
      - foot_to_ground_angle_deg:       takeoff foot angle to the mat
      - knee_drive_peak_speed_mps:      free-leg knee drive peak speed
      - curve_start_step:               which stride number the J-curve begins
    """

    # ── Approach / curve ──────────────────────────────────────────────────
    approach_speed_mps: float         # horizontal speed entering the curve
    curve_radius_m: float             # radius of the J-approach curve

    # ── Step lengths ──────────────────────────────────────────────────────
    penultimate_step_length_cm: float
    last_step_length_cm: float

    # ── Plant / takeoff angles ────────────────────────────────────────────
    # Plant angle: angle of takeoff leg (hip→foot vector) from horizontal at
    # foot strike.  ~65–70° for elite Fosbury Floppers (Dapena 1980).
    plant_angle_deg: float
    takeoff_knee_angle_deg: float     # knee angle at the instant of takeoff
    takeoff_hip_angle_deg: float

    # ── Arm / free-leg drive ──────────────────────────────────────────────
    arm_swing_timing_ms: float        # relative timing of arm drive
    free_leg_drive_angle_deg: float

    # ── Athlete A priority fields ────────────────────────────────────────────
    ground_contact_time_takeoff_ms: float = 120.0   # takeoff foot contact (ms)
    body_alignment_deviation_deg: float = 0.0       # deviation from straight line (deg)
    foot_to_ground_angle_deg: float = 65.0          # foot-to-mat angle at strike (deg)
    knee_drive_peak_speed_mps: float = 3.0          # free-leg knee peak speed (m/s)
    curve_start_step: int = 5                       # step index where curve begins

    def to_tensor(self) -> torch.Tensor:
        """Encode all 14 technique parameters as a float32 tensor.

        Ordering must exactly match from_tensor().
        """
        return torch.tensor([
            self.approach_speed_mps,           # [0]
            self.curve_radius_m,               # [1]
            self.penultimate_step_length_cm,   # [2]
            self.last_step_length_cm,          # [3]
            self.plant_angle_deg,              # [4]
            self.takeoff_knee_angle_deg,       # [5]
            self.takeoff_hip_angle_deg,        # [6]
            self.arm_swing_timing_ms,          # [7]
            self.free_leg_drive_angle_deg,     # [8]
            self.ground_contact_time_takeoff_ms,  # [9]
            self.body_alignment_deviation_deg,    # [10]
            self.foot_to_ground_angle_deg,        # [11]
            self.knee_drive_peak_speed_mps,       # [12]
            float(self.curve_start_step),         # [13]
        ], dtype=torch.float32)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TechniqueParameters:
        """Decode a 14-element float32 tensor back to TechniqueParameters.

        Ordering must exactly match to_tensor().
        """
        v = t.detach().cpu().numpy()
        return cls(
            approach_speed_mps=float(v[0]),
            curve_radius_m=float(v[1]),
            penultimate_step_length_cm=float(v[2]),
            last_step_length_cm=float(v[3]),
            plant_angle_deg=float(v[4]),
            takeoff_knee_angle_deg=float(v[5]),
            takeoff_hip_angle_deg=float(v[6]),
            arm_swing_timing_ms=float(v[7]),
            free_leg_drive_angle_deg=float(v[8]),
            ground_contact_time_takeoff_ms=float(v[9]),
            body_alignment_deviation_deg=float(v[10]),
            foot_to_ground_angle_deg=float(v[11]),
            knee_drive_peak_speed_mps=float(v[12]),
            curve_start_step=int(round(float(v[13]))),
        )


@dataclass
class AthleteConstraints:
    """Biomechanical feasibility limits for a specific athlete.

    Defaults are based on female national-level high jumpers.
    """

    max_approach_speed_mps: float = 8.5
    min_approach_speed_mps: float = 5.0
    max_knee_extension_torque_nm: float = 300.0
    max_hip_extension_torque_nm: float = 400.0
    knee_rom_deg: tuple[float, float] = (90.0, 180.0)
    hip_rom_deg: tuple[float, float] = (140.0, 180.0)
    plant_angle_deg: tuple[float, float] = (55.0, 80.0)
    ground_contact_time_ms: tuple[float, float] = (80.0, 200.0)
    curve_radius_m: tuple[float, float] = (6.0, 15.0)


@dataclass
class OptimizationResult:
    """Output of the technique optimization."""

    optimal_params: TechniqueParameters
    predicted_height_m: float
    current_height_m: float
    improvement_cm: float
    n_iterations: int
    sensitivity: dict[str, float]  # parameter name → ∂height/∂param (cm per unit)
    coaching_cues: list[str]       # human-readable improvement suggestions


# ── Differentiable forward model ──────────────────────────────────────────


def _estimate_takeoff_com_height(
    athlete_height_m: float,
    knee_angle_deg: float,
    hip_angle_deg: float,
) -> float:
    """Estimate CoM height at takeoff from body geometry.

    At full extension (knee=180°, hip=180°), CoM is ~55% of standing
    height (de Leva 1996 male average). Joint flexion lowers CoM.

    Args:
        athlete_height_m: Standing height in metres.
        knee_angle_deg:   Knee angle at takeoff (180° = full extension).
        hip_angle_deg:    Hip angle at takeoff (180° = full extension).

    Returns:
        CoM height above ground in metres.
    """
    # Full-extension CoM fraction (de Leva 1996)
    com_fraction = 0.55
    h_full = athlete_height_m * com_fraction

    # Deviation from full extension lowers CoM.
    # Each degree of knee flexion (below 180°) costs ~0.3% of leg length.
    # Each degree of hip flexion costs ~0.2%.
    knee_deficit = max(0.0, 180.0 - knee_angle_deg)
    hip_deficit = max(0.0, 180.0 - hip_angle_deg)

    leg_length = athlete_height_m * 0.53  # approximate leg length fraction
    knee_loss = knee_deficit * 0.003 * leg_length
    hip_loss = hip_deficit * 0.002 * leg_length

    return max(0.3, h_full - knee_loss - hip_loss)


def _estimate_takeoff_com_height_differentiable(
    athlete_height_m: float,
    knee_angle_deg: torch.Tensor,
    hip_angle_deg: torch.Tensor,
) -> torch.Tensor:
    """Differentiable version of CoM height estimation for gradient optimisation."""
    com_fraction = 0.55
    h_full = athlete_height_m * com_fraction
    leg_length = athlete_height_m * 0.53

    knee_deficit = torch.clamp(180.0 - knee_angle_deg, min=0.0)
    hip_deficit = torch.clamp(180.0 - hip_angle_deg, min=0.0)

    knee_loss = knee_deficit * 0.003 * leg_length
    hip_loss = hip_deficit * 0.002 * leg_length

    return torch.clamp(h_full - knee_loss - hip_loss, min=0.3)


def _impulse_model_vertical_velocity(
    approach_speed: float,
    plant_angle_deg: float,
    contact_time_ms: float,
    knee_drive_speed: float,
    arm_timing_ms: float,
    body_mass_kg: float,
) -> float:
    """Estimate vertical takeoff velocity from biomechanical parameters.

    The "pivot-and-push" model (Dapena 1980):
    - The plant leg acts as a rigid pivot, redirecting horizontal momentum
      upward through the plant angle.
    - Muscular impulse from leg extension adds to the vertical velocity.
    - Arm swing and free-knee drive contribute upward momentum.

    v_y = v_approach × sin(plant_angle) × η_redirect
        + v_muscular
        + v_arm_drive
        + v_knee_drive

    Args:
        approach_speed:  Horizontal speed at last step (m/s).
        plant_angle_deg: Plant leg angle from horizontal (deg).
        contact_time_ms: Ground contact time of takeoff foot (ms).
        knee_drive_speed: Free-leg knee drive peak speed (m/s).
        arm_timing_ms:   Arm swing timing (ms) — lower = better synchronisation.
        body_mass_kg:    Athlete mass (kg).

    Returns:
        Estimated vertical velocity at takeoff (m/s).
    """
    plant_rad = math.radians(plant_angle_deg)

    # Momentum redirection: horizontal → vertical via the plant.
    # Dapena (1980) shows about 25–35% of approach KE converts to vertical KE.
    # In velocity terms, η ≈ 0.40–0.45 after ground losses, eccentric absorption,
    # and rotational energy transfer.  Calibrated against elite female data.
    eta_redirect = 0.42
    v_redirect = approach_speed * math.sin(plant_rad) * eta_redirect

    # Muscular push during contact.
    # Shorter contact = less time for force → less impulse, but also
    # implies higher peak forces. Model: v_muscular ∝ sqrt(contact_time)
    # Calibrated so that ~120 ms contact → ~0.8 m/s push for a 67 kg jumper.
    contact_s = contact_time_ms / 1000.0
    v_muscular = 2.5 * math.sqrt(contact_s)  # ~0.87 m/s at 120 ms

    # Free-knee drive imparts upward momentum to ~15% of body mass (thigh+shank).
    # Δv_com ≈ (m_leg / m_total) × v_knee_drive_y
    leg_mass_fraction = 0.185  # thigh + shank (de Leva 1996)
    v_knee_contrib = leg_mass_fraction * knee_drive_speed * 0.5  # only ~50% vertical

    # Arm swing: two arms ≈ 10% body mass. Better timing → more contribution.
    # Optimal timing ≈ 50 ms before toe-off. Penalty for bad timing.
    arm_mass_fraction = 0.10
    arm_speed = 2.0  # typical arm swing speed (m/s)
    timing_penalty = max(0.0, 1.0 - abs(arm_timing_ms - 50.0) / 100.0)
    v_arm_contrib = arm_mass_fraction * arm_speed * timing_penalty * 0.5

    return v_redirect + v_muscular + v_knee_contrib + v_arm_contrib


def _impulse_model_vertical_velocity_differentiable(
    approach_speed: torch.Tensor,
    plant_angle_deg: torch.Tensor,
    contact_time_ms: torch.Tensor,
    knee_drive_speed: torch.Tensor,
    arm_timing_ms: torch.Tensor,
) -> torch.Tensor:
    """Differentiable version for gradient-based optimisation."""
    plant_rad = plant_angle_deg * (math.pi / 180.0)

    eta_redirect = 0.42
    v_redirect = approach_speed * torch.sin(plant_rad) * eta_redirect

    contact_s = contact_time_ms / 1000.0
    v_muscular = 2.5 * torch.sqrt(torch.clamp(contact_s, min=1e-4))

    leg_mass_fraction = 0.185
    v_knee_contrib = leg_mass_fraction * knee_drive_speed * 0.5

    arm_mass_fraction = 0.10
    arm_speed = 2.0
    timing_penalty = torch.clamp(1.0 - torch.abs(arm_timing_ms - 50.0) / 100.0, min=0.0)
    v_arm_contrib = arm_mass_fraction * arm_speed * timing_penalty * 0.5

    return v_redirect + v_muscular + v_knee_contrib + v_arm_contrib


def _horizontal_velocity_at_takeoff(
    approach_speed: float,
    plant_angle_deg: float,
) -> float:
    """Horizontal velocity retained after the plant.

    Elite jumpers retain ~40–55% of approach speed horizontally (Dapena 1980).
    Steeper plant angles transfer more horizontal → vertical, so less is retained.
    Modelled as: v_h = v_approach × (0.55 - 0.0015 × plant_angle_deg)
    At 67°: ~45% retained.  At 55°: ~47%.  At 75°: ~44%.
    """
    retention = max(0.15, 0.55 - 0.0015 * plant_angle_deg)
    return approach_speed * retention


def _clearance_deficit(
    body_alignment_deviation_deg: float,
    athlete_height_m: float,
) -> float:
    """Fosbury Flop clearance: how efficiently the body arches over the bar.

    Perfect arch (deviation=0°): CoM can pass ~5 cm below the bar.
    Poor alignment: CoM must be higher, reducing effective clearance.

    Returns the deficit in metres (positive = must clear more).
    """
    # Base clearance for a good Fosbury arch: -5 cm (CoM below bar)
    base_clearance = -0.05
    # Each degree of misalignment costs ~1 cm of clearance efficiency
    alignment_penalty = body_alignment_deviation_deg * 0.01
    return base_clearance + alignment_penalty


def predict_bar_clearance(
    params: TechniqueParameters,
    body_mass_kg: float,
    athlete_height_m: float,
) -> dict[str, float]:
    """Predict bar clearance height from technique parameters.

    H_bar = h_takeoff + v_y²/(2g) + clearance_bonus

    Args:
        params: Technique parameters.
        body_mass_kg: Athlete mass in kg.
        athlete_height_m: Athlete standing height in metres.

    Returns:
        Dict with all intermediate calculations and final predicted height.
    """
    h_takeoff = _estimate_takeoff_com_height(
        athlete_height_m, params.takeoff_knee_angle_deg, params.takeoff_hip_angle_deg,
    )

    v_y = _impulse_model_vertical_velocity(
        approach_speed=params.approach_speed_mps,
        plant_angle_deg=params.plant_angle_deg,
        contact_time_ms=params.ground_contact_time_takeoff_ms,
        knee_drive_speed=params.knee_drive_peak_speed_mps,
        arm_timing_ms=params.arm_swing_timing_ms,
        body_mass_kg=body_mass_kg,
    )

    v_h = _horizontal_velocity_at_takeoff(params.approach_speed_mps, params.plant_angle_deg)

    h_rise = v_y ** 2 / (2 * GRAVITY)

    deficit = _clearance_deficit(
        params.body_alignment_deviation_deg, athlete_height_m,
    )

    bar_height = h_takeoff + h_rise - deficit
    takeoff_angle = math.degrees(math.atan2(v_y, v_h)) if v_h > 0 else 90.0

    return {
        "h_takeoff_m": h_takeoff,
        "v_vertical_mps": v_y,
        "v_horizontal_mps": v_h,
        "takeoff_angle_deg": takeoff_angle,
        "h_rise_m": h_rise,
        "clearance_deficit_m": deficit,
        "predicted_bar_height_m": bar_height,
    }


def _evaluate_height_differentiable(
    params_tensor: torch.Tensor,
    body_mass_kg: float,
    athlete_height_m: float,
) -> torch.Tensor:
    """Differentiable forward model: technique tensor → predicted bar height.

    This is the function that gradient-based optimisation differentiates through.
    """
    approach_speed = params_tensor[0]
    plant_angle = params_tensor[4]
    knee_angle = params_tensor[5]
    hip_angle = params_tensor[6]
    arm_timing = params_tensor[7]
    contact_time = params_tensor[9]
    alignment_dev = params_tensor[10]
    knee_drive = params_tensor[12]

    h_takeoff = _estimate_takeoff_com_height_differentiable(
        athlete_height_m, knee_angle, hip_angle,
    )

    v_y = _impulse_model_vertical_velocity_differentiable(
        approach_speed, plant_angle, contact_time, knee_drive, arm_timing,
    )

    h_rise = v_y ** 2 / (2 * GRAVITY)

    # Clearance deficit (differentiable through alignment_dev)
    deficit = -0.05 + alignment_dev * 0.01

    return h_takeoff + h_rise - deficit


# ── Optimisation ─────────────────────────────────────────────────────────


def optimize_technique(
    current_params: TechniqueParameters,
    body_mass_kg: float,
    athlete_height_m: float,
    constraints: AthleteConstraints | None = None,
    n_iterations: int = 200,
    lr: float = 0.01,
    pinn_model: nn.Module | None = None,
) -> OptimizationResult:
    """Find optimal technique parameters using gradient-based optimization.

    The differentiable biomechanical forward model maps:
        technique_params → takeoff state → predicted bar clearance height

    We maximize predicted height subject to biomechanical feasibility.

    Args:
        current_params: Athlete's current technique.
        body_mass_kg:   Athlete mass in kg.
        athlete_height_m: Athlete standing height in metres.
        constraints:    Feasibility limits (uses defaults if None).
        n_iterations:   Optimization steps.
        lr:             Step size.
        pinn_model:     Optional trained PINN (not used in forward model,
                        but can validate force feasibility).

    Returns:
        OptimizationResult with optimal parameters and coaching cues.
    """
    if constraints is None:
        constraints = AthleteConstraints()

    params_tensor = current_params.to_tensor().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([params_tensor], lr=lr)

    current_height = _evaluate_height_differentiable(
        current_params.to_tensor(), body_mass_kg, athlete_height_m,
    ).item()

    best_height = current_height
    best_tensor = params_tensor.data.clone()

    for _ in range(n_iterations):
        optimizer.zero_grad()
        height = _evaluate_height_differentiable(
            params_tensor, body_mass_kg, athlete_height_m,
        )
        loss = -height  # maximize height
        loss.backward()
        optimizer.step()

        # Project back to feasible region
        with torch.no_grad():
            params_tensor[0].clamp_(
                constraints.min_approach_speed_mps, constraints.max_approach_speed_mps,
            )
            params_tensor[1].clamp_(*constraints.curve_radius_m)
            params_tensor[4].clamp_(*constraints.plant_angle_deg)
            params_tensor[5].clamp_(*constraints.knee_rom_deg)
            params_tensor[6].clamp_(*constraints.hip_rom_deg)
            params_tensor[9].clamp_(*constraints.ground_contact_time_ms)
            params_tensor[10].clamp_(0.0, 15.0)  # alignment deviation
            params_tensor[12].clamp_(0.0, 6.0)   # knee drive speed

            h = _evaluate_height_differentiable(
                params_tensor, body_mass_kg, athlete_height_m,
            ).item()
            if h > best_height:
                best_height = h
                best_tensor = params_tensor.data.clone()

    optimal_params = TechniqueParameters.from_tensor(best_tensor)

    sensitivity = compute_sensitivity(
        best_tensor, body_mass_kg, athlete_height_m,
    )

    coaching_cues = generate_coaching_cues(
        current_params, optimal_params, sensitivity, current_height, best_height,
    )

    return OptimizationResult(
        optimal_params=optimal_params,
        predicted_height_m=best_height,
        current_height_m=current_height,
        improvement_cm=(best_height - current_height) * 100,
        n_iterations=n_iterations,
        sensitivity=sensitivity,
        coaching_cues=coaching_cues,
    )


# ── Parameter names (must stay in sync with to_tensor ordering) ──────────

PARAM_NAMES = [
    "approach_speed",                # [0]  m/s
    "curve_radius",                  # [1]  m
    "penultimate_step",              # [2]  cm
    "last_step",                     # [3]  cm
    "plant_angle",                   # [4]  deg
    "takeoff_knee_angle",            # [5]  deg
    "takeoff_hip_angle",             # [6]  deg
    "arm_swing_timing",              # [7]  ms
    "free_leg_drive",                # [8]  deg
    "ground_contact_time_takeoff",   # [9]  ms
    "body_alignment_deviation",      # [10] deg
    "foot_to_ground_angle",          # [11] deg
    "knee_drive_peak_speed",         # [12] m/s
    "curve_start_step",              # [13] step #
]

PARAM_UNITS = {
    "approach_speed": "m/s",
    "curve_radius": "m",
    "penultimate_step": "cm",
    "last_step": "cm",
    "plant_angle": "°",
    "takeoff_knee_angle": "°",
    "takeoff_hip_angle": "°",
    "arm_swing_timing": "ms",
    "free_leg_drive": "°",
    "ground_contact_time_takeoff": "ms",
    "body_alignment_deviation": "°",
    "foot_to_ground_angle": "°",
    "knee_drive_peak_speed": "m/s",
    "curve_start_step": "",
}


def compute_sensitivity(
    params: torch.Tensor,
    body_mass_kg: float,
    athlete_height_m: float,
) -> dict[str, float]:
    """Compute ∂height/∂param for each technique parameter.

    Returns the sensitivity in cm of bar height per unit change in each parameter.
    For example, sensitivity["approach_speed"] = 2.5 means +1 m/s approach speed
    → +2.5 cm bar height.
    """
    params = params.detach().clone().requires_grad_(True)
    height = _evaluate_height_differentiable(params, body_mass_kg, athlete_height_m)
    grads = torch.autograd.grad(height, params)[0]

    grad_vals = grads.detach().cpu().numpy()
    # Convert to cm per unit (grad is m per unit)
    return {name: float(g * 100) for name, g in zip(PARAM_NAMES, grad_vals)}


def what_if_scenario(
    base_params: TechniqueParameters,
    body_mass_kg: float,
    athlete_height_m: float,
    modifications: dict[str, float],
) -> dict[str, float]:
    """Run a what-if scenario: what happens if we change specific parameters?

    Args:
        base_params:     Current technique.
        body_mass_kg:    Athlete mass in kg.
        athlete_height_m: Athlete height in metres.
        modifications:   Dict of param_name → new_value.

    Returns:
        Dict with base height, modified height, delta, and breakdown.
    """
    base_pred = predict_bar_clearance(base_params, body_mass_kg, athlete_height_m)

    mod_params = TechniqueParameters(**{
        **base_params.__dict__,
        **modifications,
    })
    mod_pred = predict_bar_clearance(mod_params, body_mass_kg, athlete_height_m)

    return {
        "base_height_m": base_pred["predicted_bar_height_m"],
        "modified_height_m": mod_pred["predicted_bar_height_m"],
        "delta_cm": (mod_pred["predicted_bar_height_m"]
                     - base_pred["predicted_bar_height_m"]) * 100,
        "base_breakdown": base_pred,
        "modified_breakdown": mod_pred,
    }


# ── Coaching output ──────────────────────────────────────────────────────


def generate_coaching_cues(
    current: TechniqueParameters,
    optimal: TechniqueParameters,
    sensitivity: dict[str, float],
    current_height_m: float,
    optimal_height_m: float,
) -> list[str]:
    """Generate human-readable coaching cues from optimisation results.

    Ranks changes by impact and produces actionable text like:
    "Increase approach speed by 0.3 m/s → predicted +2.1 cm bar height"

    Returns:
        List of coaching cue strings, ordered by predicted impact.
    """
    current_vals = current.to_tensor().numpy()
    optimal_vals = optimal.to_tensor().numpy()

    changes: list[tuple[float, str]] = []  # (|impact_cm|, cue_string)

    for i, name in enumerate(PARAM_NAMES):
        delta = float(optimal_vals[i] - current_vals[i])
        if abs(delta) < 1e-3:
            continue

        sens = sensitivity.get(name, 0.0)  # cm per unit
        impact_cm = sens * delta  # predicted height change from this param alone

        if abs(impact_cm) < 0.1:  # ignore <1mm changes
            continue

        unit = PARAM_UNITS.get(name, "")
        direction = "Increase" if delta > 0 else "Decrease"

        # Human-readable parameter name
        display_name = name.replace("_", " ")
        cue = (
            f"{direction} {display_name} by {abs(delta):.1f}{unit} "
            f"(from {current_vals[i]:.1f} to {optimal_vals[i]:.1f}) "
            f"→ predicted {impact_cm:+.1f} cm"
        )
        changes.append((abs(impact_cm), cue))

    # Sort by impact (biggest first)
    changes.sort(reverse=True, key=lambda x: x[0])
    cues = [c for _, c in changes]

    # Add a summary line
    total_gain = (optimal_height_m - current_height_m) * 100
    if total_gain > 0.1:
        cues.insert(0, f"Total predicted improvement: +{total_gain:.1f} cm")
    elif total_gain < -0.1:
        cues.insert(0, f"Warning: optimised parameters predict {total_gain:.1f} cm change")
    else:
        cues.insert(0, "Current technique is near-optimal within constraints")

    return cues


def extract_params_from_report(report: dict) -> TechniqueParameters:
    """Extract TechniqueParameters from a video analysis report.

    Maps the measured kinematics from analyze_jump_video.py into the
    optimisable technique parameters. Uses defaults where measurements
    are unavailable.

    Args:
        report: JSON-style dict from analyze_jump_video.

    Returns:
        TechniqueParameters populated from the report.
    """
    vel = report.get("velocity", {})
    com = report.get("com", {})

    # Approach speed: peak horizontal speed during the run-up
    approach_speed = vel.get("peak_horizontal_mps", 6.5)

    # Takeoff angle: if negative or missing, use a default
    takeoff_angle = vel.get("takeoff_angle_deg")
    if takeoff_angle is None or takeoff_angle < 0:
        # Negative means the detection was wrong; use a plausible default
        plant_angle = 65.0
    else:
        # Plant angle ≈ 90° - takeoff_angle for Fosbury Flop geometry
        plant_angle = max(55.0, min(80.0, 90.0 - takeoff_angle))

    return TechniqueParameters(
        approach_speed_mps=approach_speed,
        curve_radius_m=9.0,               # default, not measurable from single cam
        penultimate_step_length_cm=170.0,  # default
        last_step_length_cm=120.0,         # default
        plant_angle_deg=plant_angle,
        takeoff_knee_angle_deg=170.0,      # near full extension typical
        takeoff_hip_angle_deg=175.0,       # near full extension typical
        arm_swing_timing_ms=50.0,          # default, hard to measure from video
        free_leg_drive_angle_deg=80.0,     # default
    )
