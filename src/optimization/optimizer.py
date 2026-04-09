"""Technique optimization via differentiable simulation.

Uses the trained PINN as a differentiable physics simulator to:
1. Find the optimal technique parameters for a given athlete
2. Perform sensitivity analysis (which parameters matter most)
3. Run what-if scenarios (what if I change X?)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TechniqueParameters:
    """Controllable technique variables for optimization."""

    approach_speed_mps: float        # horizontal speed entering the curve
    curve_radius_m: float            # radius of the J-approach curve
    penultimate_step_length_cm: float
    last_step_length_cm: float
    plant_angle_deg: float           # angle of takeoff foot at plant
    takeoff_knee_angle_deg: float    # knee angle at the instant of takeoff
    takeoff_hip_angle_deg: float
    arm_swing_timing_ms: float       # relative timing of arm drive
    free_leg_drive_angle_deg: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.approach_speed_mps,
            self.curve_radius_m,
            self.penultimate_step_length_cm,
            self.last_step_length_cm,
            self.plant_angle_deg,
            self.takeoff_knee_angle_deg,
            self.takeoff_hip_angle_deg,
            self.arm_swing_timing_ms,
            self.free_leg_drive_angle_deg,
        ], dtype=torch.float32)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TechniqueParameters:
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
        )


@dataclass
class AthleteConstraints:
    """Biomechanical feasibility limits for a specific athlete."""

    max_approach_speed_mps: float = 8.5
    min_approach_speed_mps: float = 5.0
    max_knee_extension_torque_nm: float = 300.0
    max_hip_extension_torque_nm: float = 400.0
    knee_rom_deg: tuple[float, float] = (0.0, 160.0)
    hip_rom_deg: tuple[float, float] = (-20.0, 130.0)


@dataclass
class OptimizationResult:
    """Output of the technique optimization."""

    optimal_params: TechniqueParameters
    predicted_height_m: float
    improvement_over_current_cm: float
    n_iterations: int
    sensitivity: dict[str, float]  # parameter name → impact score


def optimize_technique(
    pinn_model: nn.Module,
    current_params: TechniqueParameters,
    anthropometrics: torch.Tensor,
    constraints: AthleteConstraints,
    n_iterations: int = 200,
    lr: float = 0.01,
) -> OptimizationResult:
    """Find optimal technique parameters using gradient-based optimization.

    The PINN serves as a differentiable simulator:
        technique_params → PINN → predicted_jump_height

    We maximize predicted height subject to biomechanical feasibility.

    Args:
        pinn_model: Trained PINN that maps params → height.
        current_params: Athlete's current technique.
        anthropometrics: Fixed athlete body parameters.
        constraints: Feasibility limits.
        n_iterations: Optimization steps.
        lr: Step size.

    Returns:
        OptimizationResult with optimal parameters and gain analysis.
    """
    params_tensor = current_params.to_tensor().requires_grad_(True)
    optimizer = torch.optim.Adam([params_tensor], lr=lr)

    current_height = _evaluate_height(pinn_model, current_params.to_tensor(), anthropometrics)

    for _ in range(n_iterations):
        optimizer.zero_grad()
        height = _evaluate_height(pinn_model, params_tensor, anthropometrics)
        loss = -height  # maximize height
        loss.backward()
        optimizer.step()

        # Project back to feasible region
        with torch.no_grad():
            params_tensor[0].clamp_(constraints.min_approach_speed_mps, constraints.max_approach_speed_mps)
            params_tensor[5].clamp_(*constraints.knee_rom_deg)
            params_tensor[6].clamp_(*constraints.hip_rom_deg)

    optimal_params = TechniqueParameters.from_tensor(params_tensor)
    optimal_height = _evaluate_height(pinn_model, params_tensor, anthropometrics).item()

    sensitivity = compute_sensitivity(pinn_model, params_tensor, anthropometrics)

    return OptimizationResult(
        optimal_params=optimal_params,
        predicted_height_m=optimal_height,
        improvement_over_current_cm=(optimal_height - current_height.item()) * 100,
        n_iterations=n_iterations,
        sensitivity=sensitivity,
    )


def compute_sensitivity(
    pinn_model: nn.Module,
    params: torch.Tensor,
    anthropometrics: torch.Tensor,
) -> dict[str, float]:
    """Compute ∂height/∂param for each technique parameter.

    This tells the athlete which parameters have the most impact.

    Returns:
        Dict mapping parameter name to normalized sensitivity score.
    """
    params = params.detach().requires_grad_(True)
    height = _evaluate_height(pinn_model, params, anthropometrics)
    grads = torch.autograd.grad(height, params)[0]

    param_names = [
        "approach_speed", "curve_radius", "penultimate_step",
        "last_step", "plant_angle", "knee_angle",
        "hip_angle", "arm_timing", "free_leg_drive",
    ]
    grad_vals = grads.detach().cpu().numpy()
    max_abs = np.abs(grad_vals).max() + 1e-8

    return {name: float(g / max_abs) for name, g in zip(param_names, grad_vals)}


def what_if_scenario(
    pinn_model: nn.Module,
    base_params: TechniqueParameters,
    anthropometrics: torch.Tensor,
    modifications: dict[str, float],
) -> dict[str, float]:
    """Run a what-if scenario: what happens if we change specific parameters?

    Args:
        pinn_model: Trained PINN.
        base_params: Current technique.
        anthropometrics: Athlete body parameters.
        modifications: Dict of param_name → new_value.

    Returns:
        Dict with base height, modified height, and delta.
    """
    base_tensor = base_params.to_tensor()
    base_height = _evaluate_height(pinn_model, base_tensor, anthropometrics).item()

    mod_params = TechniqueParameters(**{
        **base_params.__dict__,
        **modifications,
    })
    mod_tensor = mod_params.to_tensor()
    mod_height = _evaluate_height(pinn_model, mod_tensor, anthropometrics).item()

    return {
        "base_height_m": base_height,
        "modified_height_m": mod_height,
        "delta_cm": (mod_height - base_height) * 100,
    }


def _evaluate_height(
    model: nn.Module,
    params: torch.Tensor,
    anthropometrics: torch.Tensor,
) -> torch.Tensor:
    """Evaluate predicted jump height from the PINN.

    This is a placeholder — the actual implementation depends on
    how the full PINN pipeline maps technique params to height.
    """
    combined = torch.cat([params, anthropometrics])
    return model(combined.unsqueeze(0)).squeeze()
