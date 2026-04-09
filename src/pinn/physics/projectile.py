"""Flight-phase PINN: enforces projectile motion physics on CoM trajectory.

This is the simplest PINN in the system and should be built first as a
proof-of-concept. After takeoff, the CoM follows:
    ẍ = 0,  ÿ = -g,  z̈ = 0

The PINN learns to denoise observed CoM data while strictly respecting
these physics constraints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectilePINN(nn.Module):
    """Physics-Informed Neural Network for flight-phase CoM prediction.

    Input:  t (normalized time) + athlete features (optional)
    Output: [x(t), y(t), z(t)] CoM position

    Physics loss enforces:
        d²x/dt² = 0
        d²y/dt² = -g
        d²z/dt² = 0
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()

        act_fn = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}[activation]

        layers = [nn.Linear(1, hidden_dim), act_fn()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
        layers.append(nn.Linear(hidden_dim, 3))  # output: [x, y, z]

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Predict CoM position at time(s) t.

        Args:
            t: (N, 1) tensor of time values.

        Returns:
            (N, 3) predicted positions [x, y, z].
        """
        return self.net(t)


def compute_physics_loss(
    model: ProjectilePINN,
    t_collocation: torch.Tensor,
    g: float = 9.81,
) -> torch.Tensor:
    """Compute physics residual loss: how well the predictions obey Newton's laws.

    Uses automatic differentiation to compute d²y/dt²
    and penalizes deviation from free-fall dynamics.

    Args:
        model: The PINN model.
        t_collocation: (N, 1) collocation points (require grad).
        g: Gravitational acceleration in m/s².

    Returns:
        Scalar physics loss.
    """
    t_collocation = t_collocation.requires_grad_(True)
    pred = model(t_collocation)  # (N, 3)

    # Compute first derivatives
    grad_outputs = torch.ones_like(pred[:, 0:1])

    dx_dt = torch.autograd.grad(pred[:, 0:1], t_collocation, grad_outputs, create_graph=True)[0]
    dy_dt = torch.autograd.grad(pred[:, 1:2], t_collocation, grad_outputs, create_graph=True)[0]
    dz_dt = torch.autograd.grad(pred[:, 2:3], t_collocation, grad_outputs, create_graph=True)[0]

    # Compute second derivatives
    d2x_dt2 = torch.autograd.grad(dx_dt, t_collocation, grad_outputs, create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt, t_collocation, grad_outputs, create_graph=True)[0]
    d2z_dt2 = torch.autograd.grad(dz_dt, t_collocation, grad_outputs, create_graph=True)[0]

    # Physics residuals
    res_x = d2x_dt2          # should be 0
    res_y = d2y_dt2 + g      # should be 0 (d²y/dt² = -g)
    res_z = d2z_dt2          # should be 0

    return torch.mean(res_x ** 2 + res_y ** 2 + res_z ** 2)


def compute_data_loss(
    model: ProjectilePINN,
    t_data: torch.Tensor,
    y_data: torch.Tensor,
) -> torch.Tensor:
    """Compute data fidelity loss.

    Args:
        model: The PINN model.
        t_data: (N, 1) observed time points.
        y_data: (N, 3) observed CoM positions.

    Returns:
        Scalar MSE data loss.
    """
    pred = model(t_data)
    return nn.functional.mse_loss(pred, y_data)


def compute_boundary_loss(
    model: ProjectilePINN,
    t0: torch.Tensor,
    y0: torch.Tensor,
    v0: torch.Tensor,
) -> torch.Tensor:
    """Compute initial condition loss (position and velocity at takeoff).

    Args:
        model: The PINN model.
        t0: (1, 1) takeoff time.
        y0: (1, 3) known CoM position at takeoff.
        v0: (1, 3) known CoM velocity at takeoff.

    Returns:
        Scalar boundary condition loss.
    """
    t0 = t0.requires_grad_(True)
    pred = model(t0)

    # Position BC
    pos_loss = nn.functional.mse_loss(pred, y0)

    # Velocity BC (first derivative)
    grad_outputs = torch.ones_like(pred[:, 0:1])
    vel = torch.cat([
        torch.autograd.grad(pred[:, i:i+1], t0, grad_outputs, create_graph=True)[0]
        for i in range(3)
    ], dim=1)
    vel_loss = nn.functional.mse_loss(vel, v0)

    return pos_loss + vel_loss
