"""Inverse dynamics PINN: estimates ground reaction forces from kinematics.

During the takeoff phase, the PINN learns the mapping:
    kinematics (q, q̇, q̈) → joint torques τ and GRF

Physics constraint (Newton-Euler for the whole body):
    F_GRF = m * a_CoM + m * g
    τ = I * α + ω × (I * ω)  (for each joint)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InverseDynamicsPINN(nn.Module):
    """PINN for estimating GRF and joint torques from observed motion.

    This model is conditioned on the athlete's anthropometrics, making it
    personalised from the start.

    Input:  [t, body_mass, height, joint_angles(t), joint_velocities(t)]
    Output: [GRF_x, GRF_y, GRF_z, τ_ankle, τ_knee, τ_hip]
    """

    def __init__(
        self,
        input_dim: int = 15,    # t + anthropometrics + joint states
        output_dim: int = 6,    # 3 GRF components + 3 joint torques
        hidden_dim: int = 128,
        n_layers: int = 5,
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def newton_euler_residual(
    predicted_grf: torch.Tensor,
    com_acceleration: torch.Tensor,
    body_mass: float,
    g: float = 9.81,
) -> torch.Tensor:
    """Physics residual: F_GRF - m*(a + g) should equal zero.

    Args:
        predicted_grf: (N, 3) predicted ground reaction force [Fx, Fy, Fz].
        com_acceleration: (N, 3) observed CoM acceleration.
        body_mass: Athlete mass in kg.
        g: Gravitational acceleration.

    Returns:
        Scalar residual loss.
    """
    gravity = torch.tensor([0.0, -g, 0.0], device=predicted_grf.device)
    expected_grf = body_mass * (com_acceleration - gravity)
    return torch.mean((predicted_grf - expected_grf) ** 2)
