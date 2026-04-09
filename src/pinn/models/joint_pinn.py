"""Per-joint PINN module for biomechanical analysis.

Each anatomical joint (ankle, knee, hip, shoulder, etc.) gets its own PINN
that models the local dynamics. These are then coupled via the GNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class JointPINN(nn.Module):
    """Physics-informed model for a single anatomical joint.

    Learns the mapping from joint state + time → torque prediction,
    while respecting Euler-Lagrange dynamics:
        τ = M(q) * q̈ + C(q, q̇) * q̇ + G(q)

    where:
        q     = generalised joint coordinate (angle)
        M(q)  = inertia matrix
        C     = Coriolis/centrifugal terms
        G(q)  = gravitational torque

    Input:  [t, q, q̇, segment_mass, segment_length, segment_com_pos]
    Output: [q̈_predicted, τ_predicted, internal_force_x, internal_force_y, internal_force_z]
    """

    def __init__(
        self,
        joint_name: str,
        input_dim: int = 7,   # t + q + q̇ + 4 anthropometric params
        output_dim: int = 5,  # q̈, τ, Fx, Fy, Fz
        hidden_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.joint_name = joint_name

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, input_dim) joint state + anthropometrics.

        Returns:
            (batch, output_dim) predicted dynamics.
        """
        return self.net(x)

    @property
    def predicted_acceleration(self):
        """Slice index for q̈ in the output."""
        return 0

    @property
    def predicted_torque(self):
        """Slice index for τ in the output."""
        return 1

    @property
    def predicted_forces(self):
        """Slice indices for [Fx, Fy, Fz] in the output."""
        return slice(2, 5)
