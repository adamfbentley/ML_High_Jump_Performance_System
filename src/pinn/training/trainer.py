"""PINN training loop with physics-weighted loss.

Handles the balancing of data loss, physics residual loss, and
boundary condition loss during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class TrainingConfig:
    """Configuration for PINN training."""

    n_epochs: int = 5000
    learning_rate: float = 1e-3
    batch_size: int = 256

    # Loss weighting (these are the critical hyperparameters)
    lambda_data: float = 1.0
    lambda_physics: float = 1.0
    lambda_boundary: float = 10.0

    # Collocation points for physics loss
    n_collocation: int = 1000

    # Adaptive loss weighting (Neural Tangent Kernel approach)
    adaptive_weights: bool = False

    # Logging
    log_interval: int = 100


@dataclass
class TrainingResult:
    """Result of a training run."""

    final_data_loss: float
    final_physics_loss: float
    final_boundary_loss: float
    final_total_loss: float
    loss_history: list[dict[str, float]] = field(default_factory=list)
    n_epochs_completed: int = 0


def train_pinn(
    model: nn.Module,
    data_loss_fn: Callable,
    physics_loss_fn: Callable,
    boundary_loss_fn: Callable | None,
    config: TrainingConfig,
    device: torch.device | None = None,
) -> TrainingResult:
    """Train a PINN with composite loss.

    Args:
        model: The neural network (any PINN variant).
        data_loss_fn: Callable() → scalar tensor (data fidelity loss).
        physics_loss_fn: Callable() → scalar tensor (physics residual loss).
        boundary_loss_fn: Callable() → scalar tensor (initial/boundary conditions).
        config: Training hyperparameters.
        device: Torch device (defaults to CUDA if available).

    Returns:
        TrainingResult with loss history and final metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.n_epochs)

    result = TrainingResult(
        final_data_loss=0, final_physics_loss=0,
        final_boundary_loss=0, final_total_loss=0,
    )

    for epoch in range(config.n_epochs):
        optimizer.zero_grad()

        loss_data = data_loss_fn()
        loss_physics = physics_loss_fn()
        loss_boundary = boundary_loss_fn() if boundary_loss_fn is not None else torch.tensor(0.0, device=device)

        total_loss = (
            config.lambda_data * loss_data
            + config.lambda_physics * loss_physics
            + config.lambda_boundary * loss_boundary
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        record = {
            "epoch": epoch,
            "data_loss": loss_data.item(),
            "physics_loss": loss_physics.item(),
            "boundary_loss": loss_boundary.item(),
            "total_loss": total_loss.item(),
            "lr": scheduler.get_last_lr()[0],
        }
        result.loss_history.append(record)

        if epoch % config.log_interval == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Data: {loss_data.item():.6f} | "
                f"Physics: {loss_physics.item():.6f} | "
                f"BC: {loss_boundary.item():.6f} | "
                f"Total: {total_loss.item():.6f}"
            )

    result.final_data_loss = result.loss_history[-1]["data_loss"]
    result.final_physics_loss = result.loss_history[-1]["physics_loss"]
    result.final_boundary_loss = result.loss_history[-1]["boundary_loss"]
    result.final_total_loss = result.loss_history[-1]["total_loss"]
    result.n_epochs_completed = config.n_epochs

    return result
