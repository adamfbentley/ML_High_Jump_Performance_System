"""Tests for the projectile PINN (flight phase)."""

import numpy as np
import pytest
import torch

from src.pinn.physics.projectile import (
    ProjectilePINN,
    compute_data_loss,
    compute_physics_loss,
    compute_boundary_loss,
)


@pytest.fixture
def model():
    return ProjectilePINN(hidden_dim=32, n_hidden_layers=2)


@pytest.fixture
def synthetic_flight():
    """Simple parabolic trajectory for testing."""
    g = 9.81
    t = np.linspace(0, 0.5, 20)
    v0 = np.array([2.0, 4.0, 0.0])
    y0 = np.array([0.0, 1.8, 0.0])
    positions = np.column_stack([
        y0[0] + v0[0] * t,
        y0[1] + v0[1] * t - 0.5 * g * t ** 2,
        y0[2] + v0[2] * t,
    ])
    return t, positions, y0, v0


def test_model_output_shape(model):
    t = torch.randn(10, 1)
    out = model(t)
    assert out.shape == (10, 3)


def test_physics_loss_returns_scalar(model):
    t = torch.linspace(0, 1, 50).unsqueeze(1)
    loss = compute_physics_loss(model, t)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_data_loss_returns_scalar(model, synthetic_flight):
    t_np, pos_np, _, _ = synthetic_flight
    t = torch.tensor(t_np, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(pos_np, dtype=torch.float32)
    loss = compute_data_loss(model, t, y)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_boundary_loss_returns_scalar(model, synthetic_flight):
    _, _, y0, v0 = synthetic_flight
    t0 = torch.tensor([[0.0]])
    y0_t = torch.tensor([y0], dtype=torch.float32)
    v0_t = torch.tensor([v0], dtype=torch.float32)
    loss = compute_boundary_loss(model, t0, y0_t, v0_t)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_physics_loss_decreases_for_true_parabola():
    """A model that outputs a perfect parabola should have near-zero physics loss."""
    # This is a regression test — we train briefly and check physics loss drops
    model = ProjectilePINN(hidden_dim=32, n_hidden_layers=2)
    t = torch.linspace(0, 0.5, 100).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = compute_physics_loss(model, t).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = compute_physics_loss(model, t)
        loss.backward()
        optimizer.step()
    final_loss = compute_physics_loss(model, t).item()

    assert final_loss < initial_loss
