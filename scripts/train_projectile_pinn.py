"""Quick-start script: train the flight-phase projectile PINN on synthetic data.

Generates a noisy parabolic CoM trajectory and trains the PINN to recover it.
This validates the PINN approach before needing real data.

Usage:
    python scripts/train_projectile_pinn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pinn.physics.projectile import (
    ProjectilePINN,
    compute_data_loss,
    compute_physics_loss,
    compute_boundary_loss,
)
from src.pinn.training.trainer import TrainingConfig, train_pinn


def generate_synthetic_flight(
    v0: np.ndarray = np.array([2.0, 4.5, 0.3]),
    y0: np.ndarray = np.array([0.0, 1.85, 0.0]),
    fps: int = 120,
    n_frames: int = 60,
    noise_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noisy parabolic trajectory (synthetic flight phase).

    Returns:
        (times, positions) arrays.
    """
    g = 9.81
    dt = 1.0 / fps
    t = np.arange(n_frames) * dt

    x = y0[0] + v0[0] * t
    y = y0[1] + v0[1] * t - 0.5 * g * t ** 2
    z = y0[2] + v0[2] * t

    positions = np.column_stack([x, y, z])
    positions += np.random.randn(*positions.shape) * noise_std

    return t, positions


def main():
    print("=== Flight-Phase Projectile PINN Training ===\n")

    # Generate synthetic data
    v0 = np.array([2.0, 4.5, 0.3])
    y0 = np.array([0.0, 1.85, 0.0])
    t_np, pos_np = generate_synthetic_flight(v0=v0, y0=y0)
    print(f"Generated {len(t_np)} synthetic flight frames")
    print(f"  Initial position: {y0}")
    print(f"  Initial velocity: {v0}")
    print(f"  Peak height (theoretical): {y0[1] + v0[1]**2 / (2*9.81):.3f} m\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Prepare tensors
    t_data = torch.tensor(t_np, dtype=torch.float32).unsqueeze(1).to(device)
    y_data = torch.tensor(pos_np, dtype=torch.float32).to(device)
    t0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    y0_t = torch.tensor([y0], dtype=torch.float32).to(device)
    v0_t = torch.tensor([v0], dtype=torch.float32).to(device)

    # Collocation points (uniformly sampled in the time domain)
    t_colloc = torch.linspace(0, t_np[-1], 200).unsqueeze(1).to(device)

    # Build model
    model = ProjectilePINN(hidden_dim=64, n_hidden_layers=4)

    config = TrainingConfig(
        n_epochs=3000,
        learning_rate=1e-3,
        lambda_data=1.0,
        lambda_physics=0.1,
        lambda_boundary=10.0,
        log_interval=500,
    )

    print("Training...\n")
    result = train_pinn(
        model=model,
        data_loss_fn=lambda: compute_data_loss(model, t_data, y_data),
        physics_loss_fn=lambda: compute_physics_loss(model, t_colloc),
        boundary_loss_fn=lambda: compute_boundary_loss(model, t0, y0_t, v0_t),
        config=config,
        device=device,
    )

    print(f"\n=== Training Complete ===")
    print(f"  Final data loss:    {result.final_data_loss:.6f}")
    print(f"  Final physics loss: {result.final_physics_loss:.6f}")
    print(f"  Final BC loss:      {result.final_boundary_loss:.6f}")

    # Evaluate
    with torch.no_grad():
        pred = model(t_data).cpu().numpy()
    errors = np.abs(pred - pos_np)
    print(f"\n  Mean absolute error: {errors.mean():.4f} m")
    print(f"  Max absolute error:  {errors.max():.4f} m")
    print(f"  Predicted peak height: {pred[:, 1].max():.3f} m")
    print(f"  True peak height:      {pos_np[:, 1].max():.3f} m")

    # Save model
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "projectile_pinn_synthetic.pth")
    print(f"\n  Model saved: {out_dir / 'projectile_pinn_synthetic.pth'}")


if __name__ == "__main__":
    main()
