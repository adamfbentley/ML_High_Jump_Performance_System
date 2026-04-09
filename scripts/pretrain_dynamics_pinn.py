"""Pre-train the inverse dynamics PINN on public biomechanics data.

This script loads data from AddBiomechanics (and optionally BioCV/OpenCap),
builds a DynamicsDataset, and pre-trains the InverseDynamicsPINN so that it
learns general human movement dynamics (F=ma, joint torques) before ever
seeing high jump data.

Usage:
    python scripts/pretrain_dynamics_pinn.py
    python scripts/pretrain_dynamics_pinn.py --config experiments/configs/pretrain_dynamics.yaml
    python scripts/pretrain_dynamics_pinn.py --data-dir data/public/addbiomechanics --epochs 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data_pipeline.sample import MovementType
from src.data_pipeline.torch_datasets import DynamicsDataset, FlightPhaseDataset
from src.utils.constants import DATA_DIR, EXPERIMENTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────


def default_config() -> dict:
    return {
        # Data
        "datasets": ["addbiomechanics"],  # which datasets to load
        "data_root": str(DATA_DIR / "public"),
        "movement_filter": ["cmj", "drop_jump", "squat_jump", "vertical_jump",
                            "running", "sprinting"],
        "window_size": 64,
        "stride": 32,
        "max_subjects": None,  # None = all

        # Model — Inverse Dynamics PINN
        "model_type": "inverse_dynamics",  # or "projectile"
        "hidden_dim": 128,
        "n_layers": 5,

        # Training
        "epochs": 3000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "lambda_data": 1.0,
        "lambda_physics": 1.0,
        "weight_decay": 1e-5,
        "scheduler": "cosine",

        # Output
        "save_dir": str(EXPERIMENTS_DIR / "results" / "pretrain_dynamics"),
        "save_every": 500,
        "log_interval": 50,
    }


def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data Loading ───────────────────────────────────────────────────────


MOVEMENT_NAME_TO_ENUM = {m.value: m for m in MovementType}


def load_all_samples(config: dict) -> list:
    """Load BiomechanicalSamples from all configured datasets."""
    from src.data_pipeline.sample import BiomechanicalSample

    movement_filter = None
    if config.get("movement_filter"):
        movement_filter = [
            MOVEMENT_NAME_TO_ENUM.get(m, MovementType.OTHER)
            for m in config["movement_filter"]
        ]

    samples: list[BiomechanicalSample] = []
    data_root = Path(config["data_root"])
    max_subjects = config.get("max_subjects")

    for dataset_name in config["datasets"]:
        dataset_dir = data_root / dataset_name
        logger.info(f"Loading {dataset_name} from {dataset_dir}...")

        try:
            if dataset_name == "addbiomechanics":
                from src.data_pipeline.loaders.addbiomechanics import load_addbiomechanics
                for sample in load_addbiomechanics(
                    dataset_dir, movement_filter, max_subjects
                ):
                    samples.append(sample)

            elif dataset_name == "biocv":
                from src.data_pipeline.loaders.biocv import load_biocv
                for sample in load_biocv(
                    dataset_dir, movement_filter, max_subjects
                ):
                    samples.append(sample)

            elif dataset_name == "opencap":
                from src.data_pipeline.loaders.opencap import load_opencap
                for sample in load_opencap(
                    dataset_dir, movement_filter, max_subjects
                ):
                    samples.append(sample)

            else:
                logger.warning(f"Unknown dataset: {dataset_name}")

        except FileNotFoundError as e:
            logger.error(str(e))
            continue

    logger.info(f"Total samples loaded: {len(samples)}")
    return samples


# ── Model Factory ──────────────────────────────────────────────────────


def build_model(config: dict, input_dim: int):
    """Build the PINN model based on config."""
    model_type = config.get("model_type", "inverse_dynamics")

    if model_type == "inverse_dynamics":
        from src.pinn.physics.inverse_dynamics import InverseDynamicsPINN
        return InverseDynamicsPINN(
            input_dim=input_dim,
            output_dim=6,  # 3 GRF + 3 main joint torques
            hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"],
        )
    elif model_type == "projectile":
        from src.pinn.physics.projectile import ProjectilePINN
        return ProjectilePINN(
            hidden_dim=config["hidden_dim"],
            n_hidden_layers=config["n_layers"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ── Physics Loss ───────────────────────────────────────────────────────


def compute_newton_euler_loss(
    pred_grf: torch.Tensor,
    com_acc: torch.Tensor,
    body_mass: torch.Tensor,
) -> torch.Tensor:
    """F_GRF = m * (a_CoM + g) → residual should be zero.

    Args:
        pred_grf: (B, T, 3) predicted GRF (per-kg if normalized).
        com_acc: (B, T, 3) observed CoM acceleration.
        body_mass: (B,) body mass (used only if forces are absolute).

    Returns:
        Scalar physics residual loss.
    """
    g = torch.tensor([0.0, 9.81, 0.0], device=pred_grf.device)
    # If forces are normalized by mass, the equation becomes: F/m = a + g
    expected = com_acc + g.unsqueeze(0).unsqueeze(0)
    residual = pred_grf - expected
    return torch.mean(residual ** 2)


# ── Training Loop ──────────────────────────────────────────────────────


def train(config: dict):
    """Main pre-training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load data ──
    samples = load_all_samples(config)
    if not samples:
        logger.error(
            "No data loaded. Make sure you have downloaded at least one dataset.\n"
            "Run: python scripts/download_datasets.py  for instructions."
        )
        return

    # ── Build datasets ──
    model_type = config.get("model_type", "inverse_dynamics")

    if model_type == "projectile":
        dataset = FlightPhaseDataset(samples)
        logger.info(f"Flight phase dataset: {len(dataset)} trajectories")
    else:
        dataset = DynamicsDataset(
            samples,
            window_size=config["window_size"],
            stride=config["stride"],
        )
        logger.info(f"Dynamics dataset: {len(dataset)} windows")

    if len(dataset) == 0:
        logger.error("Dataset is empty after windowing. Check data format and filters.")
        return

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ── Build model ──
    # Determine input dim from first batch
    sample_batch = next(iter(loader))
    input_dim = sample_batch["input"].shape[-1] if "input" in sample_batch else 1
    model = build_model(config, input_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config['model_type']} | {n_params:,} parameters")

    # ── Optimizer ──
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"],
    )

    # ── Output dir ──
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──
    logger.info(f"\nStarting pre-training for {config['epochs']} epochs...")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batch size:   {config['batch_size']}")
    logger.info(f"  Batches/epoch: {len(loader)}")
    logger.info("")

    best_loss = float("inf")
    loss_history = []
    start_time = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        n_batches = 0

        for batch in loader:
            optimizer.zero_grad()

            if model_type == "projectile":
                # Projectile PINN: predict position from time
                t = batch["time"].unsqueeze(-1).to(device)
                pos = batch["position"].to(device)
                pred = model(t.reshape(-1, 1)).reshape(pos.shape)

                data_loss = torch.nn.functional.mse_loss(pred, pos)

                # Physics loss: d²y/dt² = -g (approximate via finite differences)
                if pos.shape[1] >= 3:
                    dt = t[:, 1, 0] - t[:, 0, 0]  # (B,)
                    dt = dt.clamp(min=1e-6)
                    acc_y = (pos[:, 2:, 1] - 2 * pos[:, 1:-1, 1] + pos[:, :-2, 1]) / (dt[:, None] ** 2)
                    physics_loss = torch.mean((acc_y + 9.81) ** 2)
                else:
                    physics_loss = torch.tensor(0.0, device=device)

            else:
                # Inverse dynamics PINN
                x = batch["input"].to(device)  # (B, T, D)
                B, T, D = x.shape

                pred = model(x.reshape(B * T, D)).reshape(B, T, -1)
                # pred: (B, T, 6) → [GRF_x, GRF_y, GRF_z, τ_ankle, τ_knee, τ_hip]

                # Data loss: match GRF if available
                data_loss = torch.tensor(0.0, device=device)
                if "target_grf" in batch:
                    target_grf = batch["target_grf"].to(device)
                    pred_grf = pred[:, :, :3]
                    data_loss = torch.nn.functional.mse_loss(pred_grf, target_grf)

                if "target_torques" in batch:
                    target_tau = batch["target_torques"].to(device)
                    # Match as many torque columns as we predict
                    n_tau = min(pred.shape[-1] - 3, target_tau.shape[-1])
                    if n_tau > 0:
                        data_loss = data_loss + torch.nn.functional.mse_loss(
                            pred[:, :, 3:3 + n_tau],
                            target_tau[:, :, :n_tau],
                        )

                # Physics loss: Newton-Euler residual
                physics_loss = torch.tensor(0.0, device=device)
                if "target_com_acc" in batch:
                    com_acc = batch["target_com_acc"].to(device)
                    body_mass = batch["body_mass"].to(device)
                    physics_loss = compute_newton_euler_loss(
                        pred[:, :, :3], com_acc, body_mass,
                    )

            # Combined loss
            total_loss = (
                config["lambda_data"] * data_loss
                + config["lambda_physics"] * physics_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()
            n_batches += 1

        scheduler.step()

        avg_data = epoch_data_loss / max(n_batches, 1)
        avg_physics = epoch_physics_loss / max(n_batches, 1)
        avg_total = avg_data + avg_physics

        loss_history.append({
            "epoch": epoch,
            "data_loss": avg_data,
            "physics_loss": avg_physics,
            "total_loss": avg_total,
            "lr": scheduler.get_last_lr()[0],
        })

        if epoch % config["log_interval"] == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch:5d}/{config['epochs']} | "
                f"Data: {avg_data:.6f} | Physics: {avg_physics:.6f} | "
                f"Total: {avg_total:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.0f}s"
            )

        # Save checkpoints
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": config,
            }, save_dir / "best_model.pth")

        if config["save_every"] and epoch > 0 and epoch % config["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                "config": config,
            }, save_dir / f"checkpoint_epoch{epoch}.pth")

    # ── Final save ──
    elapsed = time.time() - start_time
    logger.info(f"\nPre-training complete in {elapsed:.0f}s")
    logger.info(f"  Best loss: {best_loss:.6f}")
    logger.info(f"  Final data loss: {loss_history[-1]['data_loss']:.6f}")
    logger.info(f"  Final physics loss: {loss_history[-1]['physics_loss']:.6f}")

    torch.save({
        "epoch": config["epochs"] - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "config": config,
    }, save_dir / "final_model.pth")

    # Save loss history as numpy
    np.savez(
        save_dir / "loss_history.npz",
        epochs=np.array([h["epoch"] for h in loss_history]),
        data_loss=np.array([h["data_loss"] for h in loss_history]),
        physics_loss=np.array([h["physics_loss"] for h in loss_history]),
        total_loss=np.array([h["total_loss"] for h in loss_history]),
    )

    logger.info(f"  Models saved to: {save_dir}")


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train dynamics PINN on public biomechanics data"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Root directory for public datasets",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Which datasets to use (e.g., addbiomechanics biocv opencap)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model-type", choices=["inverse_dynamics", "projectile"], default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config = default_config()

    # Load YAML config if provided
    if args.config:
        yaml_config = load_yaml_config(args.config)
        config.update(yaml_config)

    # CLI overrides
    if args.data_dir:
        config["data_root"] = args.data_dir
    if args.datasets:
        config["datasets"] = args.datasets
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.model_type:
        config["model_type"] = args.model_type
    if args.max_subjects:
        config["max_subjects"] = args.max_subjects

    logger.info("=== Dynamics PINN Pre-Training ===")
    logger.info(f"Config: {config}\n")

    train(config)


if __name__ == "__main__":
    main()
