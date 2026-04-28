"""Fine-tune the pre-trained inverse-dynamics PINN on a single athlete's
extracted kinematics, producing a personalised model.

Pipeline:
    BiomechanicalSamples (.npz cache from analyze_jump_video.py --save-samples)
        → optional plausible-scale filter (Phase 9a guardrail)
        → DynamicsDataset windowing
        → load best_model.pth
        → fine-tune at lower LR
        → save data/models/personal/<athlete>_finetuned.pth

The pretrained checkpoint produced by `pretrain_dynamics_pinn.py` uses
input_dim = 7 and output_dim = 6:
    input  = [t, com_pos_xyz, com_vel_xyz]                 (1 + 3 + 3)
    output = [GRF_xyz, tau_ankle, tau_knee, tau_hip]       (3 + 3)
Both losses (data + Newton-Euler residual) match the pre-training script
so the optimiser starts from a meaningful point.

Usage:
    python scripts/finetune_personal.py \
        --samples-dir data/results/samples \
        --pretrained experiments/results/pretrain_dynamics/best_model.pth \
        --output data/models/personal/athlete_a_finetuned.pth \
        --epochs 200 --lr 1e-4

Phase 9a guardrail:
    Single-camera scale calibration is broken on ~60% of clips (peak
    CoM > 3 m, physically impossible).  Fine-tuning on those clips would
    teach the model garbage CoM dynamics.  By default we keep only clips
    whose peak CoM height is below `--max-peak-com-m` (default 3.0 m).
    This is conservative: relax once Phase 9a follow-ups land.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data_pipeline.sample import BiomechanicalSample
from src.data_pipeline.torch_datasets import DynamicsDataset
from src.pinn.physics.inverse_dynamics import InverseDynamicsPINN

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Sample loading and filtering ──────────────────────────────────────────


def load_samples(
    samples_dir: Path,
    max_peak_com_m: float | None = 3.0,
) -> list[BiomechanicalSample]:
    """Load all .npz samples from `samples_dir`, optionally filtering by scale.

    The scale filter rejects clips whose peak CoM height exceeds
    `max_peak_com_m`.  An elite female high jumper's peak CoM in flight
    is ~2.2–2.7 m; values above 3 m indicate broken scale calibration
    (see ROADMAP Phase 9a).
    """
    paths = sorted(samples_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No .npz samples in {samples_dir}.  Run "
            f"`analyze_jump_video.py <video_dir> --save-samples {samples_dir}` first."
        )

    kept: list[BiomechanicalSample] = []
    rejected: list[tuple[str, float]] = []
    for p in paths:
        sample = BiomechanicalSample.load_npz(p)
        if sample.com_position is None:
            rejected.append((p.stem, float("nan")))
            continue
        peak = float(sample.com_position[:, 1].max())
        if max_peak_com_m is not None and peak > max_peak_com_m:
            rejected.append((p.stem, peak))
            continue
        kept.append(sample)

    logger.info(f"Loaded {len(kept)} samples (rejected {len(rejected)} for scale).")
    if rejected:
        logger.info(
            "  Rejected clips (peak CoM > "
            f"{max_peak_com_m} m, see ROADMAP Phase 9a):"
        )
        for stem, peak in rejected[:10]:
            logger.info(f"    {stem}: peak={peak:.2f} m")
        if len(rejected) > 10:
            logger.info(f"    ... and {len(rejected) - 10} more")

    return kept


# ── Physics loss (matches pretrain_dynamics_pinn.compute_newton_euler_loss) ──


def newton_euler_residual(
    pred_grf_per_kg: torch.Tensor,
    com_acc: torch.Tensor,
) -> torch.Tensor:
    """F_GRF/m = a_CoM + g — residual should be zero (Y-up gravity)."""
    g = torch.tensor([0.0, 9.81, 0.0], device=pred_grf_per_kg.device)
    expected = com_acc + g.view(1, 1, 3)
    return torch.mean((pred_grf_per_kg - expected) ** 2)


# ── Training ──────────────────────────────────────────────────────────────


def finetune(
    samples: list[BiomechanicalSample],
    pretrained_path: Path,
    output_path: Path,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    window_size: int = 64,
    stride: int = 32,
    lambda_data: float = 1.0,
    lambda_physics: float = 1.0,
    log_interval: int = 10,
) -> dict:
    """Fine-tune the inverse-dynamics PINN on personal samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    dataset = DynamicsDataset(samples, window_size=window_size, stride=stride)
    if len(dataset) == 0:
        raise RuntimeError(
            f"DynamicsDataset is empty (window_size={window_size}, stride={stride}). "
            f"Check that loaded samples have com_position and grf, and that they "
            f"have at least {window_size} frames."
        )
    logger.info(f"Dynamics windows: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False,
    )

    # ── Load pretrained checkpoint (architecture matches pretraining) ─────
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = InverseDynamicsPINN(
        input_dim=7,
        output_dim=6,
        hidden_dim=cfg.get("hidden_dim", 128),
        n_layers=cfg.get("n_layers", 5),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        f"Loaded pretrained from {pretrained_path} "
        f"(epoch {ckpt.get('epoch')}, best_loss {ckpt.get('best_loss', float('nan')):.4f})"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = float("inf")
    start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_data = epoch_phys = 0.0
        n = 0
        for batch in loader:
            optimizer.zero_grad()
            x = batch["input"].to(device)            # (B, T, 7)
            B, T, D = x.shape
            pred = model(x.reshape(B * T, D)).reshape(B, T, -1)  # (B, T, 6)

            data_loss = torch.tensor(0.0, device=device)
            if "target_grf" in batch:
                tgt = batch["target_grf"].to(device)
                data_loss = data_loss + torch.nn.functional.mse_loss(pred[:, :, :3], tgt)

            physics_loss = torch.tensor(0.0, device=device)
            if "target_com_acc" in batch:
                acc = batch["target_com_acc"].to(device)
                physics_loss = newton_euler_residual(pred[:, :, :3], acc)

            total = lambda_data * data_loss + lambda_physics * physics_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_data += data_loss.item()
            epoch_phys += physics_loss.item()
            n += 1

        scheduler.step()
        avg_data = epoch_data / max(n, 1)
        avg_phys = epoch_phys / max(n, 1)
        avg_total = avg_data + avg_phys
        history.append({"epoch": epoch, "data": avg_data, "physics": avg_phys, "total": avg_total})

        if epoch % log_interval == 0 or epoch == epochs - 1:
            elapsed = time.time() - start
            logger.info(
                f"Epoch {epoch:4d}/{epochs} | data={avg_data:.4f} "
                f"physics={avg_phys:.4f} total={avg_total:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | t={elapsed:.0f}s"
            )

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "pretrained_from": str(pretrained_path),
                "lr": lr,
                "config": cfg,
            }, output_path)

    logger.info(f"Fine-tune complete: best total loss = {best_loss:.4f}")
    logger.info(f"Saved: {output_path}")
    return {"best_loss": best_loss, "history": history}


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune inverse-dynamics PINN on personal samples",
    )
    parser.add_argument(
        "--samples-dir", type=str, default="data/results/samples",
        help="Directory of .npz samples produced by analyze_jump_video.py --save-samples",
    )
    parser.add_argument(
        "--pretrained", type=str,
        default="experiments/results/pretrain_dynamics/best_model.pth",
        help="Pretrained checkpoint to fine-tune from",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/models/personal/athlete_a_finetuned.pth",
        help="Where to save the fine-tuned checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument(
        "--max-peak-com-m", type=float, default=3.0,
        help="Reject samples with peak CoM above this height "
             "(Phase 9a scale-calibration guardrail). Set to 0 to disable.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load samples and report counts, do not train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    pretrained = Path(args.pretrained)
    output = Path(args.output)

    if not pretrained.exists():
        logger.error(f"Pretrained checkpoint not found: {pretrained}")
        sys.exit(1)

    max_peak = args.max_peak_com_m if args.max_peak_com_m > 0 else None
    samples = load_samples(samples_dir, max_peak_com_m=max_peak)
    if not samples:
        logger.error(
            "No usable samples after filtering. Either widen the scale filter "
            "(--max-peak-com-m 0) or fix Phase 9a scale calibration first."
        )
        sys.exit(1)

    if args.dry_run:
        logger.info(
            f"Dry-run complete: {len(samples)} samples kept, "
            f"would fine-tune for {args.epochs} epochs at lr={args.lr} → {output}"
        )
        return

    finetune(
        samples=samples,
        pretrained_path=pretrained,
        output_path=output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        window_size=args.window_size,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
