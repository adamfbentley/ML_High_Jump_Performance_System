"""Evaluate the pretrained dynamics PINN on local public biomechanics datasets.

This is a validation/benchmark script, not a training script. It loads public
BiomechanicalSample datasets already present under ``data/public/``, builds the
same CoM-window representation used by pretraining, and compares the pretrained
PINN's GRF predictions against measured force-plate GRF.

Important: the current checkpoint may have been trained on these same local
datasets because the original pretraining run did not enforce a held-out split.
Use this script as a post-hoc benchmark. For publication-grade held-out results,
retrain with the same subject split printed here and evaluate on the held-out
subjects only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.loaders.cmj_npz import load_cmj_npz_dir  # noqa: E402
from src.data_pipeline.loaders.cod_zenodo import load_cod_zenodo  # noqa: E402
from src.data_pipeline.loaders.dvj_zenodo import load_dvj_zenodo  # noqa: E402
from src.data_pipeline.sample import BiomechanicalSample, MovementType  # noqa: E402
from src.data_pipeline.torch_datasets import DynamicsDataset  # noqa: E402
from src.pinn.physics.inverse_dynamics import InverseDynamicsPINN  # noqa: E402

logger = logging.getLogger(__name__)

GRAVITY_MPS2 = 9.81
DEFAULT_CHECKPOINT = Path("experiments/results/pretrain_dynamics/best_model.pth")
DEFAULT_OUTPUT = Path("memory/experiments/exp_004_dynamics_pinn_validation.md")
DEFAULT_JSON_OUTPUT = Path("experiments/results/pretrain_dynamics/evaluation_metrics.json")

DATASET_CHOICES = ("cmj_grf_zenodo", "cod_ik_id_zenodo", "dvj_opensim_zenodo")


@dataclass(frozen=True)
class PredictionBundle:
    target: np.ndarray
    model: np.ndarray
    mean_profile: np.ndarray
    bodyweight: np.ndarray


def _movement_filter() -> list[MovementType]:
    return [
        MovementType.COUNTERMOVEMENT_JUMP,
        MovementType.DROP_JUMP,
        MovementType.SQUAT_JUMP,
        MovementType.VERTICAL_JUMP,
        MovementType.RUNNING,
        MovementType.SPRINTING,
    ]


def _load_dataset(name: str, max_subjects: int | None) -> list[BiomechanicalSample]:
    if name == "cmj_grf_zenodo":
        # This loader's cap is trial-based, not subject-based; leave uncapped by
        # default because the dataset is small.
        return list(load_cmj_npz_dir(Path("data/public/cmj_grf_zenodo"), _movement_filter()))
    if name == "cod_ik_id_zenodo":
        return list(
            load_cod_zenodo(
                Path("data/public/cod_ik_id_zenodo"),
                _movement_filter(),
                max_subjects=max_subjects,
            )
        )
    if name == "dvj_opensim_zenodo":
        return list(
            load_dvj_zenodo(
                Path("data/public/dvj_opensim_zenodo"),
                _movement_filter(),
                max_subjects=max_subjects,
            )
        )
    raise ValueError(f"Unknown dataset: {name}")


def _subject_key(sample: BiomechanicalSample) -> str:
    return sample.subject.subject_id or sample.trial_id


def _is_validation_subject(subject_id: str, val_fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def _split_samples(
    samples: list[BiomechanicalSample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[BiomechanicalSample], list[BiomechanicalSample], list[str]]:
    subjects = sorted({_subject_key(sample) for sample in samples})
    val_subjects = [
        subject
        for subject in subjects
        if _is_validation_subject(subject, val_fraction=val_fraction, seed=seed)
    ]
    if not val_subjects and subjects:
        val_subjects = subjects[-max(1, round(len(subjects) * val_fraction)) :]
    val_set = set(val_subjects)
    train = [sample for sample in samples if _subject_key(sample) not in val_set]
    val = [sample for sample in samples if _subject_key(sample) in val_set]
    if not train and val:
        train = val
    return train, val, val_subjects


def _build_model(checkpoint_path: Path, input_dim: int) -> InverseDynamicsPINN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = InverseDynamicsPINN(
        input_dim=input_dim,
        output_dim=6,
        hidden_dim=int(config.get("hidden_dim", 128)),
        n_layers=int(config.get("n_layers", 5)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _mean_profile(train_dataset: DynamicsDataset, window_size: int) -> np.ndarray:
    targets = [
        item["target_grf"].numpy()
        for item in train_dataset
        if "target_grf" in item
    ]
    if not targets:
        return np.zeros((window_size, 3), dtype=np.float32)
    return np.mean(np.stack(targets, axis=0), axis=0).astype(np.float32)


def _predict(
    model: InverseDynamicsPINN,
    eval_dataset: DynamicsDataset,
    mean_profile: np.ndarray,
    *,
    batch_size: int,
) -> PredictionBundle:
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if "target_grf" not in batch:
                continue
            x = batch["input"].float()
            pred = model(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1], -1)
            preds.append(pred[:, :, :3].cpu().numpy())
            targets.append(batch["target_grf"].cpu().numpy())

    if not targets:
        empty = np.zeros((0, mean_profile.shape[0], 3), dtype=np.float32)
        return PredictionBundle(empty, empty, empty, empty)

    target = np.concatenate(targets, axis=0)
    model_pred = np.concatenate(preds, axis=0)
    mean_pred = np.repeat(mean_profile[None, :, :], target.shape[0], axis=0)
    bodyweight = np.zeros_like(target)
    bodyweight[:, :, 1] = GRAVITY_MPS2
    return PredictionBundle(target, model_pred, mean_pred, bodyweight)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    aa = a[finite]
    bb = b[finite]
    if float(np.std(aa)) < 1e-8 or float(np.std(bb)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {
            "vertical_rmse_bw": float("nan"),
            "vertical_mae_bw": float("nan"),
            "vertical_corr": float("nan"),
            "peak_vgrf_mae_bw": float("nan"),
            "peak_vgrf_bias_bw": float("nan"),
            "all_axis_rmse_bw": float("nan"),
        }

    err = pred - target
    vertical_err = err[:, :, 1]
    target_y = target[:, :, 1]
    pred_y = pred[:, :, 1]
    target_peak = np.nanmax(target_y, axis=1)
    pred_peak = np.nanmax(pred_y, axis=1)
    peak_err = pred_peak - target_peak
    return {
        "vertical_rmse_bw": float(np.sqrt(np.nanmean(vertical_err**2)) / GRAVITY_MPS2),
        "vertical_mae_bw": float(np.nanmean(np.abs(vertical_err)) / GRAVITY_MPS2),
        "vertical_corr": _corr(pred_y.reshape(-1), target_y.reshape(-1)),
        "peak_vgrf_mae_bw": float(np.nanmean(np.abs(peak_err)) / GRAVITY_MPS2),
        "peak_vgrf_bias_bw": float(np.nanmean(peak_err) / GRAVITY_MPS2),
        "all_axis_rmse_bw": float(np.sqrt(np.nanmean(err**2)) / GRAVITY_MPS2),
    }


def _evaluate_one_dataset(
    dataset_name: str,
    *,
    checkpoint: Path,
    window_size: int,
    stride: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
    max_subjects: int | None,
) -> dict:
    logger.info("Loading %s...", dataset_name)
    samples = _load_dataset(dataset_name, max_subjects=max_subjects)
    usable_samples = [
        sample
        for sample in samples
        if sample.grf is not None and sample.com_acceleration is not None
    ]
    train_samples, val_samples, val_subjects = _split_samples(
        usable_samples,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_dataset = DynamicsDataset(train_samples, window_size=window_size, stride=stride)
    eval_dataset = DynamicsDataset(val_samples, window_size=window_size, stride=stride)
    if len(eval_dataset) == 0:
        logger.warning("%s produced no validation windows; evaluating all usable samples.", dataset_name)
        val_samples = usable_samples
        eval_dataset = DynamicsDataset(val_samples, window_size=window_size, stride=stride)
    if len(eval_dataset) == 0:
        return {
            "dataset": dataset_name,
            "n_samples": len(samples),
            "n_usable_samples": len(usable_samples),
            "n_eval_windows": 0,
            "error": "no_eval_windows",
        }

    first = eval_dataset[0]
    input_dim = int(first["input"].shape[-1])
    model = _build_model(checkpoint, input_dim=input_dim)
    profile = _mean_profile(train_dataset if len(train_dataset) else eval_dataset, window_size)
    bundle = _predict(model, eval_dataset, profile, batch_size=batch_size)

    model_metrics = _metrics(bundle.model, bundle.target)
    mean_metrics = _metrics(bundle.mean_profile, bundle.target)
    bodyweight_metrics = _metrics(bundle.bodyweight, bundle.target)

    return {
        "dataset": dataset_name,
        "n_samples": len(samples),
        "n_usable_samples": len(usable_samples),
        "n_train_samples_for_baseline": len(train_samples),
        "n_eval_samples": len(val_samples),
        "n_eval_windows": int(bundle.target.shape[0]),
        "n_subjects": len({_subject_key(sample) for sample in usable_samples}),
        "n_validation_subjects": len(val_subjects),
        "validation_split_note": (
            "Checkpoint may have seen these subjects in the original pretraining run; "
            "this split is used for baseline construction and future retraining."
        ),
        "metrics": {
            "pretrained_pinn": model_metrics,
            "mean_profile_baseline": mean_metrics,
            "bodyweight_baseline": bodyweight_metrics,
        },
    }


def _fmt(value: float) -> str:
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_markdown(results: list[dict], output_path: Path, checkpoint: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 004: Dynamics PINN Public-Data Validation",
        "",
        "Aggregate metrics only. No private athlete data.",
        "",
        "## Scope",
        "",
        "This evaluates the pretrained inverse-dynamics PINN against local public",
        "force-plate biomechanics datasets. It validates the dynamics model, not",
        "the phone-video pose/calibration pipeline.",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Important caveat: the original checkpoint was not trained with a",
        "  formal held-out split. Treat this as a post-hoc benchmark until the",
        "  model is retrained with the subject split used here.",
        "",
        "## Metrics",
        "",
        "| Dataset | Windows | Model vRMSE (BW) | Mean vRMSE (BW) | BW vRMSE (BW) | Model peak MAE (BW) | Model vCorr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        if result.get("error"):
            lines.append(
                f"| {result['dataset']} | 0 | {result['error']} |  |  |  |  |"
            )
            continue
        metrics = result["metrics"]
        model = metrics["pretrained_pinn"]
        mean = metrics["mean_profile_baseline"]
        bw = metrics["bodyweight_baseline"]
        lines.append(
            f"| {result['dataset']} | {result['n_eval_windows']} | "
            f"{_fmt(model['vertical_rmse_bw'])} | "
            f"{_fmt(mean['vertical_rmse_bw'])} | "
            f"{_fmt(bw['vertical_rmse_bw'])} | "
            f"{_fmt(model['peak_vgrf_mae_bw'])} | "
            f"{_fmt(model['vertical_corr'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The `mean_profile_baseline` is the average GRF waveform from the",
            "  training side of the subject split.",
            "- The `bodyweight_baseline` predicts quiet standing force",
            "  (`Fy = 1 BW`) for every frame.",
            "- A useful pretrained model should beat these baselines on vertical",
            "  RMSE and peak-force error, especially on jump-like datasets.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained dynamics PINN.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to pretrained inverse-dynamics checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CHOICES,
        default=["cmj_grf_zenodo", "cod_ik_id_zenodo"],
        help="Datasets to evaluate. DVJ is slower if its cache has not been built.",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap for CoD/DVJ loader debugging. CMJ remains uncapped.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    results = [
        _evaluate_one_dataset(
            name,
            checkpoint=args.checkpoint,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_subjects=args.max_subjects,
        )
        for name in args.datasets
    ]

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(results, args.output, args.checkpoint)

    print("Dynamics PINN validation")
    print(f"  checkpoint: {args.checkpoint}")
    for result in results:
        if result.get("error"):
            print(f"  {result['dataset']}: {result['error']}")
            continue
        metrics = result["metrics"]
        model = metrics["pretrained_pinn"]
        mean = metrics["mean_profile_baseline"]
        bw = metrics["bodyweight_baseline"]
        print(
            f"  {result['dataset']}: windows={result['n_eval_windows']} "
            f"model_vRMSE={_fmt(model['vertical_rmse_bw'])} BW "
            f"mean_vRMSE={_fmt(mean['vertical_rmse_bw'])} BW "
            f"bodyweight_vRMSE={_fmt(bw['vertical_rmse_bw'])} BW "
            f"model_peakMAE={_fmt(model['peak_vgrf_mae_bw'])} BW "
            f"model_corr={_fmt(model['vertical_corr'])}"
        )
    print(f"  wrote: {args.output}")
    print(f"  wrote: {args.json_output}")


if __name__ == "__main__":
    main()
