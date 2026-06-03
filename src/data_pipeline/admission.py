"""Local admission manifest for private personal-fine-tuning samples."""

from __future__ import annotations

import json
from pathlib import Path

ADMISSION_MANIFEST_FILENAME = "_admission_manifest.json"
ADMISSION_MANIFEST_SCHEMA_VERSION = 1
ADMISSION_CACHE_POLICY = "training_grade_only"


def load_admission_manifest(samples_dir: Path) -> dict:
    """Load and validate the local admitted-only cache manifest."""
    manifest_path = samples_dir / ADMISSION_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Admitted-only cache manifest not found: {manifest_path}. "
            "Re-run analyze_jump_video.py with --save-samples into a fresh directory."
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ADMISSION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported admission manifest schema: {manifest_path}")
    if payload.get("cache_policy") != ADMISSION_CACHE_POLICY:
        raise ValueError(f"Unsupported admission cache policy: {manifest_path}")
    if not isinstance(payload.get("samples"), dict):
        raise ValueError(f"Admission manifest is missing its samples map: {manifest_path}")
    return payload


def record_admission_decision(
    samples_dir: Path,
    *,
    trial_id: str,
    admitted: bool,
    saved: bool,
    stationary_camera_confirmed: bool,
    training_grade_failures: list[str],
) -> Path:
    """Record a clip decision in the ignored local fine-tuning cache manifest."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = samples_dir / ADMISSION_MANIFEST_FILENAME
    if manifest_path.exists():
        payload = load_admission_manifest(samples_dir)
    else:
        payload = {
            "schema_version": ADMISSION_MANIFEST_SCHEMA_VERSION,
            "cache_policy": ADMISSION_CACHE_POLICY,
            "samples": {},
        }

    payload["samples"][trial_id] = {
        "admitted": bool(admitted),
        "saved": bool(saved),
        "sample_file": f"{trial_id}.npz" if saved else None,
        "stationary_camera_confirmed": bool(stationary_camera_confirmed),
        "training_grade_failures": list(training_grade_failures),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def admitted_sample_paths(samples_dir: Path) -> list[Path]:
    """Return only saved samples explicitly admitted by the local manifest."""
    payload = load_admission_manifest(samples_dir)
    paths: list[Path] = []
    for entry in payload["samples"].values():
        if entry.get("admitted") is not True or entry.get("saved") is not True:
            continue
        sample_file = entry.get("sample_file")
        relative_path = Path(sample_file) if isinstance(sample_file, str) else None
        if (
            relative_path is None
            or relative_path.name != str(relative_path)
            or relative_path.suffix.lower() != ".npz"
        ):
            raise ValueError(f"Unsafe admitted sample filename: {sample_file!r}")
        sample_path = samples_dir / relative_path
        if not sample_path.exists():
            raise FileNotFoundError(f"Admitted sample is missing: {sample_path}")
        paths.append(sample_path)
    return sorted(paths)
