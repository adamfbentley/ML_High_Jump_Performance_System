"""BioCV dataset loader.

Reads C3D motion capture files with synchronized force plate data
from the BioCV dataset. This dataset bridges video and dynamics
because it provides synchronized video + markers + forces.

Requirements:
    pip install ezc3d

Download:
    See https://doi.org/10.1038/s41597-024-03463-1
    Place files in data/public/biocv/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import numpy as np

from src.data_pipeline.sample import (
    BiomechanicalSample,
    MovementType,
    SubjectInfo,
)

logger = logging.getLogger(__name__)

# BioCV expected directory layout:
#   biocv/
#     Subject01/
#       Walking/
#         trial_01.c3d
#       Running/
#         trial_01.c3d
#       CMJ/
#         trial_01.c3d
#       Hopping/
#         trial_01.c3d

_BIOCV_MOVEMENT_MAP = {
    "walking": MovementType.WALKING,
    "walk": MovementType.WALKING,
    "running": MovementType.RUNNING,
    "run": MovementType.RUNNING,
    "cmj": MovementType.COUNTERMOVEMENT_JUMP,
    "countermovement": MovementType.COUNTERMOVEMENT_JUMP,
    "hopping": MovementType.HOPPING,
    "hop": MovementType.HOPPING,
    "jump": MovementType.VERTICAL_JUMP,
}


def _classify_biocv_movement(path: Path) -> MovementType:
    """Infer movement type from the directory structure."""
    parts_lower = [p.lower() for p in path.parts]
    for part in parts_lower:
        for keyword, mtype in _BIOCV_MOVEMENT_MAP.items():
            if keyword in part:
                return mtype
    return MovementType.OTHER


def load_c3d_file(filepath: Path, subject_id: str = "") -> BiomechanicalSample | None:
    """Load a single C3D file into a BiomechanicalSample.

    Args:
        filepath: Path to the .c3d file.
        subject_id: Identifier for the subject (e.g., "Subject01").

    Returns:
        BiomechanicalSample or None if loading fails.
    """
    try:
        import ezc3d
    except ImportError:
        raise ImportError(
            "ezc3d is required to load BioCV .c3d files.\n"
            "Install: pip install ezc3d"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        return None

    logger.info(f"Loading {filepath.name}...")

    c3d = ezc3d.c3d(str(filepath))

    # ── Extract marker data ──
    point_data = c3d["data"]["points"]  # (4, n_markers, n_frames)
    n_markers = point_data.shape[1]
    n_frames = point_data.shape[2]

    # Transpose to (n_frames, n_markers, 3) — drop the 4th row (residual)
    marker_positions = point_data[:3, :, :].transpose(2, 1, 0)  # (T, M, 3)

    # Convert from mm to metres
    marker_positions = marker_positions / 1000.0

    # Marker names
    marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]
    if len(marker_names) > n_markers:
        marker_names = marker_names[:n_markers]

    # Frame rate
    point_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]

    # ── Extract force plate data (if available) ──
    grf = None
    cop = None

    try:
        analog_data = c3d["data"]["analogs"]  # (1, n_channels, n_analog_frames)

        # Force plate parameters
        fp_params = c3d["parameters"].get("FORCE_PLATFORM", {})
        n_force_plates = 0
        if "USED" in fp_params:
            n_force_plates = fp_params["USED"]["value"][0]

        if n_force_plates > 0 and analog_data.shape[1] >= 6:
            analog_rate = c3d["parameters"]["ANALOG"]["RATE"]["value"][0]
            ratio = int(analog_rate / point_rate)

            # Extract Fx, Fy, Fz channels (first force plate)
            # Standard C3D order: Fx, Fy, Fz, Mx, My, Mz per plate
            raw_forces = analog_data[0, :, :]  # (n_channels, n_analog_frames)

            # Sum forces from all plates, downsample to marker rate
            total_force = np.zeros((n_frames, 3))
            for plate in range(min(n_force_plates, raw_forces.shape[0] // 6)):
                ch_start = plate * 6
                for axis in range(3):
                    channel = raw_forces[ch_start + axis, :]
                    # Downsample by averaging each block of `ratio` samples
                    for f in range(min(n_frames, len(channel) // ratio)):
                        total_force[f, axis] += np.mean(
                            channel[f * ratio:(f + 1) * ratio]
                        )

            grf = total_force
            logger.debug(f"  Extracted GRF from {n_force_plates} force plate(s)")

    except (KeyError, IndexError) as e:
        logger.debug(f"  No force plate data in {filepath.name}: {e}")

    if n_frames < 10:
        return None

    # ── Estimate CoM from markers ──
    # Simple approximation: mean of all valid markers weighted equally.
    # A proper segment model needs marker-to-segment mapping which varies
    # by lab setup. For pre-training, this approximation is acceptable.
    valid_mask = np.all(np.isfinite(marker_positions), axis=-1)  # (T, M)
    com_position = np.zeros((n_frames, 3))
    for t in range(n_frames):
        valid_markers = marker_positions[t, valid_mask[t]]
        if len(valid_markers) > 0:
            com_position[t] = valid_markers.mean(axis=0)

    dt = 1.0 / point_rate
    com_velocity = np.gradient(com_position, dt, axis=0)
    com_acceleration = np.gradient(com_velocity, dt, axis=0)

    subject = SubjectInfo(
        subject_id=subject_id or filepath.parent.parent.name,
    )

    sample = BiomechanicalSample(
        dataset_name="biocv",
        trial_id=f"{subject.subject_id}_{filepath.stem}",
        subject=subject,
        movement_type=_classify_biocv_movement(filepath),
        fps=point_rate,
        marker_positions=marker_positions,
        marker_names=marker_names,
        com_position=com_position,
        com_velocity=com_velocity,
        com_acceleration=com_acceleration,
        grf=grf,
    )

    logger.info(
        f"  {filepath.stem}: {n_frames} frames @ {point_rate:.0f} Hz, "
        f"{n_markers} markers ({sample.movement_type.value})"
    )
    return sample


def load_biocv(
    data_dir: Path | str | None = None,
    movement_filter: list[MovementType] | None = None,
    max_subjects: int | None = None,
) -> Generator[BiomechanicalSample, None, None]:
    """Iterate over all BioCV samples.

    Args:
        data_dir: Root directory. Defaults to data/public/biocv/.
        movement_filter: Only yield matching movement types.
        max_subjects: Limit subjects for debugging.

    Yields:
        BiomechanicalSample for each trial.
    """
    if data_dir is None:
        from src.data_pipeline.registry import DATASET_REGISTRY
        data_dir = DATASET_REGISTRY["biocv"].local_dir

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"BioCV data directory not found: {data_dir}\n"
            f"Download from: https://doi.org/10.1038/s41597-024-03463-1"
        )

    # Find subject directories
    subject_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not subject_dirs:
        # Flat layout — just glob for .c3d files
        c3d_files = sorted(data_dir.glob("**/*.c3d"))
        subject_dirs = []  # fall through to file-level iteration
        for c3d_path in c3d_files:
            sample = load_c3d_file(c3d_path)
            if sample is not None:
                if movement_filter and sample.movement_type not in movement_filter:
                    continue
                yield sample
        return

    for i, subject_dir in enumerate(subject_dirs):
        if max_subjects is not None and i >= max_subjects:
            break

        c3d_files = sorted(subject_dir.glob("**/*.c3d"))
        for c3d_path in c3d_files:
            try:
                sample = load_c3d_file(c3d_path, subject_id=subject_dir.name)
            except Exception as e:
                logger.warning(f"Failed to load {c3d_path}: {e}")
                continue

            if sample is not None:
                if movement_filter and sample.movement_type not in movement_filter:
                    continue
                yield sample
