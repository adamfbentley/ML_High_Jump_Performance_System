"""OpenCap dataset loader.

Reads .trc (marker trajectories) and .mot (motion/force) files from
the OpenCap platform and validation datasets. OpenCap uses markerless
pose estimation from smartphone video → OpenSim kinematics/dynamics.

Requirements:
    No special dependencies beyond numpy/pandas.

Download:
    Validation data from https://simtk.org/projects/opencap
    Place files in data/public/opencap/
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

_OPENCAP_MOVEMENT_MAP = {
    "dj": MovementType.DROP_JUMP,
    "drop_jump": MovementType.DROP_JUMP,
    "dropjump": MovementType.DROP_JUMP,
    "squat": MovementType.SQUAT_JUMP,
    "squatjump": MovementType.SQUAT_JUMP,
    "cmj": MovementType.COUNTERMOVEMENT_JUMP,
    "walk": MovementType.WALKING,
    "run": MovementType.RUNNING,
    "jog": MovementType.RUNNING,
    "sprint": MovementType.SPRINTING,
}


def _classify_opencap_movement(name: str) -> MovementType:
    name_lower = name.lower()
    for keyword, mtype in _OPENCAP_MOVEMENT_MAP.items():
        if keyword in name_lower:
            return mtype
    return MovementType.OTHER


def load_trc_file(filepath: Path) -> tuple[float, list[str], np.ndarray]:
    """Parse a .trc (Track Row Column) marker file.

    The TRC format is a tab-separated text file with a header block,
    followed by time + 3D marker position columns.

    Args:
        filepath: Path to the .trc file.

    Returns:
        (fps, marker_names, positions) where positions is (T, n_markers, 3).
    """
    filepath = Path(filepath)
    lines = filepath.read_text().splitlines()

    # Line 3 (0-indexed line 2): DataRate  CameraRate  NumFrames  NumMarkers ...
    header_parts = lines[2].split("\t")
    fps = float(header_parts[0])

    # Line 4 (0-indexed line 3): marker names (tab-separated, every 3rd is a marker)
    marker_line = lines[3].split("\t")
    marker_names = [m.strip() for m in marker_line[2:] if m.strip()]

    # Data starts at line 6 (0-indexed line 5)
    data_lines = lines[5:]
    frames = []
    for line in data_lines:
        if not line.strip():
            continue
        vals = line.split("\t")
        if len(vals) < 3:
            continue
        # vals[0] = frame number, vals[1] = time, rest = X Y Z per marker
        coords = [float(v) if v.strip() else np.nan for v in vals[2:]]
        frames.append(coords)

    raw = np.array(frames)  # (T, n_markers * 3)

    # Reshape to (T, n_markers, 3)
    n_coords = raw.shape[1]
    n_markers = n_coords // 3
    positions = raw[:, :n_markers * 3].reshape(-1, n_markers, 3)

    # TRC is typically in mm → convert to meters
    if np.nanmax(np.abs(positions)) > 10:
        positions = positions / 1000.0

    return fps, marker_names[:n_markers], positions


def load_mot_file(filepath: Path) -> tuple[float, list[str], np.ndarray]:
    """Parse an OpenSim .mot (motion/storage) file.

    These contain time-series data: joint angles, forces, or moments.

    Args:
        filepath: Path to the .mot file.

    Returns:
        (fps, column_names, data) where data is (T, n_columns).
    """
    filepath = Path(filepath)
    lines = filepath.read_text().splitlines()

    # Find 'endheader' line
    header_end = 0
    n_rows = 0
    n_cols = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break
        if line.startswith("nRows"):
            n_rows = int(line.split("=")[1].strip())
        if line.startswith("nColumns"):
            n_cols = int(line.split("=")[1].strip())

    # Column names are on the line after 'endheader'
    col_line = lines[header_end + 1]
    column_names = col_line.split("\t")
    if len(column_names) == 1:
        column_names = col_line.split()

    # Data starts after column names
    data_lines = lines[header_end + 2:]
    rows = []
    for line in data_lines:
        if not line.strip():
            continue
        vals = line.split("\t")
        if len(vals) == 1:
            vals = line.split()
        rows.append([float(v) for v in vals])

    data = np.array(rows)  # (T, n_cols)

    # Estimate fps from time column (first column)
    if data.shape[0] > 1:
        dt = np.median(np.diff(data[:, 0]))
        fps = 1.0 / dt if dt > 0 else 100.0
    else:
        fps = 100.0

    return fps, column_names, data


def load_opencap_trial(
    trc_path: Path,
    mot_path: Path | None = None,
    subject_id: str = "",
) -> BiomechanicalSample | None:
    """Load an OpenCap trial from .trc and optional .mot files.

    Args:
        trc_path: Path to the .trc marker file.
        mot_path: Optional path to .mot file with forces/kinematics.
        subject_id: Subject identifier.

    Returns:
        BiomechanicalSample or None.
    """
    try:
        fps, marker_names, positions = load_trc_file(trc_path)
    except Exception as e:
        logger.warning(f"Failed to parse {trc_path.name}: {e}")
        return None

    n_frames = positions.shape[0]
    if n_frames < 10:
        return None

    subject = SubjectInfo(subject_id=subject_id or trc_path.parent.name)

    # CoM: approximate as mean of all valid markers
    valid_mask = np.all(np.isfinite(positions), axis=-1)
    com_position = np.zeros((n_frames, 3))
    for t in range(n_frames):
        valid = positions[t, valid_mask[t]]
        if len(valid) > 0:
            com_position[t] = valid.mean(axis=0)

    dt = 1.0 / fps
    com_velocity = np.gradient(com_position, dt, axis=0)
    com_acceleration = np.gradient(com_velocity, dt, axis=0)

    # Load dynamics from .mot file if available
    grf = None
    joint_angles = None
    joint_names = []

    if mot_path is not None and mot_path.exists():
        try:
            mot_fps, col_names, mot_data = load_mot_file(mot_path)

            # Look for GRF columns
            grf_cols = []
            for axis_name in ["ground_force_vx", "ground_force_vy", "ground_force_vz"]:
                for j, name in enumerate(col_names):
                    if axis_name in name.lower():
                        grf_cols.append(j)
                        break

            if len(grf_cols) == 3:
                raw_grf = mot_data[:, grf_cols]
                # Resample to marker rate if needed
                if raw_grf.shape[0] != n_frames:
                    indices = np.linspace(0, raw_grf.shape[0] - 1, n_frames).astype(int)
                    grf = raw_grf[indices]
                else:
                    grf = raw_grf

            # Look for joint angle columns (skip time column)
            angle_col_names = [
                n for n in col_names[1:]
                if any(k in n.lower() for k in ["angle", "flexion", "extension", "rotation"])
            ]
            if angle_col_names:
                angle_indices = [col_names.index(n) for n in angle_col_names]
                raw_angles = mot_data[:, angle_indices]
                if raw_angles.shape[0] != n_frames:
                    indices = np.linspace(0, raw_angles.shape[0] - 1, n_frames).astype(int)
                    joint_angles = np.radians(raw_angles[indices])
                else:
                    joint_angles = np.radians(raw_angles)
                joint_names = angle_col_names

        except Exception as e:
            logger.warning(f"Failed to parse {mot_path.name}: {e}")

    sample = BiomechanicalSample(
        dataset_name="opencap",
        trial_id=f"{subject.subject_id}_{trc_path.stem}",
        subject=subject,
        movement_type=_classify_opencap_movement(trc_path.stem),
        fps=fps,
        marker_positions=positions,
        marker_names=marker_names,
        com_position=com_position,
        com_velocity=com_velocity,
        com_acceleration=com_acceleration,
        grf=grf,
        joint_angles=joint_angles,
        joint_names=joint_names,
    )

    logger.info(
        f"  {trc_path.stem}: {n_frames} frames @ {fps:.0f} Hz "
        f"({sample.movement_type.value})"
    )
    return sample


def load_opencap(
    data_dir: Path | str | None = None,
    movement_filter: list[MovementType] | None = None,
    max_subjects: int | None = None,
) -> Generator[BiomechanicalSample, None, None]:
    """Iterate over all OpenCap trials in a directory.

    Expects .trc files, with optional paired .mot files (same stem).

    Args:
        data_dir: Root directory. Defaults to data/public/opencap/.
        movement_filter: Only yield matching movement types.
        max_subjects: Limit subjects for debugging.

    Yields:
        BiomechanicalSample for each trial.
    """
    if data_dir is None:
        from src.data_pipeline.registry import DATASET_REGISTRY
        data_dir = DATASET_REGISTRY["opencap"].local_dir

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"OpenCap data directory not found: {data_dir}\n"
            f"Download from: https://simtk.org/projects/opencap"
        )

    trc_files = sorted(data_dir.glob("**/*.trc"))
    if not trc_files:
        raise FileNotFoundError(f"No .trc files found in {data_dir}")

    logger.info(f"Found {len(trc_files)} .trc files in {data_dir}")

    # Track unique subject directories
    seen_subjects = set()
    for trc_path in trc_files:
        subject_id = trc_path.parent.name
        if max_subjects and subject_id not in seen_subjects:
            if len(seen_subjects) >= max_subjects:
                break
            seen_subjects.add(subject_id)

        # Look for paired .mot file
        mot_path = trc_path.with_suffix(".mot")
        if not mot_path.exists():
            # Try _grf.mot or _forces.mot variants
            for suffix in ["_grf.mot", "_forces.mot", "_ik.mot"]:
                candidate = trc_path.with_name(trc_path.stem + suffix)
                if candidate.exists():
                    mot_path = candidate
                    break
            else:
                mot_path = None

        try:
            sample = load_opencap_trial(trc_path, mot_path, subject_id)
        except Exception as e:
            logger.warning(f"Failed to load {trc_path}: {e}")
            continue

        if sample is not None:
            if movement_filter and sample.movement_type not in movement_filter:
                continue
            yield sample
