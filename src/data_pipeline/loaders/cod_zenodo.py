"""Change-of-direction (CoD) Zenodo dataset loader.

Reads the Nitschke et al. (2022) CoD dataset from Zenodo record 6949012.
Contains running and v-cut change-of-direction trials with full inverse
kinematics, inverse dynamics, and measured ground reaction forces.

Reference:
    Nitschke, M. et al. (2022). Optical motion capturing of change of
    direction motions reconstructed with inverse kinematics and dynamics
    and optimal control simulation. Zenodo.
    https://doi.org/10.5281/zenodo.6949012

Expected layout after extraction::

    data/public/cod_ik_id_zenodo/
        metaInfo.csv
        Participant_02/
            Participant_02.osim
            curvedslowrunning/
                trial0100_inverse_methods_coordinates.mot
                trial0100_inverse_methods_moments.sto
                trial0100_measured_GRFs.mot
                ...
            straightslowrunning/
                ...
            vcut/
                ...
        Participant_03/
            ...
"""

from __future__ import annotations

import csv
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

_COD_MOVEMENT_MAP = {
    "curvedslowrunning": MovementType.RUNNING,
    "straightslowrunning": MovementType.RUNNING,
    "vcut": MovementType.OTHER,
}

# Translational DOFs that should NOT be converted from degrees
_TRANSLATIONAL_DOFS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}


def _parse_opensim_file(filepath: Path) -> tuple[float, list[str], np.ndarray]:
    """Parse an OpenSim .mot or .sto text file.

    Returns:
        (fps, column_names, data_array).
    """
    header_done = False
    col_names: list[str] = []
    rows: list[list[float]] = []

    with filepath.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_done:
                if line.lower() == "endheader":
                    header_done = True
                continue
            if not col_names:
                col_names = line.split("\t")
                if len(col_names) == 1:
                    col_names = line.split()
                continue
            try:
                vals = line.split("\t")
                if len(vals) == 1:
                    vals = line.split()
                rows.append([float(v) for v in vals])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No data rows in {filepath}")

    data = np.array(rows, dtype=np.float64)

    if data.shape[0] > 1 and col_names and col_names[0].lower() == "time":
        dt = float(np.median(np.diff(data[:, 0])))
        fps = 1.0 / dt if dt > 0 else 100.0
    else:
        fps = 100.0

    return fps, col_names, data


def _load_meta_info(csv_path: Path) -> dict[str, dict]:
    """Load metaInfo.csv and return per-participant demographics.

    Returns:
        Dict mapping participant_id (e.g. "02") to
        {height_m, mass_kg, age, sex}.
    """
    meta: dict[str, dict] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["Participant_ID"]
            if pid not in meta:
                meta[pid] = {
                    "height_m": float(row["Bodyheight_in_m"]),
                    "mass_kg": float(row["Bodymass_in_kg"]),
                    "age": int(row["Age_in_y"]),
                    "sex": "M" if row["Sex"] == "male" else "F",
                }
    return meta


def _discover_trials(participant_dir: Path) -> list[tuple[str, Path, Path, Path]]:
    """Find all trials in a participant folder.

    Returns:
        List of (motion_type, ik_path, id_path, grf_path) tuples.
    """
    trials = []
    for motion_dir in sorted(participant_dir.iterdir()):
        if not motion_dir.is_dir():
            continue
        motion_type = motion_dir.name

        ik_files = sorted(motion_dir.glob("*_inverse_methods_coordinates.mot"))
        for ik_path in ik_files:
            # Extract base trial name (e.g., "trial0100")
            base = ik_path.stem.replace("_inverse_methods_coordinates", "")
            id_path = motion_dir / f"{base}_inverse_methods_moments.sto"
            grf_path = motion_dir / f"{base}_measured_GRFs.mot"

            if id_path.exists() and grf_path.exists():
                trials.append((motion_type, ik_path, id_path, grf_path))
            else:
                logger.debug(f"Skipping {base}: missing ID or GRF file")

    return trials


def _load_trial(
    motion_type: str,
    ik_path: Path,
    id_path: Path,
    grf_path: Path,
    subject: SubjectInfo,
) -> BiomechanicalSample | None:
    """Load a single CoD trial from its three data files."""
    try:
        ik_fps, ik_cols, ik_data = _parse_opensim_file(ik_path)
        id_fps, id_cols, id_data = _parse_opensim_file(id_path)
        grf_fps, grf_cols, grf_data = _parse_opensim_file(grf_path)
    except Exception as e:
        logger.warning(f"Failed to parse {ik_path.stem}: {e}")
        return None

    n_frames = ik_data.shape[0]
    if n_frames < 10:
        return None

    # ── Joint angles (IK) ──
    # Skip the time column. Convert degrees to radians (except translations).
    time_col = 1 if ik_cols[0].lower() == "time" else 0
    joint_names = ik_cols[time_col:]
    raw_angles = ik_data[:, time_col:]

    joint_angles = np.zeros_like(raw_angles)
    for j, jname in enumerate(joint_names):
        if jname in _TRANSLATIONAL_DOFS:
            joint_angles[:, j] = raw_angles[:, j]  # already in metres
        else:
            joint_angles[:, j] = np.deg2rad(raw_angles[:, j])

    # Compute angular velocities and accelerations
    dt = 1.0 / ik_fps
    joint_vels = np.gradient(joint_angles, dt, axis=0)
    joint_accs = np.gradient(joint_vels, dt, axis=0)

    # ── Joint torques (ID) ──
    id_start = 1 if id_cols[0].lower() == "time" else 0
    id_joint_cols = id_cols[id_start:]
    n_id_frames = min(n_frames, id_data.shape[0])

    joint_torques = np.zeros((n_frames, len(joint_names)))
    for ji, jname in enumerate(joint_names):
        # Try to match column name with or without _moment/_force suffix
        for ci, cname in enumerate(id_joint_cols):
            stripped = cname.replace("_moment", "").replace("_force", "")
            if stripped == jname:
                joint_torques[:n_id_frames, ji] = id_data[:n_id_frames, id_start + ci]
                break

    # ── GRF ──
    # Sum left + right foot forces. Downsample from force plate rate to IK rate.
    g_start = 1 if grf_cols[0].lower() == "time" else 0
    grf_col_lower = [c.lower() for c in grf_cols[g_start:]]

    def _find_col(prefix: str, component: str) -> int | None:
        target = f"{prefix}{component}"
        for k, c in enumerate(grf_col_lower):
            if c == target:
                return g_start + k
        return None

    grf_array = np.zeros((n_frames, 3))
    n_grf_frames = grf_data.shape[0]

    # Right foot: ground_force_v{x,y,z}
    # Left foot: l_ground_force_v{x,y,z}
    for ax_idx, ax in enumerate(["vx", "vy", "vz"]):
        r_col = _find_col("ground_force_", ax)
        l_col = _find_col("l_ground_force_", ax)
        total = np.zeros(n_grf_frames)
        if r_col is not None:
            total += grf_data[:, r_col]
        if l_col is not None:
            total += grf_data[:, l_col]

        # Downsample to IK frame count
        if n_grf_frames != n_frames:
            indices = np.linspace(0, n_grf_frames - 1, n_frames).astype(int)
            grf_array[:, ax_idx] = total[indices]
        else:
            grf_array[:, ax_idx] = total

    # ── CoM estimation from pelvis translation ──
    # pelvis_tx/ty/tz approximate whole-body CoM in the absence of
    # bodyKinematics output. This is a reasonable proxy for running/cutting.
    com_position = None
    com_velocity = None
    com_acceleration = None
    pelvis_cols = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
    pelvis_indices = []
    for pc in pelvis_cols:
        for ji, jn in enumerate(joint_names):
            if jn == pc:
                pelvis_indices.append(ji)
                break
    if len(pelvis_indices) == 3:
        com_position = joint_angles[:, pelvis_indices]  # metres (not radians)
        com_velocity = np.gradient(com_position, dt, axis=0)
        com_acceleration = np.gradient(com_velocity, dt, axis=0)

    trial_base = ik_path.stem.replace("_inverse_methods_coordinates", "")
    movement_type = _COD_MOVEMENT_MAP.get(motion_type, MovementType.OTHER)

    sample = BiomechanicalSample(
        dataset_name="cod_ik_id_zenodo",
        trial_id=f"{subject.subject_id}_{motion_type}_{trial_base}",
        subject=subject,
        movement_type=movement_type,
        fps=ik_fps,
        joint_angles=joint_angles,
        joint_angular_velocities=joint_vels,
        joint_angular_accelerations=joint_accs,
        joint_names=list(joint_names),
        joint_torques=joint_torques,
        grf=grf_array,
        com_position=com_position,
        com_velocity=com_velocity,
        com_acceleration=com_acceleration,
    )

    logger.info(
        f"  {trial_base} ({motion_type}): {n_frames} frames @ {ik_fps:.0f} Hz "
        f"| GRF max={np.max(grf_array[:, 1]):.0f} N"
    )
    return sample


def load_cod_zenodo(
    data_dir: Path | str | None = None,
    movement_filter: list[MovementType] | None = None,
    max_subjects: int | None = None,
) -> Generator[BiomechanicalSample, None, None]:
    """Iterate over all CoD Zenodo trials.

    Args:
        data_dir: Root directory. Defaults to data/public/cod_ik_id_zenodo/.
        movement_filter: Only yield matching movement types.
        max_subjects: Limit number of participants for debugging.

    Yields:
        BiomechanicalSample for each trial.
    """
    if data_dir is None:
        from src.data_pipeline.registry import DATASET_REGISTRY
        data_dir = DATASET_REGISTRY["cod_ik_id_zenodo"].local_dir

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"CoD data directory not found: {data_dir}\n"
            f"Download from: https://zenodo.org/records/6949012"
        )

    # Load participant demographics
    meta_path = data_dir / "metaInfo.csv"
    meta = _load_meta_info(meta_path) if meta_path.exists() else {}

    participant_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("Participant_")
    )
    if not participant_dirs:
        raise FileNotFoundError(f"No Participant_* folders in {data_dir}")

    logger.info(f"Found {len(participant_dirs)} participants in {data_dir}")

    loaded = 0
    for pdir in participant_dirs:
        if max_subjects is not None and loaded >= max_subjects:
            break

        pid = pdir.name.replace("Participant_", "")
        info = meta.get(pid, {})

        subject = SubjectInfo(
            subject_id=pdir.name,
            body_mass_kg=info.get("mass_kg"),
            height_m=info.get("height_m"),
            sex=info.get("sex"),
        )

        trials = _discover_trials(pdir)
        if not trials:
            continue

        loaded += 1

        for motion_type, ik_path, id_path, grf_path in trials:
            sample = _load_trial(motion_type, ik_path, id_path, grf_path, subject)
            if sample is None:
                continue
            if movement_filter and sample.movement_type not in movement_filter:
                continue
            yield sample
