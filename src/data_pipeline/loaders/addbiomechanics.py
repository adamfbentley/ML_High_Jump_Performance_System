"""AddBiomechanics dataset loader.

Supports two export formats from https://addbiomechanics.org/download_data.html:

1. **B3D format** (.b3d binary files) — richest data, but requires nimblephysics
   which is Linux/macOS only.  Place files in data/public/addbiomechanics/*.b3d

2. **OpenSim text format** (Windows-compatible, no extra packages needed).
   Download the "OpenSim Results" export from the website.  Expected layout::

       data/public/addbiomechanics/
           subject_001/
               subject.json              # optional demographics
               IK/  *.mot                # joint angles (inverse kinematics)
               ID/  *.sto                # joint torques (inverse dynamics)
               GRF/ *.mot                # ground reaction forces
               bodyKinematics/
                   *_pos_global.sto      # CoM position
                   *_vel_global.sto      # CoM velocity
                   *_acc_global.sto      # CoM acceleration

The loader auto-detects which format is available when given a directory.
On Windows, OpenSim text format is used automatically.
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

# Joint names in the standard AddBiomechanics OpenSim model order.
# These map to the Rajagopal 2015 full-body model used by default.
ADDBIOMECH_JOINT_NAMES = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
    "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "arm_flex_r", "arm_add_r", "arm_rot_r",
    "elbow_flex_r", "pro_sup_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l",
    "elbow_flex_l", "pro_sup_l",
]

# Keywords in trial names that suggest jumping movements
_JUMP_KEYWORDS = {"jump", "cmj", "drop", "squat_jump", "hop", "plyometric", "dj"}
_RUN_KEYWORDS = {"run", "sprint", "jog", "gait_fast"}
_WALK_KEYWORDS = {"walk", "gait"}


def _classify_movement(trial_name: str) -> MovementType:
    """Guess movement type from the trial/file name."""
    name_lower = trial_name.lower()
    if any(k in name_lower for k in _JUMP_KEYWORDS):
        if "drop" in name_lower:
            return MovementType.DROP_JUMP
        if "squat" in name_lower:
            return MovementType.SQUAT_JUMP
        if "cmj" in name_lower or "counter" in name_lower:
            return MovementType.COUNTERMOVEMENT_JUMP
        return MovementType.VERTICAL_JUMP
    if any(k in name_lower for k in _RUN_KEYWORDS):
        if "sprint" in name_lower:
            return MovementType.SPRINTING
        return MovementType.RUNNING
    if any(k in name_lower for k in _WALK_KEYWORDS):
        return MovementType.WALKING
    return MovementType.OTHER


def load_b3d_file(filepath: Path) -> list[BiomechanicalSample]:
    """Load a single .b3d file and return one BiomechanicalSample per trial.

    Args:
        filepath: Path to the .b3d binary file.

    Returns:
        List of BiomechanicalSample (one per trial in the file).
    """
    try:
        import nimblephysics as nimble
    except ImportError:
        raise ImportError(
            "nimblephysics is required to load AddBiomechanics .b3d files.\n"
            "Install: pip install nimblephysics\n"
            "See: https://nimblephysics.org/docs/install"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"B3D file not found: {filepath}")

    logger.info(f"Loading {filepath.name}...")

    subject_on_disk = nimble.biomechanics.SubjectOnDisk(str(filepath))

    # Extract subject info
    n_dofs = subject_on_disk.getNumDofs()
    mass_kg = subject_on_disk.getMassKg()
    height_m = subject_on_disk.getHeightM()
    sex = subject_on_disk.getBiologicalSex()

    subject = SubjectInfo(
        subject_id=filepath.stem,
        body_mass_kg=mass_kg if mass_kg > 0 else None,
        height_m=height_m if height_m > 0 else None,
        sex="M" if sex == "male" else ("F" if sex == "female" else None),
    )

    samples = []
    n_trials = subject_on_disk.getNumTrials()

    for trial_idx in range(n_trials):
        trial_name = subject_on_disk.getTrialName(trial_idx)
        n_frames = subject_on_disk.getTrialLength(trial_idx)
        timestep = subject_on_disk.getTrialTimestep(trial_idx)
        fps = 1.0 / timestep if timestep > 0 else 100.0

        if n_frames < 10:
            logger.debug(f"  Skipping trial {trial_name}: only {n_frames} frames")
            continue

        # Load all frames for this trial
        frames = subject_on_disk.readFrames(
            trial=trial_idx,
            startFrame=0,
            numFramesToRead=n_frames,
        )

        # Pre-allocate arrays
        joint_angles = np.zeros((n_frames, n_dofs))
        joint_velocities = np.zeros((n_frames, n_dofs))
        joint_accelerations = np.zeros((n_frames, n_dofs))
        joint_torques = np.zeros((n_frames, n_dofs))
        grf_data = []
        com_pos = np.zeros((n_frames, 3))
        com_vel = np.zeros((n_frames, 3))
        com_acc = np.zeros((n_frames, 3))

        valid_frames = 0
        for i, frame in enumerate(frames):
            if frame is None:
                continue

            joint_angles[i] = frame.pos
            joint_velocities[i] = frame.vel
            joint_accelerations[i] = frame.acc
            joint_torques[i] = frame.tau
            com_pos[i] = frame.comPos
            com_vel[i] = frame.comVel
            com_acc[i] = frame.comAcc

            # GRF: sum of all contact forces
            ground_forces = frame.groundContactForce
            if len(ground_forces) >= 3:
                # Sum all contact plate forces (each is 3D)
                total_grf = np.zeros(3)
                for f_idx in range(0, len(ground_forces), 3):
                    total_grf += ground_forces[f_idx:f_idx + 3]
                grf_data.append(total_grf)
            else:
                grf_data.append(np.zeros(3))

            valid_frames += 1

        if valid_frames < 10:
            continue

        grf_array = np.array(grf_data[:n_frames])

        sample = BiomechanicalSample(
            dataset_name="addbiomechanics",
            trial_id=f"{filepath.stem}_{trial_name}",
            subject=subject,
            movement_type=_classify_movement(trial_name),
            fps=fps,
            joint_angles=joint_angles,
            joint_angular_velocities=joint_velocities,
            joint_angular_accelerations=joint_accelerations,
            joint_names=ADDBIOMECH_JOINT_NAMES[:n_dofs],
            com_position=com_pos,
            com_velocity=com_vel,
            com_acceleration=com_acc,
            grf=grf_array,
            joint_torques=joint_torques,
        )

        warnings = sample.validate()
        if warnings:
            logger.debug(f"  Trial {trial_name} warnings: {warnings}")

        samples.append(sample)
        logger.info(
            f"  Trial {trial_name}: {n_frames} frames @ {fps:.0f} Hz "
            f"({sample.movement_type.value})"
        )

    logger.info(f"Loaded {len(samples)} trials from {filepath.name}")
    return samples


def load_addbiomechanics(
    data_dir: Path | str | None = None,
    movement_filter: list[MovementType] | None = None,
    max_subjects: int | None = None,
) -> Generator[BiomechanicalSample, None, None]:
    """Iterate over all AddBiomechanics samples in a directory.

    Auto-detects the format present:
    - If ``.b3d`` files are found → uses the nimblephysics loader (Linux/macOS).
    - If subject sub-directories with ``IK/`` folders are found → uses the
      OpenSim text-file loader (Windows-compatible, no extra packages).

    Args:
        data_dir: Root directory containing exported data. Defaults to
                  ``data/public/addbiomechanics/``.
        movement_filter: Only yield samples matching these movement types.
        max_subjects: Limit number of subjects processed (for debugging).

    Yields:
        BiomechanicalSample for each trial.
    """
    if data_dir is None:
        from src.data_pipeline.registry import DATASET_REGISTRY
        data_dir = DATASET_REGISTRY["addbiomechanics"].local_dir

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"AddBiomechanics data directory not found: {data_dir}\n"
            f"Download from: https://addbiomechanics.org/download_data.html\n"
            f"Place exported data in: {data_dir}"
        )

    # Auto-detect format -------------------------------------------------------
    b3d_files = sorted(data_dir.glob("**/*.b3d"))
    opensim_subjects = [
        p for p in sorted(data_dir.iterdir())
        if p.is_dir() and (p / "IK").exists()
    ]

    if b3d_files:
        logger.info(f"Found {len(b3d_files)} .b3d files → using nimblephysics loader")
        yield from _load_b3d_format(b3d_files, movement_filter, max_subjects)
    elif opensim_subjects:
        logger.info(
            f"Found {len(opensim_subjects)} OpenSim subject folders "
            f"→ using text-file loader (Windows-compatible)"
        )
        yield from _load_opensim_format(opensim_subjects, movement_filter, max_subjects)
    else:
        raise FileNotFoundError(
            f"No AddBiomechanics data found in {data_dir}.\n"
            "Expected either:\n"
            "  • .b3d files (requires nimblephysics, Linux/macOS only)\n"
            "  • subject_*/IK/*.mot folders (OpenSim export, works on Windows)\n"
            "See: https://addbiomechanics.org/download_data.html"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_b3d_format(
    b3d_files: list[Path],
    movement_filter: list[MovementType] | None,
    max_subjects: int | None,
) -> Generator[BiomechanicalSample, None, None]:
    """Load samples from .b3d files (Linux/macOS only, requires nimblephysics)."""
    for i, b3d_path in enumerate(b3d_files):
        if max_subjects is not None and i >= max_subjects:
            break
        try:
            samples = load_b3d_file(b3d_path)
        except Exception as e:
            logger.warning(f"Failed to load {b3d_path.name}: {e}")
            continue
        for sample in samples:
            if movement_filter is None or sample.movement_type in movement_filter:
                yield sample


def _load_opensim_format(
    subject_dirs: list[Path],
    movement_filter: list[MovementType] | None,
    max_subjects: int | None,
) -> Generator[BiomechanicalSample, None, None]:
    """Load samples from OpenSim text exports (Windows-compatible)."""
    for i, subject_dir in enumerate(subject_dirs):
        if max_subjects is not None and i >= max_subjects:
            break
        try:
            yield from _load_opensim_subject(subject_dir, movement_filter)
        except Exception as e:
            logger.warning(f"Failed to load subject {subject_dir.name}: {e}")


def _parse_opensim_file(filepath: Path) -> tuple[float, list[str], np.ndarray]:
    """Parse an OpenSim .mot or .sto text file.

    Returns:
        Tuple of (fps, column_names, data_array).
        fps is estimated from the time column (column 0).
    """
    filepath = Path(filepath)
    header_done = False
    n_rows = None
    col_names: list[str] = []
    rows: list[list[float]] = []

    with filepath.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if not header_done:
                low = line.lower()
                if low == "endheader":
                    header_done = True
                    continue
                if low.startswith("nrows"):
                    try:
                        n_rows = int(line.split("=")[1].strip())
                    except (IndexError, ValueError):
                        pass
                continue

            # First non-header line is the column names
            if not col_names:
                col_names = line.split()
                continue

            # Data rows
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No data rows found in {filepath}")

    data = np.array(rows, dtype=np.float64)

    # Estimate fps from time column (column 0)
    if data.shape[0] > 1 and len(col_names) > 0 and col_names[0].lower() == "time":
        dt = float(np.median(np.diff(data[:, 0])))
        fps = 1.0 / dt if dt > 0 else 100.0
    else:
        fps = 100.0

    return fps, col_names, data


def _load_opensim_subject(
    subject_dir: Path,
    movement_filter: list[MovementType] | None,
) -> Generator[BiomechanicalSample, None, None]:
    """Load all trials from one AddBiomechanics OpenSim export subject folder."""
    import json

    # Optional: read subject demographics from subject.json
    subject_json = subject_dir / "subject.json"
    mass_kg: float | None = None
    height_m: float | None = None
    sex: str | None = None
    if subject_json.exists():
        try:
            meta = json.loads(subject_json.read_text())
            mass_kg = float(meta.get("massKg", 0) or 0) or None
            height_m = float(meta.get("heightM", 0) or 0) or None
            sex_raw = str(meta.get("sex", "")).lower()
            sex = "M" if sex_raw == "male" else ("F" if sex_raw == "female" else None)
        except Exception:
            pass

    subject = SubjectInfo(
        subject_id=subject_dir.name,
        body_mass_kg=mass_kg,
        height_m=height_m,
        sex=sex,
    )

    ik_dir = subject_dir / "IK"
    id_dir = subject_dir / "ID"
    grf_dir = subject_dir / "GRF"
    bk_dir = subject_dir / "bodyKinematics"

    ik_files = sorted(ik_dir.glob("*.mot")) if ik_dir.exists() else []
    if not ik_files:
        logger.debug(f"No IK .mot files in {ik_dir}")
        return

    for ik_file in ik_files:
        trial_name = ik_file.stem  # e.g. "subject01_cmj_01_ik" or "trial_01_ik"
        # Strip common suffixes to get a clean base name for pairing
        base = trial_name
        for suffix in ("_ik", "_IK", "_kinematics"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        movement_type = _classify_movement(trial_name)
        if movement_filter is not None and movement_type not in movement_filter:
            continue

        try:
            fps, ik_cols, ik_data = _parse_opensim_file(ik_file)
        except Exception as e:
            logger.warning(f"  Skipping {ik_file.name}: {e}")
            continue

        n_frames = ik_data.shape[0]
        if n_frames < 10:
            continue

        # Joint angles: all columns except 'time'
        time_col = ik_cols[0].lower() == "time"
        data_start = 1 if time_col else 0
        joint_names = ik_cols[data_start:]
        joint_angles = np.deg2rad(ik_data[:, data_start:])  # degrees → radians

        # Torques from ID folder
        joint_torques: np.ndarray | None = None
        id_candidates = list((id_dir).glob(f"{base}*.sto")) if id_dir.exists() else []
        if id_candidates:
            try:
                _, id_cols, id_data = _parse_opensim_file(id_candidates[0])
                t_start = 1 if id_cols[0].lower() == "time" else 0
                # Align columns to IK joint order where possible
                id_joint_cols = id_cols[t_start:]
                torque_arr = np.zeros((n_frames, len(joint_names)))
                for ji, jname in enumerate(joint_names):
                    matches = [k for k, c in enumerate(id_joint_cols) if c == jname]
                    if matches:
                        src_col = t_start + matches[0]
                        torque_arr[:, ji] = id_data[:min(n_frames, id_data.shape[0]), src_col]
                joint_torques = torque_arr
            except Exception as e:
                logger.debug(f"  Could not load ID for {base}: {e}")

        # GRF from GRF folder
        grf: np.ndarray | None = None
        grf_candidates = (
            list(grf_dir.glob(f"{base}*.mot")) if grf_dir.exists() else []
        )
        if grf_candidates:
            try:
                _, grf_cols, grf_data = _parse_opensim_file(grf_candidates[0])
                g_start = 1 if grf_cols[0].lower() == "time" else 0
                grf_cols_lower = [c.lower() for c in grf_cols[g_start:]]
                # Sum left + right vertical GRF (columns containing 'vy' or 'ground_force_vy')
                vy_indices = [k for k, c in enumerate(grf_cols_lower) if "vy" in c or "fz" in c]
                fx_indices = [k for k, c in enumerate(grf_cols_lower) if ("vx" in c and "vy" not in c) or "fx" in c]
                fz_indices = [k for k, c in enumerate(grf_cols_lower) if ("vz" in c) or ("fz" in c and "vy" not in c)]
                grf_frames = min(n_frames, grf_data.shape[0])
                grf_arr = np.zeros((n_frames, 3))
                if vy_indices:
                    grf_arr[:grf_frames, 1] = grf_data[:grf_frames, g_start + vy_indices[0]]
                if fx_indices:
                    grf_arr[:grf_frames, 0] = grf_data[:grf_frames, g_start + fx_indices[0]]
                if fz_indices:
                    grf_arr[:grf_frames, 2] = grf_data[:grf_frames, g_start + fz_indices[0]]
                grf = grf_arr
            except Exception as e:
                logger.debug(f"  Could not load GRF for {base}: {e}")

        # CoM kinematics from bodyKinematics folder
        com_pos = com_vel = com_acc = None
        if bk_dir.exists():
            for suffix, attr in [("_pos_global", "pos"), ("_vel_global", "vel"), ("_acc_global", "acc")]:
                candidates = list(bk_dir.glob(f"{base}*{suffix}*.sto"))
                if not candidates:
                    continue
                try:
                    _, bk_cols, bk_data = _parse_opensim_file(candidates[0])
                    bk_start = 1 if bk_cols[0].lower() == "time" else 0
                    arr = bk_data[:min(n_frames, bk_data.shape[0]), bk_start:bk_start + 3]
                    padded = np.zeros((n_frames, 3))
                    padded[:arr.shape[0]] = arr
                    if attr == "pos":
                        com_pos = padded
                    elif attr == "vel":
                        com_vel = padded
                    else:
                        com_acc = padded
                except Exception:
                    pass

        sample = BiomechanicalSample(
            dataset_name="addbiomechanics",
            trial_id=f"{subject_dir.name}_{trial_name}",
            subject=subject,
            movement_type=movement_type,
            fps=fps,
            joint_angles=joint_angles,
            joint_names=joint_names,
            joint_torques=joint_torques,
            grf=grf,
            com_position=com_pos,
            com_velocity=com_vel,
            com_acceleration=com_acc,
        )

        warnings = sample.validate()
        if warnings:
            logger.debug(f"  {trial_name} warnings: {warnings}")

        logger.info(
            f"  {trial_name}: {n_frames} frames @ {fps:.0f} Hz ({movement_type.value})"
        )
        yield sample
