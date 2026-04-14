"""Drop Vertical Jump (DVJ) Zenodo dataset loader.

Reads the DVJ dataset from Zenodo record 18503500.
Contains marker-based motion capture (.trc) and ground reaction forces (.mot)
for drop vertical jump trials at different heights and VR conditions.

Reference:
    Wu, J. et al. (2025). A Comprehensive Dataset of Drop Vertical Jump
    Biomechanics: Integrating Kinematics, EMG, and Force Plate Data.
    Zenodo. https://doi.org/10.5281/zenodo.18503500

Expected layout after extraction::

    data/public/dvj_opensim_zenodo/
        Data/
            Kinematic Data/
                Subject1/
                    Opensim/
                        30cm/
                            1.trc, 1.mot, 2.trc, 2.mot, ...
                        VR0cm/
                            ...
                        VR10cm/ VR30cm/ VR50cm/
                    C3D/ ASCII/
                Subject2/ ...
            Data Processing/
                participant log.xlsx
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

# All DVJ conditions map to DROP_JUMP
_CONDITION_MAP = {
    "30cm": MovementType.DROP_JUMP,
    "VR0cm": MovementType.DROP_JUMP,
    "VR10cm": MovementType.DROP_JUMP,
    "VR30cm": MovementType.DROP_JUMP,
    "VR50cm": MovementType.DROP_JUMP,
}

# Pelvis markers used to approximate whole-body CoM
_PELVIS_MARKERS = ["LASI", "RASI", "LPSI", "RPSI"]


def _load_participant_log(log_path: Path) -> dict[int, dict]:
    """Load participant demographics from the Excel log.

    Returns:
        Dict mapping subject ID (int) to {mass_kg, height_m, sex, age}.
    """
    try:
        import openpyxl
    except ImportError:
        logger.warning("openpyxl not installed — cannot read participant log")
        return {}

    wb = openpyxl.load_workbook(log_path, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(min_row=1, values_only=True))
    wb.close()
    if not rows:
        return {}

    header = [str(h).strip() if h else "" for h in rows[0]]
    info: dict[int, dict] = {}

    for row in rows[1:]:
        if row[0] is None:
            continue
        vals = dict(zip(header, row))
        sid = int(vals.get("ID", 0))
        if sid == 0:
            continue

        height_cm = vals.get("Height(cm)")
        weight_kg = vals.get("Weight(Kg)")
        gender = vals.get("Gender", "")

        info[sid] = {
            "mass_kg": float(weight_kg) if weight_kg else None,
            "height_m": float(height_cm) / 100.0 if height_cm else None,
            "sex": "M" if str(gender).lower() == "male" else "F",
            "age": int(vals.get("Age(years)", 0)) or None,
        }

    return info


def _parse_trc(filepath: Path) -> tuple[float, list[str], np.ndarray]:
    """Parse a .trc motion-capture marker file.

    Returns:
        (fps, marker_names, positions) where positions is (T, N_markers, 3)
        in metres.
    """
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"TRC file too short: {filepath}")

    # Line 3 has metadata: DataRate CameraRate NumFrames NumMarkers Units ...
    meta = lines[2].split()
    fps = float(meta[0])
    num_frames = int(meta[2])
    num_markers = int(meta[3])
    units = meta[4].lower() if len(meta) > 4 else "mm"

    # Line 4 has marker names (tab-separated), starts with Frame# Time
    marker_line = lines[3].split("\t")
    marker_names = [m.strip() for m in marker_line[2:] if m.strip()]
    # Remove duplicates from the 3-column expansion
    marker_names = marker_names[:num_markers]

    # Data starts at line 7 (0-indexed: line 6)
    rows = []
    for line in lines[6:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            vals = [float(v) if v.strip() else float("nan") for v in parts]
            rows.append(vals)
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No data rows in {filepath}")

    data = np.array(rows, dtype=np.float64)
    # Skip Frame# (col 0) and Time (col 1)
    marker_data = data[:, 2:]

    n_cols = marker_data.shape[1]
    actual_markers = n_cols // 3
    if actual_markers < num_markers:
        num_markers = actual_markers

    positions = marker_data[:, : num_markers * 3].reshape(-1, num_markers, 3)

    # Convert units to metres
    if units == "mm":
        positions /= 1000.0

    return fps, marker_names, positions


def _parse_grf_mot(filepath: Path) -> tuple[float, np.ndarray]:
    """Parse a .mot GRF file.

    Returns:
        (fps, grf_array) where grf_array is (T, 3) total GRF in N.
        Sums force plate 1 and force plate 2.
    """
    header_done = False
    col_names: list[str] = []
    rows: list[list[float]] = []

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
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

    # Map column names to indices
    col_lower = {c.lower(): i for i, c in enumerate(col_names)}

    grf_total = np.zeros((data.shape[0], 3))

    # Force plate 1 (right): ground_force1_vx/vy/vz
    for ax_idx, ax in enumerate(["vx", "vy", "vz"]):
        c1 = col_lower.get(f"ground_force1_{ax}")
        c2 = col_lower.get(f"ground_force2_{ax}")
        if c1 is not None:
            grf_total[:, ax_idx] += data[:, c1]
        if c2 is not None:
            grf_total[:, ax_idx] += data[:, c2]

    # Compute fps
    if data.shape[0] > 1:
        dt = float(np.median(np.diff(data[:, 0])))
        fps = 1.0 / dt if dt > 0 else 1000.0
    else:
        fps = 1000.0

    return fps, grf_total


def _estimate_com_from_pelvis(
    positions: np.ndarray, marker_names: list[str]
) -> np.ndarray | None:
    """Approximate CoM from pelvis marker centroid (T, 3) in metres."""
    indices = []
    for pm in _PELVIS_MARKERS:
        for i, mn in enumerate(marker_names):
            if mn.upper() == pm.upper():
                indices.append(i)
                break

    if len(indices) < 2:
        return None

    pelvis_pos = positions[:, indices, :]  # (T, n_pelvis, 3)
    # Mask NaN markers
    valid = ~np.isnan(pelvis_pos).any(axis=2)  # (T, n_pelvis)
    com = np.nanmean(pelvis_pos, axis=1)  # (T, 3)

    # If all markers invalid for a frame, propagate NaN
    all_invalid = ~valid.any(axis=1)
    com[all_invalid] = np.nan

    return com


def _load_trial(
    trc_path: Path,
    mot_path: Path,
    condition: str,
    subject: SubjectInfo,
) -> BiomechanicalSample | None:
    """Load a single DVJ trial from marker + GRF files."""
    try:
        marker_fps, marker_names, positions = _parse_trc(trc_path)
        grf_fps, grf_raw = _parse_grf_mot(mot_path)
    except Exception as e:
        logger.warning(f"Failed to parse {trc_path.stem}: {e}")
        return None

    n_frames = positions.shape[0]
    if n_frames < 10:
        return None

    dt = 1.0 / marker_fps

    # ── CoM from pelvis markers ──
    com_position = _estimate_com_from_pelvis(positions, marker_names)
    com_velocity = None
    com_acceleration = None

    if com_position is not None and not np.isnan(com_position).all():
        # Interpolate any NaN frames
        for ax in range(3):
            nans = np.isnan(com_position[:, ax])
            if nans.any() and not nans.all():
                good = ~nans
                com_position[nans, ax] = np.interp(
                    np.where(nans)[0], np.where(good)[0], com_position[good, ax]
                )

        com_velocity = np.gradient(com_position, dt, axis=0)
        com_acceleration = np.gradient(com_velocity, dt, axis=0)

    # ── GRF: downsample to marker frame rate ──
    n_grf = grf_raw.shape[0]
    if n_grf != n_frames:
        indices = np.linspace(0, n_grf - 1, n_frames).astype(int)
        grf = grf_raw[indices]
    else:
        grf = grf_raw

    # ── Basic joint angles from markers (knee angle estimation) ──
    # This gives the PINN some kinematic context beyond just CoM
    joint_angles = None
    joint_names = None

    # Estimate knee angles from thigh-shank-ankle markers
    knee_angles = _estimate_knee_angles(positions, marker_names)
    if knee_angles is not None:
        joint_angles = knee_angles  # (T, 2) left, right knee
        joint_names = ["knee_angle_l", "knee_angle_r"]

    joint_vels = None
    joint_accs = None
    if joint_angles is not None:
        joint_vels = np.gradient(joint_angles, dt, axis=0)
        joint_accs = np.gradient(joint_vels, dt, axis=0)

    trial_name = trc_path.stem
    movement_type = _CONDITION_MAP.get(condition, MovementType.DROP_JUMP)

    sample = BiomechanicalSample(
        dataset_name="dvj_opensim_zenodo",
        trial_id=f"{subject.subject_id}_{condition}_{trial_name}",
        subject=subject,
        movement_type=movement_type,
        fps=marker_fps,
        joint_angles=joint_angles,
        joint_angular_velocities=joint_vels,
        joint_angular_accelerations=joint_accs,
        joint_names=joint_names,
        joint_torques=None,  # no ID output in this dataset
        grf=grf,
        com_position=com_position,
        com_velocity=com_velocity,
        com_acceleration=com_acceleration,
    )

    peak_grf = np.max(grf[:, 1]) if grf.shape[1] > 1 else np.max(grf)
    logger.info(
        f"  {trial_name} ({condition}): {n_frames} frames @ {marker_fps:.0f} Hz "
        f"| GRF peak={peak_grf:.0f} N"
    )

    # Skip trials where force plate did not record (no physics signal)
    if peak_grf < 100.0:
        logger.debug(f"  Skipping {trial_name}: GRF peak {peak_grf:.0f} N < 100 N threshold")
        return None

    return sample


def _estimate_knee_angles(
    positions: np.ndarray, marker_names: list[str]
) -> np.ndarray | None:
    """Estimate left and right knee flexion angles from markers.

    Uses hip-knee-ankle marker triplets. Returns (T, 2) in radians.
    """
    name_map = {mn.upper(): i for i, mn in enumerate(marker_names)}

    results = []
    for side_prefix in ["L", "R"]:
        # Hip: midpoint of ASI + PSI, or just thigh marker
        hip_markers = [f"{side_prefix}ASI", f"{side_prefix}PSI"]
        knee_marker = f"{side_prefix}KNE"
        ankle_marker = f"{side_prefix}ANK"

        hip_idx = [name_map.get(m) for m in hip_markers]
        knee_idx = name_map.get(knee_marker)
        ankle_idx = name_map.get(ankle_marker)

        if knee_idx is None or ankle_idx is None:
            return None

        if all(i is not None for i in hip_idx):
            hip_pos = np.nanmean(positions[:, hip_idx, :], axis=1)  # (T, 3)
        elif name_map.get(f"{side_prefix}THI") is not None:
            hip_pos = positions[:, name_map[f"{side_prefix}THI"], :]
        else:
            return None

        knee_pos = positions[:, knee_idx, :]
        ankle_pos = positions[:, ankle_idx, :]

        # Vectors
        thigh = hip_pos - knee_pos  # (T, 3)
        shank = ankle_pos - knee_pos  # (T, 3)

        # Dot product → angle
        dot = np.sum(thigh * shank, axis=1)
        thigh_norm = np.linalg.norm(thigh, axis=1)
        shank_norm = np.linalg.norm(shank, axis=1)

        denom = thigh_norm * shank_norm
        valid = denom > 1e-8
        cos_angle = np.ones(len(dot))
        cos_angle[valid] = np.clip(dot[valid] / denom[valid], -1, 1)

        # Convert to knee flexion (pi - included angle)
        angle = np.pi - np.arccos(cos_angle)
        results.append(angle)

    return np.column_stack(results)  # (T, 2)


def _save_cache(samples: list["BiomechanicalSample"], cache_path: Path) -> None:
    """Serialise a list of BiomechanicalSamples to a compressed .npz cache."""
    import pickle
    # Use pickle for full object fidelity (numpy arrays + Python attrs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"DVJ cache written: {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")


def _load_cache(cache_path: Path) -> list["BiomechanicalSample"] | None:
    """Load cache produced by _save_cache; returns None if missing or corrupt."""
    import pickle
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
        logger.info(f"DVJ cache loaded: {len(samples)} samples from {cache_path}")
        return samples
    except Exception as e:
        logger.warning(f"DVJ cache corrupt, rebuilding: {e}")
        return None


def load_dvj_zenodo(
    data_dir: Path | str | None = None,
    movement_filter: list[MovementType] | None = None,
    max_subjects: int | None = None,
    use_cache: bool = True,
) -> Generator["BiomechanicalSample", None, None]:
    """Iterate over all DVJ Zenodo trials.

    Uses OpenSim .trc (markers) + .mot (GRF) files.  On first call the full
    dataset (~4 h on Windows) is parsed and saved to ``dvj_cache.pkl`` in
    ``data_dir``.  Subsequent calls load from cache in ~30 s.

    Args:
        data_dir: Root directory. Defaults to data/public/dvj_opensim_zenodo/.
        movement_filter: Only yield matching movement types.
        max_subjects: Limit number of subjects (bypasses/ignores cache).
        use_cache: If True and max_subjects is None, read/write a cache file.

    Yields:
        BiomechanicalSample for each trial.
    """
    if data_dir is None:
        from src.data_pipeline.registry import DATASET_REGISTRY
        data_dir = DATASET_REGISTRY["dvj_opensim_zenodo"].local_dir

    data_dir = Path(data_dir)
    kin_dir = data_dir / "Data" / "Kinematic Data"

    if not kin_dir.exists():
        raise FileNotFoundError(
            f"DVJ kinematic data not found: {kin_dir}\n"
            f"Download from: https://zenodo.org/records/18503500"
        )

    # ── Cache: skip slow text parsing on repeat runs ──────────────────────
    cache_path = data_dir / "dvj_cache.pkl"
    if use_cache and max_subjects is None:
        cached = _load_cache(cache_path)
        if cached is not None:
            for s in cached:
                if movement_filter is None or s.movement_type in movement_filter:
                    yield s
            return

    # Load participant demographics
    log_path = data_dir / "Data" / "Data Processing" / "participant log.xlsx"
    participant_info = _load_participant_log(log_path) if log_path.exists() else {}

    subject_dirs = sorted(
        p for p in kin_dir.iterdir()
        if p.is_dir() and p.name.startswith("Subject")
    )
    if not subject_dirs:
        raise FileNotFoundError(f"No Subject* folders in {kin_dir}")

    logger.info(f"Found {len(subject_dirs)} subjects in {kin_dir}")

    # ── Parse all files, collect into a list, then yield ─────────────────
    # This allows us to write the cache after a full parse (one-time cost).
    all_samples: list[BiomechanicalSample] = []
    loaded = 0
    for sdir in subject_dirs:
        if max_subjects is not None and loaded >= max_subjects:
            break

        # Extract subject ID number
        try:
            sid = int(sdir.name.replace("Subject", ""))
        except ValueError:
            continue

        info = participant_info.get(sid, {})
        subject = SubjectInfo(
            subject_id=sdir.name,
            body_mass_kg=info.get("mass_kg"),
            height_m=info.get("height_m"),
            sex=info.get("sex"),
        )

        opensim_dir = sdir / "Opensim"
        if not opensim_dir.exists():
            continue

        has_trials = False
        for cond_dir in sorted(opensim_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            condition = cond_dir.name
            if condition not in _CONDITION_MAP:
                continue

            trc_files = sorted(cond_dir.glob("*.trc"))
            for trc_path in trc_files:
                # Skip calibration files
                if "cal" in trc_path.stem.lower():
                    continue

                mot_path = trc_path.with_suffix(".mot")
                if not mot_path.exists():
                    continue

                sample = _load_trial(trc_path, mot_path, condition, subject)
                if sample is None:
                    continue

                has_trials = True
                all_samples.append(sample)

        if has_trials:
            loaded += 1

    # ── Save cache before yielding (only full-dataset runs) ───────────────
    if use_cache and max_subjects is None and all_samples:
        _save_cache(all_samples, cache_path)

    # ── Yield, applying movement filter ───────────────────────────────────
    for s in all_samples:
        if movement_filter is None or s.movement_type in movement_filter:
            yield s
