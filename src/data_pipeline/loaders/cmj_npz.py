"""CMJ accelerometer + vGRF dataset loader (Zenodo 19136480).

Loads ``cmj_dataset_both.npz`` (or the per-condition variants) produced by the
``acc2grf_prediction`` preprocessing pipeline (White et al., *Sports Biomechanics*,
submitted).  The file is CC-BY-4.0 and directly downloadable from Zenodo.

Dataset summary
---------------
- 67 participants, 663 countermovement jump trials
- Lower-back triaxial accelerometer (L5, Delsys Trigno) at 250 Hz in g units
- Vertical GRF from Kistler force plates, stored at 1000 Hz, body-weight (BW) units
- 500-sample (2 s) pre-takeoff window aligned to the instant of takeoff
- Group demographics: 73.1 ± 13.1 kg, 1.74 ± 0.10 m, 21.6 ± 1.5 years

NPZ keys (from ``src/data_loader.py`` in the source repository):
    acc_signals   — object array of (n_timesteps, 3) float32 arrays, g units
    acc_takeoff   — (N,) int array, takeoff index within each acc_signal
    grf_signals   — object array of (n_timesteps,) float32 arrays, BW units, 1000 Hz
    subject_ids   — (N,) int array, 0-indexed participant IDs
    jump_height   — (N,) float64, metres
    peak_power    — (N,) float64, W/kg
    n_subjects    — scalar

Physics notes
-------------
* Vertical GRF convention (Y-up coordinate system, see constants.py):

    F_vGRF = m * (a_COM_y + g)          Newton's 2nd law

  Rearranging to BW-normalised form:

    vGRF_bw = F_vGRF / (m * g) = a_COM_y / g + 1

  Therefore:

    a_COM_y  = (vGRF_bw - 1.0) * g     [m/s²]
    F_vGRF_N = vGRF_bw * mass_kg * g   [N]

* No individual body masses are stored in the NPZ.  The ``DATASET_MEAN_MASS_KG``
  group mean (73.1 kg, White et al.) is used when converting BW → Newtons.  If you
  only need BW-normalised physics (dimensionless F=ma) the absolute mass is not
  required.

* The raw accelerometer records *specific force* (a_COM - g_vec) in the sensor
  frame.  Sensor-frame to lab-frame mapping is unknown, so we do NOT use the raw
  accelerometer for ``com_acceleration``.  Instead we derive COM vertical
  acceleration from the vGRF signal via Newton's law — this is more physically
  accurate because the L5 sensor is not located at the whole-body COM.

  Horizontal GRF components are not available in this dataset; they are stored
  as zero in ``BiomechanicalSample.grf``.
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

# ── Dataset metadata ──────────────────────────────────────────────────────────

DATASET_NAME = "cmj_grf_zenodo"

# Group means from White et al. (submitted). Used to convert BW → N when
# individual masses are absent from the NPZ file.
DATASET_MEAN_MASS_KG: float = 73.1
DATASET_MEAN_HEIGHT_M: float = 1.74
DATASET_MEAN_AGE_YEARS: float = 21.6

_GRAVITY: float = 9.81          # m/s²  (matches constants.GRAVITY_MPS2)
_TARGET_HZ: float = 250.0       # Hz  (ACC native rate; GRF downsampled 1000→250)
_WINDOW_SAMPLES: int = 500      # samples (2 s pre-takeoff window)
_GRF_DOWNSAMPLE: int = 4        # factor: 1000 Hz → 250 Hz

# Default filenames produced by the upstream prepare_dataset.py script
_DEFAULT_FILENAMES = (
    "cmj_dataset_both.npz",
    "cmj_dataset_noarms.npz",
    "cmj_dataset_arms.npz",
)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _align_acc_window(
    acc: np.ndarray,
    takeoff_idx: int,
    window: int = _WINDOW_SAMPLES,
) -> np.ndarray:
    """Extract ``window`` samples ending at ``takeoff_idx`` from ``acc``.

    If fewer than ``window`` samples exist before takeoff the signal is
    left-padded by repeating the first available sample (matches the
    behaviour of the upstream ``_align_signal_at_takeoff`` helper).

    Args:
        acc:          (n_timesteps, 3) accelerometer signal in g.
        takeoff_idx:  Sample index of the takeoff instant.
        window:       Number of samples to extract (default 500).

    Returns:
        (window, 3) float32 array.
    """
    n = min(takeoff_idx, len(acc))  # don't overshoot the array
    out = np.empty((window, 3), dtype=np.float32)

    if n >= window:
        start = n - window
        out[:] = acc[start:n, :]
    else:
        pad = window - n
        out[:pad, :] = acc[0, :]       # repeat first sample
        out[pad:, :] = acc[:n, :]

    return out


def _align_grf_window(
    grf_250hz: np.ndarray,
    window: int = _WINDOW_SAMPLES,
) -> np.ndarray:
    """Extract the last ``window`` samples from a downsampled GRF signal.

    The GRF signal is already time-aligned so that its end corresponds to
    takeoff.  We take the trailing ``window`` samples (padding the start with
    the first sample if the signal is shorter).

    Args:
        grf_250hz:  (n_timesteps,) GRF in BW units at 250 Hz.
        window:     Number of samples to extract.

    Returns:
        (window,) float32 array.
    """
    n = len(grf_250hz)
    out = np.empty(window, dtype=np.float32)

    if n >= window:
        out[:] = grf_250hz[-window:]
    else:
        pad = window - n
        out[:pad] = grf_250hz[0]       # repeat first sample (pre-movement rest)
        out[pad:] = grf_250hz

    return out


# ── Public API ────────────────────────────────────────────────────────────────


def load_cmj_npz(
    npz_path: Path | str,
    movement_filter: list[MovementType] | None = None,
    max_trials: int | None = None,
) -> Generator[BiomechanicalSample, None, None]:
    """Yield :class:`BiomechanicalSample` objects from a CMJ NPZ file.

    Each trial produces one sample containing:

    * ``grf``              — (500, 3) vertical GRF in Newtons (Y component only)
    * ``com_acceleration`` — (500, 3) vertical COM acceleration in m/s²
                             (derived from Newton's 2nd law, Y component only)
    * ``fps``              — 250.0
    * ``movement_type``    — :attr:`MovementType.COUNTERMOVEMENT_JUMP`
    * ``subject``          — group-mean anthropometrics; ``subject_id`` is the
                             0-indexed participant ID from the dataset

    Args:
        npz_path:         Path to the ``.npz`` file.
        movement_filter:  If given, only yield samples whose movement type is
                          in this list.  CMJ samples pass when the filter
                          includes :attr:`MovementType.COUNTERMOVEMENT_JUMP`.
        max_trials:       Optional cap on the number of trials loaded.

    Yields:
        :class:`BiomechanicalSample` — one per trial.

    Raises:
        FileNotFoundError: If ``npz_path`` does not exist.
        KeyError: If expected NPZ keys are missing (indicates an incompatible
                  file version).
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"CMJ NPZ file not found: {npz_path}")

    # Early-exit if the caller only wants non-CMJ movements
    if (
        movement_filter is not None
        and MovementType.COUNTERMOVEMENT_JUMP not in movement_filter
    ):
        logger.debug("CMJ movement type not in filter — skipping %s", npz_path.name)
        return

    logger.info("Loading CMJ NPZ: %s", npz_path)
    data = np.load(npz_path, allow_pickle=True)

    # ── Validate required keys ──────────────────────────────────────────
    required_keys = {"acc_signals", "acc_takeoff", "grf_signals", "subject_ids",
                     "jump_height", "peak_power"}
    missing = required_keys - set(data.files)
    if missing:
        raise KeyError(
            f"CMJ NPZ is missing expected keys: {missing}. "
            f"Available keys: {data.files}. "
            "This file may have been generated by an incompatible version of "
            "scripts/prepare_dataset.py."
        )

    acc_signals: np.ndarray = data["acc_signals"]    # object array of (n, 3)
    acc_takeoff: np.ndarray = data["acc_takeoff"]    # (N,) int
    grf_signals: np.ndarray = data["grf_signals"]    # object array of (n,) @ 1000 Hz
    subject_ids: np.ndarray = data["subject_ids"]    # (N,) int
    jump_heights: np.ndarray = data["jump_height"].astype(np.float64)  # (N,)
    peak_powers: np.ndarray = data["peak_power"].astype(np.float64)   # (N,)
    n_total = len(acc_signals)

    logger.info("  %d trials from %s", n_total, npz_path.name)

    n_loaded = 0
    for i in range(n_total):
        if max_trials is not None and n_loaded >= max_trials:
            break

        acc_raw = np.asarray(acc_signals[i], dtype=np.float32)   # (n_t, 3) in g
        grf_raw = np.asarray(grf_signals[i], dtype=np.float32)   # (n_g,) in BW @ 1000 Hz
        takeoff_idx = int(acc_takeoff[i])
        subject_id_int = int(subject_ids[i])

        # ── Downsample GRF from 1000 Hz → 250 Hz ──────────────────────
        grf_250hz = grf_raw[::_GRF_DOWNSAMPLE]  # (n_g//4,) in BW

        # ── Align to 500-sample pre-takeoff window ─────────────────────
        acc_window = _align_acc_window(acc_raw, takeoff_idx)   # (500, 3) g
        grf_window = _align_grf_window(grf_250hz)              # (500,) BW

        T = _WINDOW_SAMPLES  # 500

        # ── Convert vertical GRF: BW → Newtons ─────────────────────────
        # F_vGRF_N = vGRF_bw * mass_kg * g
        # No individual masses in this dataset → group mean 73.1 kg
        mass_kg = DATASET_MEAN_MASS_KG
        vgrf_newtons = grf_window * mass_kg * _GRAVITY  # (500,) N, vertical only

        # ── Derive vertical COM acceleration from Newton's 2nd law ──────
        # F_vGRF = m * (a_COM_y + g)  ⟹  a_COM_y = (vGRF_bw - 1) * g
        # This is physics-principled and more accurate than using the L5
        # IMU signal (which is measured in the sensor frame, not at COM).
        a_com_y = (grf_window - 1.0) * _GRAVITY  # (500,) m/s²

        # ── Assemble multi-axis arrays (Y = vertical, X/Z unknown) ────
        grf_3d = np.zeros((T, 3), dtype=np.float32)
        grf_3d[:, 1] = vgrf_newtons          # Y axis is vertical (Y-up convention)

        com_acc_3d = np.zeros((T, 3), dtype=np.float32)
        com_acc_3d[:, 1] = a_com_y           # Y component from Newton's law

        # ── Subject info ───────────────────────────────────────────────
        subject = SubjectInfo(
            subject_id=f"cmj_zenodo_subj{subject_id_int:03d}",
            body_mass_kg=mass_kg,          # group mean — not individual
            height_m=DATASET_MEAN_HEIGHT_M,
            age_years=DATASET_MEAN_AGE_YEARS,
        )

        # ── Build sample ───────────────────────────────────────────────
        # jump_height and peak_power are trial-level metadata stored in
        # com_position and com_velocity scalar slots via a convention of
        # recording the integrated jump impulse as a check variable.  We
        # don't have time-series COM position/velocity, so those fields
        # remain None; the metadata is embedded in the trial_id.
        sample = BiomechanicalSample(
            dataset_name=DATASET_NAME,
            trial_id=f"subj{subject_id_int:03d}_trial{i:04d}_"
                     f"jh{jump_heights[i]:.3f}m_"
                     f"pp{peak_powers[i]:.1f}Wkg",
            subject=subject,
            movement_type=MovementType.COUNTERMOVEMENT_JUMP,
            fps=_TARGET_HZ,
            grf=grf_3d,
            com_acceleration=com_acc_3d,
        )

        n_loaded += 1
        yield sample

    logger.info("  Yielded %d / %d trials", n_loaded, n_total)


def load_cmj_npz_dir(
    dataset_dir: Path | str,
    movement_filter: list[MovementType] | None = None,
    max_trials: int | None = None,
) -> Generator[BiomechanicalSample, None, None]:
    """Scan a directory for known CMJ NPZ filenames and yield samples.

    Tries :data:`_DEFAULT_FILENAMES` in order and loads the first match.
    If multiple files are present (unlikely) they are all loaded.

    Args:
        dataset_dir:     Directory containing one or more ``.npz`` files.
        movement_filter: Forwarded to :func:`load_cmj_npz`.
        max_trials:      Optional trial cap per file.

    Yields:
        :class:`BiomechanicalSample` objects.

    Raises:
        FileNotFoundError: If no recognised NPZ file is found in
                           ``dataset_dir``.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"CMJ dataset directory not found: {dataset_dir}")

    found_any = False
    for name in _DEFAULT_FILENAMES:
        p = dataset_dir / name
        if p.is_file():
            found_any = True
            yield from load_cmj_npz(p, movement_filter, max_trials)

    # Fall back: any .npz file in the directory
    if not found_any:
        npz_files = sorted(dataset_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"No CMJ NPZ files found in {dataset_dir}. "
                f"Expected one of {_DEFAULT_FILENAMES}. "
                "Run: python scripts/download_datasets.py --dataset cmj_grf_zenodo --auto"
            )
        for p in npz_files:
            found_any = True
            yield from load_cmj_npz(p, movement_filter, max_trials)
