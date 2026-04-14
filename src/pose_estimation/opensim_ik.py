"""Optional OpenSim inverse kinematics for refined joint angles.

Runs OpenSim's IK solver to post-process MediaPipe 3D landmarks through a
musculoskeletal model, producing biomechanically consistent joint angles that
respect anatomical constraints (joint limits, coupling, segment inertia).

Because opensim is compiled against numpy 1.x and the main pipeline uses numpy 2.x
with PyTorch, we run OpenSim IK as a **subprocess** using a conda environment where
opensim is installed.  Data is exchanged via TRC / MOT files on disk.

Setup:
    conda create -n opensim_ik python=3.11 opensim=4.5 -c opensim-org -c defaults
    Download model: data/models/population/RajagopalLaiUhlrich2023.osim

Usage:
    from src.pose_estimation.opensim_ik import run_opensim_ik, is_opensim_available

    if is_opensim_available():
        joint_angles, joint_names = run_opensim_ik(
            landmarks_3d, fps, height_m=1.75, mass_kg=65.0
        )

References:
    Seth, A. et al. (2018). OpenSim: Simulating musculoskeletal dynamics
    and neuromuscular control to study human and animal movement.
    PLoS Computational Biology, 14(7), e1006223.

    Rajagopal, A. et al. (2016). Full-body musculoskeletal model for
    muscle-driven simulation of human gait.  IEEE TBME.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────

# Default model path (relative to project root)
_DEFAULT_MODEL = "data/models/population/RajagopalLaiUhlrich2023.osim"

# Subprocess script (ships with the project)
_IK_SCRIPT = "scripts/opensim_ik_subprocess.py"

# Conda environment name where opensim is installed
_CONDA_ENV_NAME = "opensim_ik"

# BlazePose landmark indices → Rajagopal model marker names
# These must EXACTLY match markers defined in the .osim model.
# Rajagopal2016/LaiUhlrich2023 marker names verified from model MarkerSet.
_BLAZEPOSE_TO_OPENSIM_MARKERS = {
    0: "C7",               # nose → approximate as C7 (neck)
    11: "LSJC",            # left shoulder → left shoulder joint center
    12: "RSJC",            # right shoulder → right shoulder joint center
    13: "LEJC",            # left elbow → left elbow joint center
    14: "REJC",            # right elbow → right elbow joint center
    15: "LFAradius",       # left wrist → left forearm radius
    16: "RFAradius",       # right wrist → right forearm radius
    23: "LASI",            # left hip → left ASIS
    24: "RASI",            # right hip → right ASIS
    25: "LKJC",            # left knee → left knee joint center
    26: "RKJC",            # right knee → right knee joint center
    27: "LAJC",            # left ankle → left ankle joint center
    28: "RAJC",            # right ankle → right ankle joint center
    31: "LTOE",            # left foot → left toe
    32: "RTOE",            # right foot → right toe
}


def _find_conda_python() -> Path | None:
    """Locate the Python executable in the opensim_ik conda environment."""
    home = Path(os.path.expanduser("~"))
    candidates = [
        home / "anaconda3" / "envs" / _CONDA_ENV_NAME / "python.exe",
        home / "miniconda3" / "envs" / _CONDA_ENV_NAME / "python.exe",
        home / "miniforge3" / "envs" / _CONDA_ENV_NAME / "python.exe",
        Path("C:/ProgramData/anaconda3/envs") / _CONDA_ENV_NAME / "python.exe",
        Path("C:/ProgramData/miniconda3/envs") / _CONDA_ENV_NAME / "python.exe",
        # Linux / macOS
        home / "anaconda3" / "envs" / _CONDA_ENV_NAME / "bin" / "python",
        home / "miniconda3" / "envs" / _CONDA_ENV_NAME / "bin" / "python",
        home / "miniforge3" / "envs" / _CONDA_ENV_NAME / "bin" / "python",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Try conda run as fallback
    conda_exe = shutil.which("conda")
    if conda_exe:
        try:
            result = subprocess.run(
                [conda_exe, "run", "-n", _CONDA_ENV_NAME, "python", "-c",
                 "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                exe = result.stdout.strip()
                if Path(exe).exists():
                    return Path(exe)
        except Exception:
            pass

    return None


def is_opensim_available() -> bool:
    """Check whether OpenSim IK can run (conda env + model file exist)."""
    conda_python = _find_conda_python()
    if conda_python is None:
        return False
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / _DEFAULT_MODEL
    script_path = project_root / _IK_SCRIPT
    return model_path.exists() and script_path.exists()


def _write_trc(
    landmarks: np.ndarray,
    fps: float,
    output_path: Path,
) -> list[str]:
    """Write a subset of BlazePose landmarks as a .trc marker file.

    Args:
        landmarks: (T, 33, 3+) array — MediaPipe world landmarks in metres.
        fps: Sampling rate.
        output_path: Where to write the TRC file.

    Returns:
        List of marker names in column order.
    """
    marker_names = []
    col_data = []  # list of (T, 3) arrays

    for bp_idx, osim_name in sorted(_BLAZEPOSE_TO_OPENSIM_MARKERS.items()):
        marker_names.append(osim_name)
        col_data.append(landmarks[:, bp_idx, :3])

    T = landmarks.shape[0]
    n_markers = len(marker_names)

    with open(output_path, "w") as f:
        # Header block (TRC v2 format)
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(output_path.name))
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{T}\t{n_markers}\tm\t{fps}\t1\t{T}\n")

        # Marker name row
        header = "Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\n"
        f.write(header)

        # X/Y/Z sub-header
        xyz_row = "\t\t" + "\t".join(
            f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(n_markers)
        ) + "\n"
        f.write(xyz_row)
        f.write("\n")  # blank line before data

        # Data rows
        for t in range(T):
            time = t / fps
            vals = [f"{t+1}", f"{time:.6f}"]
            for c in col_data:
                vals.extend([f"{c[t, 0]:.6f}", f"{c[t, 1]:.6f}", f"{c[t, 2]:.6f}"])
            f.write("\t".join(vals) + "\n")

    return marker_names


def _parse_mot_file(mot_path: Path) -> tuple[np.ndarray, list[str]]:
    """Parse an OpenSim .mot file into joint angles array.

    Args:
        mot_path: Path to the .mot output from IK.

    Returns:
        (angles_rad, joint_names): (T, J) array in radians, list of J names.
    """
    lines = mot_path.read_text().splitlines()

    # Find 'endheader'
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break

    # Column names on line after endheader
    col_line = lines[header_end + 1]
    col_names = col_line.split("\t")
    if len(col_names) == 1:
        col_names = col_line.split()

    # Joint names = everything except 'time'
    joint_names = [n for n in col_names if n.strip().lower() != "time"]

    # Data rows
    rows = []
    for line in lines[header_end + 2:]:
        if not line.strip():
            continue
        vals = line.split("\t")
        if len(vals) == 1:
            vals = line.split()
        rows.append([float(v) for v in vals])

    data = np.array(rows)  # (T, n_cols) — first col is time

    # Extract joint angle columns (skip time col at index 0)
    time_idx = 0
    for i, name in enumerate(col_names):
        if name.strip().lower() == "time":
            time_idx = i
            break

    angle_cols = [i for i in range(data.shape[1]) if i != time_idx]
    angles_deg = data[:, angle_cols]

    # OpenSim outputs degrees — convert to radians
    angles_rad = np.deg2rad(angles_deg)

    return angles_rad, joint_names


def run_opensim_ik(
    landmarks: np.ndarray,
    fps: float,
    height_m: float = 1.75,
    mass_kg: float = 65.0,
    model_file: str | Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Run OpenSim inverse kinematics on MediaPipe landmarks.

    Runs IK as a subprocess using the opensim_ik conda environment to avoid
    numpy version conflicts.  Data is exchanged via TRC/MOT files.

    Args:
        landmarks: (T, 33, 3+) MediaPipe world landmarks in metres.
        fps: Frames per second.
        height_m: Athlete standing height (for model scaling).
        mass_kg: Body mass (for model scaling).
        model_file: Path to OpenSim .osim model file.  If None, uses
            data/models/population/RajagopalLaiUhlrich2023.osim.

    Returns:
        joint_angles: (T, J) array of joint angles in radians.
        joint_names: List of J joint/coordinate names.

    Raises:
        RuntimeError: If the conda env or model is not found, or IK fails.
    """
    # Locate conda Python
    conda_python = _find_conda_python()
    if conda_python is None:
        raise RuntimeError(
            f"Conda environment '{_CONDA_ENV_NAME}' not found. "
            f"Create it with: conda create -n {_CONDA_ENV_NAME} python=3.11 "
            f"opensim=4.5 -c opensim-org -c defaults"
        )

    # Locate project root and paths
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / _IK_SCRIPT
    if not script_path.exists():
        raise RuntimeError(f"IK subprocess script not found: {script_path}")

    if model_file is None:
        model_file = project_root / _DEFAULT_MODEL
    model_file = Path(model_file)
    if not model_file.exists():
        raise RuntimeError(
            f"OpenSim model not found: {model_file}\n"
            f"Download from: https://github.com/opensim-org/opensim-models"
        )

    with tempfile.TemporaryDirectory(prefix="hj_opensim_") as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Write landmarks as TRC
        trc_path = tmpdir / "markers.trc"
        _write_trc(landmarks, fps, trc_path)

        # 2. Run IK via subprocess
        mot_path = tmpdir / "ik_results.mot"
        cmd = [
            str(conda_python),
            str(script_path),
            str(trc_path),
            str(model_file),
            str(mot_path),
            "--height_m", str(height_m),
            "--mass_kg", str(mass_kg),
        ]

        logger.info(f"  Running OpenSim IK subprocess ({landmarks.shape[0]} frames)...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"OpenSim IK failed (exit code {result.returncode}):\n{stderr}"
            )

        # Parse JSON status from stdout
        try:
            status = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            status = {"status": "unknown"}

        if status.get("status") == "error":
            raise RuntimeError(f"OpenSim IK error: {status.get('message', 'unknown')}")

        logger.info(
            f"  OpenSim IK: {status.get('n_frames', '?')} frames, "
            f"{status.get('n_coordinates', '?')} coordinates"
        )

        # 3. Parse MOT results
        if not mot_path.exists():
            raise RuntimeError("OpenSim IK did not produce output MOT file")

        angles_rad, joint_names = _parse_mot_file(mot_path)

    return angles_rad, joint_names
