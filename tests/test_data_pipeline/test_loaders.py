"""Tests for OpenCap .trc and .mot file parsers (no network/data needed)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data_pipeline.loaders.opencap import load_trc_file, load_mot_file


def _write_trc(path: Path, n_frames: int = 20, n_markers: int = 3) -> None:
    """Write a minimal valid .trc file for testing."""
    marker_names = [f"M{i}" for i in range(n_markers)]

    lines = [
        "PathFileType\t4\t(X/Y/Z)\ttest.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigNumFrames",
        f"100.0\t100.0\t{n_frames}\t{n_markers}\tmm\t100.0\t{n_frames}",
        "Frame#\tTime\t" + "\t".join(marker_names),
        "\t\t" + "\t".join(["X\tY\tZ"] * n_markers),
    ]

    for f in range(n_frames):
        t = f / 100.0
        vals = [f"{np.random.randn() * 1000:.4f}" for _ in range(n_markers * 3)]
        lines.append(f"{f + 1}\t{t:.4f}\t" + "\t".join(vals))

    path.write_text("\n".join(lines))


def _write_mot(path: Path, n_frames: int = 20) -> None:
    """Write a minimal valid .mot file for testing."""
    columns = ["time", "hip_flexion_r", "knee_angle_r", "ankle_angle_r"]

    lines = [
        "Motion file",
        f"nRows={n_frames}",
        f"nColumns={len(columns)}",
        "inDegrees=yes",
        "endheader",
        "\t".join(columns),
    ]

    for f in range(n_frames):
        t = f / 100.0
        vals = [f"{t:.4f}"] + [f"{np.random.randn() * 30:.4f}" for _ in range(len(columns) - 1)]
        lines.append("\t".join(vals))

    path.write_text("\n".join(lines))


def test_load_trc_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        trc_path = Path(tmpdir) / "test.trc"
        _write_trc(trc_path, n_frames=30, n_markers=5)

        fps, names, positions = load_trc_file(trc_path)

        assert fps == 100.0
        assert len(names) == 5
        assert positions.shape == (30, 5, 3)
        # Should be in meters (converted from mm)
        assert np.max(np.abs(positions)) < 5.0


def test_load_mot_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        mot_path = Path(tmpdir) / "test.mot"
        _write_mot(mot_path, n_frames=50)

        fps, col_names, data = load_mot_file(mot_path)

        assert fps > 0
        assert len(col_names) == 4
        assert data.shape[0] == 50
        assert data.shape[1] == 4
