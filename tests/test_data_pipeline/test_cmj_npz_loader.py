"""Tests for the CMJ NPZ dataset loader (src/data_pipeline/loaders/cmj_npz.py).

All tests are self-contained — they create synthetic NPZ data matching the
format of Zenodo record 19136480 and do NOT require downloading any files.

Physics under test:
    Newton's 2nd law (Y-up):  F_vGRF = m * (a_COM_y + g)
    BW-normalised:            vGRF_bw = a_COM_y / g + 1
    Inverted:                 a_COM_y = (vGRF_bw - 1) * g
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data_pipeline.loaders.cmj_npz import (
    DATASET_MEAN_MASS_KG,
    DATASET_NAME,
    _GRAVITY,
    _WINDOW_SAMPLES,
    _align_acc_window,
    _align_grf_window,
    load_cmj_npz,
    load_cmj_npz_dir,
)
from src.data_pipeline.sample import MovementType


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_fake_npz(
    tmp_dir: Path,
    n_trials: int = 5,
    signal_len: int = 800,       # > 500 so takeoff can be at index 600
    takeoff_at: int = 600,
    filename: str = "cmj_dataset_both.npz",
) -> Path:
    """Create a synthetic NPZ file with the structure of the real dataset.

    GRF is a simple sinusoidal pattern in BW units (never negative).
    ACC is random noise around the expected specific-force baseline of 1g.
    """
    rng = np.random.default_rng(42)
    n_subjects = 3

    # Build variable-length object arrays
    acc_signals = np.empty(n_trials, dtype=object)
    grf_signals = np.empty(n_trials, dtype=object)
    for i in range(n_trials):
        # ACC: triaxial, upright standing ≈ [0, 1, 0] g (Y ≈ gravitational axis)
        acc = rng.standard_normal((signal_len, 3)).astype(np.float32) * 0.05
        acc[:, 1] += 1.0       # Y-axis ≈ 1g during quiet standing
        acc_signals[i] = acc

        # GRF: at 1000 Hz, BW units.  Quiet phase ≈ 1 BW; peak ≈ 3 BW
        t = np.linspace(0, 2 * np.pi, signal_len * 4)  # 4× upsample for 1000 Hz
        grf = 1.0 + 0.5 * np.sin(t).astype(np.float32)
        grf_signals[i] = grf

    acc_takeoff = np.full(n_trials, takeoff_at, dtype=np.int64)
    subject_ids = np.array([i % n_subjects for i in range(n_trials)], dtype=np.int64)
    jump_height = rng.uniform(0.30, 0.55, size=n_trials)
    peak_power = rng.uniform(30.0, 60.0, size=n_trials)

    npz_path = tmp_dir / filename
    np.savez(
        npz_path,
        acc_signals=acc_signals,
        acc_takeoff=acc_takeoff,
        grf_signals=grf_signals,
        subject_ids=subject_ids,
        jump_height=jump_height,
        peak_power=peak_power,
        n_subjects=np.int64(n_subjects),
    )
    return npz_path


# ── Unit tests for internal alignment helpers ─────────────────────────────────

def test_align_acc_window_exact():
    """Signal exactly 500 samples before takeoff → no padding."""
    acc = np.arange(600 * 3, dtype=np.float32).reshape(600, 3)
    window = _align_acc_window(acc, takeoff_idx=500, window=500)
    assert window.shape == (500, 3)
    # Should match acc[0:500]
    np.testing.assert_array_equal(window, acc[0:500])


def test_align_acc_window_partial():
    """Only 200 samples before takeoff → first 300 are padded with acc[0]."""
    acc = np.ones((200, 3), dtype=np.float32) * 99.0
    acc[0, :] = 1.0   # first sample is the padding value
    window = _align_acc_window(acc, takeoff_idx=200, window=500)
    assert window.shape == (500, 3)
    pad_len = 500 - 200
    # Padded region (first 300) should repeat the first sample (value 1.0)
    np.testing.assert_array_equal(window[:pad_len, :], 1.0)
    # Data region starts with acc[0]=1.0 then acc[1:]=99.0
    np.testing.assert_array_equal(window[pad_len:, :], acc[0:200, :])


def test_align_grf_window_exact():
    """Signal exactly 500 samples → no padding."""
    grf = np.arange(500, dtype=np.float32)
    window = _align_grf_window(grf, window=500)
    assert window.shape == (500,)
    np.testing.assert_array_equal(window, grf)


def test_align_grf_window_too_long():
    """Signal longer than window → take last 500 samples."""
    grf = np.arange(800, dtype=np.float32)
    window = _align_grf_window(grf, window=500)
    assert window.shape == (500,)
    np.testing.assert_array_equal(window, grf[300:])   # last 500 of 800


def test_align_grf_window_too_short():
    """Signal shorter than window → pad start with first sample."""
    grf = np.ones(100, dtype=np.float32) * 5.0
    grf[0] = 2.5
    window = _align_grf_window(grf, window=500)
    assert window.shape == (500,)
    assert window[0] == pytest.approx(2.5)   # first sample used for padding


# ── Integration tests for the loader ─────────────────────────────────────────

def test_load_cmj_npz_yields_samples():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_fake_npz(Path(tmpdir), n_trials=5)
        samples = list(load_cmj_npz(npz_path))
    assert len(samples) == 5


def test_load_cmj_npz_sample_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_fake_npz(Path(tmpdir), n_trials=3)
        samples = list(load_cmj_npz(npz_path))

    for s in samples:
        assert s.dataset_name == DATASET_NAME
        assert s.movement_type == MovementType.COUNTERMOVEMENT_JUMP
        assert s.fps == 250.0
        assert s.subject is not None
        assert s.subject.body_mass_kg == pytest.approx(DATASET_MEAN_MASS_KG)


def test_load_cmj_npz_array_shapes():
    """grf and com_acceleration must both be (500, 3)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_fake_npz(Path(tmpdir), n_trials=4)
        samples = list(load_cmj_npz(npz_path))

    for s in samples:
        assert s.grf is not None
        assert s.com_acceleration is not None
        assert s.grf.shape == (_WINDOW_SAMPLES, 3)
        assert s.com_acceleration.shape == (_WINDOW_SAMPLES, 3)


def test_load_cmj_npz_grf_units():
    """Vertical GRF should be in Newtons (>> 1 for standing, Y component non-zero)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_fake_npz(Path(tmpdir), n_trials=3)
        samples = list(load_cmj_npz(npz_path))

    for s in samples:
        vgrf_N = s.grf[:, 1]
        # During upright standing ≈ body weight = 73.1 * 9.81 ≈ 717 N
        expected_bw_N = DATASET_MEAN_MASS_KG * _GRAVITY
        assert np.mean(vgrf_N) > expected_bw_N * 0.5, (
            "Mean vGRF is suspiciously low — possible unit error"
        )
        # X and Z components must be zero (not available for this dataset)
        np.testing.assert_array_equal(s.grf[:, 0], 0.0)
        np.testing.assert_array_equal(s.grf[:, 2], 0.0)


def test_newton_second_law_consistency():
    """Physics check: F_vGRF = m * (a_COM_y + g).

    For a synthetic vGRF of constant 1.5 BW the COM vertical acceleration
    should be (1.5 - 1.0) * 9.81 = 4.905 m/s² throughout the window.
    """
    rng = np.random.default_rng(0)
    n_trials = 2
    signal_len = 700
    takeoff_at = 600

    # Build synthetic NPZ with constant 1.5 BW GRF
    acc_signals = np.empty(n_trials, dtype=object)
    grf_signals = np.empty(n_trials, dtype=object)
    for i in range(n_trials):
        acc_signals[i] = np.ones((signal_len, 3), dtype=np.float32)
        grf_at_1000hz = np.full(signal_len * 4, 1.5, dtype=np.float32)
        grf_signals[i] = grf_at_1000hz

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test.npz"
        np.savez(
            npz_path,
            acc_signals=acc_signals,
            acc_takeoff=np.full(n_trials, takeoff_at, dtype=np.int64),
            grf_signals=grf_signals,
            subject_ids=np.zeros(n_trials, dtype=np.int64),
            jump_height=np.zeros(n_trials),
            peak_power=np.zeros(n_trials),
            n_subjects=np.int64(1),
        )
        samples = list(load_cmj_npz(npz_path))

    for s in samples:
        vgrf_N = s.grf[:, 1]
        a_com_y = s.com_acceleration[:, 1]

        # Newton's 2nd law: F = m * (a + g)
        m = DATASET_MEAN_MASS_KG
        expected_force = m * (a_com_y + _GRAVITY)
        np.testing.assert_allclose(vgrf_N, expected_force, rtol=1e-5)

        # Expected acceleration from (1.5 BW - 1.0) * g = 4.905 m/s²
        np.testing.assert_allclose(a_com_y, 0.5 * _GRAVITY, rtol=1e-5)


def test_load_cmj_npz_max_trials():
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_fake_npz(Path(tmpdir), n_trials=10)
        samples = list(load_cmj_npz(npz_path, max_trials=3))
    assert len(samples) == 3


def test_load_cmj_npz_movement_filter_pass():
    """CMJ movement passes a filter that includes COUNTERMOVEMENT_JUMP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_fake_npz(Path(tmpdir), n_trials=4)
        samples = list(
            load_cmj_npz(
                npz_path,
                movement_filter=[MovementType.COUNTERMOVEMENT_JUMP, MovementType.DROP_JUMP],
            )
        )
    assert len(samples) == 4


def test_load_cmj_npz_movement_filter_reject():
    """No CMJ samples when filter contains only other movement types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_fake_npz(Path(tmpdir), n_trials=4)
        samples = list(
            load_cmj_npz(
                npz_path,
                movement_filter=[MovementType.RUNNING, MovementType.WALKING],
            )
        )
    assert len(samples) == 0


def test_load_cmj_npz_file_not_found():
    with pytest.raises(FileNotFoundError):
        list(load_cmj_npz(Path("/nonexistent/path/cmj.npz")))


def test_load_cmj_npz_missing_key():
    """NPZ missing required key should raise KeyError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_npz = Path(tmpdir) / "bad.npz"
        np.savez(bad_npz, acc_signals=np.array([]))  # missing keys
        with pytest.raises(KeyError, match="missing expected keys"):
            list(load_cmj_npz(bad_npz))


def test_load_cmj_npz_dir_finds_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _make_fake_npz(tmp, n_trials=3, filename="cmj_dataset_both.npz")
        samples = list(load_cmj_npz_dir(tmp))
    assert len(samples) == 3


def test_load_cmj_npz_dir_not_found():
    with pytest.raises(FileNotFoundError):
        list(load_cmj_npz_dir(Path("/nonexistent/dir")))


def test_short_signal_pads_correctly():
    """Takeoff at sample 100 (< 500) — window should be padded."""
    rng = np.random.default_rng(7)
    n_trials = 1
    signal_len = 150
    takeoff_at = 100

    acc_signals = np.empty(1, dtype=object)
    grf_signals = np.empty(1, dtype=object)
    acc_signals[0] = rng.standard_normal((signal_len, 3)).astype(np.float32)
    grf_signals[0] = np.ones(signal_len * 4, dtype=np.float32)  # constant 1 BW

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "short.npz"
        np.savez(
            npz_path,
            acc_signals=acc_signals,
            acc_takeoff=np.array([takeoff_at], dtype=np.int64),
            grf_signals=grf_signals,
            subject_ids=np.zeros(1, dtype=np.int64),
            jump_height=np.zeros(1),
            peak_power=np.zeros(1),
            n_subjects=np.int64(1),
        )
        samples = list(load_cmj_npz(npz_path))

    assert len(samples) == 1
    s = samples[0]
    assert s.grf.shape == (500, 3)
    assert s.com_acceleration.shape == (500, 3)
    # Constant 1 BW → vGRF_N = 1 * mass * g — known exactly
    expected_N = DATASET_MEAN_MASS_KG * _GRAVITY
    np.testing.assert_allclose(s.grf[:, 1], expected_N, rtol=1e-5)
    # COM acceleration: (1.0 - 1.0) * g = 0 m/s² (quiet standing)
    np.testing.assert_allclose(s.com_acceleration[:, 1], 0.0, atol=1e-5)
