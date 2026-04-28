"""Tests for the data pipeline: sample, registry, transforms, and datasets."""

import numpy as np
import pytest

from src.data_pipeline.sample import (
    BiomechanicalSample,
    MovementType,
    SubjectInfo,
    MOVEMENT_RELEVANCE,
)
from src.data_pipeline.registry import (
    DATASET_REGISTRY,
    get_dataset_info,
    list_datasets,
    list_dynamics_datasets,
    list_pose_datasets,
)
from src.data_pipeline.transforms import (
    normalize_by_body_mass,
    normalize_by_height,
    window_sample,
    lowpass_filter,
    compute_derivatives,
)


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_sample(
    n_frames: int = 100,
    n_joints: int = 10,
    has_dynamics: bool = True,
    has_poses: bool = False,
    mass: float = 80.0,
    height: float = 1.90,
) -> BiomechanicalSample:
    """Create a synthetic BiomechanicalSample for testing."""
    subject = SubjectInfo(
        subject_id="test_subject",
        body_mass_kg=mass,
        height_m=height,
        sex="M",
    )

    sample = BiomechanicalSample(
        dataset_name="test",
        trial_id="test_trial_001",
        subject=subject,
        movement_type=MovementType.COUNTERMOVEMENT_JUMP,
        fps=100.0,
        joint_angles=np.random.randn(n_frames, n_joints).astype(np.float32),
        joint_angular_velocities=np.random.randn(n_frames, n_joints).astype(np.float32),
        joint_angular_accelerations=np.random.randn(n_frames, n_joints).astype(np.float32),
        joint_names=[f"joint_{i}" for i in range(n_joints)],
        com_position=np.random.randn(n_frames, 3).astype(np.float32),
        com_velocity=np.random.randn(n_frames, 3).astype(np.float32),
        com_acceleration=np.random.randn(n_frames, 3).astype(np.float32),
    )

    if has_dynamics:
        sample.grf = np.random.randn(n_frames, 3).astype(np.float32) * 500
        sample.joint_torques = np.random.randn(n_frames, n_joints).astype(np.float32) * 50

    if has_poses:
        sample.pose_2d = np.random.randn(n_frames, 17, 3).astype(np.float32)
        sample.pose_3d = np.random.randn(n_frames, 17, 3).astype(np.float32)

    return sample


# ── BiomechanicalSample tests ──────────────────────────────────────────


def test_sample_n_frames():
    sample = _make_sample(n_frames=50)
    assert sample.n_frames == 50


def test_sample_duration():
    sample = _make_sample(n_frames=100)
    assert abs(sample.duration_s - 1.0) < 1e-5  # 100 frames at 100 Hz = 1s


def test_sample_has_dynamics():
    sample = _make_sample(has_dynamics=True)
    assert sample.has_dynamics


def test_sample_has_kinematics():
    sample = _make_sample()
    assert sample.has_kinematics


def test_sample_relevance_score():
    sample = _make_sample()
    sample.movement_type = MovementType.COUNTERMOVEMENT_JUMP
    assert sample.relevance_score == MOVEMENT_RELEVANCE[MovementType.COUNTERMOVEMENT_JUMP]


def test_sample_get_window():
    sample = _make_sample(n_frames=100)
    window = sample.get_window(10, 30)
    assert window.n_frames == 20
    assert window.joint_angles.shape[0] == 20
    assert window.grf.shape[0] == 20
    assert window.com_position.shape[0] == 20


def test_sample_validate_clean():
    sample = _make_sample()
    warnings = sample.validate()
    assert len(warnings) == 0


def test_sample_validate_missing_fps():
    sample = _make_sample()
    sample.fps = 0
    warnings = sample.validate()
    assert any("fps" in w for w in warnings)


def test_sample_save_load_roundtrip(tmp_path):
    """save_npz → load_npz must recover all populated fields exactly."""
    sample = _make_sample(n_frames=80, n_joints=8, has_dynamics=True, has_poses=True)
    path = tmp_path / "sample.npz"
    sample.save_npz(path)
    loaded = BiomechanicalSample.load_npz(path)

    assert loaded.dataset_name == sample.dataset_name
    assert loaded.trial_id == sample.trial_id
    assert loaded.movement_type == sample.movement_type
    assert loaded.fps == sample.fps
    assert loaded.joint_names == sample.joint_names
    assert loaded.subject.body_mass_kg == sample.subject.body_mass_kg
    assert loaded.subject.height_m == sample.subject.height_m
    for attr in BiomechanicalSample._ARRAY_FIELDS:
        original = getattr(sample, attr)
        recovered = getattr(loaded, attr)
        if original is None:
            assert recovered is None
        else:
            np.testing.assert_array_equal(original, recovered)


def test_sample_save_load_omits_none_arrays(tmp_path):
    """Fields that are None on the source sample must remain None after load."""
    sample = _make_sample(n_frames=10, has_dynamics=False, has_poses=False)
    path = tmp_path / "sample.npz"
    sample.save_npz(path)
    loaded = BiomechanicalSample.load_npz(path)
    assert loaded.grf is None
    assert loaded.joint_torques is None
    assert loaded.pose_2d is None
    assert loaded.pose_3d is None


def test_sample_validate_missing_mass():
    sample = _make_sample()
    sample.subject.body_mass_kg = None
    warnings = sample.validate()
    assert any("mass" in w for w in warnings)


# ── Registry tests ─────────────────────────────────────────────────────


def test_registry_has_addbiomechanics():
    info = get_dataset_info("addbiomechanics")
    assert info.has_dynamics
    assert info.has_grf
    assert info.priority == 10


def test_registry_has_biocv():
    info = get_dataset_info("biocv")
    assert info.has_video
    assert info.has_grf


def test_registry_list_sorted():
    datasets = list_datasets(sort_by_priority=True)
    priorities = [d.priority for d in datasets]
    assert priorities == sorted(priorities, reverse=True)


def test_registry_dynamics_datasets():
    dynamics = list_dynamics_datasets()
    assert all(d.has_dynamics for d in dynamics)
    assert len(dynamics) >= 3  # addbiomechanics, biocv, opencap


def test_registry_pose_datasets():
    poses = list_pose_datasets()
    assert all(d.has_3d_poses or d.has_2d_poses for d in poses)


def test_registry_unknown_dataset():
    with pytest.raises(KeyError):
        get_dataset_info("nonexistent_dataset")


# ── Transform tests ────────────────────────────────────────────────────


def test_normalize_by_body_mass():
    sample = _make_sample(mass=80.0)
    original_grf = sample.grf.copy()
    normalize_by_body_mass(sample)
    np.testing.assert_allclose(sample.grf, original_grf / 80.0)


def test_normalize_by_height():
    sample = _make_sample(height=1.90)
    original_com = sample.com_position.copy()
    normalize_by_height(sample)
    np.testing.assert_allclose(sample.com_position, original_com / 1.90)


def test_window_sample():
    sample = _make_sample(n_frames=100)
    windows = window_sample(sample, window_size=20, stride=10)
    # (100 - 20) / 10 + 1 = 9 windows
    assert len(windows) == 9
    for w in windows:
        assert w.n_frames == 20


def test_window_sample_no_overlap():
    sample = _make_sample(n_frames=100)
    windows = window_sample(sample, window_size=25)
    assert len(windows) == 4  # 100 // 25


def test_window_too_short():
    sample = _make_sample(n_frames=10)
    windows = window_sample(sample, window_size=20)
    assert len(windows) == 0


def test_lowpass_filter_shape():
    data = np.random.randn(200, 3).astype(np.float32)
    filtered = lowpass_filter(data, fps=100.0, cutoff_hz=12.0)
    assert filtered.shape == data.shape


def test_lowpass_filter_reduces_noise():
    t = np.linspace(0, 1, 500)
    # Clean sine + high-frequency noise
    clean = np.sin(2 * np.pi * 5 * t)
    noisy = clean + 0.3 * np.sin(2 * np.pi * 40 * t)

    filtered = lowpass_filter(noisy, fps=500.0, cutoff_hz=12.0)
    # Filtered should be closer to the clean signal than the noisy version
    error_noisy = np.mean((noisy - clean) ** 2)
    error_filtered = np.mean((filtered - clean) ** 2)
    assert error_filtered < error_noisy


def test_compute_derivatives_constant_position():
    # Constant position → zero velocity and acceleration
    n = 100
    pos = np.ones((n, 3)) * 5.0
    vel, acc = compute_derivatives(pos, fps=100.0, filter_cutoff_hz=None)
    np.testing.assert_allclose(vel, 0.0, atol=1e-10)
    np.testing.assert_allclose(acc, 0.0, atol=1e-10)


def test_compute_derivatives_linear_position():
    # Linear position → constant velocity, zero acceleration
    n = 200
    t = np.linspace(0, 2, n)
    pos = np.column_stack([t * 3, t * 0, t * 0])  # 3 m/s in x
    vel, acc = compute_derivatives(pos, fps=100.0, filter_cutoff_hz=None)
    # Central part (avoid edge effects from gradient)
    np.testing.assert_allclose(vel[10:-10, 0], 3.0, atol=0.05)
    np.testing.assert_allclose(acc[20:-20, 0], 0.0, atol=0.5)
