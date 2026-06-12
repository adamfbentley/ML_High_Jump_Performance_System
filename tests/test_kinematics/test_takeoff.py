"""Tests for takeoff biomechanics analysis.

Covers: TakeoffMetrics new fields, body alignment deviation,
foot-to-ground angle, arm drive metrics, and free-knee drive metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scripts.analyze_jump_video import (
    _KEY_JOINT_INDICES,
    _cache_admitted_sample,
    _calibration_source,
    _quality_block,
    _validate_takeoff_anchor,
    _windowed_peak_com_frame,
    _write_json_report,
    generate_report,
    parse_bar_height,
    pose_validity_pct,
    resolve_bar_height,
    select_takeoff_frame_details,
    takeoff_window_pose_validity_pct,
)
from src.data_pipeline.admission import admitted_sample_paths, load_admission_manifest
from src.data_pipeline.sample import BiomechanicalSample, MovementType, SubjectInfo
from src.kinematics.takeoff_analysis import (
    TakeoffMetrics,
    compute_arm_drive_metrics,
    compute_body_alignment_deviation,
    compute_foot_to_ground_angle,
    compute_free_knee_drive_metrics,
    compute_impulse,
    compute_takeoff_angle,
    estimate_grf_from_com,
    predict_max_com_height,
)

# ── TakeoffMetrics new fields ─────────────────────────────────────────────

def test_takeoff_metrics_new_fields_have_defaults():
    """New Athlete A-priority fields should all default to 0.0."""
    m = TakeoffMetrics(
        ground_contact_time_ms=120.0,
        time_to_peak_force_ms=80.0,
        horizontal_velocity_mps=6.5,
        vertical_velocity_mps=3.2,
        resultant_velocity_mps=7.2,
        takeoff_angle_deg=26.0,
        takeoff_distance_from_bar_cm=60.0,
        com_height_at_takeoff_cm=105.0,
        peak_vertical_grf_bw=3.5,
        average_vertical_grf_bw=2.1,
        braking_impulse_ns=80.0,
        propulsive_impulse_ns=320.0,
        knee_angle_at_takeoff_deg=170.0,
        hip_angle_at_takeoff_deg=175.0,
        ankle_angle_at_takeoff_deg=110.0,
        trunk_lean_deg=5.0,
    )
    assert m.body_alignment_deviation_deg == 0.0
    assert m.foot_to_ground_angle_deg == 0.0
    assert m.arm_drive_peak_speed_mps == 0.0
    assert m.arm_drive_peak_timing_ms == 0.0
    assert m.free_knee_drive_peak_speed_mps == 0.0
    assert m.free_knee_drive_peak_timing_ms == 0.0


def test_write_json_report_creates_explicit_parent_directory(tmp_path):
    """Explicit report destinations may use a new nested result directory."""
    report_path = tmp_path / "stationary_rerun" / "report.json"

    _write_json_report(report_path, {"processed": 5})

    assert report_path.read_text() == '{\n  "processed": 5\n}'


# ── compute_body_alignment_deviation ─────────────────────────────────────

def _make_straight_body_landmarks() -> np.ndarray:
    """All joints on the Y-axis (perfectly straight vertical body)."""
    landmarks = np.zeros((33, 3))
    # Right side (takeoff = right): ankle=28, knee=26, hip=24, shoulder=12, nose=0
    landmarks[28] = [0.0, 0.0, 0.0]    # ankle (ground)
    landmarks[26] = [0.0, 0.45, 0.0]   # knee
    landmarks[24] = [0.0, 0.90, 0.0]   # hip
    landmarks[12] = [0.0, 1.40, 0.0]   # shoulder
    landmarks[0]  = [0.0, 1.75, 0.0]   # nose / head
    return landmarks


def test_body_alignment_straight_line_gives_zero():
    """Perfectly straight body → ~0° alignment deviation."""
    landmarks = _make_straight_body_landmarks()
    dev = compute_body_alignment_deviation(landmarks, takeoff_side="right")
    assert dev < 1.0, f"Expected ~0°, got {dev:.3f}°"


def test_body_alignment_bent_knee_gives_positive():
    """Knee displaced forward by 10 cm → non-zero alignment deviation."""
    landmarks = _make_straight_body_landmarks()
    landmarks[26, 0] = 0.10   # push right knee 10 cm forward (X)
    dev = compute_body_alignment_deviation(landmarks, takeoff_side="right")
    assert dev > 0.5, f"Expected positive deviation, got {dev:.3f}°"


def test_body_alignment_left_side():
    """Left-side takeoff should work symmetrically."""
    landmarks = np.zeros((33, 3))
    landmarks[27] = [0.0, 0.0, 0.0]    # left ankle
    landmarks[25] = [0.0, 0.45, 0.0]   # left knee
    landmarks[23] = [0.0, 0.90, 0.0]   # left hip
    landmarks[11] = [0.0, 1.40, 0.0]   # left shoulder
    landmarks[0]  = [0.0, 1.75, 0.0]   # head
    dev = compute_body_alignment_deviation(landmarks, takeoff_side="left")
    assert dev < 1.0, f"Expected ~0°, got {dev:.3f}°"


# ── compute_foot_to_ground_angle ──────────────────────────────────────────

def test_foot_to_ground_angle_horizontal_foot():
    """Foot perfectly horizontal (ankle and toe at same height) → 0°."""
    ankle = np.array([0.0, 0.0, 0.0])
    toe   = np.array([0.2, 0.0, 0.0])
    angle = compute_foot_to_ground_angle(ankle, toe)
    assert abs(angle) < 1e-5


def test_foot_to_ground_angle_positive_plantarflexion():
    """Toe raised above heel → positive angle."""
    ankle = np.array([0.0, 0.0, 0.0])
    toe   = np.array([0.2, 0.1, 0.0])   # toe 10 cm higher than heel
    angle = compute_foot_to_ground_angle(ankle, toe)
    expected = np.degrees(np.arctan2(0.1, 0.2))
    assert abs(angle - expected) < 0.5


def test_foot_to_ground_angle_negative_dorsiflexion():
    """Heel raised above toe → negative angle (dorsiflexed/heel strike)."""
    ankle = np.array([0.0, 0.05, 0.0])  # ankle (heel) 5 cm high
    toe   = np.array([0.2, 0.0, 0.0])   # toe lower
    angle = compute_foot_to_ground_angle(ankle, toe)
    assert angle < 0.0


# ── compute_arm_drive_metrics ─────────────────────────────────────────────

def test_arm_drive_peak_speed_constant_returns_nonzero():
    """Wrist accelerating from rest → peak speed at end."""
    fps = 100.0
    n = 20
    t = np.arange(n) / fps
    # Wrist moves in Y with constant acceleration
    wrist = np.column_stack([np.zeros(n), 0.5 * 10.0 * t ** 2, np.zeros(n)])
    peak_speed, timing_ms = compute_arm_drive_metrics(wrist, fps)
    assert peak_speed > 0.0
    # Peak should be at the end (maximum speed at last frame)
    assert timing_ms >= (n - 2) / fps * 1000.0


def test_arm_drive_single_frame_returns_zero():
    """Single-frame input cannot produce a velocity → return (0, 0)."""
    wrist = np.zeros((1, 3))
    peak_speed, timing_ms = compute_arm_drive_metrics(wrist, fps=100.0)
    assert peak_speed == 0.0
    assert timing_ms == 0.0


def test_arm_drive_timing_in_ms():
    """Peak speed at frame 10 of 100 Hz sequence → timing = 100 ms."""
    fps = 100.0
    n = 20
    # Constant forward wrist motion, then sudden speed at frame 10
    wrist = np.zeros((n, 3))
    wrist[10:, 0] = np.linspace(0, 1, n - 10)  # only X component moves after frame 10
    peak_speed, timing_ms = compute_arm_drive_metrics(wrist, fps)
    # The peak frame should be around 10
    assert 50.0 <= timing_ms <= 200.0, f"Unexpected timing {timing_ms} ms"


# ── compute_free_knee_drive_metrics ───────────────────────────────────────

def test_free_knee_drive_rising_knee():
    """Knee accelerating upward → positive peak speed."""
    fps = 100.0
    n = 20
    t = np.arange(n) / fps
    knee = np.column_stack([np.zeros(n), 5.0 * t, np.zeros(n)])  # constant 5 m/s up
    peak_speed, timing_ms = compute_free_knee_drive_metrics(knee, fps)
    assert peak_speed > 0.0


def test_free_knee_drive_only_upward_component():
    """Only upward (positive Y) velocity is counted; downward is excluded."""
    fps = 100.0
    n = 10
    # Knee moves downward → upward speed should be 0
    knee = np.column_stack([np.zeros(n), -np.linspace(0, 1, n), np.zeros(n)])
    peak_speed, _ = compute_free_knee_drive_metrics(knee, fps)
    assert peak_speed == 0.0


def test_free_knee_drive_single_frame_returns_zero():
    knee = np.zeros((1, 3))
    peak_speed, timing = compute_free_knee_drive_metrics(knee, fps=100.0)
    assert peak_speed == 0.0
    assert timing == 0.0


# ── Legacy functions still work ───────────────────────────────────────────

def test_estimate_grf_static():
    """At rest (zero acceleration), GRF should equal body weight."""
    mass = 70.0
    acc = np.zeros((1, 3))
    grf = estimate_grf_from_com(acc, mass)
    # GRF_y should be m*g = 70 * 9.81
    np.testing.assert_allclose(grf[0, 1], mass * 9.81, rtol=1e-5)


def test_compute_takeoff_angle_45_degrees():
    """Equal horizontal and vertical velocity → 45° takeoff angle."""
    vel = np.array([3.0, 3.0, 0.0])
    angle = compute_takeoff_angle(vel)
    assert abs(angle - 45.0) < 0.01


def test_predict_max_com_height_physics():
    """h_max = h0 + vy²/(2g).  Verify against manual calculation."""
    h0 = 1.05
    vy = 4.0
    expected = h0 + vy ** 2 / (2 * 9.81)
    assert abs(predict_max_com_height(h0, vy) - expected) < 1e-6


def test_compute_impulse_constant_force():
    """Constant vertical force F over T seconds → impulse ≈ F*T.

    The trapezoidal rule integrates over N-1 intervals from N points, so the
    result is F * (N-1)/fps rather than F * N/fps.  The boundary error is
    exactly one time-step: F * dt = 700 * 0.01 = 7 N·s.
    """
    fps = 100.0
    n = 100
    f = np.zeros((n, 3))
    f[:, 1] = 700.0  # 700 N vertical
    impulse = compute_impulse(f, fps, axis=1)
    # Trapezoidal integration: F * (N-1) * dt
    expected = 700.0 * (n - 1) / fps
    assert abs(impulse - expected) < 0.01   # floating-point tolerance only


# ── Video report takeoff-frame selection ─────────────────────────────────

def _make_video_sample_for_takeoff_frame_test(
    *,
    contact_slice: slice | None,
    velocity_spike_frame: int,
    true_takeoff_frame: int,
) -> BiomechanicalSample:
    """Synthetic calibrated video sample with foot landmark Y in metres.

    Contact is expressed via the heel/forefoot landmarks (the real ground-
    contact surfaces), with the ankle held elevated even during contact to
    model the plantarflexed toe-off pose — the case an ankle-only detector
    misses.  Foot landmarks: ankles 27/28, heels 29/30, toes 31/32.
    """
    n = 30
    fps = 100.0
    pose_3d = np.zeros((n, 33, 3), dtype=float)
    # All foot landmarks start off the ground.
    for _idx in (27, 28, 29, 30, 31, 32):
        pose_3d[:, _idx, 1] = 0.25
    if contact_slice is not None:
        # Right-foot heel + forefoot on the ground; ankle stays elevated.
        pose_3d[contact_slice, 30, 1] = 0.02  # right heel
        pose_3d[contact_slice, 32, 1] = 0.02  # right forefoot

    com_position = np.zeros((n, 3), dtype=float)
    com_position[:, 1] = np.linspace(1.0, 1.8, n)
    com_velocity = np.zeros((n, 3), dtype=float)
    com_velocity[:, 0] = 3.0
    com_velocity[:, 1] = 0.5
    com_velocity[true_takeoff_frame, 1] = 3.4
    com_velocity[velocity_spike_frame, 1] = 20.0
    grf = np.zeros((n, 3), dtype=float)
    grf[:, 1] = 65.0 * 9.81

    return BiomechanicalSample(
        dataset_name="unit_test",
        trial_id="synthetic_takeoff",
        subject=SubjectInfo(subject_id="athlete", body_mass_kg=65.0, height_m=1.75),
        movement_type=MovementType.HIGH_JUMP,
        fps=fps,
        com_position=com_position,
        com_velocity=com_velocity,
        grf=grf,
        pose_3d=pose_3d,
    )


def test_generate_report_uses_last_ground_contact_for_takeoff_frame():
    """Contact-anchored takeoff should ignore later single-frame vy spikes."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=slice(8, 12),
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )

    report = generate_report(sample, pinn_grf=None)

    assert report["takeoff_frame"] == 11
    assert report["velocity"]["takeoff_vertical_mps"] == 3.4


def test_generate_report_falls_back_to_argmax_vy_without_ground_contacts():
    """Very short or failed pose clips still produce a takeoff estimate."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=None,
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )

    report = generate_report(sample, pinn_grf=None)

    assert report["takeoff_frame"] == 20
    assert report["velocity"]["takeoff_vertical_mps"] == 20.0


def test_generate_report_uses_scene_corrected_horizontal_velocity_for_angle():
    """Once scene calibration corrects X velocity, the report angle follows vh_scene."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=slice(8, 12),
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )
    sample.com_velocity[:, 0] = 4.5
    sample.com_velocity[:, 1] = 0.5
    sample.com_velocity[11, 1] = 3.5
    sample.com_velocity[20, 1] = 20.0

    report = generate_report(
        sample,
        pinn_grf=None,
        calibration_info={"method": "scene_homography", "anchor_coverage_pct": 100.0},
    )
    image_relative_angle = float(np.degrees(np.arctan2(3.5, 0.5)))

    assert report["velocity"]["takeoff_angle_deg"] == pytest.approx(37.9, abs=0.1)
    assert image_relative_angle > 80.0
    assert report["calibration"]["method"] == "scene_homography"


def test_parse_bar_height_accepts_numeric_video_extension():
    assert parse_bar_height("session_attempt_1.88.mp4") == 1.88


def test_resolve_bar_height_prefers_explicit_override():
    assert resolve_bar_height("session_attempt_1.88.mp4", 1.75) == 1.75


def test_resolve_bar_height_rejects_non_positive_override():
    with pytest.raises(ValueError, match="bar height override must be positive"):
        resolve_bar_height("session_attempt_1.88.mp4", 0.0)


# ── Change 1: stationary_camera admission source ──────────────────────────

def test_calibration_source_stationary_camera_when_confirmed():
    """Reviewed stationary anatomical footage credits the fixed camera source."""
    source = _calibration_source(
        {"method": "anatomical"},
        capture_mode="stationary",
        stationary_camera_confirmed=True,
    )
    assert source == "stationary_camera"


def test_calibration_source_stationary_camera_requires_confirmation():
    """Stationary mode alone is not durable confirmation of a fixed camera."""
    source = _calibration_source({"method": "anatomical"}, capture_mode="stationary")
    assert source == "none"


def test_calibration_source_handheld_default_returns_none():
    """Default handheld mode with anatomical method → 'none' (no scene-fixed source)."""
    source = _calibration_source({"method": "anatomical"}, capture_mode="handheld")
    assert source == "none"


def test_calibration_source_egomotion_takes_precedence():
    """egomotion method always wins, regardless of capture_mode."""
    source = _calibration_source({"method": "egomotion"}, capture_mode="stationary")
    assert source == "egomotion"


def test_quality_block_confirmed_stationary_does_not_append_no_scene_source():
    """Confirmed stationary capture removes the missing scene-source failure."""
    result = _quality_block(
        pose_pct=100.0,
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="stationary",
        stationary_camera_confirmed=True,
        takeoff_anchor_review_passed=True,
    )
    assert "no_scene_fixed_horizontal_source" not in result["training_grade_failures"]
    assert result["scene_fixed_horizontal_source"] == "stationary_camera"
    assert result["stationary_camera_confirmed"] is True


def test_quality_block_unconfirmed_stationary_appends_no_scene_source():
    """Stationary workflow without operator confirmation is not admission-grade."""
    result = _quality_block(
        pose_pct=100.0,
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="stationary",
        stationary_camera_confirmed=False,
        takeoff_anchor_review_passed=True,
    )
    assert "no_scene_fixed_horizontal_source" in result["training_grade_failures"]
    assert result["stationary_camera_confirmed"] is False


def test_quality_block_handheld_appends_no_scene_source():
    """handheld default still fails the scene-fixed horizontal source gate."""
    result = _quality_block(
        pose_pct=100.0,
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="handheld",
        takeoff_anchor_review_passed=True,
    )
    assert "no_scene_fixed_horizontal_source" in result["training_grade_failures"]


def test_generate_report_stationary_credited_in_calibration_block():
    """calibration block should carry capture_mode and scene_fixed_horizontal_source
    when stationary is asserted."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=slice(8, 12),
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )
    report = generate_report(
        sample,
        pinn_grf=None,
        calibration_info={
            "method": "anatomical",
            "capture_mode": "stationary",
            "stationary_camera_confirmed": True,
            "scene_fixed_horizontal_source": "stationary_camera",
            "scale_info": {},
        },
        pose_validity_pct_value=100.0,
        capture_mode="stationary",
        stationary_camera_confirmed=True,
    )
    assert report["calibration"]["capture_mode"] == "stationary"
    assert report["calibration"]["stationary_camera_confirmed"] is True
    assert report["calibration"]["scene_fixed_horizontal_source"] == "stationary_camera"
    assert report["quality"]["scene_fixed_horizontal_source"] == "stationary_camera"
    assert report["quality"]["stationary_camera_confirmed"] is True
    assert "no_scene_fixed_horizontal_source" not in report["quality"]["training_grade_failures"]


# ── Change 2: takeoff-window correctness ──────────────────────────────────

def _make_com_vel_for_anchor_test(n: int, fps: float, vy_at_frame: dict) -> np.ndarray:
    """com_velocity array with custom vy values at specific frames."""
    vel = np.zeros((n, 3), dtype=float)
    vel[:, 0] = 3.0
    vel[:, 1] = 2.0  # default upward
    for frame, vy in vy_at_frame.items():
        vel[frame, 1] = vy
    return vel


def test_validate_takeoff_anchor_passes_true_toeoff():
    """Candidate at frame 10 with vy=3.5, peak at frame 20: plausible toe-off."""
    fps = 30.0
    n = 30
    com_vel = _make_com_vel_for_anchor_test(n, fps, {10: 3.5})
    assert _validate_takeoff_anchor(
        candidate_frame=10, peak_com_frame=20, com_vel=com_vel, fps=fps
    )


def test_validate_takeoff_anchor_rejects_approach_stride():
    """Candidate 50 frames before peak with vy=0.1 (approach stride) is rejected."""
    fps = 30.0
    n = 60
    # vy=0.1 is below the minimum launch threshold and cannot be toe-off.
    com_vel = _make_com_vel_for_anchor_test(n, fps, {5: 0.1})
    assert not _validate_takeoff_anchor(
        candidate_frame=5, peak_com_frame=55, com_vel=com_vel, fps=fps
    )


def test_validate_takeoff_anchor_rejects_negative_vy():
    """Negative vy at candidate frame → not a toe-off."""
    fps = 30.0
    n = 20
    com_vel = _make_com_vel_for_anchor_test(n, fps, {5: -0.5})
    assert not _validate_takeoff_anchor(
        candidate_frame=5, peak_com_frame=15, com_vel=com_vel, fps=fps
    )


def test_validate_takeoff_anchor_rejects_weak_positive_vy_near_apex():
    """A nearby approach-stride contact with weak upward speed is not toe-off."""
    fps = 30.0
    n = 20
    com_vel = _make_com_vel_for_anchor_test(n, fps, {5: 0.1})
    assert not _validate_takeoff_anchor(
        candidate_frame=5, peak_com_frame=10, com_vel=com_vel, fps=fps
    )


def test_validate_takeoff_anchor_rejects_candidate_at_or_after_peak():
    """If peak_com_frame <= candidate_frame, the contact cannot be toe-off."""
    fps = 30.0
    n = 20
    com_vel = _make_com_vel_for_anchor_test(n, fps, {15: 3.0})
    assert not _validate_takeoff_anchor(
        candidate_frame=15, peak_com_frame=10, com_vel=com_vel, fps=fps
    )


def test_select_takeoff_frame_details_anchor_review_passes_for_valid_contact():
    """True toe-off contact → anchor_review_passed=True."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=slice(8, 12),
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )
    _frame, detected, _count, anchor_ok = select_takeoff_frame_details(
        sample, fallback_frame=20
    )
    assert detected is True
    assert anchor_ok is True


def test_windowed_peak_com_frame_keeps_consistent_global_peak():
    """When the global CoM peak follows the vy peak, keep the global apex."""
    com_pos = np.zeros((12, 3), dtype=float)
    com_pos[:, 1] = np.linspace(1.0, 1.6, 12)
    com_vel = np.zeros((12, 3), dtype=float)
    com_vel[5, 1] = 3.0
    assert _windowed_peak_com_frame(com_pos, com_vel) == 11


def test_windowed_peak_com_frame_recovers_after_vy_peak():
    """Approach-stride Y contamination before vy peak should not define flight apex."""
    com_pos = np.zeros((12, 3), dtype=float)
    com_pos[:, 1] = 1.0
    com_pos[2, 1] = 2.0   # contaminated early approach peak
    com_pos[9, 1] = 1.8   # plausible flight peak after launch velocity peak
    com_vel = np.zeros((12, 3), dtype=float)
    com_vel[5, 1] = 3.0
    assert _windowed_peak_com_frame(com_pos, com_vel) == 9


def test_select_takeoff_frame_details_detects_plantarflexed_toeoff_contact():
    """Forefoot-down / ankle-elevated plant is detected (regression).

    Models the stationary clip-3 failure mode: at toe-off the malleolus
    (ankle, idx 27/28) stays well above the support band while the heel and
    forefoot load the ground.  An ankle-height detector finds no contact; the
    foot-landmark detector must register the plant and pass the anchor review.
    """
    n, fps = 30, 100.0
    pose_3d = np.zeros((n, 33, 3), dtype=float)
    for idx in (27, 28, 29, 30, 31, 32):
        pose_3d[:, idx, 1] = 0.30
    # Ankle never enters the 5 cm ground band even during the plant.
    pose_3d[:, 27, 1] = 0.12
    pose_3d[:, 28, 1] = 0.12
    contact = slice(8, 12)
    pose_3d[contact, 30, 1] = 0.02  # right heel on ground
    pose_3d[contact, 32, 1] = 0.02  # right forefoot on ground

    com_position = np.zeros((n, 3), dtype=float)
    com_position[:, 1] = np.linspace(1.0, 1.8, n)
    com_velocity = np.zeros((n, 3), dtype=float)
    com_velocity[:, 0] = 3.0
    com_velocity[:, 1] = 0.5
    com_velocity[11, 1] = 3.4
    com_velocity[20, 1] = 20.0
    grf = np.zeros((n, 3), dtype=float)
    grf[:, 1] = 65.0 * 9.81
    sample = BiomechanicalSample(
        dataset_name="unit_test",
        trial_id="plantarflexed_toeoff",
        subject=SubjectInfo(subject_id="athlete", body_mass_kg=65.0, height_m=1.75),
        movement_type=MovementType.HIGH_JUMP,
        fps=fps,
        com_position=com_position,
        com_velocity=com_velocity,
        grf=grf,
        pose_3d=pose_3d,
    )
    frame, detected, count, anchor_ok = select_takeoff_frame_details(
        sample, fallback_frame=20
    )
    assert detected is True
    assert count >= 1
    # Toe-off = last frame of the ground-contact interval (slice end is exclusive).
    assert frame == 11
    assert anchor_ok is True


def test_select_takeoff_frame_details_no_contact_flags_anchor_failed():
    """No contact detected → anchor_review_passed=False (argmax fallback is not safe)."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=None,
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )
    _frame, detected, _count, anchor_ok = select_takeoff_frame_details(
        sample, fallback_frame=20
    )
    assert detected is False
    assert anchor_ok is False


def _make_no_contact_anchor_fail_sample() -> BiomechanicalSample:
    """Sample where no contacts AND velocity-based anchor fails.

    The CoM peaks at frame 5 (parabolic, immediately descending after).
    The highest vy is also at frame 5 (the velocity spike).
    _windowed_peak_com_frame: vy_peak=5, global_peak=5 → returns 5.
    _validate_takeoff_anchor(5, 5, ...): peak_com_frame(5) <= candidate(5) → False.
    """
    n, fps = 30, 100.0
    pose_3d = np.zeros((n, 33, 3), dtype=float)
    # All foot landmarks off the ground → no contacts detected.
    for _idx in (27, 28, 29, 30, 31, 32):
        pose_3d[:, _idx, 1] = 0.25

    # CoM peaks at frame 5 then descends
    t = np.arange(n) / fps
    peak_t = 5 / fps
    g = 9.81
    com_position = np.zeros((n, 3), dtype=float)
    com_position[:, 1] = np.maximum(1.0, 1.2 - 0.5 * g * (t - peak_t) ** 2)

    com_velocity = np.zeros((n, 3), dtype=float)
    com_velocity[:, 0] = 3.0
    com_velocity[:, 1] = 0.5
    com_velocity[5, 1] = 3.5  # vy spike at frame 5 (same as the CoM peak)

    grf = np.zeros((n, 3), dtype=float)
    grf[:, 1] = 65.0 * 9.81
    return BiomechanicalSample(
        dataset_name="unit_test",
        trial_id="anchor_fail",
        subject=SubjectInfo(subject_id="athlete", body_mass_kg=65.0, height_m=1.75),
        movement_type=MovementType.HIGH_JUMP,
        fps=fps,
        com_position=com_position,
        com_velocity=com_velocity,
        grf=grf,
        pose_3d=pose_3d,
    )


def test_select_takeoff_frame_details_no_contact_velocity_anchor_fails():
    """No contact AND velocity anchor fails when CoM peaks at same frame as vy peak."""
    sample = _make_no_contact_anchor_fail_sample()
    # fallback = argmax(vy) = 5 (the spike)
    _frame, detected, _count, anchor_ok = select_takeoff_frame_details(
        sample, fallback_frame=5
    )
    assert detected is False
    assert anchor_ok is False  # peak_com_frame == candidate_frame → fails


def test_generate_report_no_contact_velocity_anchor_fails_still_rejects():
    """When no contact AND velocity anchor fails, anchor_review_failed + no_contact added."""
    sample = _make_no_contact_anchor_fail_sample()
    # Use fallback_frame=5 so argmax(vy) selects the co-located peak frame
    # Force argmax(vy)=5 by making com_velocity[5,1] the highest
    report = generate_report(sample, pinn_grf=None)
    assert report["quality"]["takeoff_anchor_review_passed"] is False
    failures = report["quality"]["training_grade_failures"]
    assert "takeoff_anchor_review_failed" in failures
    assert "no_contact_interval" in failures


def test_generate_report_no_contact_includes_anchor_review_failure():
    """When no contact is detected, report.quality should include anchor review failure."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=None,
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )
    report = generate_report(sample, pinn_grf=None)
    assert report["quality"]["takeoff_anchor_review_passed"] is False
    assert "takeoff_anchor_review_failed" in report["quality"]["training_grade_failures"]
    assert "no_contact_interval" in report["quality"]["training_grade_failures"]


def test_generate_report_valid_contact_anchor_review_passed():
    """Valid toe-off contact → anchor review passes in report."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=slice(8, 12),
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )
    report = generate_report(sample, pinn_grf=None)
    assert report["quality"]["takeoff_anchor_review_passed"] is True
    assert "takeoff_anchor_review_failed" not in report["quality"]["training_grade_failures"]


def test_quality_block_anchor_review_failure_rejects_kinematics_grade():
    """A questionable toe-off frame cannot emit kinematics-grade metrics."""
    result = _quality_block(
        pose_pct=100.0,
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="stationary",
        stationary_camera_confirmed=True,
        takeoff_anchor_review_passed=False,
    )
    assert result["kinematics_grade"] is False
    assert "takeoff_anchor_review_failed" in result["kinematics_grade_failures"]


# ── Change 4: stricter key-joint pose_validity metric ─────────────────────

def test_pose_validity_pct_all_key_joints_visible_returns_100():
    """All 8 key joints visible on every frame → 100 %."""
    lm = np.zeros((10, 33, 3), dtype=np.float32)
    lm[:, _KEY_JOINT_INDICES, 2] = 0.9  # all key joints visible
    assert pose_validity_pct(lm) == pytest.approx(100.0)


def test_pose_validity_pct_missing_one_key_joint_returns_0():
    """One key joint invisible → every frame fails → 0 %."""
    lm = np.zeros((10, 33, 3), dtype=np.float32)
    lm[:, _KEY_JOINT_INDICES, 2] = 0.9
    lm[:, 27, 2] = 0.0  # left ankle invisible (index 27 is in _KEY_JOINT_INDICES)
    assert pose_validity_pct(lm) == pytest.approx(0.0)


def test_pose_validity_pct_many_non_key_joints_visible_does_not_pass():
    """Old metric (>=4 of 33) would pass; new strict metric should not when key joints absent."""
    lm = np.zeros((5, 33, 3), dtype=np.float32)
    # Make 20 NON-key landmarks visible — old metric would pass, new must fail
    non_key_indices = [i for i in range(33) if i not in _KEY_JOINT_INDICES.tolist()]
    for idx in non_key_indices[:20]:
        lm[:, idx, 2] = 0.9
    # key joints remain invisible → strict metric should return 0
    assert pose_validity_pct(lm) == pytest.approx(0.0)


def test_pose_validity_pct_partial_frames():
    """5 of 10 frames have all key joints visible → 50 %."""
    lm = np.zeros((10, 33, 3), dtype=np.float32)
    lm[:5, _KEY_JOINT_INDICES, 2] = 0.9  # first 5 frames valid
    assert pose_validity_pct(lm) == pytest.approx(50.0)


def test_quality_block_strict_validity_fails_below_60_pct():
    """Pose validity below 60 % → pose_validity_below_threshold in training failures."""
    result = _quality_block(
        pose_pct=55.0,
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="stationary",
        stationary_camera_confirmed=True,
        takeoff_anchor_review_passed=True,
    )
    assert "pose_validity_below_threshold" in result["training_grade_failures"]


def test_quality_block_all_gates_pass_stationary():
    """All gates satisfied for a stationary clip → training_grade True."""
    result = _quality_block(
        pose_pct=70.0,
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="stationary",
        stationary_camera_confirmed=True,
        takeoff_anchor_review_passed=True,
    )
    assert result["training_grade"] is True
    assert result["training_grade_failures"] == []


# ── Windowed pose_validity metric ─────────────────────────────────────────

def _make_approach_plus_jump_landmarks(
    n_approach: int = 60,
    n_jump: int = 60,
) -> np.ndarray:
    """Simulate a clip where approach frames have no detection, jump has full coverage."""
    n_frames = n_approach + n_jump
    lm = np.zeros((n_frames, 33, 3), dtype=np.float32)
    # Jump window: all key joints visible
    lm[n_approach:, _KEY_JOINT_INDICES, 2] = 0.9
    return lm


def test_windowed_pct_ignores_approach_frames():
    """Global pct = 50%; windowed around takeoff (frame 90) = 100%."""
    lm = _make_approach_plus_jump_landmarks(n_approach=60, n_jump=60)
    global_pct = pose_validity_pct(lm)
    windowed = takeoff_window_pose_validity_pct(lm, takeoff_frame=90, half_window=30)
    assert global_pct == pytest.approx(50.0)
    assert windowed == pytest.approx(100.0)


def test_windowed_pct_all_zeros_for_missing_window():
    """Takeoff frame beyond all detections → windowed returns 0."""
    lm = _make_approach_plus_jump_landmarks(n_approach=60, n_jump=60)
    windowed = takeoff_window_pose_validity_pct(lm, takeoff_frame=20, half_window=10)
    assert windowed == pytest.approx(0.0)


def test_quality_block_windowed_overrides_global_for_gate():
    """Low global pct but high windowed → gate passes (windowed takes priority)."""
    result = _quality_block(
        pose_pct=30.0,           # global: would fail gate
        windowed_pose_pct=75.0,  # windowed: passes gate
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="stationary",
        stationary_camera_confirmed=True,
        takeoff_anchor_review_passed=True,
    )
    assert "pose_validity_below_threshold" not in result["training_grade_failures"]
    assert result["takeoff_window_pose_validity_pct"] == pytest.approx(75.0)


def test_quality_block_windowed_fails_when_below_threshold():
    """High global pct but low windowed → gate still fails."""
    result = _quality_block(
        pose_pct=90.0,           # global: would pass
        windowed_pose_pct=40.0,  # windowed: fails gate
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="stationary",
        stationary_camera_confirmed=True,
        takeoff_anchor_review_passed=True,
    )
    assert "pose_validity_below_threshold" in result["training_grade_failures"]


def test_quality_block_reports_both_global_and_windowed():
    """Quality block must always report both pose_validity_pct and windowed."""
    result = _quality_block(
        pose_pct=35.0,
        windowed_pose_pct=70.0,
        contact_interval_detected=True,
        calibration_info={"method": "anatomical", "scale_info": {}},
        peak_com_height_m=1.5,
        takeoff_horizontal_mps=3.5,
        takeoff_angle_deg=43.0,
        capture_mode="stationary",
        stationary_camera_confirmed=True,
        takeoff_anchor_review_passed=True,
    )
    assert result["pose_validity_pct"] == pytest.approx(35.0)
    assert result["takeoff_window_pose_validity_pct"] == pytest.approx(70.0)


# ── Admitted-only sample caching ──────────────────────────────────────────

def test_cache_admitted_sample_writes_training_grade_npz(tmp_path):
    """Training-grade clips are cached for later personal fine-tuning."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=slice(8, 12),
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )
    report = {"quality": {"training_grade": True}}

    sample_path = _cache_admitted_sample(sample, report, tmp_path / "samples")

    assert sample_path is not None
    assert sample_path.exists()
    assert report["sample_cache"]["requested"] is True
    assert report["sample_cache"]["admitted"] is True
    assert report["sample_cache"]["saved"] is True
    assert report["sample_cache"]["decision_reason"] == "training_grade_passed"
    assert report["sample_cache"]["manifest"].endswith("_admission_manifest.json")
    assert report["sample_npz"] == str(sample_path)
    assert admitted_sample_paths(tmp_path / "samples") == [sample_path]


def test_cache_admitted_sample_skips_failed_clip(tmp_path):
    """Rejected clips must not quietly appear in the fine-tuning cache."""
    sample = _make_video_sample_for_takeoff_frame_test(
        contact_slice=None,
        true_takeoff_frame=11,
        velocity_spike_frame=20,
    )
    report = {"quality": {"training_grade": False}}
    samples_dir = tmp_path / "samples"

    sample_path = _cache_admitted_sample(sample, report, samples_dir)

    assert sample_path is None
    assert samples_dir.exists()
    assert "sample_npz" not in report
    assert report["sample_cache"]["requested"] is True
    assert report["sample_cache"]["admitted"] is False
    assert report["sample_cache"]["saved"] is False
    assert report["sample_cache"]["decision_reason"] == "training_grade_required"
    assert report["sample_cache"]["manifest"].endswith("_admission_manifest.json")
    assert admitted_sample_paths(samples_dir) == []
    manifest = load_admission_manifest(samples_dir)
    entry = manifest["samples"][sample.trial_id]
    assert entry["admitted"] is False
    assert entry["saved"] is False


def test_admitted_sample_paths_rejects_legacy_cache_without_manifest(tmp_path):
    """Fine-tuning must not silently consume a pre-admission mixed cache."""
    (tmp_path / "legacy.npz").write_bytes(b"not a real sample")

    with pytest.raises(FileNotFoundError, match="Admitted-only cache manifest"):
        admitted_sample_paths(tmp_path)
