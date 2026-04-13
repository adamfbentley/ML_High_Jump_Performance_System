"""Tests for TechniqueParameters and optimizer utilities.

Covers: to_tensor / from_tensor round-trip, tensor length,
new Imogen-priority fields, and parameter ordering consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.optimizer import TechniqueParameters


# ── Tensor length ─────────────────────────────────────────────────────────

def _default_params() -> TechniqueParameters:
    return TechniqueParameters(
        approach_speed_mps=7.0,
        curve_radius_m=9.5,
        penultimate_step_length_cm=170.0,
        last_step_length_cm=120.0,
        plant_angle_deg=67.0,
        takeoff_knee_angle_deg=170.0,
        takeoff_hip_angle_deg=175.0,
        arm_swing_timing_ms=50.0,
        free_leg_drive_angle_deg=80.0,
    )


def test_to_tensor_length():
    """to_tensor() must produce exactly 14 elements (9 original + 5 new)."""
    tp = _default_params()
    t = tp.to_tensor()
    assert t.shape == (14,), f"Expected tensor of length 14, got {t.shape}"


def test_from_tensor_round_trip():
    """from_tensor(to_tensor(tp)) must recover all original values."""
    original = TechniqueParameters(
        approach_speed_mps=7.2,
        curve_radius_m=10.0,
        penultimate_step_length_cm=165.0,
        last_step_length_cm=115.0,
        plant_angle_deg=68.0,
        takeoff_knee_angle_deg=168.0,
        takeoff_hip_angle_deg=172.0,
        arm_swing_timing_ms=45.0,
        free_leg_drive_angle_deg=82.0,
        ground_contact_time_takeoff_ms=115.0,
        body_alignment_deviation_deg=2.5,
        foot_to_ground_angle_deg=64.0,
        knee_drive_peak_speed_mps=3.5,
        curve_start_step=4,
    )
    recovered = TechniqueParameters.from_tensor(original.to_tensor())

    assert abs(recovered.approach_speed_mps - 7.2) < 1e-4
    assert abs(recovered.curve_radius_m - 10.0) < 1e-4
    assert abs(recovered.penultimate_step_length_cm - 165.0) < 1e-4
    assert abs(recovered.last_step_length_cm - 115.0) < 1e-4
    assert abs(recovered.plant_angle_deg - 68.0) < 1e-4
    assert abs(recovered.takeoff_knee_angle_deg - 168.0) < 1e-4
    assert abs(recovered.takeoff_hip_angle_deg - 172.0) < 1e-4
    assert abs(recovered.arm_swing_timing_ms - 45.0) < 1e-4
    assert abs(recovered.free_leg_drive_angle_deg - 82.0) < 1e-4
    assert abs(recovered.ground_contact_time_takeoff_ms - 115.0) < 1e-2
    assert abs(recovered.body_alignment_deviation_deg - 2.5) < 1e-4
    assert abs(recovered.foot_to_ground_angle_deg - 64.0) < 1e-4
    assert abs(recovered.knee_drive_peak_speed_mps - 3.5) < 1e-4
    assert recovered.curve_start_step == 4


def test_new_fields_have_sensible_defaults():
    """Default values for new fields should be physiologically plausible."""
    tp = _default_params()
    assert 50.0 <= tp.ground_contact_time_takeoff_ms <= 300.0
    assert tp.body_alignment_deviation_deg == 0.0
    assert 50.0 <= tp.foot_to_ground_angle_deg <= 90.0
    assert tp.knee_drive_peak_speed_mps > 0.0
    assert tp.curve_start_step >= 1


def test_tensor_ordering_first_nine_unchanged():
    """The first 9 tensor elements should match the original 9 fields in order."""
    tp = TechniqueParameters(
        approach_speed_mps=7.0,
        curve_radius_m=9.5,
        penultimate_step_length_cm=170.0,
        last_step_length_cm=120.0,
        plant_angle_deg=67.0,
        takeoff_knee_angle_deg=170.0,
        takeoff_hip_angle_deg=175.0,
        arm_swing_timing_ms=50.0,
        free_leg_drive_angle_deg=80.0,
    )
    t = tp.to_tensor().numpy()
    assert abs(t[0] - 7.0) < 1e-5    # approach_speed_mps
    assert abs(t[1] - 9.5) < 1e-5    # curve_radius_m
    assert abs(t[2] - 170.0) < 1e-5  # penultimate_step_length_cm
    assert abs(t[3] - 120.0) < 1e-5  # last_step_length_cm
    assert abs(t[4] - 67.0) < 1e-5   # plant_angle_deg
    assert abs(t[5] - 170.0) < 1e-5  # takeoff_knee_angle_deg
    assert abs(t[6] - 175.0) < 1e-5  # takeoff_hip_angle_deg
    assert abs(t[7] - 50.0) < 1e-5   # arm_swing_timing_ms
    assert abs(t[8] - 80.0) < 1e-5   # free_leg_drive_angle_deg


def test_new_fields_at_positions_9_to_13():
    """Positions 9–13 in tensor correspond to the five new Imogen-priority fields."""
    tp = TechniqueParameters(
        approach_speed_mps=7.0,
        curve_radius_m=9.5,
        penultimate_step_length_cm=170.0,
        last_step_length_cm=120.0,
        plant_angle_deg=67.0,
        takeoff_knee_angle_deg=170.0,
        takeoff_hip_angle_deg=175.0,
        arm_swing_timing_ms=50.0,
        free_leg_drive_angle_deg=80.0,
        ground_contact_time_takeoff_ms=130.0,
        body_alignment_deviation_deg=3.0,
        foot_to_ground_angle_deg=66.0,
        knee_drive_peak_speed_mps=3.2,
        curve_start_step=6,
    )
    t = tp.to_tensor().numpy()
    assert abs(t[9]  - 130.0) < 1e-4   # ground_contact_time_takeoff_ms
    assert abs(t[10] - 3.0)   < 1e-4   # body_alignment_deviation_deg
    assert abs(t[11] - 66.0)  < 1e-4   # foot_to_ground_angle_deg
    assert abs(t[12] - 3.2)   < 1e-4   # knee_drive_peak_speed_mps
    assert abs(t[13] - 6.0)   < 1e-4   # curve_start_step


def test_curve_start_step_round_trips_as_int():
    """curve_start_step must survive the float tensor round-trip as an integer."""
    for step in [1, 3, 5, 8]:
        tp = _default_params()
        tp.curve_start_step = step
        recovered = TechniqueParameters.from_tensor(tp.to_tensor())
        assert recovered.curve_start_step == step, (
            f"Expected curve_start_step={step}, got {recovered.curve_start_step}"
        )
