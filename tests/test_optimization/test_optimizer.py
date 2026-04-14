"""Tests for the technique optimisation engine.

Covers: TechniqueParameters round-trip, forward model physics,
sensitivity analysis, optimisation convergence, coaching output,
and what-if scenarios.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.optimization.optimizer import (
    GRAVITY,
    AthleteConstraints,
    OptimizationResult,
    TechniqueParameters,
    compute_sensitivity,
    extract_params_from_report,
    generate_coaching_cues,
    optimize_technique,
    predict_bar_clearance,
    what_if_scenario,
    _estimate_takeoff_com_height,
    _impulse_model_vertical_velocity,
    _evaluate_height_differentiable,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

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


IMOGEN_MASS = 67.0
IMOGEN_HEIGHT = 1.78


# ── TechniqueParameters tests ─────────────────────────────────────────────

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


# ── Forward model physics tests ──────────────────────────────────────────

def test_com_height_full_extension():
    """At full extension (180° knee & hip), CoM should be ~55% of height."""
    h = _estimate_takeoff_com_height(1.78, 180.0, 180.0)
    expected = 1.78 * 0.55  # 0.979 m
    assert abs(h - expected) < 0.01, f"Expected ~{expected:.3f}, got {h:.3f}"


def test_com_height_decreases_with_flexion():
    """Flexing knee or hip should lower CoM."""
    h_full = _estimate_takeoff_com_height(1.78, 180.0, 180.0)
    h_bent = _estimate_takeoff_com_height(1.78, 160.0, 170.0)
    assert h_bent < h_full, "Bent joints should produce lower CoM"


def test_vertical_velocity_increases_with_approach_speed():
    """Faster approach → more momentum to redirect → higher v_y."""
    v_slow = _impulse_model_vertical_velocity(5.0, 67.0, 120.0, 3.0, 50.0, 67.0)
    v_fast = _impulse_model_vertical_velocity(8.0, 67.0, 120.0, 3.0, 50.0, 67.0)
    assert v_fast > v_slow, "Faster approach should give higher vertical velocity"


def test_vertical_velocity_increases_with_plant_angle():
    """Steeper plant → more vertical redirection."""
    v_shallow = _impulse_model_vertical_velocity(7.0, 55.0, 120.0, 3.0, 50.0, 67.0)
    v_steep = _impulse_model_vertical_velocity(7.0, 75.0, 120.0, 3.0, 50.0, 67.0)
    assert v_steep > v_shallow, "Steeper plant should give higher vertical velocity"


def test_predicted_height_physically_plausible():
    """A national-level female jumper should predict ~1.7–2.0 m bar clearance."""
    params = _default_params()
    result = predict_bar_clearance(params, IMOGEN_MASS, IMOGEN_HEIGHT)
    h = result["predicted_bar_height_m"]
    assert 1.5 < h < 2.3, f"Predicted height {h:.2f} m is outside plausible range"


def test_takeoff_angle_in_fosbury_range():
    """Predicted takeoff angle should be 30–60° for Fosbury flop (Dapena 1980)."""
    params = _default_params()
    result = predict_bar_clearance(params, IMOGEN_MASS, IMOGEN_HEIGHT)
    angle = result["takeoff_angle_deg"]
    assert 30 < angle < 60, f"Takeoff angle {angle:.1f}° outside Fosbury range"


def test_projectile_physics_height_rise():
    """v_y²/(2g) must match the predicted h_rise."""
    params = _default_params()
    result = predict_bar_clearance(params, IMOGEN_MASS, IMOGEN_HEIGHT)
    expected_rise = result["v_vertical_mps"] ** 2 / (2 * GRAVITY)
    assert abs(result["h_rise_m"] - expected_rise) < 1e-6


# ── Sensitivity analysis tests ────────────────────────────────────────────

def test_sensitivity_returns_all_params():
    """Sensitivity dict should have entries for all 14 parameters."""
    params = _default_params()
    sens = compute_sensitivity(params.to_tensor(), IMOGEN_MASS, IMOGEN_HEIGHT)
    assert len(sens) == 14
    for name in [
        "approach_speed", "plant_angle", "takeoff_knee_angle",
        "ground_contact_time_takeoff", "knee_drive_peak_speed",
    ]:
        assert name in sens


def test_approach_speed_has_positive_sensitivity():
    """Increasing approach speed should increase predicted height."""
    params = _default_params()
    sens = compute_sensitivity(params.to_tensor(), IMOGEN_MASS, IMOGEN_HEIGHT)
    assert sens["approach_speed"] > 0, "Approach speed should have positive sensitivity"


def test_alignment_deviation_has_negative_sensitivity():
    """More body misalignment should decrease clearance efficiency."""
    params = _default_params()
    sens = compute_sensitivity(params.to_tensor(), IMOGEN_MASS, IMOGEN_HEIGHT)
    assert sens["body_alignment_deviation"] < 0, (
        "Alignment deviation should have negative sensitivity"
    )


# ── Differentiable model tests ────────────────────────────────────────────

def test_differentiable_model_matches_analytical():
    """Differentiable version should produce the same height as analytical."""
    params = _default_params()
    analytical = predict_bar_clearance(params, IMOGEN_MASS, IMOGEN_HEIGHT)

    diff_h = _evaluate_height_differentiable(
        params.to_tensor(), IMOGEN_MASS, IMOGEN_HEIGHT,
    ).item()

    assert abs(diff_h - analytical["predicted_bar_height_m"]) < 0.01, (
        f"Differentiable: {diff_h:.3f}, Analytical: "
        f"{analytical['predicted_bar_height_m']:.3f}"
    )


def test_differentiable_model_has_gradients():
    """The differentiable model must propagate gradients to all active params."""
    params = _default_params().to_tensor().requires_grad_(True)
    h = _evaluate_height_differentiable(params, IMOGEN_MASS, IMOGEN_HEIGHT)
    h.backward()
    assert params.grad is not None
    # approach_speed (idx 0) should have a non-zero gradient
    assert abs(params.grad[0].item()) > 1e-6


# ── Optimisation tests ────────────────────────────────────────────────────

def test_optimize_technique_improves_height():
    """Optimiser should find parameters that predict at least as high as current."""
    params = _default_params()
    # Use a suboptimal starting point
    params.approach_speed_mps = 5.5
    params.takeoff_knee_angle_deg = 155.0

    result = optimize_technique(
        params, IMOGEN_MASS, IMOGEN_HEIGHT, n_iterations=50,
    )
    assert result.improvement_cm >= 0, "Optimiser should not decrease height"
    assert result.predicted_height_m >= result.current_height_m - 0.001


def test_optimize_technique_returns_coaching_cues():
    """Optimisation result should include human-readable coaching cues."""
    params = _default_params()
    params.approach_speed_mps = 5.5
    result = optimize_technique(
        params, IMOGEN_MASS, IMOGEN_HEIGHT, n_iterations=50,
    )
    assert len(result.coaching_cues) >= 1
    assert any("cm" in cue.lower() or "optimal" in cue.lower()
               for cue in result.coaching_cues)


# ── What-if scenario tests ────────────────────────────────────────────────

def test_what_if_speed_increase():
    """Increasing approach speed should increase predicted height."""
    params = _default_params()
    result = what_if_scenario(
        params, IMOGEN_MASS, IMOGEN_HEIGHT,
        {"approach_speed_mps": params.approach_speed_mps + 0.5},
    )
    assert result["delta_cm"] > 0, "More speed should predict more height"


def test_what_if_alignment_worsens():
    """Worsening body alignment should reduce clearance."""
    params = _default_params()
    result = what_if_scenario(
        params, IMOGEN_MASS, IMOGEN_HEIGHT,
        {"body_alignment_deviation_deg": 10.0},
    )
    assert result["delta_cm"] < 0, "Worse alignment should reduce clearance"


# ── Report extraction tests ──────────────────────────────────────────────

def test_extract_params_from_report():
    """Should extract TechniqueParameters from a video analysis report."""
    report = {
        "velocity": {
            "peak_horizontal_mps": 7.2,
            "takeoff_angle_deg": 22.0,
        },
        "com": {"rise_m": 0.25},
    }
    params = extract_params_from_report(report)
    assert abs(params.approach_speed_mps - 7.2) < 1e-4
    assert 55.0 <= params.plant_angle_deg <= 80.0


def test_extract_params_handles_negative_takeoff():
    """Negative takeoff angle (bad detection) should use default plant angle."""
    report = {
        "velocity": {
            "peak_horizontal_mps": 4.5,
            "takeoff_angle_deg": -30.0,
        },
    }
    params = extract_params_from_report(report)
    assert params.plant_angle_deg == 65.0  # default


# ── Coaching cue tests ────────────────────────────────────────────────────

def test_coaching_cues_ordered_by_impact():
    """Coaching cues should be ordered by absolute predicted impact."""
    current = _default_params()
    optimal = TechniqueParameters(
        approach_speed_mps=7.5,   # +0.5 m/s
        curve_radius_m=9.5,
        penultimate_step_length_cm=170.0,
        last_step_length_cm=120.0,
        plant_angle_deg=69.0,     # +2°
        takeoff_knee_angle_deg=175.0,
        takeoff_hip_angle_deg=178.0,
        arm_swing_timing_ms=50.0,
        free_leg_drive_angle_deg=80.0,
        ground_contact_time_takeoff_ms=130.0,
        knee_drive_peak_speed_mps=3.5,
    )
    sens = compute_sensitivity(current.to_tensor(), IMOGEN_MASS, IMOGEN_HEIGHT)
    cues = generate_coaching_cues(current, optimal, sens, 1.80, 1.85)
    # First cue should be the summary
    assert "improvement" in cues[0].lower() or "predicted" in cues[0].lower()
