"""Tests for Phase 9c scene-fixed calibration."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_jump_video import generate_report
from src.data_pipeline.sample import BiomechanicalSample, MovementType, SubjectInfo
from src.pose_estimation.egomotion import CameraMotion
from src.pose_estimation.scale_calibration import calibrate_landmarks_with_scene
from src.pose_estimation.scene_calibration import (
    SceneAnchors,
    fit_per_frame_homography,
    homography_valid_mask,
    warp_landmarks_to_scene,
)


def _anchor_sequence(
    n_frames: int = 1,
    *,
    pan_px_per_frame: float = 0.0,
    missing_frame: int | None = None,
    bar_height_m: float | None = 1.78,
    valid_fraction: float = 1.0,
) -> SceneAnchors:
    left_base = np.zeros((n_frames, 2), dtype=float)
    right_base = np.zeros((n_frames, 2), dtype=float)
    left_top = np.zeros((n_frames, 2), dtype=float)
    right_top = np.zeros((n_frames, 2), dtype=float)
    for t in range(n_frames):
        dx = pan_px_per_frame * t
        left_base[t] = [100.0 + dx, 500.0]
        right_base[t] = [500.0 + dx, 500.0]
        left_top[t] = [100.0 + dx, 140.0]
        right_top[t] = [500.0 + dx, 140.0]

    confidence = np.ones(n_frames, dtype=float)
    n_valid = int(round(n_frames * valid_fraction))
    if n_valid < n_frames:
        confidence[n_valid:] = 0.2
    if missing_frame is not None:
        right_top[missing_frame] = np.nan
        confidence[missing_frame] = 0.2

    return SceneAnchors(
        upright_left_base_px=left_base,
        upright_right_base_px=right_base,
        upright_left_top_px=left_top,
        upright_right_top_px=right_top,
        confidence=confidence,
        upright_separation_m=4.02,
        bar_height_m=bar_height_m,
    )


def _controlled_landmarks(
    homographies: np.ndarray,
    scene_x: np.ndarray,
    scene_y: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    segment_pixels: float = 40.0,
) -> np.ndarray:
    landmarks_2d = np.zeros((len(scene_x), 33, 3), dtype=np.float32)
    landmarks_2d[:, :, 2] = 1.0
    for t in range(len(scene_x)):
        px = _project_scene_to_pixel(homographies[t], np.array([scene_x[t], scene_y[t]]))
        landmarks_2d[t, :, 0] = px[0] / image_width
        landmarks_2d[t, :, 1] = px[1] / image_height

        # Keep a visible thigh segment at a known pixel length so anatomical
        # mpp diagnostics are controlled in these scene-decision tests.
        landmarks_2d[t, 23, 0] = 200.0 / image_width
        landmarks_2d[t, 23, 1] = 300.0 / image_height
        landmarks_2d[t, 25, 0] = (200.0 + segment_pixels) / image_width
        landmarks_2d[t, 25, 1] = 300.0 / image_height
    return landmarks_2d


def _blank_world_landmarks(n_frames: int) -> np.ndarray:
    landmarks_3d = np.zeros((n_frames, 33, 4), dtype=np.float32)
    landmarks_3d[:, :, 3] = 1.0
    return landmarks_3d


def _camera_motion(
    n_frames: int,
    tx_per_frame_px: float,
    *,
    confidence: np.ndarray | None = None,
    inlier_counts: np.ndarray | None = None,
) -> CameraMotion:
    pairwise = np.zeros((n_frames, 2, 3), dtype=float)
    cumulative = np.zeros((n_frames, 2, 3), dtype=float)
    pairwise[:, 0, 0] = 1.0
    pairwise[:, 1, 1] = 1.0
    cumulative[:, 0, 0] = 1.0
    cumulative[:, 1, 1] = 1.0
    for t in range(n_frames):
        pairwise[t, 0, 2] = -tx_per_frame_px if t > 0 else 0.0
        cumulative[t, 0, 2] = tx_per_frame_px * t
    if confidence is None:
        confidence = np.full(n_frames, 0.8, dtype=float)
        confidence[0] = 1.0
    if inlier_counts is None:
        inlier_counts = np.full(n_frames, 30, dtype=int)
        inlier_counts[0] = 0
    return CameraMotion(
        pairwise_affines=pairwise,
        cumulative_affines=cumulative,
        confidence=confidence,
        inlier_counts=inlier_counts,
        feature_counts=np.full(n_frames, 40, dtype=int),
    )


def _project_scene_to_pixel(homography: np.ndarray, scene_xy: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(homography)
    xy1 = np.array([scene_xy[0], scene_xy[1], 1.0], dtype=float)
    px = inv @ xy1
    return px[:2] / px[2]


def test_perfect_homography_recovers_anchor_scene_coordinates():
    anchors = _anchor_sequence()
    homographies = fit_per_frame_homography(anchors)
    valid = homography_valid_mask(homographies, anchors.confidence)

    anchor_landmarks = np.zeros((1, 4, 3), dtype=float)
    anchor_landmarks[0, :, :2] = np.array(
        [
            anchors.upright_left_base_px[0],
            anchors.upright_right_base_px[0],
            anchors.upright_left_top_px[0],
            anchors.upright_right_top_px[0],
        ]
    )
    anchor_landmarks[:, :, 2] = 1.0

    warped = warp_landmarks_to_scene(anchor_landmarks, homographies, valid)

    expected = np.array(
        [
            [-2.01, 0.0],
            [2.01, 0.0],
            [-2.01, 1.78],
            [2.01, 1.78],
        ]
    )
    np.testing.assert_allclose(warped[0, :, :2], expected, atol=1e-3)


def test_homography_removes_synthetic_camera_pan():
    anchors = _anchor_sequence(n_frames=6, pan_px_per_frame=25.0)
    homographies = fit_per_frame_homography(anchors)
    valid = homography_valid_mask(homographies, anchors.confidence)

    scene_point = np.array([0.65, 1.05], dtype=float)
    landmarks_px = np.zeros((6, 1, 3), dtype=float)
    for t in range(6):
        landmarks_px[t, 0, :2] = _project_scene_to_pixel(homographies[t], scene_point)
        landmarks_px[t, 0, 2] = 1.0

    warped = warp_landmarks_to_scene(landmarks_px, homographies, valid)

    np.testing.assert_allclose(warped[:, 0, :2], np.tile(scene_point, (6, 1)), atol=1e-3)


def test_missing_anchor_marks_frame_invalid_for_fallback():
    anchors = _anchor_sequence(n_frames=4, missing_frame=2)
    homographies = fit_per_frame_homography(anchors)
    valid = homography_valid_mask(homographies, anchors.confidence)

    assert valid.tolist() == [True, True, False, True]


def test_bar_height_anchor_recovers_scene_y():
    anchors = _anchor_sequence(bar_height_m=1.78)
    homographies = fit_per_frame_homography(anchors)
    valid = homography_valid_mask(homographies, anchors.confidence)
    top_midpoint_px = np.array([[[300.0, 140.0, 1.0]]])

    warped = warp_landmarks_to_scene(top_midpoint_px, homographies, valid)

    assert warped[0, 0, 1] == pytest.approx(1.78, abs=0.005)


def test_scene_calibration_prefers_homography_and_interpolates_occlusion():
    n_frames = 6
    full_anchors = _anchor_sequence(n_frames=n_frames, pan_px_per_frame=20.0, bar_height_m=1.78)
    anchors = _anchor_sequence(
        n_frames=n_frames,
        pan_px_per_frame=20.0,
        missing_frame=3,
        bar_height_m=1.78,
    )
    homographies = fit_per_frame_homography(full_anchors)

    image_width, image_height = 800, 600
    scene_x = np.linspace(-0.4, 0.6, n_frames)
    scene_y = np.full(n_frames, 1.0)
    scene_mpp = anchors.upright_separation_m / 400.0

    landmarks_2d = _controlled_landmarks(
        homographies,
        scene_x,
        scene_y,
        image_width=image_width,
        image_height=image_height,
    )

    calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=scene_mpp * 40.0,
        scene_anchors=anchors,
    )

    assert info["method"] == "scene_homography"
    assert info["scene_anchor_decision_reason"] == "accepted"
    assert info["anchor_coverage_pct"] == pytest.approx(5 / 6 * 100, abs=0.01)
    np.testing.assert_allclose(calibrated[:, 0, 0], scene_x, atol=1e-3)
    assert np.isfinite(calibrated[3, 0, :2]).all()


def test_scene_calibration_rejects_low_coverage_clip():
    n_frames = 10
    anchors = _anchor_sequence(n_frames=n_frames, valid_fraction=0.3)
    homographies = fit_per_frame_homography(anchors)
    image_width, image_height = 800, 600
    scene_x = np.linspace(-0.2, 0.2, n_frames)
    scene_y = np.full(n_frames, 1.0)
    scene_mpp = anchors.upright_separation_m / 400.0
    landmarks_2d = _controlled_landmarks(
        homographies,
        scene_x,
        scene_y,
        image_width=image_width,
        image_height=image_height,
    )

    calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=scene_mpp * 40.0,
        scene_anchors=anchors,
    )

    assert info["method"] == "anatomical"
    assert info["scene_anchor_decision_reason"] == "coverage_below_threshold"
    assert info["anchor_coverage_pct"] == pytest.approx(30.0)
    assert np.isfinite(calibrated).all()


def test_scene_calibration_rejects_ratio_out_of_range_clip():
    n_frames = 10
    anchors = _anchor_sequence(n_frames=n_frames, valid_fraction=0.8)
    homographies = fit_per_frame_homography(anchors)
    image_width, image_height = 800, 600
    scene_x = np.linspace(-0.2, 0.2, n_frames)
    scene_y = np.full(n_frames, 1.0)
    scene_mpp = anchors.upright_separation_m / 400.0
    landmarks_2d = _controlled_landmarks(
        homographies,
        scene_x,
        scene_y,
        image_width=image_width,
        image_height=image_height,
    )

    calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=2.0 * scene_mpp * 40.0,
        scene_anchors=anchors,
    )

    assert info["method"] == "anatomical"
    assert info["scene_anchor_decision_reason"] == "ratio_out_of_range"
    assert info["scene_anatomical_scale_ratio"] == pytest.approx(0.5)
    assert np.isfinite(calibrated).all()


def test_truth_style_homography_bypasses_anatomical_ratio_gate():
    n_frames = 10
    anchors = _anchor_sequence(n_frames=n_frames)
    homographies = fit_per_frame_homography(anchors)
    image_width, image_height = 800, 600
    scene_x = np.linspace(-0.2, 0.2, n_frames)
    scene_y = np.full(n_frames, 1.0)
    scene_mpp = anchors.upright_separation_m / 400.0
    landmarks_2d = _controlled_landmarks(
        homographies,
        scene_x,
        scene_y,
        image_width=image_width,
        image_height=image_height,
    )

    _calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=2.0 * scene_mpp * 40.0,
        scene_anchors=anchors,
    )
    assert info["scene_anchor_decision_reason"] == "ratio_out_of_range"

    landmarks_px = landmarks_2d.astype(float).copy()
    landmarks_px[:, :, 0] *= image_width
    landmarks_px[:, :, 1] *= image_height
    truth_homographies = fit_per_frame_homography(anchors)
    truth_valid = homography_valid_mask(truth_homographies, anchors.confidence)
    scene_coords = warp_landmarks_to_scene(
        landmarks_px,
        truth_homographies,
        truth_valid,
    )

    assert np.isfinite(scene_coords[:, 0, :2]).all()
    np.testing.assert_allclose(scene_coords[:, 0, 0], scene_x, atol=1e-3)
    np.testing.assert_allclose(scene_coords[:, 0, 1], scene_y, atol=1e-3)


def test_scene_calibration_accepted_gap_keeps_trajectory_continuous():
    n_frames = 20
    fps = 30.0
    image_width, image_height = 800, 600
    full_anchors = _anchor_sequence(n_frames=n_frames, pan_px_per_frame=8.0)
    anchors = _anchor_sequence(n_frames=n_frames, pan_px_per_frame=8.0)
    anchors.confidence[8:12] = 0.2
    homographies = fit_per_frame_homography(full_anchors)
    scene_x = np.arange(n_frames, dtype=float) * (5.0 / fps)
    scene_y = np.full(n_frames, 1.0)
    scene_mpp = anchors.upright_separation_m / 400.0
    landmarks_2d = _controlled_landmarks(
        homographies,
        scene_x,
        scene_y,
        image_width=image_width,
        image_height=image_height,
    )

    calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=scene_mpp * 40.0,
        scene_anchors=anchors,
    )

    frame_to_frame = np.linalg.norm(np.diff(calibrated[:, 0, :2], axis=0), axis=1)
    expected_step = 5.0 / fps
    discontinuity = np.abs(frame_to_frame - expected_step)

    assert info["method"] == "scene_homography"
    assert info["scene_anchor_decision_reason"] == "accepted"
    assert info["anchor_coverage_pct"] == pytest.approx(80.0)
    assert np.max(discontinuity) < 0.05


def test_bar_occluded_gap_longer_than_four_frames_rejects_scene_path():
    n_frames = 20
    image_width, image_height = 800, 600
    anchors = _anchor_sequence(n_frames=n_frames, pan_px_per_frame=8.0)
    anchors.confidence[7:13] = 0.2
    homographies = fit_per_frame_homography(anchors)
    scene_x = np.linspace(-0.2, 0.2, n_frames)
    scene_y = np.full(n_frames, 1.0)
    scene_mpp = anchors.upright_separation_m / 400.0
    landmarks_2d = _controlled_landmarks(
        homographies,
        scene_x,
        scene_y,
        image_width=image_width,
        image_height=image_height,
    )

    _calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=scene_mpp * 40.0,
        scene_anchors=anchors,
    )

    assert info["method"] == "anatomical"
    assert info["scene_anchor_decision_reason"] == "coverage_below_threshold"
    assert info["scene_anchor_longest_invalid_gap_frames"] == 6


def test_egomotion_accepted_recovers_horizontal_velocity():
    n_frames = 10
    fps = 30.0
    image_width, image_height = 800, 600
    mpp = 0.01
    scene_x = 300.0 + np.arange(n_frames, dtype=float)
    image_x = scene_x - 5.0 * np.arange(n_frames, dtype=float)
    landmarks_2d = _controlled_landmarks(
        np.repeat(np.eye(3, dtype=float)[None, :, :], n_frames, axis=0),
        image_x,
        np.full(n_frames, 300.0),
        image_width=image_width,
        image_height=image_height,
        segment_pixels=40.0,
    )
    confidence = np.full(n_frames, 0.8, dtype=float)
    confidence[0] = 1.0
    confidence[4:6] = 0.2
    inliers = np.full(n_frames, 30, dtype=int)
    inliers[0] = 0
    inliers[4:6] = 0

    calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=mpp * 40.0,
        camera_motion=_camera_motion(n_frames, 5.0, confidence=confidence, inlier_counts=inliers),
    )

    vx = np.gradient(calibrated[:, 0, 0], 1.0 / fps)

    assert info["method"] == "egomotion"
    assert info["egomotion_coverage_pct"] == pytest.approx(80.0)
    assert np.median(vx) == pytest.approx(1.0 * mpp * fps, rel=0.05)


def test_egomotion_preserves_anatomical_vertical_coordinates():
    n_frames = 8
    image_width, image_height = 800, 600
    mpp = 0.01
    landmarks_2d = _controlled_landmarks(
        np.repeat(np.eye(3, dtype=float)[None, :, :], n_frames, axis=0),
        300.0 + np.arange(n_frames, dtype=float),
        np.linspace(320.0, 260.0, n_frames),
        image_width=image_width,
        image_height=image_height,
        segment_pixels=40.0,
    )
    camera_motion = _camera_motion(n_frames, 5.0)
    camera_motion.cumulative_affines[:, 1, 2] = 30.0 * np.arange(n_frames)

    anatomical, _fallback_info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=mpp * 40.0,
    )
    calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=mpp * 40.0,
        camera_motion=camera_motion,
    )

    assert info["method"] == "egomotion"
    np.testing.assert_allclose(calibrated[:, :, 1], anatomical[:, :, 1], atol=1e-6)
    assert calibrated[-1, 0, 0] != pytest.approx(anatomical[-1, 0, 0])


def test_egomotion_rejected_for_low_coverage():
    n_frames = 10
    image_width, image_height = 800, 600
    landmarks_2d = np.zeros((n_frames, 33, 3), dtype=np.float32)
    landmarks_2d[:, :, 2] = 1.0
    confidence = np.full(n_frames, 0.2, dtype=float)
    confidence[:3] = 0.8
    confidence[0] = 1.0
    inliers = np.zeros(n_frames, dtype=int)
    inliers[:3] = 30
    inliers[0] = 0

    _calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=0.4,
        camera_motion=_camera_motion(n_frames, 5.0, confidence=confidence, inlier_counts=inliers),
    )

    assert info["method"] == "anatomical"
    assert info["egomotion_decision_reason"] == "coverage_below_threshold"


def test_egomotion_rejected_for_low_confidence():
    n_frames = 10
    image_width, image_height = 800, 600
    landmarks_2d = np.zeros((n_frames, 33, 3), dtype=np.float32)
    landmarks_2d[:, :, 2] = 1.0
    confidence = np.full(n_frames, 0.4, dtype=float)
    confidence[0] = 1.0
    confidence[8:] = 0.2
    inliers = np.full(n_frames, 30, dtype=int)
    inliers[0] = 0
    inliers[8:] = 0

    _calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=0.4,
        camera_motion=_camera_motion(n_frames, 5.0, confidence=confidence, inlier_counts=inliers),
    )

    assert info["method"] == "anatomical"
    assert info["egomotion_decision_reason"] == "confidence_below_threshold"


def test_egomotion_takes_priority_over_accepted_scene_homography():
    n_frames = 10
    image_width, image_height = 800, 600
    anchors = _anchor_sequence(n_frames=n_frames, valid_fraction=0.8)
    homographies = fit_per_frame_homography(anchors)
    scene_x = np.linspace(-0.2, 0.2, n_frames)
    scene_y = np.full(n_frames, 1.0)
    scene_mpp = anchors.upright_separation_m / 400.0
    landmarks_2d = _controlled_landmarks(
        homographies,
        scene_x,
        scene_y,
        image_width=image_width,
        image_height=image_height,
    )

    _calibrated, info = calibrate_landmarks_with_scene(
        landmarks_2d,
        _blank_world_landmarks(n_frames),
        height_m=1.78,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=scene_mpp * 40.0,
        scene_anchors=anchors,
        camera_motion=_camera_motion(n_frames, 0.0),
    )

    assert info["method"] == "egomotion"
    assert info["egomotion_decision_reason"] == "accepted"
    assert info["scene_anchor_decision_reason"] == "accepted"


def test_report_emits_egomotion_and_scene_decision_reasons_when_both_accepted():
    n_frames = 10
    fps = 30.0
    com_position = np.zeros((n_frames, 3), dtype=float)
    com_position[:, 1] = np.linspace(1.0, 1.8, n_frames)
    com_velocity = np.zeros((n_frames, 3), dtype=float)
    com_velocity[:, 0] = 4.5
    com_velocity[:, 1] = 0.5
    com_velocity[5, 1] = 3.5
    grf = np.zeros((n_frames, 3), dtype=float)
    grf[:, 1] = 67.0 * 9.81
    pose_3d = np.zeros((n_frames, 33, 3), dtype=float)
    pose_3d[:, 27, 1] = 0.2
    pose_3d[:, 28, 1] = 0.2
    pose_3d[3:6, 28, 1] = 0.02
    sample = BiomechanicalSample(
        dataset_name="unit_test",
        trial_id="both_accepted",
        subject=SubjectInfo(subject_id="athlete", body_mass_kg=67.0, height_m=1.78),
        movement_type=MovementType.HIGH_JUMP,
        fps=fps,
        com_position=com_position,
        com_velocity=com_velocity,
        grf=grf,
        pose_3d=pose_3d,
    )

    report = generate_report(
        sample,
        pinn_grf=None,
        calibration_info={
            "method": "egomotion",
            "egomotion_decision_reason": "accepted",
            "scene_anchor_decision_reason": "accepted",
            "egomotion_coverage_pct": 80.0,
            "anchor_coverage_pct": 80.0,
            "scale_info": {},
        },
        pose_validity_pct_value=100.0,
    )

    assert report["calibration"]["egomotion_decision_reason"] == "accepted"
    assert report["calibration"]["scene_anchor_decision_reason"] == "accepted"
    assert report["quality"]["scene_fixed_horizontal_source"] == "egomotion"
