"""Tests for stable takeoff-window physics fitting."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.analyze_stable_takeoff_window import (
    BarPlaneFitConfig,
    CameraModel,
    StableWindowAnchors,
    apparatus_object_points,
    fit_bar_plane_projectile,
    fit_projectile_to_com_pixels,
    project_world_points,
    solve_camera_from_anchors,
)


def _synthetic_camera(width: int = 1280, height: int = 720, focal_px: float = 950.0):
    pytest.importorskip("cv2")
    camera_matrix = np.array(
        [
            [focal_px, 0.0, width / 2.0],
            [0.0, focal_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rvec = np.array([[0.05], [-0.08], [0.01]], dtype=float)
    tvec = np.array([[0.15], [-0.75], [8.0]], dtype=float)
    return CameraModel(
        camera_matrix=camera_matrix,
        rvec=rvec,
        tvec=tvec,
        image_width=width,
        image_height=height,
        focal_length_px=focal_px,
        anchor_reprojection_rms_px=0.0,
    )


def _anchors_from_camera(camera: CameraModel) -> StableWindowAnchors:
    image_points = project_world_points(apparatus_object_points(1.75, 4.02), camera)
    return StableWindowAnchors(
        frame_index=0,
        bar_height_m=1.75,
        upright_separation_m=4.02,
        image_points_px=image_points,
    )


def test_solve_camera_from_anchors_reprojects_labels():
    camera = _synthetic_camera()
    anchors = _anchors_from_camera(camera)

    solved = solve_camera_from_anchors(
        anchors,
        image_width=camera.image_width,
        image_height=camera.image_height,
        focal_length_px=camera.focal_length_px,
    )

    assert solved.anchor_reprojection_rms_px < 1e-3


def test_fit_projectile_to_com_pixels_recovers_launch_velocity():
    camera = _synthetic_camera()
    anchors = _anchors_from_camera(camera)
    fps = 60.0
    takeoff_frame = 4
    n_frames = 34
    frame_indices = np.arange(n_frames)
    times = (frame_indices - takeoff_frame) / fps

    true_p0 = np.array([-0.45, 1.05, 1.20])
    true_v0 = np.array([2.8, 3.3, 1.15])
    world = true_p0[None, :] + times[:, None] * true_v0[None, :]
    world[:, 1] -= 0.5 * 9.81 * times**2
    com_px = project_world_points(world, camera)
    valid = np.ones(n_frames, dtype=bool)

    fit = fit_projectile_to_com_pixels(
        com_px,
        valid,
        takeoff_frame=takeoff_frame,
        fps=fps,
        camera=camera,
        anchors=anchors,
        athlete_height_m=1.75,
    )

    assert fit["accepted"] is True
    assert fit["projectile_reprojection_rms_px"] < 0.5
    assert fit["takeoff_vertical_mps"] == pytest.approx(true_v0[1], abs=0.15)
    assert fit["takeoff_horizontal_mps"] == pytest.approx(
        float(np.hypot(true_v0[0], true_v0[2])),
        abs=0.20,
    )
    assert fit["takeoff_angle_deg"] == pytest.approx(47.6, abs=1.5)


def test_fit_bar_plane_projectile_recovers_inplane_launch_velocity():
    camera = _synthetic_camera()
    anchors = _anchors_from_camera(camera)
    fps = 60.0
    takeoff_frame = 3
    n_frames = 30
    frame_indices = np.arange(n_frames)
    times = (frame_indices - takeoff_frame) / fps

    # An in-plane (Z=0, the apparatus plane) gravity parabola: X along the bar,
    # Y up. The bar-plane solver should recover vx and vy without any depth/focal
    # information.
    x0, y0 = -0.40, 1.05
    vx_true, vy_true = 2.6, 3.2
    world = np.zeros((n_frames, 3), dtype=float)
    world[:, 0] = x0 + vx_true * times
    world[:, 1] = y0 + vy_true * times - 0.5 * 9.81 * times**2
    world[:, 2] = 0.0
    com_px = project_world_points(world, camera)
    valid = np.ones(n_frames, dtype=bool)

    fit = fit_bar_plane_projectile(
        com_px,
        valid,
        takeoff_frame=takeoff_frame,
        fps=fps,
        anchors=anchors,
        athlete_height_m=1.75,
        config=BarPlaneFitConfig(fit_window_s=0.4),
    )

    assert fit["solver"] == "bar_plane"
    assert fit["accepted"] is True
    assert fit["bar_plane_reprojection_rms_m"] < 0.02
    assert fit["takeoff_vertical_mps"] == pytest.approx(vy_true, abs=0.1)
    assert fit["takeoff_horizontal_mps"] == pytest.approx(vx_true, abs=0.1)
    assert fit["takeoff_angle_deg"] == pytest.approx(
        float(np.degrees(np.arctan2(vy_true, vx_true))), abs=1.0
    )


def test_fit_bar_plane_projectile_rejects_too_few_frames():
    camera = _synthetic_camera()
    anchors = _anchors_from_camera(camera)
    com_px = np.full((8, 2), np.nan)
    com_px[3:5] = [640.0, 360.0]
    valid = np.zeros(8, dtype=bool)
    valid[3:5] = True

    fit = fit_bar_plane_projectile(
        com_px, valid, takeoff_frame=3, fps=60.0, anchors=anchors, athlete_height_m=1.75
    )
    assert fit["accepted"] is False
    assert "insufficient_fit_frames" in fit["decision_reasons"]
