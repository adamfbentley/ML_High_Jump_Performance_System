"""Tests for background egomotion compensation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pose_estimation.egomotion import (
    CameraMotion,
    camera_motion_valid_mask,
    estimate_camera_motion,
    warp_landmarks_remove_camera_motion,
)


def _textured_canvas(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(900):
        x = int(rng.integers(8, width - 8))
        y = int(rng.integers(8, height - 8))
        colour = int(rng.integers(80, 255))
        if int(rng.integers(0, 2)) == 0:
            cv2.rectangle(canvas, (x - 3, y - 3), (x + 4, y + 4), (colour, colour, colour), -1)
        else:
            cv2.circle(canvas, (x, y), 4, (colour, colour, colour), -1)
    return canvas


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _box_landmarks(
    n_frames: int,
    width: int,
    height: int,
    centres: np.ndarray,
    *,
    box_w: float = 42.0,
    box_h: float = 58.0,
) -> np.ndarray:
    landmarks = np.zeros((n_frames, 33, 3), dtype=np.float32)
    indices = (11, 12, 23, 24, 25, 26, 27, 28)
    offsets = np.array(
        [
            [-box_w / 2, -box_h / 2],
            [box_w / 2, -box_h / 2],
            [-box_w / 2, 0.0],
            [box_w / 2, 0.0],
            [-box_w / 2, box_h / 2],
            [box_w / 2, box_h / 2],
            [-box_w / 4, box_h / 2],
            [box_w / 4, box_h / 2],
        ],
        dtype=np.float32,
    )
    for t in range(n_frames):
        pts = centres[t] + offsets
        for idx, pt in zip(indices, pts):
            landmarks[t, idx, 0] = pt[0] / width
            landmarks[t, idx, 1] = pt[1] / height
            landmarks[t, idx, 2] = 1.0
    return landmarks


def _pan_frames(
    n_frames: int,
    width: int,
    height: int,
    pan_px_per_frame: float,
    *,
    draw_athlete: bool = False,
    athlete_centres: np.ndarray | None = None,
    athlete_checker: bool = False,
) -> list[np.ndarray]:
    margin = int(abs(pan_px_per_frame) * (n_frames + 2) + 80)
    canvas = _textured_canvas(width + margin, height)
    frames: list[np.ndarray] = []
    for t in range(n_frames):
        shift = int(round(pan_px_per_frame * t))
        frame = canvas[:, shift : shift + width].copy()
        if draw_athlete and athlete_centres is not None:
            cx, cy = athlete_centres[t]
            left, top = int(round(cx - 24)), int(round(cy - 32))
            right, bottom = int(round(cx + 24)), int(round(cy + 32))
            if athlete_checker:
                for yy in range(top, bottom, 8):
                    for xx in range(left, right, 8):
                        colour = 255 if ((xx + yy) // 8) % 2 == 0 else 30
                        cv2.rectangle(frame, (xx, yy), (xx + 5, yy + 5), (colour, 40, 200), -1)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (30, 220, 220), -1)
                cv2.circle(frame, (int(round(cx)), int(round(cy))), 10, (240, 40, 40), -1)
        frames.append(frame)
    return frames


def test_static_scene_pairwise_affines_are_identity(tmp_path: Path):
    n_frames, width, height = 10, 320, 240
    frame = _textured_canvas(width, height)
    video_path = tmp_path / "static.avi"
    _write_video(video_path, [frame.copy() for _ in range(n_frames)])
    landmarks = np.zeros((n_frames, 33, 3), dtype=np.float32)

    motion = estimate_camera_motion(video_path, landmarks, image_width=width, image_height=height)

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.max(np.abs(motion.pairwise_affines - identity)) < 0.01


def test_horizontal_pan_recovers_pairwise_translation(tmp_path: Path):
    n_frames, width, height = 12, 320, 240
    video_path = tmp_path / "pan.avi"
    centres = np.tile(np.array([[160.0, 120.0]], dtype=np.float32), (n_frames, 1))
    _write_video(video_path, _pan_frames(n_frames, width, height, 20.0))
    landmarks = _box_landmarks(n_frames, width, height, centres)

    motion = estimate_camera_motion(video_path, landmarks, image_width=width, image_height=height)

    assert np.median(motion.pairwise_affines[1:, 0, 2]) == pytest.approx(-20.0, abs=1.5)
    assert np.median(motion.pairwise_affines[1:, 1, 2]) == pytest.approx(0.0, abs=1.5)
    assert np.median(motion.confidence[1:]) > 0.7


def test_warp_removes_camera_pan_from_moving_athlete(tmp_path: Path):
    n_frames, width, height = 12, 320, 240
    video_path = tmp_path / "pan_athlete.avi"
    scene_x = 260.0 + np.arange(n_frames) * 1.0
    image_x = scene_x - np.arange(n_frames) * 20.0
    centres = np.column_stack([image_x, np.full(n_frames, 120.0)]).astype(np.float32)
    _write_video(
        video_path,
        _pan_frames(
            n_frames,
            width,
            height,
            20.0,
            draw_athlete=True,
            athlete_centres=centres,
        ),
    )
    landmarks = _box_landmarks(n_frames, width, height, centres)

    motion = estimate_camera_motion(video_path, landmarks, image_width=width, image_height=height)
    landmarks_px = landmarks.copy().astype(np.float64)
    landmarks_px[:, :, 0] *= width
    landmarks_px[:, :, 1] *= height
    warped = warp_landmarks_remove_camera_motion(
        landmarks_px,
        motion,
        camera_motion_valid_mask(motion),
    )

    np.testing.assert_allclose(warped[:, 23, 0], scene_x - 21.0, atol=1.0)


def test_athlete_mask_keeps_camera_motion_fit_on_background(tmp_path: Path):
    n_frames, width, height = 14, 320, 240
    video_path = tmp_path / "dominant_athlete.avi"
    athlete_x = 140.0 + np.arange(n_frames) * 7.0
    centres = np.column_stack([athlete_x, np.full(n_frames, 120.0)]).astype(np.float32)
    _write_video(
        video_path,
        _pan_frames(
            n_frames,
            width,
            height,
            8.0,
            draw_athlete=True,
            athlete_centres=centres,
            athlete_checker=True,
        ),
    )
    landmarks = _box_landmarks(n_frames, width, height, centres, box_w=64.0, box_h=82.0)

    motion = estimate_camera_motion(video_path, landmarks, image_width=width, image_height=height)

    assert np.median(motion.pairwise_affines[1:, 0, 2]) == pytest.approx(-8.0, abs=1.5)


def test_close_athlete_over_half_frame_still_fits_background_motion(tmp_path: Path):
    n_frames, width, height = 14, 320, 240
    video_path = tmp_path / "close_athlete.avi"
    centres = np.tile(np.array([[160.0, 120.0]], dtype=np.float32), (n_frames, 1))
    _write_video(
        video_path,
        _pan_frames(
            n_frames,
            width,
            height,
            6.0,
            draw_athlete=True,
            athlete_centres=centres,
            athlete_checker=True,
        ),
    )
    landmarks = _box_landmarks(n_frames, width, height, centres, box_w=230.0, box_h=180.0)

    motion = estimate_camera_motion(
        video_path,
        landmarks,
        image_width=width,
        image_height=height,
        athlete_padding_pct=0.02,
        min_inlier_count=10,
    )

    assert np.median(motion.pairwise_affines[1:, 0, 2]) == pytest.approx(-6.0, abs=1.75)
    assert np.median(motion.inlier_counts[1:]) >= 10


def test_missing_pose_frame_keeps_cumulative_trajectory_continuous(tmp_path: Path):
    n_frames, width, height = 14, 320, 240
    video_path = tmp_path / "missing_pose.avi"
    centres = np.tile(np.array([[160.0, 120.0]], dtype=np.float32), (n_frames, 1))
    _write_video(video_path, _pan_frames(n_frames, width, height, 10.0))
    landmarks = _box_landmarks(n_frames, width, height, centres)
    landmarks[6, :, 2] = 0.0

    motion = estimate_camera_motion(video_path, landmarks, image_width=width, image_height=height)

    x_steps = np.diff(motion.cumulative_affines[:, 0, 2])
    assert motion.frames_without_mask == 1
    assert abs(x_steps[6]) < 1e-6
    assert np.max(np.abs(x_steps[7:9])) < 15.0


def test_camera_motion_valid_mask_mixed_quality():
    motion = CameraMotion(
        pairwise_affines=np.zeros((4, 2, 3)),
        cumulative_affines=np.zeros((4, 2, 3)),
        confidence=np.array([1.0, 0.8, 0.4, 0.7]),
        inlier_counts=np.array([0, 30, 30, 10]),
        feature_counts=np.array([0, 50, 50, 50]),
    )

    assert camera_motion_valid_mask(motion).tolist() == [True, True, False, False]
