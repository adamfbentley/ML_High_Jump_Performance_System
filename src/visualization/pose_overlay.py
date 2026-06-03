"""Render MediaPipe BlazePose landmark overlays on top of video frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

from src.pose_estimation.estimators.mediapipe_estimator import (
    MediaPipeEstimator,
    _nominal_fps_from_timestamps,
)

_LINE_COLOR = (80, 220, 120)
_POINT_COLOR = (40, 180, 255)
_LOW_CONFIDENCE_COLOR = (100, 100, 100)
_TEXT_COLOR = (255, 255, 255)
_INVALID_TEXT_COLOR = (0, 120, 255)

# BlazePose skeletal edges (subset of the canonical MediaPipe topology).
_POSE_CONNECTIONS = [
    (0, 11), (0, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (27, 31), (28, 32),
]


def _decoded_video_timing(video_path: Path) -> tuple[float, int]:
    """Return nominal decoded FPS and actual decoded frame count."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    reported_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    timestamps_ms: list[float] = []
    while cap.grab():
        timestamps_ms.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
    cap.release()

    fps = _nominal_fps_from_timestamps(timestamps_ms, reported_fps)
    return fps, len(timestamps_ms)


def _draw_pose_landmarks(
    frame_bgr: np.ndarray,
    landmarks: list,
    min_visibility: float = 0.2,
) -> None:
    """Draw BlazePose landmarks and connections onto a frame."""
    frame_h, frame_w = frame_bgr.shape[:2]

    for start_idx, end_idx in _POSE_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        if start.visibility < min_visibility or end.visibility < min_visibility:
            continue

        start_xy = (int(start.x * frame_w), int(start.y * frame_h))
        end_xy = (int(end.x * frame_w), int(end.y * frame_h))
        cv2.line(frame_bgr, start_xy, end_xy, _LINE_COLOR, 2, cv2.LINE_AA)

    for landmark in landmarks:
        x_px = int(landmark.x * frame_w)
        y_px = int(landmark.y * frame_h)
        color = _POINT_COLOR if landmark.visibility >= min_visibility else _LOW_CONFIDENCE_COLOR
        radius = 4 if landmark.visibility >= min_visibility else 2
        cv2.circle(frame_bgr, (x_px, y_px), radius, color, -1, cv2.LINE_AA)


def render_mediapipe_pose_overlay(
    video_path: str | Path,
    output_path: str | Path,
    model_path: str | Path | None = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> dict[str, float | int | str]:
    """Render a new video with BlazePose landmarks overlaid on each frame."""
    if mp is None:
        raise ImportError("mediapipe is not installed. Run: pip install mediapipe")

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    estimator = MediaPipeEstimator(
        model_path=model_path,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps, total_frames = _decoded_video_timing(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open overlay writer: {output_path}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=estimator._model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )

    valid_frames = 0
    processed_frames = 0
    previous_timestamp_ms = -1

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            decoded_timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if not np.isfinite(decoded_timestamp_ms) or (
                frame_idx > 0 and decoded_timestamp_ms <= 0
            ):
                decoded_timestamp_ms = frame_idx * 1000.0 / fps if fps > 0 else frame_idx
            timestamp_ms = max(previous_timestamp_ms + 1, int(round(decoded_timestamp_ms)))
            previous_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_valid = False
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                _draw_pose_landmarks(frame_bgr, landmarks)

                key_joints = [11, 12, 23, 24, 25, 26, 27, 28]
                pose_valid = all(landmarks[j].visibility > 0.5 for j in key_joints)
                if pose_valid:
                    valid_frames += 1

            processed_frames += 1

            status_text = "valid" if pose_valid else "low-confidence"
            status_color = _TEXT_COLOR if pose_valid else _INVALID_TEXT_COLOR
            cv2.putText(
                frame_bgr,
                f"frame {frame_idx + 1}/{max(total_frames, 1)}",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                _TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                f"pose: {status_text}",
                (18, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            writer.write(frame_bgr)
            frame_idx += 1

    cap.release()
    writer.release()

    valid_ratio = valid_frames / processed_frames if processed_frames else 0.0
    return {
        "video": str(video_path),
        "output": str(output_path),
        "frames": processed_frames,
        "fps": fps,
        "valid_frames": valid_frames,
        "valid_ratio": valid_ratio,
    }
