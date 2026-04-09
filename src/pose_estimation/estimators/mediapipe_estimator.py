"""MediaPipe BlazePose estimator for 2D landmark detection.

Extracts 33 body landmarks from monocular video frames using
Google's MediaPipe BlazePose pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


@dataclass
class PoseFrame:
    """Single frame of detected pose landmarks."""

    frame_index: int
    timestamp_ms: float
    landmarks_2d: np.ndarray  # (33, 3) — x, y, visibility
    landmarks_3d: Optional[np.ndarray] = None  # (33, 4) — x, y, z, visibility

    @property
    def is_valid(self) -> bool:
        """Check if enough key landmarks were detected with high confidence."""
        min_visibility = 0.5
        key_joints = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulders, hips, knees, ankles
        return all(self.landmarks_2d[j, 2] > min_visibility for j in key_joints)


@dataclass
class PoseSequence:
    """Time-ordered sequence of pose frames from one video."""

    video_path: str
    fps: float
    frames: list[PoseFrame] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0.0

    def to_numpy(self) -> np.ndarray:
        """Stack all 2D landmarks into (T, 33, 3) array."""
        return np.stack([f.landmarks_2d for f in self.frames])


class MediaPipeEstimator:
    """Wrapper around MediaPipe BlazePose for high jump video analysis.

    Usage:
        estimator = MediaPipeEstimator(model_complexity=2)
        sequence = estimator.process_video("jump_001.mp4")
    """

    # BlazePose landmark indices for high-jump-relevant joints
    LANDMARK_NAMES = {
        0: "nose", 11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
        23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
        31: "left_foot_index", 32: "right_foot_index",
    }

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if mp is None:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def process_video(self, video_path: str | Path) -> PoseSequence:
        """Run pose estimation on every frame of a video file.

        Args:
            video_path: Path to the video file (mp4, mov, etc.)

        Returns:
            PoseSequence with detected landmarks for each frame.
        """
        import cv2

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sequence = PoseSequence(video_path=str(video_path), fps=fps)

        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    landmarks_2d = np.array(
                        [[l.x, l.y, l.visibility] for l in lm]
                    )
                    landmarks_3d = None
                    if results.pose_world_landmarks:
                        wl = results.pose_world_landmarks.landmark
                        landmarks_3d = np.array(
                            [[l.x, l.y, l.z, l.visibility] for l in wl]
                        )

                    pose_frame = PoseFrame(
                        frame_index=frame_idx,
                        timestamp_ms=(frame_idx / fps) * 1000 if fps > 0 else 0,
                        landmarks_2d=landmarks_2d,
                        landmarks_3d=landmarks_3d,
                    )
                    sequence.frames.append(pose_frame)

                frame_idx += 1

        cap.release()
        return sequence
