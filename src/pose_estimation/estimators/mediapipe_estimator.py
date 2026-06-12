"""MediaPipe BlazePose estimator for 2D landmark detection.

Extracts 33 body landmarks from monocular video frames using
Google's MediaPipe BlazePose pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.constants import N_BLAZEPOSE_LANDMARKS

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


@dataclass
class PoseFrame:
    """Single frame of detected pose landmarks."""

    frame_index: int
    timestamp_ms: float
    landmarks_2d: np.ndarray  # (33, 3) — x, y, visibility
    landmarks_3d: np.ndarray | None = None  # (33, 4) — x, y, z, visibility

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


def _nominal_fps_from_timestamps(
    timestamps_ms: list[float],
    fallback_fps: float,
) -> float:
    """Estimate nominal FPS from decoded frame timestamps.

    iPhone MOV containers can report an average frame rate that differs from
    the decoded cadence. Kinematic derivatives need the cadence of the decoded
    frames, so prefer the median positive timestamp interval.
    """
    if len(timestamps_ms) >= 2:
        deltas_ms = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
        valid = deltas_ms[np.isfinite(deltas_ms) & (deltas_ms > 1e-3)]
        if valid.size:
            median_delta_ms = float(np.median(valid))
            fps = 1000.0 / median_delta_ms
            if np.isfinite(fps) and fps > 0:
                return fps
    return float(fallback_fps)


def _missing_pose_frame(frame_index: int, timestamp_ms: float) -> PoseFrame:
    """Represent an undetected frame without collapsing the video timeline."""
    return PoseFrame(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        landmarks_2d=np.zeros((N_BLAZEPOSE_LANDMARKS, 3), dtype=np.float32),
        landmarks_3d=np.zeros((N_BLAZEPOSE_LANDMARKS, 4), dtype=np.float32),
    )


def remap_normalized_to_full_frame(
    landmarks_in_crop: np.ndarray,
    bbox_norm: tuple[float, float, float, float],
) -> np.ndarray:
    """Remap 2D landmarks from crop-normalised space to full-frame normalised coords.

    Pure function — tested independently of MediaPipe.

    landmarks_in_crop: (33, 3) — x, y normalised in [0, 1] of the crop extent,
        plus visibility channel (unchanged).
    bbox_norm: (x1, y1, x2, y2) — crop boundaries as fractions of the original frame.
    Returns: (33, 3) with x, y normalised to the original full-frame [0, 1].

    Correctness note: downstream scale_calibration reads pixel projections via
    image_width / image_height, so the returned coordinates MUST be in the
    original full-frame normalised space.
    """
    x1, y1, x2, y2 = bbox_norm
    crop_w = x2 - x1
    crop_h = y2 - y1
    result = landmarks_in_crop.copy()
    result[:, 0] = landmarks_in_crop[:, 0] * crop_w + x1
    result[:, 1] = landmarks_in_crop[:, 1] * crop_h + y1
    return result


def _bbox_from_landmarks_2d(
    landmarks_2d: np.ndarray,
    min_visibility: float = 0.3,
) -> tuple[float, float, float, float] | None:
    """Normalised (x1,y1,x2,y2) enclosing all landmarks above min_visibility."""
    visible = landmarks_2d[landmarks_2d[:, 2] >= min_visibility]
    if visible.shape[0] == 0:
        return None
    return (
        float(visible[:, 0].min()),
        float(visible[:, 1].min()),
        float(visible[:, 0].max()),
        float(visible[:, 1].max()),
    )


def _aggregate_smoothed_crop(
    sequence: PoseSequence,
    margin: float = 0.20,
    min_visibility: float = 0.3,
) -> tuple[float, float, float, float] | None:
    """Union of per-frame landmark bboxes with margin, for two-pass ROI cropping.

    Returns normalised (x1, y1, x2, y2) in [0, 1], or None when no landmarks
    were detected in any frame of the pass-1 sequence.
    """
    all_x: list[float] = []
    all_y: list[float] = []
    for frame in sequence.frames:
        bb = _bbox_from_landmarks_2d(frame.landmarks_2d, min_visibility)
        if bb is not None:
            all_x += [bb[0], bb[2]]
            all_y += [bb[1], bb[3]]
    if not all_x:
        return None

    x1, x2 = min(all_x), max(all_x)
    y1, y2 = min(all_y), max(all_y)
    w, h = x2 - x1, y2 - y1

    # Apply relative margin; guarantee an absolute minimum half-width of 5 % so
    # near-degenerate detections (few visible landmarks on a wide-angle clip) still
    # produce a usable crop region.
    half_min = 0.05
    x1 = max(0.0, min(x1 - w * margin, x1 - half_min))
    x2 = min(1.0, max(x2 + w * margin, x2 + half_min))
    y1 = max(0.0, min(y1 - h * margin, y1 - half_min))
    y2 = min(1.0, max(y2 + h * margin, y2 + half_min))

    # Final sanity: reject if still degenerate after clamping to [0,1]
    if (x2 - x1) < 0.05 or (y2 - y1) < 0.05:
        return None
    return x1, y1, x2, y2


def _estimate_motion_apex_frame(
    sequence: PoseSequence,
    min_visibility: float = 0.3,
) -> int | None:
    """Estimate the flight apex frame from full-frame 2D hip landmarks.

    Image y increases downward, so the minimum visible hip-centre y is the
    highest body position. This is only used to choose a crop window; it is not
    used for physics or report metrics.
    """
    candidates: list[tuple[int, float]] = []
    for index, frame in enumerate(sequence.frames):
        hips = frame.landmarks_2d[[23, 24]]
        visible = hips[:, 2] >= min_visibility
        if np.any(visible):
            candidates.append((index, float(np.mean(hips[visible, 1]))))

    if candidates:
        return min(candidates, key=lambda item: item[1])[0]

    # Fallback: if hips are unavailable, use the frame whose visible-landmark
    # bbox reaches highest in the image.
    fallback: list[tuple[int, float]] = []
    for index, frame in enumerate(sequence.frames):
        bb = _bbox_from_landmarks_2d(frame.landmarks_2d, min_visibility)
        if bb is not None:
            fallback.append((index, bb[1]))
    if not fallback:
        return None
    return min(fallback, key=lambda item: item[1])[0]


def _expand_crop_to_min_size(
    bbox: tuple[float, float, float, float],
    *,
    min_width: float,
    min_height: float,
) -> tuple[float, float, float, float]:
    """Expand a normalised bbox around its centre to a minimum size."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width < min_width:
        centre_x = (x1 + x2) / 2.0
        half_width = min_width / 2.0
        x1 = max(0.0, centre_x - half_width)
        x2 = min(1.0, centre_x + half_width)
        if x2 - x1 < min_width:
            if x1 == 0.0:
                x2 = min(1.0, min_width)
            elif x2 == 1.0:
                x1 = max(0.0, 1.0 - min_width)
    if height < min_height:
        centre_y = (y1 + y2) / 2.0
        half_height = min_height / 2.0
        y1 = max(0.0, centre_y - half_height)
        y2 = min(1.0, centre_y + half_height)
        if y2 - y1 < min_height:
            if y1 == 0.0:
                y2 = min(1.0, min_height)
            elif y2 == 1.0:
                y1 = max(0.0, 1.0 - min_height)
    return x1, y1, x2, y2


def _aggregate_takeoff_crop(
    sequence: PoseSequence,
    *,
    pre_window_s: float = 1.2,
    post_window_s: float = 0.7,
    min_visibility: float = 0.3,
) -> tuple[float, float, float, float] | None:
    """Crop around the estimated takeoff/flight window, not the whole run-up.

    The full-clip crop has to include the approach path, which can keep the
    athlete small. For current stationary footage, the critical evidence is the
    final stride, plant, takeoff, and early flight. We estimate the flight apex
    from pass-1 landmarks, collect a generous time window around it, then expand
    the crop upward so shoulders and flight extension are not clipped.
    """
    apex_frame = _estimate_motion_apex_frame(sequence, min_visibility=min_visibility)
    if apex_frame is None:
        return None

    fps = sequence.fps if sequence.fps > 0 else 30.0
    start = max(0, apex_frame - int(round(pre_window_s * fps)))
    end = min(len(sequence.frames), apex_frame + int(round(post_window_s * fps)) + 1)

    all_x: list[float] = []
    all_y: list[float] = []
    for frame in sequence.frames[start:end]:
        bb = _bbox_from_landmarks_2d(frame.landmarks_2d, min_visibility)
        if bb is not None:
            all_x += [bb[0], bb[2]]
            all_y += [bb[1], bb[3]]
    if not all_x:
        return None

    x1, x2 = min(all_x), max(all_x)
    y1, y2 = min(all_y), max(all_y)
    width = x2 - x1
    height = y2 - y1

    # Horizontal margin keeps curve/run-up drift in frame. The larger upward
    # margin protects shoulders and head during flight, which the whole-clip ROI
    # previously risked trimming.
    x_margin = max(width * 0.30, 0.08)
    top_margin = max(height * 0.85, 0.14)
    bottom_margin = max(height * 0.45, 0.08)
    bbox = (
        max(0.0, x1 - x_margin),
        max(0.0, y1 - top_margin),
        min(1.0, x2 + x_margin),
        min(1.0, y2 + bottom_margin),
    )
    bbox = _expand_crop_to_min_size(bbox, min_width=0.22, min_height=0.35)
    if (bbox[2] - bbox[0]) < 0.05 or (bbox[3] - bbox[1]) < 0.05:
        return None
    return bbox


def _normalise_roi_crop_mode(roi_crop: bool | str) -> str:
    """Map legacy bool and CLI string values to one crop mode."""
    if isinstance(roi_crop, bool):
        return "full" if roi_crop else "off"
    if roi_crop == "on":
        return "full"
    if roi_crop in {"off", "full", "takeoff"}:
        return roi_crop
    raise ValueError(f"Unsupported ROI crop mode: {roi_crop!r}")


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

    _DEFAULT_MODEL = "data/models/mediapipe/pose_landmarker_heavy.task"

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ):
        if mp is None:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Resolve model path — search from project root or cwd
        if model_path is not None:
            self._model_path = str(Path(model_path))
        else:
            # Walk up from this file to find the project root
            candidates = [
                Path(__file__).resolve().parents[3] / self._DEFAULT_MODEL,
                Path.cwd() / self._DEFAULT_MODEL,
            ]
            for c in candidates:
                if c.exists():
                    self._model_path = str(c)
                    break
            else:
                raise FileNotFoundError(
                    f"Pose landmarker model not found. Download it to {self._DEFAULT_MODEL}\n"
                    "See: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker"
                )

    def process_video(self, video_path: str | Path, roi_crop: bool | str = False) -> PoseSequence:
        """Run pose estimation on every frame of a video file.

        Uses the MediaPipe Tasks PoseLandmarker API (≥0.10.14).

        When roi_crop=True/"on"/"full", a two-pass strategy locates the athlete
        across the whole clip. When roi_crop="takeoff", pass 2 uses a tighter
        crop around the estimated takeoff/flight window.

        Args:
            video_path: Path to the video file (mp4, mov, etc.)
            roi_crop: False/"off", True/"on"/"full", or "takeoff".

        Returns:
            PoseSequence with one frame per decoded source frame. Undetected
            frames are filled with zero-visibility placeholders. fps is derived
            from median decoded timestamp spacing (not container average).
        """
        crop_mode = _normalise_roi_crop_mode(roi_crop)
        if crop_mode == "off":
            return self._run_pose_detection(Path(video_path), crop_bbox_norm=None)

        # Two-pass: coarse full-frame pass → stable crop bbox → re-detect on crop
        pass1 = self._run_pose_detection(Path(video_path), crop_bbox_norm=None)
        bbox_norm = (
            _aggregate_takeoff_crop(pass1)
            if crop_mode == "takeoff"
            else _aggregate_smoothed_crop(pass1)
        )
        if bbox_norm is None:
            logger.warning("ROI crop pass 1 yielded no landmarks; returning full-frame result")
            return pass1
        logger.info(
            "ROI crop (%s): pass-2 bbox (normalised) x=[%.3f,%.3f] y=[%.3f,%.3f]",
            crop_mode, bbox_norm[0], bbox_norm[2], bbox_norm[1], bbox_norm[3],
        )
        return self._run_pose_detection(Path(video_path), crop_bbox_norm=bbox_norm)

    def _run_pose_detection(
        self,
        video_path: Path,
        crop_bbox_norm: tuple[float, float, float, float] | None,
    ) -> PoseSequence:
        """Core detection loop for one pass over the video.

        When crop_bbox_norm (x1,y1,x2,y2 normalised) is given, each frame is
        cropped to that region before pose detection and the resulting 2D
        landmarks are remapped back to full-frame normalised coordinates via
        remap_normalized_to_full_frame. The 3D world landmarks (metric,
        hip-centred) are passed through without remapping.

        Preserves invariants:
          - One output frame per decoded source frame.
          - Zero-visibility _missing_pose_frame placeholder for undetected frames.
          - fps from median decoded timestamp spacing.
        """
        import cv2
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Precompute pixel-space crop extents once (stable across all frames)
        crop_px: tuple[int, int, int, int] | None = None
        if crop_bbox_norm is not None:
            x1_n, y1_n, x2_n, y2_n = crop_bbox_norm
            crop_px = (
                max(0, int(x1_n * vid_w)),
                max(0, int(y1_n * vid_h)),
                min(vid_w, int(x2_n * vid_w)),
                min(vid_h, int(y2_n * vid_h)),
            )

        sequence = PoseSequence(video_path=str(video_path), fps=reported_fps)
        decoded_timestamps_ms: list[float] = []
        previous_timestamp_ms = -1

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )

        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                decoded_timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if not np.isfinite(decoded_timestamp_ms) or (
                    frame_idx > 0 and decoded_timestamp_ms <= 0
                ):
                    decoded_timestamp_ms = (
                        frame_idx * 1000.0 / reported_fps
                        if reported_fps > 0
                        else float(frame_idx)
                    )
                decoded_timestamps_ms.append(decoded_timestamp_ms)
                timestamp_ms = max(previous_timestamp_ms + 1, int(round(decoded_timestamp_ms)))
                previous_timestamp_ms = timestamp_ms

                # Crop to athlete ROI when requested
                frame_input = (
                    frame
                    if crop_px is None
                    else frame[crop_px[1]:crop_px[3], crop_px[0]:crop_px[2]]
                )

                rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    # 2D normalised landmarks (x, y in [0,1] of input image, visibility)
                    lm = result.pose_landmarks[0]
                    landmarks_2d_detected = np.array(
                        [[landmark.x, landmark.y, landmark.visibility] for landmark in lm],
                        dtype=np.float32,
                    )

                    # Remap from crop-normalised to full-frame-normalised when cropped
                    if crop_bbox_norm is not None:
                        landmarks_2d = remap_normalized_to_full_frame(
                            landmarks_2d_detected, crop_bbox_norm
                        )
                    else:
                        landmarks_2d = landmarks_2d_detected

                    # 3D world landmarks (metric, hip-centred) — no spatial remap needed
                    landmarks_3d = None
                    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                        wl = result.pose_world_landmarks[0]
                        landmarks_3d = np.array(
                            [
                                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                                for landmark in wl
                            ],
                            dtype=np.float32,
                        )

                    pose_frame = PoseFrame(
                        frame_index=frame_idx,
                        timestamp_ms=float(timestamp_ms),
                        landmarks_2d=landmarks_2d,
                        landmarks_3d=landmarks_3d,
                    )
                else:
                    pose_frame = _missing_pose_frame(
                        frame_index=frame_idx,
                        timestamp_ms=float(timestamp_ms),
                    )
                sequence.frames.append(pose_frame)
                frame_idx += 1

        cap.release()
        sequence.fps = _nominal_fps_from_timestamps(decoded_timestamps_ms, reported_fps)
        return sequence
