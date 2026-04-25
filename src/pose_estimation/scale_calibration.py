"""Scale calibration: convert BlazePose pixel coordinates to real-world metres.

Uses the athlete's known height (and optionally bar height from filename) to
establish a metres-per-pixel scale factor.  This solves the fundamental problem
that BlazePose world landmarks are hip-centred and camera-relative — they do
not give absolute position, velocity, or height in real-world units.

Approach
--------
1. Find frames in the early approach where the athlete is roughly upright.
2. Measure the pixel-space distance from ankle midpoint to nose (≈ standing height).
3. ``scale = known_height_m / pixel_height``.
4. Apply that scale to the 2D normalised landmarks → real-world 2D positions.
5. Recombine with depth from world landmarks (also rescaled) → calibrated 3D.

If the bar height is known (parsed from filename), an independent scale estimate
is possible once the bar is detected.  For now we use athlete height only.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def estimate_standing_pixel_height(
    landmarks_2d: np.ndarray,
    *,
    n_candidate_frames: int = 30,
    min_visibility: float = 0.5,
) -> tuple[float, int]:
    """Find the athlete's pixel height from frames where she is most upright.

    We look at the first ``n_candidate_frames`` valid frames (typically the
    approach phase) and pick the frame where the vertical span from ankles to
    nose is largest — that's the most upright posture.

    Args:
        landmarks_2d: (T, 33, 3) normalised landmarks (x, y, visibility).
        n_candidate_frames: How many early frames to consider.
        min_visibility: Minimum visibility for key landmarks.

    Returns:
        (pixel_height, best_frame_idx) where pixel_height is in normalised
        [0, 1] y-units.
    """
    nose_idx = 0
    left_ankle_idx, right_ankle_idx = 27, 28
    left_hip_idx, right_hip_idx = 23, 24
    left_shoulder_idx, right_shoulder_idx = 11, 12

    best_height = 0.0
    best_frame = 0

    n_frames = landmarks_2d.shape[0]
    end = min(n_frames, n_candidate_frames * 3)  # scan a wider window

    for i in range(end):
        lm = landmarks_2d[i]

        # Check visibility of key landmarks
        key_idxs = [nose_idx, left_ankle_idx, right_ankle_idx,
                    left_hip_idx, right_hip_idx,
                    left_shoulder_idx, right_shoulder_idx]
        if any(lm[j, 2] < min_visibility for j in key_idxs):
            continue

        # Ankle midpoint (y)
        ankle_y = (lm[left_ankle_idx, 1] + lm[right_ankle_idx, 1]) / 2
        nose_y = lm[nose_idx, 1]

        # In normalised coords, y increases downward, so ankle_y > nose_y
        # when the person is upright.
        pixel_height = abs(ankle_y - nose_y)

        if pixel_height > best_height:
            best_height = pixel_height
            best_frame = i

    return best_height, best_frame


def compute_scale_factor(
    landmarks_2d: np.ndarray,
    height_m: float,
    image_height_px: int = 1,
    n_candidate_frames: int = 30,
) -> float:
    """Compute metres-per-normalised-unit from the athlete's known height.

    The returned scale converts normalised y-coordinates (0–1 range) to metres.

    Args:
        landmarks_2d: (T, 33, 3) normalised landmarks.
        height_m: Athlete's known standing height in metres.
        image_height_px: Actual pixel height of the video (for px conversion).
                         If 1, the scale is in metres-per-normalised-unit.
        n_candidate_frames: Passed to ``estimate_standing_pixel_height``.

    Returns:
        Scale factor such that ``real_y = (landmark_y - offset) * scale``.
    """
    pixel_height, best_frame = estimate_standing_pixel_height(
        landmarks_2d, n_candidate_frames=n_candidate_frames,
    )

    if pixel_height < 0.05:
        logger.warning(
            "Could not find a frame with sufficient standing height — "
            "falling back to approximate scale"
        )
        # Assume the athlete fills about 60% of the frame vertically
        pixel_height = 0.6

    # nose-to-ankle is roughly 95% of full standing height (head-top overshoot)
    effective_height_m = height_m * 0.95
    scale = effective_height_m / pixel_height

    logger.info(
        f"  Scale calibration: nose-ankle span = {pixel_height:.3f} (normalised) "
        f"in frame {best_frame} → {scale:.2f} m/unit"
    )
    return scale


def calibrate_landmarks_to_world(
    landmarks_2d: np.ndarray,
    landmarks_3d_world: np.ndarray,
    height_m: float,
    image_width: int = 1920,
    image_height: int = 1080,
) -> np.ndarray:
    """Produce calibrated 3D landmarks in real-world metres.

    Strategy:
    - Y from 2D normalised coordinates × scale_y → real metres (vertical).
    - X from 2D normalised coordinates × scale_x → real metres (horizontal),
      corrected for aspect ratio.
    - Z (depth) from BlazePose world landmarks, rescaled to match.
    - Origin: ground level at the ankle midpoint in the calibration frame.

    Note: horizontal (X) positions are unreliable when the camera pans to
    follow the athlete.  Vertical (Y) measurements are well-calibrated.

    Args:
        landmarks_2d:       (T, 33, 3)  normalised (x, y, visibility)
        landmarks_3d_world: (T, 33, 4)  BlazePose world landmarks (x, y, z, vis)
        height_m: Athlete's known height.
        image_width:  Video frame width in pixels (for aspect ratio correction).
        image_height: Video frame height in pixels.

    Returns:
        calibrated_3d: (T, 33, 4) — (x_metres, y_metres, z_metres, visibility)
    """
    T, J = landmarks_2d.shape[0], landmarks_2d.shape[1]

    # 1. Compute vertical scale (metres per normalised y-unit)
    scale_y = compute_scale_factor(landmarks_2d, height_m)

    # 2. Aspect ratio correction for horizontal axis.
    #    Normalised x spans [0,1] over image_width pixels,
    #    normalised y spans [0,1] over image_height pixels.
    #    So 1 normalised x-unit = (image_width/image_height) * 1 normalised y-unit
    #    in physical space (assuming square pixels).
    aspect_ratio = image_width / image_height
    scale_x = scale_y * aspect_ratio

    # 3. Convert 2D normalised → metres
    #    x_norm is horizontal (right-positive), y_norm is vertical (down-positive)
    #    We want Y-up, so flip y.
    x_m = landmarks_2d[:, :, 0] * scale_x        # (T, 33) horizontal
    y_m = (1.0 - landmarks_2d[:, :, 1]) * scale_y  # (T, 33) vertical, flipped to Y-up

    # 4. Set ground reference: lowest ankle position = 0
    left_ankle_y = y_m[:, 27]
    right_ankle_y = y_m[:, 28]
    ground_level = min(left_ankle_y.min(), right_ankle_y.min())
    y_m -= ground_level

    # 5. Rescale world-landmark depth (z) to match
    #    World landmarks have their own arbitrary scale; we rescale so that
    #    the vertical span in world coords maps to the same span in our calibrated y.
    world_y = landmarks_3d_world[:, :, 1]  # hip-centred, y-axis
    world_span = world_y.max() - world_y.min()
    calib_span = y_m.max() - y_m.min()
    if world_span > 1e-6:
        depth_scale = calib_span / world_span
    else:
        depth_scale = scale_y  # fallback

    z_m = landmarks_3d_world[:, :, 2] * depth_scale  # (T, 33)

    # 6. Pack into (T, 33, 4) with visibility from 2D
    calibrated = np.zeros((T, J, 4), dtype=np.float32)
    calibrated[:, :, 0] = x_m
    calibrated[:, :, 1] = y_m
    calibrated[:, :, 2] = z_m
    calibrated[:, :, 3] = landmarks_2d[:, :, 2]  # visibility

    return calibrated
