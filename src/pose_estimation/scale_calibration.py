"""Scale calibration: convert BlazePose pixel coordinates to real-world metres.

Solves the fundamental problem that BlazePose world landmarks are hip-centred
and camera-relative — they don't give absolute position, velocity, or height
in real-world units.

Two strategies, in priority order:

1. **Multi-segment per-frame** (preferred — Phase 9a).  When the athlete's
   measured segment lengths are known (`SubjectInfo.thigh_length_m`,
   `shank_length_m`), the metres-per-pixel scale is computed *per frame* from
   every visible high-confidence segment, then aggregated by median.  This
   handles depth changes through the run-up (pixel-per-metre grows as the
   athlete approaches the camera) and is robust to single-frame occlusion.
   Scale is smoothed with a rolling median across frames before being applied.

2. **Single-frame nose-ankle** (legacy fallback).  When no segment lengths are
   available, fall back to picking the most upright early-approach frame and
   using nose-to-ankle as ~95% of full standing height.  Brittle when the
   chosen frame has any occlusion or lean.

Reference: bone segment lengths are biomechanically rigid across postures, so
a thigh seen at any approach distance projects to a pixel-length proportional
only to depth — making it a strong scale invariant.  Foreshortened limbs
(leg pointing at the camera) produce *higher* scale estimates that the
per-frame median across multiple segments rejects naturally.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from src.pose_estimation.egomotion import (
    CameraMotion,
    camera_motion_valid_mask,
    warp_landmarks_remove_camera_motion,
)
from src.pose_estimation.scene_calibration import (
    SceneAnchors,
    estimate_scene_scale_mpp,
    fit_per_frame_homography,
    homography_valid_mask,
    scene_anchor_diagnostics,
    warp_landmarks_to_scene,
)

logger = logging.getLogger(__name__)


# BlazePose landmark indices for skeletal segments of known length.
_SEGMENT_REFERENCES = (
    # (name, proximal_idx, distal_idx, anthropometric_attr_on_SubjectInfo)
    ("thigh_left",  23, 25, "thigh_length_m"),
    ("thigh_right", 24, 26, "thigh_length_m"),
    ("shank_left",  25, 27, "shank_length_m"),
    ("shank_right", 26, 28, "shank_length_m"),
)


def compute_per_frame_scale_mpp(
    landmarks_2d: np.ndarray,
    image_width: int,
    image_height: int,
    thigh_length_m: float | None = None,
    shank_length_m: float | None = None,
    min_visibility: float = 0.7,
    projection_percentile: float = 95.0,
) -> tuple[np.ndarray | None, dict]:
    """Single-scalar metres-per-pixel scale, broadcast across frames.

    Computes one video-wide metres-per-pixel value derived from the *in-plane*
    pixel projection of each known-length body segment.  Returned as a
    (T,)-shaped array (constant across frames) for caller compatibility.

    Why a single scalar rather than a per-frame estimate
    ----------------------------------------------------
    Per-frame projection of a leg segment is biased downward by foreshortening
    whenever the leg rotates out of the camera plane (which happens every
    stride during run-up).  The bias is asymmetric — projection can only get
    *shorter* than the true length, never longer — so taking the 95th
    percentile of pixel projections across the video selects the frames where
    the segment was closest to the camera plane (least foreshortened).  Those
    frames give the true in-plane projection at the scene's typical depth.

    A naive per-frame median across segments still leaves frame-to-frame mpp
    swings of ~30 % in real footage, which destroys finite-difference velocity
    estimates downstream.  A single stable scalar is far more useful.

    Args:
        landmarks_2d:         (T, 33, 3) normalised (x, y, visibility).
        image_width:          Video frame width in pixels.
        image_height:         Video frame height in pixels.
        thigh_length_m:       Measured hip→knee length in metres.
        shank_length_m:       Measured knee→ankle length in metres.
        min_visibility:       Per-landmark confidence threshold.
        projection_percentile: Percentile of pixel projections used as the
                              in-plane reference per segment.  Defaults to 95
                              (robust to noise, close to true max projection).

    Returns:
        (per_frame_mpp, info)
            per_frame_mpp: (T,) array of identical mpp values, or None if no
                           segment lengths were supplied.
            info: dict with diagnostics (n_valid_frames per segment,
                  per-segment in-plane pixel lengths, derived scalar mpp).
    """
    seg_lengths = {
        "thigh_length_m": thigh_length_m,
        "shank_length_m": shank_length_m,
    }
    refs = [
        (name, a, b, seg_lengths[attr])
        for (name, a, b, attr) in _SEGMENT_REFERENCES
        if seg_lengths.get(attr) is not None
    ]
    if not refs:
        return None, {"warning": "no segment lengths supplied — multi-segment scale unavailable"}

    T = landmarks_2d.shape[0]
    per_segment_pixel_lengths: dict[str, list[float]] = {name: [] for name, _, _, _ in refs}

    for t in range(T):
        for name, idx_a, idx_b, _ in refs:
            la = landmarks_2d[t, idx_a]
            lb = landmarks_2d[t, idx_b]
            if la[2] < min_visibility or lb[2] < min_visibility:
                continue
            dx_pix = (la[0] - lb[0]) * image_width
            dy_pix = (la[1] - lb[1]) * image_height
            d_pix = float(np.hypot(dx_pix, dy_pix))
            if d_pix < 1.0:
                continue
            per_segment_pixel_lengths[name].append(d_pix)

    # In-plane projection per segment via percentile of observed pixel lengths.
    seg_scales_mpp: list[float] = []
    seg_diagnostics: dict[str, dict] = {}
    for name, _, _, length_m in refs:
        pix_lens = per_segment_pixel_lengths[name]
        if not pix_lens:
            continue
        in_plane_pixels = float(np.percentile(pix_lens, projection_percentile))
        if in_plane_pixels < 1.0:
            continue
        seg_scales_mpp.append(length_m / in_plane_pixels)
        seg_diagnostics[name] = {
            "n_frames": len(pix_lens),
            "p50_pixels": float(np.percentile(pix_lens, 50)),
            "p95_pixels": float(np.percentile(pix_lens, 95)),
            "p100_pixels": float(np.max(pix_lens)),
            "length_m": length_m,
            "scale_mpp": length_m / in_plane_pixels,
        }

    if not seg_scales_mpp:
        return None, {
            "warning": "no frames had any visible reference segment",
            "segments_used": {n: 0 for n in per_segment_pixel_lengths},
        }

    video_mpp = float(np.median(seg_scales_mpp))
    per_frame_mpp = np.full(T, video_mpp, dtype=np.float64)

    info = {
        "n_valid_frames": sum(len(v) for v in per_segment_pixel_lengths.values()),
        "scalar_scale_mpp": video_mpp,
        "segments_used": {n: len(v) for n, v in per_segment_pixel_lengths.items()},
        "per_segment": seg_diagnostics,
    }
    return per_frame_mpp, info


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
    thigh_length_m: float | None = None,
    shank_length_m: float | None = None,
) -> np.ndarray:
    """Produce calibrated 3D landmarks in real-world metres.

    Scale strategy:
        If `thigh_length_m` and/or `shank_length_m` are provided, use the
        Phase 9a multi-segment per-frame scale (`compute_per_frame_scale_mpp`).
        Otherwise fall back to the legacy single-frame nose-ankle estimate
        (`compute_scale_factor`), which is brittle on partial occlusion.

    Coordinate output:
        - X horizontal (right-positive in pixel frame)
        - Y vertical, Y-up (flipped from BlazePose's Y-down convention)
        - Z depth from BlazePose world landmarks, rescaled to the calibrated
          vertical span so units are consistent.
        - Origin: lowest ankle position over the video maps to Y = 0 (ground).

    Note: horizontal X positions are unreliable when the camera pans to follow
    the athlete.  Vertical Y is well-calibrated; per-frame mpp also tracks
    depth-induced scale changes during the run-up.

    Args:
        landmarks_2d:       (T, 33, 3)  normalised (x, y, visibility)
        landmarks_3d_world: (T, 33, 4)  BlazePose world landmarks (x, y, z, vis)
        height_m: Athlete's standing height in metres (used as fallback ref).
        image_width:  Video frame width in pixels.
        image_height: Video frame height in pixels.
        thigh_length_m: Measured hip→knee length (Athlete A: 0.45 m).
        shank_length_m: Measured knee→ankle length (Athlete A: 0.45 m).

    Returns:
        calibrated_3d: (T, 33, 4) — (x_metres, y_metres, z_metres, visibility)
    """
    T = landmarks_2d.shape[0]

    # 1. Compute per-frame metres-per-pixel scale.
    per_frame_mpp, info = compute_per_frame_scale_mpp(
        landmarks_2d,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=thigh_length_m,
        shank_length_m=shank_length_m,
    )

    if per_frame_mpp is None:
        # Fall back to legacy nose-ankle approach.
        scale_y = compute_scale_factor(landmarks_2d, height_m)
        per_frame_mpp = np.full(T, scale_y / image_height, dtype=np.float64)
        if "warning" in info:
            logger.warning(f"  {info['warning']}; using legacy nose-ankle scale.")
    else:
        per_seg = info.get("per_segment", {})
        seg_summary = ", ".join(
            f"{n}={d['scale_mpp'] * 1000:.3f}@p95={d['p95_pixels']:.0f}px"
            for n, d in per_seg.items()
        )
        logger.info(
            f"  Multi-segment scale: {info['scalar_scale_mpp'] * 1000:.4f} mm/px "
            f"(median across segments). Per-seg: {seg_summary}"
        )

    landmarks_px = _landmarks_to_pixel_space(landmarks_2d, image_width, image_height)
    return _calibrate_pixel_landmarks_to_world(
        landmarks_2d,
        landmarks_3d_world,
        landmarks_px,
        per_frame_mpp,
        image_height=image_height,
    )


def _landmarks_to_pixel_space(
    landmarks_2d: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert normalised BlazePose landmarks to pixel x/y/visibility."""
    landmarks_px = np.zeros_like(landmarks_2d, dtype=np.float64)
    landmarks_px[:, :, 0] = landmarks_2d[:, :, 0].astype(np.float64) * image_width
    landmarks_px[:, :, 1] = landmarks_2d[:, :, 1].astype(np.float64) * image_height
    landmarks_px[:, :, 2] = landmarks_2d[:, :, 2]
    return landmarks_px


def _calibrate_pixel_landmarks_to_world(
    landmarks_2d: np.ndarray,
    landmarks_3d_world: np.ndarray,
    landmarks_px: np.ndarray,
    per_frame_mpp: np.ndarray,
    *,
    image_height: int,
) -> np.ndarray:
    """Pack pixel-space x/y landmarks into the canonical metres coordinate frame."""
    n_frames, n_joints = landmarks_2d.shape[0], landmarks_2d.shape[1]
    mpp = per_frame_mpp[:n_frames, None]
    source_px = np.asarray(landmarks_px[:n_frames], dtype=np.float64)
    x_m = source_px[:, :, 0] * mpp
    y_m = (image_height - source_px[:, :, 1]) * mpp

    # 5th-percentile visible ankle reference preserves the Phase 9a ground
    # convention while allowing x/y pixels to come from an egomotion-warped
    # frame-0 reference.
    ankle_vis = landmarks_2d[:n_frames, [27, 28], 2]
    ankle_y_pair = y_m[:, [27, 28]]
    visible = (ankle_vis >= 0.5) & np.isfinite(ankle_y_pair)
    visible_ankle_y = ankle_y_pair[visible]
    finite_ankle_y = ankle_y_pair[np.isfinite(ankle_y_pair)]
    if visible_ankle_y.size > 0:
        ground_level = float(np.percentile(visible_ankle_y, 5.0))
    elif finite_ankle_y.size > 0:
        ground_level = float(np.percentile(finite_ankle_y, 5.0))
    else:
        ground_level = 0.0
    y_m -= ground_level

    world_y = landmarks_3d_world[:n_frames, :, 1].astype(np.float64)
    finite_world_y = world_y[np.isfinite(world_y)]
    finite_y_m = y_m[np.isfinite(y_m)]
    if finite_world_y.size > 0 and finite_y_m.size > 0:
        world_span = float(finite_world_y.max() - finite_world_y.min())
        calib_span = float(finite_y_m.max() - finite_y_m.min())
    else:
        world_span = 0.0
        calib_span = 0.0
    if world_span > 1e-6 and calib_span > 1e-6:
        depth_scale = calib_span / world_span
    else:
        depth_scale = float(np.nanmedian(per_frame_mpp) * image_height)
    z_m = landmarks_3d_world[:n_frames, :, 2].astype(np.float64) * depth_scale

    calibrated = np.zeros((n_frames, n_joints, 4), dtype=np.float32)
    calibrated[:, :, 0] = x_m.astype(np.float32)
    calibrated[:, :, 1] = y_m.astype(np.float32)
    calibrated[:, :, 2] = z_m.astype(np.float32)
    calibrated[:, :, 3] = landmarks_2d[:n_frames, :, 2]
    return calibrated


def _estimate_bar_height_from_anatomical_scale(
    anchors: SceneAnchors,
    anatomical_mpp: float,
) -> float | None:
    """Estimate bar height when filename metadata is unavailable.

    This keeps scene-anchor pan correction available on untagged clips while
    marking the vertical scene scale as anatomical-derived rather than
    filename-derived.
    """
    if anatomical_mpp <= 0 or not np.isfinite(anatomical_mpp):
        return None
    heights_px: list[float] = []
    anchor_pairs = (
        (anchors.upright_left_base_px, anchors.upright_left_top_px),
        (anchors.upright_right_base_px, anchors.upright_right_top_px),
    )
    for base_arr, top_arr in anchor_pairs:
        if base_arr is None or top_arr is None:
            continue
        base = np.asarray(base_arr, dtype=np.float64)
        top = np.asarray(top_arr, dtype=np.float64)
        valid = np.isfinite(base).all(axis=1) & np.isfinite(top).all(axis=1)
        pixel_heights = base[valid, 1] - top[valid, 1]
        heights_px.extend(float(v) for v in pixel_heights if v > 1.0)
    if not heights_px:
        return None
    height_m = float(np.median(heights_px) * anatomical_mpp)
    if 0.8 <= height_m <= 2.5:
        return height_m
    return None


def _anatomical_scale_mpp_for_info(
    landmarks_2d: np.ndarray,
    height_m: float,
    image_width: int,
    image_height: int,
    thigh_length_m: float | None,
    shank_length_m: float | None,
) -> tuple[np.ndarray, dict]:
    """Return the anatomical mpp series used for diagnostics."""
    per_frame_mpp, info = compute_per_frame_scale_mpp(
        landmarks_2d,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=thigh_length_m,
        shank_length_m=shank_length_m,
    )
    if per_frame_mpp is None:
        scale_y = compute_scale_factor(landmarks_2d, height_m)
        per_frame_mpp = np.full(landmarks_2d.shape[0], scale_y / image_height, dtype=np.float64)
    return per_frame_mpp, info


def _densify_homographies(
    homographies: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Linearly fill invalid homography frames from nearest valid neighbours."""
    if homographies.ndim != 3 or homographies.shape[1:] != (3, 3):
        raise ValueError("homographies must have shape (T, 3, 3)")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (homographies.shape[0],):
        raise ValueError("valid_mask must have shape (T,)")
    if not np.any(valid):
        return homographies.copy()

    frame_idx = np.arange(homographies.shape[0], dtype=np.float64)
    valid_idx = frame_idx[valid]
    dense = np.empty_like(homographies, dtype=np.float64)
    for row in range(3):
        for col in range(3):
            values = homographies[valid, row, col]
            dense[:, row, col] = np.interp(frame_idx, valid_idx, values)

    denom = dense[:, 2, 2]
    normalisable = np.isfinite(dense).all(axis=(1, 2)) & (np.abs(denom) > 1e-12)
    dense[normalisable] = dense[normalisable] / denom[normalisable, None, None]
    return dense


def _densify_affines(
    affines: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Linearly fill invalid affine frames from nearest valid neighbours."""
    if affines.ndim != 3 or affines.shape[1:] != (2, 3):
        raise ValueError("affines must have shape (T, 2, 3)")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (affines.shape[0],):
        raise ValueError("valid_mask must have shape (T,)")
    if not np.any(valid):
        return affines.copy()

    frame_idx = np.arange(affines.shape[0], dtype=np.float64)
    valid_idx = frame_idx[valid]
    dense = np.empty_like(affines, dtype=np.float64)
    for row in range(2):
        for col in range(3):
            dense[:, row, col] = np.interp(frame_idx, valid_idx, affines[valid, row, col])
    return dense


def _longest_false_run(mask: np.ndarray) -> int:
    """Return the longest contiguous run of False values in a boolean mask."""
    values = np.asarray(mask, dtype=bool)
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def calibrate_landmarks_with_scene(
    landmarks_2d: np.ndarray,
    landmarks_3d_world: np.ndarray,
    height_m: float,
    *,
    image_width: int = 1920,
    image_height: int = 1080,
    thigh_length_m: float | None = None,
    shank_length_m: float | None = None,
    scene_anchors: SceneAnchors | None = None,
    camera_motion: CameraMotion | None = None,
    min_anchor_confidence: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Calibrate landmarks, preferring egomotion or scene homography when reliable.

    The existing anatomical calibration remains the fallback for all frames.
    Camera-motion and scene homography paths are accepted or rejected at clip
    level so coordinates never switch frame-by-frame.  Egomotion is used only
    for horizontal pixels because the Phase 9 blocker is scene-fixed horizontal
    velocity; anatomical vertical pixels preserve the established Y-up height
    convention.

    Returns:
        ``(calibrated_landmarks, calibration_info)``.
    """
    fallback = calibrate_landmarks_to_world(
        landmarks_2d,
        landmarks_3d_world,
        height_m,
        image_width=image_width,
        image_height=image_height,
        thigh_length_m=thigh_length_m,
        shank_length_m=shank_length_m,
    )
    info: dict = {
        "method": "anatomical",
        "anchor_coverage_pct": 0.0,
        "anchor_valid_counts_histogram": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},
        "anchor_four_point_valid_pct": 0.0,
        "crossbar_confirmed_pct": 0.0,
        "scene_anatomical_scale_ratio": None,
        "scene_anchor_decision_reason": "coverage_below_threshold",
        "egomotion_coverage_pct": 0.0,
        "egomotion_median_confidence": None,
        "egomotion_p10_confidence": None,
        "egomotion_min_inliers": None,
        "egomotion_frames_without_mask": 0,
        "egomotion_decision_reason": "unavailable",
    }
    if scene_anchors is None and camera_motion is None:
        return fallback, info

    anatomical_mpp, scale_info = _anatomical_scale_mpp_for_info(
        landmarks_2d,
        height_m,
        image_width,
        image_height,
        thigh_length_m,
        shank_length_m,
    )

    egomotion_valid: np.ndarray | None = None
    egomotion_decision_reason = "unavailable"
    if camera_motion is not None:
        n_motion = min(fallback.shape[0], camera_motion.cumulative_affines.shape[0])
        egomotion_valid = camera_motion_valid_mask(
            camera_motion,
            min_confidence=0.0,
        )[:n_motion]
        egomotion_coverage = float(np.mean(egomotion_valid)) if n_motion else 0.0
        if np.any(egomotion_valid):
            valid_confidence = camera_motion.confidence[:n_motion][egomotion_valid]
            egomotion_median_confidence = float(np.nanmedian(valid_confidence))
            egomotion_p10_confidence = float(np.nanpercentile(valid_confidence, 10.0))
        else:
            egomotion_median_confidence = None
            egomotion_p10_confidence = None

        if egomotion_coverage < 0.60:
            egomotion_decision_reason = "coverage_below_threshold"
        elif egomotion_median_confidence is None or egomotion_median_confidence < 0.60:
            egomotion_decision_reason = "confidence_below_threshold"
        else:
            egomotion_decision_reason = "accepted"

        info.update(
            {
                "egomotion_coverage_pct": round(egomotion_coverage * 100.0, 2),
                "egomotion_median_confidence": egomotion_median_confidence,
                "egomotion_p10_confidence": egomotion_p10_confidence,
                "egomotion_min_inliers": int(camera_motion.min_inlier_count),
                "egomotion_frames_without_mask": int(camera_motion.frames_without_mask),
                "egomotion_decision_reason": egomotion_decision_reason,
            }
        )

    homographies = np.empty((0, 3, 3), dtype=np.float64)
    valid = np.zeros(fallback.shape[0], dtype=bool)
    scene_mpp = np.full(fallback.shape[0], np.nan, dtype=np.float64)
    ratio = None
    bar_height_source = "none"
    bar_height_m = None
    scene_decision_reason = "coverage_below_threshold"
    if scene_anchors is not None:
        bar_height_source = "filename" if scene_anchors.bar_height_m is not None else "none"
        anchors_for_fit = scene_anchors
        if scene_anchors.bar_height_m is None:
            estimated_bar_height = _estimate_bar_height_from_anatomical_scale(
                scene_anchors,
                float(np.nanmedian(anatomical_mpp)),
            )
            if estimated_bar_height is not None:
                anchors_for_fit = replace(scene_anchors, bar_height_m=estimated_bar_height)
                bar_height_source = "anatomical_estimate"
        bar_height_m = anchors_for_fit.bar_height_m

        homographies = fit_per_frame_homography(anchors_for_fit)
        n_scene = min(fallback.shape[0], homographies.shape[0])
        valid = homography_valid_mask(
            homographies[:n_scene],
            confidence=anchors_for_fit.confidence[:n_scene],
            min_confidence=min_anchor_confidence,
        )

        scene_mpp = estimate_scene_scale_mpp(anchors_for_fit)[:n_scene]
        ratio_valid = valid & np.isfinite(scene_mpp) & np.isfinite(anatomical_mpp[:n_scene])
        ratio_valid &= anatomical_mpp[:n_scene] > 0
        if np.any(ratio_valid):
            ratio = float(
                np.nanmedian(scene_mpp[ratio_valid] / anatomical_mpp[:n_scene][ratio_valid])
            )

        scene_coverage = float(np.mean(valid)) if n_scene else 0.0
        longest_invalid_gap = _longest_false_run(valid)
        if scene_coverage < 0.60:
            scene_decision_reason = "coverage_below_threshold"
        elif longest_invalid_gap > 4:
            scene_decision_reason = "coverage_below_threshold"
        elif ratio is None or not (0.8 <= ratio <= 1.2):
            scene_decision_reason = "ratio_out_of_range"
        else:
            scene_decision_reason = "accepted"
    else:
        scene_coverage = 0.0
        longest_invalid_gap = 0

    info.update(
        {
            "anchor_coverage_pct": round(scene_coverage * 100.0, 2),
            "scene_anatomical_scale_ratio": ratio,
            "scene_anchor_decision_reason": scene_decision_reason,
            "scene_anchor_longest_invalid_gap_frames": int(longest_invalid_gap),
            "bar_height_source": bar_height_source,
            "bar_height_m": bar_height_m,
            "anatomical_scale_mpp": float(np.nanmedian(anatomical_mpp)),
            "scene_scale_mpp": (
                float(np.nanmedian(scene_mpp)) if np.isfinite(scene_mpp).any() else None
            ),
            "scale_info": scale_info,
        }
    )
    if scene_anchors is not None:
        info.update(scene_anchor_diagnostics(scene_anchors))

    if (
        camera_motion is not None
        and egomotion_valid is not None
        and egomotion_decision_reason == "accepted"
    ):
        n_motion = min(fallback.shape[0], camera_motion.cumulative_affines.shape[0])
        dense_cumulative = _densify_affines(
            camera_motion.cumulative_affines[:n_motion],
            egomotion_valid[:n_motion],
        )
        dense_motion = replace(
            camera_motion,
            cumulative_affines=dense_cumulative,
            pairwise_affines=camera_motion.pairwise_affines[:n_motion],
            confidence=camera_motion.confidence[:n_motion],
            inlier_counts=camera_motion.inlier_counts[:n_motion],
            feature_counts=camera_motion.feature_counts[:n_motion],
        )
        landmarks_px = _landmarks_to_pixel_space(
            landmarks_2d[:n_motion],
            image_width,
            image_height,
        )
        warped_px = warp_landmarks_remove_camera_motion(
            landmarks_px,
            dense_motion,
            np.ones(n_motion, dtype=bool),
        )
        source_px = landmarks_px.copy()
        source_px[:, :, 0] = warped_px[:, :, 0]
        calibrated = _calibrate_pixel_landmarks_to_world(
            landmarks_2d[:n_motion],
            landmarks_3d_world[:n_motion],
            source_px,
            anatomical_mpp[:n_motion],
            image_height=image_height,
        )
        info["method"] = "egomotion"
        return calibrated, info

    if scene_decision_reason != "accepted":
        return fallback, info

    n_frames = min(fallback.shape[0], homographies.shape[0])
    landmarks_px = _landmarks_to_pixel_space(landmarks_2d[:n_frames], image_width, image_height)
    dense_homographies = _densify_homographies(homographies[:n_frames], valid)
    scene_xy = warp_landmarks_to_scene(
        landmarks_px,
        dense_homographies,
        np.ones(n_frames, dtype=bool),
    )

    calibrated = fallback.copy()
    for t in range(n_frames):
        finite = np.isfinite(scene_xy[t, :, 0]) & np.isfinite(scene_xy[t, :, 1])
        calibrated[t, finite, 0] = scene_xy[t, finite, 0].astype(np.float32)
        calibrated[t, finite, 1] = scene_xy[t, finite, 1].astype(np.float32)
        calibrated[t, :, 3] = landmarks_2d[t, :, 2]

    info["method"] = "scene_homography"
    return calibrated, info
