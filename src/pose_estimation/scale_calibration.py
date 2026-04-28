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

import numpy as np

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
        thigh_length_m: Measured hip→knee length (Imogen: 0.43 m).
        shank_length_m: Measured knee→ankle length (Imogen: 0.47 m).

    Returns:
        calibrated_3d: (T, 33, 4) — (x_metres, y_metres, z_metres, visibility)
    """
    T, J = landmarks_2d.shape[0], landmarks_2d.shape[1]

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

    # 2. Per-frame conversion of normalised coords → metres.
    mpp = per_frame_mpp[:, None]  # (T, 1) for broadcasting over landmarks
    x_m = landmarks_2d[:, :, 0].astype(np.float64) * image_width * mpp
    y_m = (1.0 - landmarks_2d[:, :, 1].astype(np.float64)) * image_height * mpp

    # 3. Ground reference: 5th percentile of ankle Y across the video, taken
    #    from frames where the ankle landmark has high visibility.  Using
    #    `min()` was unsafe: a single outlier frame (tracking error pushing an
    #    ankle landmark off-frame) corrupted ground level by metres, which
    #    then inflated *every* downstream Y including peak CoM.  The 5th
    #    percentile gives essentially-the-minimum across well-tracked ground
    #    contacts while rejecting solitary outliers.
    ankle_vis = landmarks_2d[:, [27, 28], 2]  # (T, 2)
    ankle_y_pair = y_m[:, [27, 28]]           # (T, 2)
    visible_ankle_y = ankle_y_pair[ankle_vis >= 0.5]
    if visible_ankle_y.size > 0:
        ground_level = float(np.percentile(visible_ankle_y, 5.0))
    else:
        ground_level = float(np.percentile(ankle_y_pair, 5.0))
    y_m -= ground_level

    # 4. Rescale world-landmark depth (z) so its vertical span matches ours.
    world_y = landmarks_3d_world[:, :, 1]  # hip-centred, y-axis
    world_span = float(world_y.max() - world_y.min())
    calib_span = float(y_m.max() - y_m.min())
    if world_span > 1e-6 and calib_span > 1e-6:
        depth_scale = calib_span / world_span
    else:
        # Fallback: use median mpp · image_height as a coarse depth scale.
        depth_scale = float(np.nanmedian(per_frame_mpp) * image_height)
    z_m = landmarks_3d_world[:, :, 2].astype(np.float64) * depth_scale

    # 5. Pack into (T, 33, 4) with visibility from 2D.
    calibrated = np.zeros((T, J, 4), dtype=np.float32)
    calibrated[:, :, 0] = x_m.astype(np.float32)
    calibrated[:, :, 1] = y_m.astype(np.float32)
    calibrated[:, :, 2] = z_m.astype(np.float32)
    calibrated[:, :, 3] = landmarks_2d[:, :, 2]  # visibility

    return calibrated
