"""Background camera-motion estimation and stable-window detection.

A fixed-camera takeoff physics solver assumes a *stationary* camera across its
fit window: one fixed camera pose is reused for every frame.  Most of the
high-jump corpus is panned during the run-up but the camera is quasi-stationary
during the biomechanically critical plant→takeoff→early-flight window
(~0.2–0.5 s).  (The gravity/PnP projectile-fit physics that consumed this is
parked on branch `parked/moving-footage-physics`; this module is retained as
standalone camera-geometry tooling.)

This module turns "the camera is probably still during the jump" into a measured,
checkable quantity:

1.  Mask the athlete (foreground) so only the **background** drives motion
    estimation — a moving athlete must not be read as camera motion.
2.  Estimate a robust homography between consecutive frames from background
    features (a pure pan/tilt about the optical centre is the infinite
    homography; the planar track is also a homography, so both are captured).
3.  Reduce each homography to a scalar **background displacement** (median motion
    of an image-point grid, in pixels/frame).
4.  `find_stable_window` returns the longest contiguous run below a px/frame
    threshold — the window a single fixed camera pose can legitimately model.

This is Phase A of ``memory/plans/moving_footage_physics_plan.md``.  Phase B
(ground-plane stabilization) reuses :func:`estimate_homography` to warp the
window into a common reference frame.

Image coordinates are pixels ``(x, y)`` with ``y`` down.  No metric scale is
involved here; this is purely about camera stillness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "MotionConfig",
    "StableWindow",
    "CameraMotionResult",
    "StabilizedWindow",
    "estimate_homography",
    "background_displacement",
    "estimate_camera_motion",
    "find_stable_window",
    "analyze_camera_motion",
    "stabilize_window",
    "remap_points_through_homography",
    "warp_frame_to_reference",
]


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for camera-motion estimation") from exc
    return cv2


@dataclass(frozen=True)
class MotionConfig:
    """Tuning for background motion estimation and stable-window detection."""

    max_features: int = 1500
    fast_threshold: int = 12
    ratio_test: float = 0.78  # Lowe ratio for descriptor matching
    min_matches: int = 18
    ransac_reproj_px: float = 3.0
    grid: int = 6  # NxN grid of probe points for the displacement metric

    # stable-window detection
    max_disp_px_per_frame: float = 2.5  # background motion under this = "still"
    min_window_s: float = 0.25
    fail_disp_px: float = 1.0e6  # sentinel displacement for a failed estimate


@dataclass(frozen=True)
class StableWindow:
    """A contiguous run of near-stationary frames (inclusive indices)."""

    start: int
    end: int
    ref_index: int
    n_frames: int
    mean_disp_px: float
    max_disp_px: float
    duration_s: float


@dataclass(frozen=True)
class CameraMotionResult:
    """Per-frame background-motion diagnostics for a clip."""

    disp_px: np.ndarray  # (T,) consecutive-frame background displacement; [0]=0
    n_inliers: np.ndarray  # (T,) RANSAC inliers backing each estimate; [0]=0
    ok: np.ndarray  # (T,) bool: estimate succeeded
    fps: float
    stable_window: StableWindow | None


# --------------------------------------------------------------------------- #
# Low-level estimation
# --------------------------------------------------------------------------- #
def _to_gray(frame: np.ndarray):
    cv2 = _require_cv2()
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.shape[2] >= 3:
        return cv2.cvtColor(arr[:, :, :3].astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
    raise ValueError("frame must be grayscale or BGR")


def _background_mask(
    foreground_mask: np.ndarray | None,
    *,
    height: int,
    width: int,
):
    """Return a uint8 ORB mask (255 = use as background, 0 = exclude athlete)."""
    if foreground_mask is None:
        return None
    fg = np.asarray(foreground_mask)
    if fg.shape[:2] != (height, width):
        raise ValueError("foreground_mask must match frame (H, W)")
    mask = np.where(fg.astype(bool), 0, 255).astype(np.uint8)
    return mask


def estimate_homography(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    *,
    foreground_mask_a: np.ndarray | None = None,
    foreground_mask_b: np.ndarray | None = None,
    config: MotionConfig = MotionConfig(),
) -> tuple[np.ndarray | None, int]:
    """Estimate the homography mapping ``frame_a`` pixels onto ``frame_b``.

    Uses ORB features on the **background** only (athlete excluded via the
    foreground masks) and a RANSAC homography fit.

    Returns ``(H, n_inliers)`` with ``H`` a ``(3, 3)`` matrix, or ``(None, 0)``
    when too few reliable matches are found.
    """
    cv2 = _require_cv2()
    gray_a = _to_gray(frame_a)
    gray_b = _to_gray(frame_b)
    height, width = gray_a.shape[:2]
    mask_a = _background_mask(foreground_mask_a, height=height, width=width)
    mask_b = _background_mask(foreground_mask_b, height=height, width=width)

    orb = cv2.ORB_create(nfeatures=config.max_features, fastThreshold=config.fast_threshold)
    kp_a, des_a = orb.detectAndCompute(gray_a, mask_a)
    kp_b, des_b = orb.detectAndCompute(gray_b, mask_b)
    if des_a is None or des_b is None or len(kp_a) < config.min_matches or len(kp_b) < config.min_matches:
        return None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good: list = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < config.ratio_test * n.distance:
            good.append(m)
    if len(good) < config.min_matches:
        return None, 0

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h_mat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, config.ransac_reproj_px)
    if h_mat is None or not np.isfinite(h_mat).all() or abs(h_mat[2, 2]) < 1e-12:
        return None, 0
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    if n_inliers < config.min_matches:
        return None, 0
    return (h_mat / h_mat[2, 2]).astype(np.float64), n_inliers


def background_displacement(
    h_mat: np.ndarray,
    *,
    width: int,
    height: int,
    grid: int = 6,
) -> float:
    """Median pixel displacement of a probe grid under ``h_mat`` vs identity.

    A scalar summary of how far the background moves between the two frames.
    """
    cv2 = _require_cv2()
    xs = np.linspace(0.12 * width, 0.88 * width, grid)
    ys = np.linspace(0.12 * height, 0.88 * height, grid)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, np.asarray(h_mat, dtype=np.float64)).reshape(-1, 2)
    disp = np.linalg.norm(warped - pts.reshape(-1, 2), axis=1)
    return float(np.median(disp))


# --------------------------------------------------------------------------- #
# Per-clip motion + stable window
# --------------------------------------------------------------------------- #
def estimate_camera_motion(
    frames: Sequence[np.ndarray],
    *,
    foreground_masks: Sequence[np.ndarray | None] | None = None,
    fps: float,
    config: MotionConfig = MotionConfig(),
) -> CameraMotionResult:
    """Consecutive-frame background displacement across a clip.

    ``disp_px[t]`` is the background motion between frame ``t-1`` and ``t``
    (``disp_px[0] = 0``).  Failed estimates are marked ``ok=False`` and given
    ``config.fail_disp_px`` so they break any stable window.
    """
    frame_list = list(frames)
    n = len(frame_list)
    disp = np.zeros(n, dtype=np.float64)
    inliers = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    if n > 0:
        ok[0] = True
    height, width = (frame_list[0].shape[:2] if n else (0, 0))

    def _mask(idx: int):
        if foreground_masks is None or idx >= len(foreground_masks):
            return None
        return foreground_masks[idx]

    for t in range(1, n):
        h_mat, n_in = estimate_homography(
            frame_list[t - 1],
            frame_list[t],
            foreground_mask_a=_mask(t - 1),
            foreground_mask_b=_mask(t),
            config=config,
        )
        if h_mat is None:
            disp[t] = config.fail_disp_px
            inliers[t] = 0
            ok[t] = False
            continue
        disp[t] = background_displacement(h_mat, width=width, height=height, grid=config.grid)
        inliers[t] = n_in
        ok[t] = True

    window = find_stable_window(disp, fps=fps, config=config)
    return CameraMotionResult(disp_px=disp, n_inliers=inliers, ok=ok, fps=fps, stable_window=window)


def find_stable_window(
    disp_px: np.ndarray,
    *,
    fps: float,
    config: MotionConfig = MotionConfig(),
) -> StableWindow | None:
    """Longest contiguous run of frames with background motion below threshold.

    A window of length ``L`` needs ``L`` frames whose entry displacement (motion
    *into* the frame) is ``<= max_disp_px_per_frame``.  The first frame of the
    window seeds it (its own entry motion is not required to be small, since the
    relevant constraint is stillness *across* the window).
    """
    disp = np.asarray(disp_px, dtype=np.float64)
    n = len(disp)
    if n == 0:
        return None
    min_frames = max(2, int(np.ceil(config.min_window_s * fps)))

    still = disp <= config.max_disp_px_per_frame  # motion into frame t is small
    best: tuple[int, int] | None = None
    start = 1
    while start < n:
        if not still[start]:
            start += 1
            continue
        end = start
        while end + 1 < n and still[end + 1]:
            end += 1
        # window covers [start-1, end]: frames start..end each entered with small
        # motion, so start-1..end are mutually near-stationary.
        win_start = start - 1
        win_end = end
        if best is None or (win_end - win_start) > (best[1] - best[0]):
            best = (win_start, win_end)
        start = end + 1

    if best is None:
        return None
    win_start, win_end = best
    n_frames = win_end - win_start + 1
    if n_frames < min_frames:
        return None
    seg = disp[win_start + 1 : win_end + 1]  # entry motions within the window
    mean_disp = float(np.mean(seg)) if seg.size else 0.0
    max_disp = float(np.max(seg)) if seg.size else 0.0
    return StableWindow(
        start=int(win_start),
        end=int(win_end),
        ref_index=int((win_start + win_end) // 2),
        n_frames=int(n_frames),
        mean_disp_px=mean_disp,
        max_disp_px=max_disp,
        duration_s=float(n_frames / fps) if fps > 0 else float("nan"),
    )


def analyze_camera_motion(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    foreground_masks: Sequence[np.ndarray | None] | None = None,
    config: MotionConfig = MotionConfig(),
) -> CameraMotionResult:
    """Convenience wrapper: estimate motion and locate the stable window.

    NOTE (2026-06-14): ``stable_window`` here is the *longest* near-stationary
    run, which on full match clips is usually the static wide shot before the
    run-up — not the takeoff.  To analyse a jump, locate the takeoff frame from
    athlete kinematics first and pass an explicit takeoff-centred window to
    :func:`stabilize_window`.  Stabilization handles pan/zoom, so the takeoff
    window does not have to be "still" — only short and background-trackable.
    """
    return estimate_camera_motion(
        frames, foreground_masks=foreground_masks, fps=fps, config=config
    )


# --------------------------------------------------------------------------- #
# Phase B: ground-plane / rotational stabilization of an explicit window
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StabilizedWindow:
    """Per-frame homographies that warp a window of frames onto a reference.

    ``homographies[k]`` maps ``frame_indices[k]`` pixels into the reference
    frame.  ``residual_px[k]`` is the leftover background motion after warping
    (a stabilization-quality check); ``inf`` marks frames that could not be
    registered.
    """

    ref_index: int
    frame_indices: list[int]
    homographies: list[np.ndarray | None]
    inliers: list[int]
    residual_px: list[float]
    mean_residual_px: float
    max_residual_px: float
    n_registered: int


def warp_frame_to_reference(
    frame: np.ndarray,
    h_mat: np.ndarray,
    *,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Warp ``frame`` into the reference view using ``h_mat`` (frame→ref)."""
    cv2 = _require_cv2()
    arr = np.asarray(frame)
    height, width = arr.shape[:2]
    out_size = size if size is not None else (width, height)
    return cv2.warpPerspective(
        arr, np.asarray(h_mat, dtype=np.float64), out_size, flags=cv2.INTER_LINEAR
    )


def remap_points_through_homography(points_px: np.ndarray, h_mat: np.ndarray) -> np.ndarray:
    """Map image points ``(N, 2)`` through a homography (e.g. frame→ref).

    Non-finite inputs map to ``NaN`` outputs (preserved missing landmarks).
    """
    cv2 = _require_cv2()
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    finite = np.isfinite(pts).all(axis=1)
    if np.any(finite):
        mapped = cv2.perspectiveTransform(
            pts[finite].reshape(-1, 1, 2), np.asarray(h_mat, dtype=np.float64)
        ).reshape(-1, 2)
        out[finite] = mapped
    return out


def stabilize_window(
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    *,
    ref_index: int,
    foreground_masks: Sequence[np.ndarray | None] | None = None,
    config: MotionConfig = MotionConfig(),
) -> StabilizedWindow:
    """Register each window frame to a reference frame via background homography.

    ``frames`` is indexed by absolute frame index (the full decoded clip or a
    dict-like sequence); only ``frame_indices`` are registered.  This converts a
    panning/zooming takeoff window into an effectively stationary one: the
    apparatus and ground lock to the reference, and athlete landmarks can be
    remapped with :func:`remap_points_through_homography`.

    The residual is measured by warping each frame to the reference and
    re-estimating leftover background motion — a direct, honest "did it
    stabilize?" number rather than a trusted assumption.
    """
    indices = [int(i) for i in frame_indices]
    if ref_index not in indices:
        raise ValueError("ref_index must be one of frame_indices")

    def _mask(idx: int):
        if foreground_masks is None or idx >= len(foreground_masks):
            return None
        return foreground_masks[idx]

    ref_frame = frames[ref_index]
    ref_mask = _mask(ref_index)
    height, width = np.asarray(ref_frame).shape[:2]

    homographies: list[np.ndarray | None] = []
    inliers: list[int] = []
    residual_px: list[float] = []
    identity = np.eye(3, dtype=np.float64)

    for idx in indices:
        if idx == ref_index:
            homographies.append(identity.copy())
            inliers.append(config.max_features)
            residual_px.append(0.0)
            continue
        h_mat, n_in = estimate_homography(
            frames[idx],
            ref_frame,
            foreground_mask_a=_mask(idx),
            foreground_mask_b=ref_mask,
            config=config,
        )
        if h_mat is None:
            homographies.append(None)
            inliers.append(0)
            residual_px.append(float("inf"))
            continue
        warped = warp_frame_to_reference(frames[idx], h_mat, size=(width, height))
        h_check, _ = estimate_homography(
            warped, ref_frame, foreground_mask_b=ref_mask, config=config
        )
        residual = (
            background_displacement(h_check, width=width, height=height, grid=config.grid)
            if h_check is not None
            else float("inf")
        )
        homographies.append(h_mat)
        inliers.append(n_in)
        residual_px.append(residual)

    finite_resid = [r for r in residual_px if np.isfinite(r)]
    return StabilizedWindow(
        ref_index=int(ref_index),
        frame_indices=indices,
        homographies=homographies,
        inliers=inliers,
        residual_px=residual_px,
        mean_residual_px=float(np.mean(finite_resid)) if finite_resid else float("inf"),
        max_residual_px=float(np.max(finite_resid)) if finite_resid else float("inf"),
        n_registered=int(sum(1 for h in homographies if h is not None)),
    )
