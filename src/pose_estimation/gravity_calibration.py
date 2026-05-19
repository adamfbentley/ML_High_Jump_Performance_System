"""Gravity-anchored metres-per-pixel calibration for airborne motion.

During true flight, the athlete's centre of mass is a projectile: its image
trajectory should have a constant acceleration whose magnitude is gravity
expressed in pixels/s^2.  This module estimates metres-per-pixel from that
curvature as a validation-only scale source.  It is intentionally not wired
into the production path yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pose_estimation.egomotion import CameraMotion, warp_landmarks_remove_camera_motion
from src.pose_estimation.skeleton.com_estimation import compute_com_trajectory


@dataclass(frozen=True)
class GravityMppConfig:
    """Quality gates for gravity-derived mpp."""

    gravity_mps2: float = 9.81
    min_flight_frames: int = 10
    max_flight_s: float = 0.9
    min_downward_fraction: float = 0.65
    max_horizontal_accel_fraction: float = 0.45
    min_y_r_squared: float = 0.75


DEFAULT_GRAVITY_MPP_CONFIG = GravityMppConfig()


def normalised_landmarks_to_pixel_space(
    landmarks_2d_normalised: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert normalised BlazePose landmarks to pixel x/y/visibility."""
    landmarks_px = np.asarray(landmarks_2d_normalised, dtype=np.float64).copy()
    landmarks_px[:, :, 0] *= float(image_width)
    landmarks_px[:, :, 1] *= float(image_height)
    return landmarks_px


def stabilise_landmarks_with_camera_motion(
    landmarks_px: np.ndarray,
    camera_motion: CameraMotion | None,
) -> np.ndarray:
    """Map pixel landmarks into frame-0 coordinates when egomotion is available."""
    if camera_motion is None:
        return np.asarray(landmarks_px, dtype=np.float64)
    n_frames = min(landmarks_px.shape[0], camera_motion.cumulative_affines.shape[0])
    return warp_landmarks_remove_camera_motion(
        np.asarray(landmarks_px[:n_frames], dtype=np.float64),
        camera_motion,
        np.ones(n_frames, dtype=bool),
    )


def com_pixel_trajectory_y_up(
    landmarks_px: np.ndarray,
    *,
    image_height: int,
) -> np.ndarray:
    """Return CoM trajectory in pixel units with Y-up."""
    pose_px = np.zeros((landmarks_px.shape[0], landmarks_px.shape[1], 4), dtype=np.float64)
    pose_px[:, :, 0] = landmarks_px[:, :, 0]
    pose_px[:, :, 1] = float(image_height) - landmarks_px[:, :, 1]
    pose_px[:, :, 2] = 0.0
    pose_px[:, :, 3] = landmarks_px[:, :, 2]
    return compute_com_trajectory(pose_px, fps=1.0)["position"]


def fit_gravity_mpp_from_com_pixels(
    com_px_y_up: np.ndarray,
    fps: float,
    *,
    start_frame: int,
    end_frame: int,
    config: GravityMppConfig = DEFAULT_GRAVITY_MPP_CONFIG,
) -> tuple[float | None, dict]:
    """Fit a 2D projectile parabola and derive metres-per-pixel from gravity."""
    n_total = len(com_px_y_up)
    start = max(0, int(start_frame))
    end = min(n_total, int(end_frame))
    if end <= start:
        return None, {
            "method": "gravity_mpp",
            "accepted": False,
            "decision_reason": "empty_window",
            "flight_start_frame": start,
            "flight_end_frame": end,
            "n_fit_frames": 0,
        }

    frame_idx = np.arange(start, end, dtype=np.float64)
    xy = np.asarray(com_px_y_up[start:end, :2], dtype=np.float64)
    finite = np.isfinite(xy).all(axis=1)
    frame_idx = frame_idx[finite]
    xy = xy[finite]
    if len(xy) < config.min_flight_frames:
        return None, {
            "method": "gravity_mpp",
            "accepted": False,
            "decision_reason": "insufficient_flight_frames",
            "flight_start_frame": start,
            "flight_end_frame": end,
            "n_fit_frames": int(len(xy)),
        }

    t = (frame_idx - frame_idx[0]) / float(fps)
    x_coeffs = np.polyfit(t, xy[:, 0], 2)
    y_coeffs = np.polyfit(t, xy[:, 1], 2)
    x_fit = np.polyval(x_coeffs, t)
    y_fit = np.polyval(y_coeffs, t)
    residual = xy - np.column_stack([x_fit, y_fit])

    accel_px_s2 = np.array([2.0 * x_coeffs[0], 2.0 * y_coeffs[0]], dtype=np.float64)
    gravity_px_s2 = float(np.linalg.norm(accel_px_s2))
    if not np.isfinite(gravity_px_s2) or gravity_px_s2 <= 1e-9:
        return None, {
            "method": "gravity_mpp",
            "accepted": False,
            "decision_reason": "degenerate_acceleration",
            "flight_start_frame": start,
            "flight_end_frame": end,
            "n_fit_frames": int(len(xy)),
        }

    downward_fraction = float((-accel_px_s2[1]) / gravity_px_s2)
    horizontal_fraction = float(abs(accel_px_s2[0]) / gravity_px_s2)
    mpp = float(config.gravity_mps2 / gravity_px_s2)

    y_var = float(np.sum((xy[:, 1] - np.mean(xy[:, 1])) ** 2))
    y_resid = float(np.sum((xy[:, 1] - y_fit) ** 2))
    y_r2 = float(1.0 - y_resid / y_var) if y_var > 1e-12 else 0.0
    rms_residual_px = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    failures: list[str] = []
    if downward_fraction < config.min_downward_fraction:
        failures.append("acceleration_not_downward")
    if horizontal_fraction > config.max_horizontal_accel_fraction:
        failures.append("horizontal_acceleration_high")
    if y_r2 < config.min_y_r_squared:
        failures.append("parabola_fit_low_r2")

    return mpp, {
        "method": "gravity_mpp",
        "accepted": not failures,
        "decision_reason": "accepted" if not failures else ",".join(failures),
        "flight_start_frame": start,
        "flight_end_frame": end,
        "n_fit_frames": int(len(xy)),
        "mpp": mpp,
        "gravity_px_s2": gravity_px_s2,
        "accel_x_px_s2": float(accel_px_s2[0]),
        "accel_y_px_s2": float(accel_px_s2[1]),
        "downward_fraction": downward_fraction,
        "horizontal_accel_fraction": horizontal_fraction,
        "y_r_squared": y_r2,
        "rms_residual_px": rms_residual_px,
    }


def compute_gravity_anchored_mpp(
    landmarks_2d_normalised: np.ndarray,
    *,
    fps: float,
    image_width: int,
    image_height: int,
    takeoff_frame: int,
    camera_motion: CameraMotion | None = None,
    config: GravityMppConfig = DEFAULT_GRAVITY_MPP_CONFIG,
) -> tuple[float | None, dict, np.ndarray]:
    """Compute gravity-derived mpp from the post-takeoff CoM pixel trajectory.

    Returns ``(mpp, info, landmarks_px_used)``.  ``landmarks_px_used`` is the
    original pixel trajectory or the egomotion-stabilised trajectory, matching
    the trajectory used for the fit.
    """
    landmarks_px = normalised_landmarks_to_pixel_space(
        landmarks_2d_normalised,
        image_width,
        image_height,
    )
    landmarks_px_used = stabilise_landmarks_with_camera_motion(landmarks_px, camera_motion)
    com_px = com_pixel_trajectory_y_up(landmarks_px_used, image_height=image_height)

    max_frames = max(config.min_flight_frames, int(round(config.max_flight_s * fps)))
    start = int(takeoff_frame)
    end = min(len(com_px), start + max_frames + 1)
    mpp, info = fit_gravity_mpp_from_com_pixels(
        com_px,
        fps,
        start_frame=start,
        end_frame=end,
        config=config,
    )
    info["camera_stabilized"] = camera_motion is not None
    return mpp, info, landmarks_px_used


def calibrate_pixel_landmarks_with_constant_mpp(
    landmarks_2d: np.ndarray,
    landmarks_3d_world: np.ndarray,
    landmarks_px: np.ndarray,
    mpp: float,
    *,
    image_height: int,
) -> np.ndarray:
    """Pack pixel landmarks into metres using a constant mpp.

    This mirrors the Phase 9a ground-reference convention without touching the
    production calibration path.
    """
    n_frames, n_joints = landmarks_2d.shape[0], landmarks_2d.shape[1]
    source_px = np.asarray(landmarks_px[:n_frames], dtype=np.float64)
    x_m = source_px[:, :, 0] * float(mpp)
    y_m = (float(image_height) - source_px[:, :, 1]) * float(mpp)

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
    depth_scale = calib_span / world_span if world_span > 1e-6 and calib_span > 1e-6 else 1.0
    z_m = landmarks_3d_world[:n_frames, :, 2].astype(np.float64) * depth_scale

    calibrated = np.zeros((n_frames, n_joints, 4), dtype=np.float32)
    calibrated[:, :, 0] = x_m.astype(np.float32)
    calibrated[:, :, 1] = y_m.astype(np.float32)
    calibrated[:, :, 2] = z_m.astype(np.float32)
    calibrated[:, :, 3] = landmarks_2d[:n_frames, :, 2]
    return calibrated


def calibrate_landmarks_with_gravity_mpp(
    landmarks_2d: np.ndarray,
    landmarks_3d_world: np.ndarray,
    *,
    fps: float,
    image_width: int,
    image_height: int,
    takeoff_frame: int,
    camera_motion: CameraMotion | None = None,
    config: GravityMppConfig = DEFAULT_GRAVITY_MPP_CONFIG,
) -> tuple[np.ndarray, dict]:
    """Return landmarks calibrated with egomotion-stabilised gravity mpp."""
    mpp, info, landmarks_px_used = compute_gravity_anchored_mpp(
        landmarks_2d,
        fps=fps,
        image_width=image_width,
        image_height=image_height,
        takeoff_frame=takeoff_frame,
        camera_motion=camera_motion,
        config=config,
    )
    if mpp is None:
        failed = np.full_like(landmarks_3d_world, np.nan, dtype=np.float32)
        return failed, {**info, "method": "gravity_mpp_unavailable"}

    calibrated = calibrate_pixel_landmarks_with_constant_mpp(
        landmarks_2d,
        landmarks_3d_world,
        landmarks_px_used,
        mpp,
        image_height=image_height,
    )
    method = "egomotion_gravity_mpp" if camera_motion is not None else "gravity_mpp"
    return calibrated, {**info, "method": method}
