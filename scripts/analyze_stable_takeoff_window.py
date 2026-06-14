"""Extract takeoff-window physics from a stable final segment.

This is an experimental analyser for clips whose run-up was panned but whose
final plant/takeoff/early-flight window is effectively stationary. It estimates
takeoff-window-only physics from:

- MediaPipe 2D landmarks;
- one manually labelled apparatus anchor frame;
- known bar height and upright separation;
- heel/forefoot contact-selected toe-off;
- post-toe-off projectile motion under SI gravity.

It does not attempt to recover full run-up physics from panned frames.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_jump_video import (  # noqa: E402
    _KEY_JOINT_INDICES,
    DEFAULT_QUALITY_GATES,
    MIN_TAKEOFF_LAUNCH_VERTICAL_MPS,
    build_sample,
    compute_kinematics,
    extract_poses,
    pose_validity_pct,
    select_takeoff_frame_details,
    takeoff_window_pose_validity_pct,
)
from src.pose_estimation.scale_calibration import calibrate_landmarks_to_world  # noqa: E402
from src.pose_estimation.skeleton.com_estimation import compute_com_trajectory  # noqa: E402
from src.pose_estimation.skeleton.landmark_postprocessor import (  # noqa: E402
    PostProcessorConfig,
    postprocess_landmarks,
)

logger = logging.getLogger(__name__)

GRAVITY_MPS2 = 9.81
DEFAULT_UPRIGHT_SEPARATION_M = 4.02
ANCHOR_NAMES = ("left_base", "right_base", "left_top", "right_top")


@dataclass(frozen=True)
class StableWindowAnchors:
    """One stable-window apparatus labelling packet."""

    frame_index: int
    bar_height_m: float
    upright_separation_m: float
    image_points_px: np.ndarray  # (4, 2), order = ANCHOR_NAMES


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera model mapping world metres to image pixels."""

    camera_matrix: np.ndarray  # (3, 3)
    rvec: np.ndarray  # (3, 1)
    tvec: np.ndarray  # (3, 1)
    image_width: int
    image_height: int
    focal_length_px: float
    anchor_reprojection_rms_px: float


@dataclass(frozen=True)
class ProjectileFitConfig:
    """Quality settings for the stable-window projectile fit."""

    pixel_sigma: float = 12.0
    min_fit_frames: int = 10
    max_fit_s: float = 0.9
    max_reprojection_rms_px: float = 35.0
    com_height_prior_fraction: float = 0.58
    com_height_prior_sigma_m: float = 0.30
    min_camera_depth_m: float = 0.10
    # Soft physical cap on horizontal takeoff speed.  A single camera barely
    # observes motion along its optical axis, so the projectile fit can run the
    # out-of-plane velocity away to absurd values (hundreds of m/s) while still
    # matching the image.  A hinge well above the elite band (~4-5 m/s) leaves
    # legitimate fits untouched but removes the depth-runaway degeneracy.
    max_horizontal_speed_mps: float = 9.0
    horizontal_speed_penalty_sigma_mps: float = 0.75
    max_peak_com_m: float = DEFAULT_QUALITY_GATES.max_peak_com_m
    min_takeoff_horizontal_mps: float = DEFAULT_QUALITY_GATES.min_takeoff_horizontal_mps
    max_takeoff_horizontal_mps: float = DEFAULT_QUALITY_GATES.max_takeoff_horizontal_mps
    min_takeoff_angle_deg: float = DEFAULT_QUALITY_GATES.min_takeoff_angle_deg
    max_takeoff_angle_deg: float = DEFAULT_QUALITY_GATES.max_takeoff_angle_deg
    min_launch_vertical_mps: float = MIN_TAKEOFF_LAUNCH_VERTICAL_MPS


DEFAULT_PROJECTILE_FIT_CONFIG = ProjectileFitConfig()


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for stable-window camera fitting") from exc
    return cv2


def apparatus_object_points(
    bar_height_m: float,
    upright_separation_m: float = DEFAULT_UPRIGHT_SEPARATION_M,
) -> np.ndarray:
    """Return apparatus anchors in a bar-centred world frame.

    X runs along the crossbar, Y is up, and Z is runway-depth. The labelled
    apparatus points live on the Z=0 plane.
    """
    half_width = upright_separation_m / 2.0
    return np.array(
        [
            [-half_width, 0.0, 0.0],
            [half_width, 0.0, 0.0],
            [-half_width, bar_height_m, 0.0],
            [half_width, bar_height_m, 0.0],
        ],
        dtype=np.float64,
    )


def focal_length_from_fov_px(image_width: int, fov_deg: float) -> float:
    """Convert horizontal field-of-view to focal length in pixels."""
    if not (10.0 <= fov_deg <= 140.0):
        raise ValueError("fov_deg must be in [10, 140]")
    return float((0.5 * image_width) / np.tan(np.deg2rad(fov_deg) / 2.0))


def load_anchor_json(
    path: Path,
    *,
    bar_height_override_m: float | None = None,
) -> StableWindowAnchors:
    """Load a one-frame manual apparatus anchor JSON.

    Expected shape:

    ```json
    {
      "frame_index": 0,
      "bar_height_m": 1.75,
      "upright_separation_m": 4.02,
      "points_px": {
        "left_base": [100, 900],
        "right_base": [1800, 900],
        "left_top": [100, 330],
        "right_top": [1800, 330]
      }
    }
    ```
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    bar_height_m = bar_height_override_m or data.get("bar_height_m")
    if bar_height_m is None or float(bar_height_m) <= 0:
        raise ValueError("anchor JSON requires positive bar_height_m or --bar-height")

    points = data.get("points_px")
    if not isinstance(points, dict):
        raise ValueError("anchor JSON requires points_px object")
    image_points = []
    for name in ANCHOR_NAMES:
        point = np.asarray(points.get(name), dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"points_px.{name} must be [x_px, y_px]")
        image_points.append(point)

    return StableWindowAnchors(
        frame_index=int(data.get("frame_index", 0)),
        bar_height_m=float(bar_height_m),
        upright_separation_m=float(data.get("upright_separation_m", DEFAULT_UPRIGHT_SEPARATION_M)),
        image_points_px=np.asarray(image_points, dtype=np.float64),
    )


def solve_camera_from_anchors(
    anchors: StableWindowAnchors,
    *,
    image_width: int,
    image_height: int,
    focal_length_px: float,
) -> CameraModel:
    """Estimate camera pose from labelled apparatus points and fixed focal length."""
    cv2 = _require_cv2()
    if focal_length_px <= 1:
        raise ValueError("focal_length_px must be positive")

    camera_matrix = np.array(
        [
            [focal_length_px, 0.0, image_width / 2.0],
            [0.0, focal_length_px, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        apparatus_object_points(anchors.bar_height_m, anchors.upright_separation_m),
        anchors.image_points_px,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Could not solve camera pose from apparatus anchors")

    projected = project_world_points(
        apparatus_object_points(anchors.bar_height_m, anchors.upright_separation_m),
        CameraModel(
            camera_matrix=camera_matrix,
            rvec=rvec,
            tvec=tvec,
            image_width=image_width,
            image_height=image_height,
            focal_length_px=float(focal_length_px),
            anchor_reprojection_rms_px=0.0,
        ),
    )
    anchor_rms = float(np.sqrt(np.mean(np.sum((projected - anchors.image_points_px) ** 2, axis=1))))
    return CameraModel(
        camera_matrix=camera_matrix,
        rvec=rvec,
        tvec=tvec,
        image_width=image_width,
        image_height=image_height,
        focal_length_px=float(focal_length_px),
        anchor_reprojection_rms_px=anchor_rms,
    )


def project_world_points(points_m: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Project world-space metres to image pixels."""
    cv2 = _require_cv2()
    projected, _jac = cv2.projectPoints(
        np.asarray(points_m, dtype=np.float64),
        camera.rvec,
        camera.tvec,
        camera.camera_matrix,
        np.zeros(5, dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def world_to_camera(points_m: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Transform world-space points into the OpenCV camera frame."""
    cv2 = _require_cv2()
    rotation, _jac = cv2.Rodrigues(camera.rvec)
    return (rotation @ np.asarray(points_m, dtype=np.float64).T).T + camera.tvec.reshape(1, 3)


def com_pixel_trajectory(
    landmarks_2d: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Estimate 2D whole-body CoM in source pixels from MediaPipe landmarks."""
    pose_px = np.zeros((landmarks_2d.shape[0], landmarks_2d.shape[1], 4), dtype=np.float64)
    pose_px[:, :, 0] = landmarks_2d[:, :, 0] * float(image_width)
    pose_px[:, :, 1] = landmarks_2d[:, :, 1] * float(image_height)
    pose_px[:, :, 2] = 0.0
    pose_px[:, :, 3] = landmarks_2d[:, :, 2]
    return compute_com_trajectory(pose_px, fps=1.0)["position"][:, :2]


def key_joint_valid_mask(landmarks_2d: np.ndarray, min_visibility: float = 0.5) -> np.ndarray:
    """Frames where all key joints needed for stable takeoff physics are visible."""
    if landmarks_2d.ndim != 3 or landmarks_2d.shape[1] <= int(np.max(_KEY_JOINT_INDICES)):
        return np.zeros(landmarks_2d.shape[0], dtype=bool)
    return np.all(landmarks_2d[:, _KEY_JOINT_INDICES, 2] >= min_visibility, axis=1)


def projectile_positions(params: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Evaluate 3D projectile position for parameters [p0_xyz, v0_xyz]."""
    p0 = params[:3]
    v0 = params[3:6]
    positions = p0[None, :] + times_s[:, None] * v0[None, :]
    positions[:, 1] -= 0.5 * GRAVITY_MPS2 * times_s**2
    return positions


def _initialise_projectile_params(
    obs_px: np.ndarray,
    frame_indices: np.ndarray,
    takeoff_frame: int,
    fps: float,
    anchors: StableWindowAnchors,
    depth_m: float,
    vz_mps: float,
) -> np.ndarray:
    """Initialise projectile from image-to-apparatus-plane homography."""
    cv2 = _require_cv2()
    h_mat, _status = cv2.findHomography(
        anchors.image_points_px,
        apparatus_object_points(anchors.bar_height_m, anchors.upright_separation_m)[:, :2],
        method=0,
    )
    if h_mat is None:
        xy_scene = np.zeros((len(obs_px), 2), dtype=np.float64)
    else:
        warped = cv2.perspectiveTransform(obs_px.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
        xy_scene = warped.astype(np.float64)

    times = (frame_indices - takeoff_frame) / fps
    p0_xy = xy_scene[0]
    if len(times) >= 2 and np.ptp(times) > 1e-9:
        coeff_x = np.polyfit(times, xy_scene[:, 0], min(1, len(times) - 1))
        coeff_y = np.polyfit(times, xy_scene[:, 1], min(2, len(times) - 1))
        vx0 = float(coeff_x[0]) if len(coeff_x) == 2 else 0.0
        vy0 = float(coeff_y[-2]) if len(coeff_y) >= 2 else 3.0
    else:
        vx0, vy0 = 0.0, 3.0
    return np.array([p0_xy[0], p0_xy[1], depth_m, vx0, vy0, vz_mps], dtype=np.float64)


def _projectile_residuals(
    params: np.ndarray,
    *,
    times_s: np.ndarray,
    obs_px: np.ndarray,
    camera: CameraModel,
    athlete_height_m: float,
    config: ProjectileFitConfig,
) -> np.ndarray:
    positions = projectile_positions(params, times_s)
    projected = project_world_points(positions, camera)
    residuals = ((projected - obs_px) / config.pixel_sigma).ravel()

    # Depth-behind-camera penalty.  Always appended (zero when satisfied) so the
    # residual vector keeps a constant length — a conditional append changes the
    # residual dimension between iterations and breaks scipy's robust losses.
    camera_depth = world_to_camera(positions, camera)[:, 2]
    depth_penalty = np.minimum(0.0, camera_depth - config.min_camera_depth_m)
    residuals = np.concatenate([residuals, depth_penalty / config.min_camera_depth_m])

    # Soft physical cap on horizontal takeoff speed (out-of-plane degeneracy).
    horizontal_speed = float(np.hypot(params[3], params[5]))
    speed_excess = max(0.0, horizontal_speed - config.max_horizontal_speed_mps)
    residuals = np.concatenate(
        [residuals, np.array([speed_excess / config.horizontal_speed_penalty_sigma_mps])]
    )

    com_height_prior_m = athlete_height_m * config.com_height_prior_fraction
    residuals = np.concatenate(
        [residuals, np.array([(params[1] - com_height_prior_m) / config.com_height_prior_sigma_m])]
    )
    return residuals


def fit_projectile_to_com_pixels(
    com_px: np.ndarray,
    valid_mask: np.ndarray,
    *,
    takeoff_frame: int,
    fps: float,
    camera: CameraModel,
    anchors: StableWindowAnchors,
    athlete_height_m: float,
    config: ProjectileFitConfig = DEFAULT_PROJECTILE_FIT_CONFIG,
) -> dict[str, Any]:
    """Fit a 3D projectile whose image projection matches post-toe-off CoM pixels."""
    n_frames = len(com_px)
    fit_end = min(n_frames, takeoff_frame + int(round(config.max_fit_s * fps)) + 1)
    frame_indices = np.arange(takeoff_frame, fit_end, dtype=int)
    frame_indices = frame_indices[valid_mask[takeoff_frame:fit_end]]
    obs_px = np.asarray(com_px[frame_indices], dtype=np.float64)
    finite = np.isfinite(obs_px).all(axis=1)
    frame_indices = frame_indices[finite]
    obs_px = obs_px[finite]

    if len(frame_indices) < config.min_fit_frames:
        return {
            "accepted": False,
            "decision_reasons": ["insufficient_fit_frames"],
            "n_fit_frames": int(len(frame_indices)),
            "takeoff_frame": int(takeoff_frame),
        }

    times_s = (frame_indices - takeoff_frame) / float(fps)
    starts = [
        _initialise_projectile_params(obs_px, frame_indices, takeoff_frame, fps, anchors, z0, vz0)
        for z0 in (-3.0, -1.0, 0.0, 1.0, 3.0)
        for vz0 in (-3.0, 0.0, 3.0)
    ]

    best = None
    for start in starts:
        result = least_squares(
            _projectile_residuals,
            start,
            kwargs={
                "times_s": times_s,
                "obs_px": obs_px,
                "camera": camera,
                "athlete_height_m": athlete_height_m,
                "config": config,
            },
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=1000,
        )
        if best is None or result.cost < best.cost:
            best = result

    assert best is not None
    params = best.x.astype(float)
    positions = projectile_positions(params, times_s)
    projected = project_world_points(positions, camera)
    reproj_rms = float(np.sqrt(np.mean(np.sum((projected - obs_px) ** 2, axis=1))))

    vx, vy, vz = params[3:6]
    horizontal_mps = float(np.hypot(vx, vz))
    angle_deg = float(np.degrees(np.arctan2(vy, horizontal_mps)))
    t_apex_s = max(0.0, float(vy / GRAVITY_MPS2))
    peak_com_m = float(params[1] + vy * t_apex_s - 0.5 * GRAVITY_MPS2 * t_apex_s**2)

    failures: list[str] = []
    if reproj_rms > config.max_reprojection_rms_px:
        failures.append("projectile_reprojection_high")
    if vy < config.min_launch_vertical_mps:
        failures.append("launch_vertical_below_threshold")
    if not (
        config.min_takeoff_horizontal_mps
        <= horizontal_mps
        <= config.max_takeoff_horizontal_mps
    ):
        failures.append("takeoff_horizontal_out_of_range")
    if not (config.min_takeoff_angle_deg <= angle_deg <= config.max_takeoff_angle_deg):
        failures.append("takeoff_angle_out_of_range")
    if peak_com_m > config.max_peak_com_m:
        failures.append("peak_com_above_guardrail")

    return {
        "accepted": not failures,
        "decision_reasons": failures,
        "n_fit_frames": int(len(frame_indices)),
        "fit_frame_start": int(frame_indices[0]),
        "fit_frame_end": int(frame_indices[-1]),
        "takeoff_frame": int(takeoff_frame),
        "focal_length_px": round(float(camera.focal_length_px), 3),
        "anchor_reprojection_rms_px": round(camera.anchor_reprojection_rms_px, 3),
        "projectile_reprojection_rms_px": round(reproj_rms, 3),
        "p0_m": [round(float(value), 4) for value in params[:3]],
        "v0_mps": [round(float(value), 4) for value in params[3:6]],
        "takeoff_horizontal_mps": round(horizontal_mps, 3),
        "takeoff_vertical_mps": round(float(vy), 3),
        "takeoff_angle_deg": round(angle_deg, 2),
        "time_to_apex_s": round(t_apex_s, 3),
        "peak_com_height_m": round(peak_com_m, 3),
        "optimizer_cost": round(float(best.cost), 6),
    }


def _focal_sweep_values(base_focal_px: float, sweep: str | None) -> list[float]:
    if not sweep:
        return [float(base_focal_px)]
    factors = [float(part.strip()) for part in sweep.split(",") if part.strip()]
    if not factors:
        raise ValueError("--focal-sweep must contain at least one factor")
    return [float(base_focal_px) * factor for factor in factors]


def _summarise_sweep(fits: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [fit for fit in fits if "takeoff_angle_deg" in fit]
    if not complete:
        return {}
    metrics = ("takeoff_horizontal_mps", "takeoff_vertical_mps", "takeoff_angle_deg")
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = [float(fit[metric]) for fit in complete]
        summary[metric] = {
            "min": round(float(np.min(values)), 3),
            "median": round(float(np.median(values)), 3),
            "max": round(float(np.max(values)), 3),
        }
    summary["accepted_all_focal_sweep"] = bool(
        complete and all(fit["accepted"] for fit in complete)
    )
    return summary


def analyse_stable_takeoff_window(
    video_path: Path,
    *,
    anchor_json: Path,
    bar_height_m: float | None,
    body_mass_kg: float,
    height_m: float,
    thigh_length_m: float,
    shank_length_m: float,
    roi_crop: str,
    takeoff_frame_override: int | None,
    focal_length_px: float | None,
    fov_deg: float,
    focal_sweep: str | None,
) -> dict[str, Any]:
    """Run the stable-window physics analysis for one video."""
    landmarks_2d, landmarks_3d_world, fps, width, height = extract_poses(
        video_path,
        use_roi_crop=roi_crop,
    )
    anchors = load_anchor_json(anchor_json, bar_height_override_m=bar_height_m)
    base_focal = focal_length_px or focal_length_from_fov_px(width, fov_deg)

    calibrated = calibrate_landmarks_to_world(
        landmarks_2d,
        landmarks_3d_world,
        height_m=height_m,
        image_width=width,
        image_height=height,
        thigh_length_m=thigh_length_m,
        shank_length_m=shank_length_m,
    )
    postprocessed = postprocess_landmarks(
        calibrated,
        fps=fps,
        config=PostProcessorConfig(
            do_segment_enforce=True,
            height_m=height_m,
            segment_enforce_weight=0.8,
        ),
    )
    kinematics = compute_kinematics(postprocessed, fps, body_mass_kg)
    sample = build_sample(video_path, postprocessed, kinematics, fps, body_mass_kg, height_m)
    fallback = int(np.argmax(sample.com_velocity[:, 1]))
    selected_frame, contact_detected, contact_count, anchor_ok = select_takeoff_frame_details(
        sample,
        fallback_frame=fallback,
    )
    takeoff_frame = int(
        takeoff_frame_override if takeoff_frame_override is not None else selected_frame
    )

    valid_mask = key_joint_valid_mask(landmarks_2d)
    com_px = com_pixel_trajectory(landmarks_2d, width, height)
    fits = []
    for focal in _focal_sweep_values(base_focal, focal_sweep):
        camera = solve_camera_from_anchors(
            anchors,
            image_width=width,
            image_height=height,
            focal_length_px=focal,
        )
        fit = fit_projectile_to_com_pixels(
            com_px,
            valid_mask,
            takeoff_frame=takeoff_frame,
            fps=fps,
            camera=camera,
            anchors=anchors,
            athlete_height_m=height_m,
        )
        fits.append(fit)

    complete_fits = [fit for fit in fits if "projectile_reprojection_rms_px" in fit]
    best_fit = min(
        complete_fits,
        key=lambda fit: fit["projectile_reprojection_rms_px"],
        default=fits[0],
    )
    quality_failures = list(best_fit.get("decision_reasons", []))
    if not contact_detected:
        quality_failures.append("no_contact_interval")
    if not anchor_ok and takeoff_frame_override is None:
        quality_failures.append("takeoff_anchor_review_failed")

    return {
        "schema_version": 1,
        "analysis_mode": "stable_takeoff_window",
        "takeoff_window_only": True,
        "video": video_path.stem,
        "frames": int(len(landmarks_2d)),
        "fps": round(float(fps), 3),
        "image_size_px": [int(width), int(height)],
        "pose": {
            "pose_validity_pct": round(pose_validity_pct(landmarks_2d), 2),
            "takeoff_window_pose_validity_pct": round(
                takeoff_window_pose_validity_pct(landmarks_2d, takeoff_frame=takeoff_frame),
                2,
            ),
            "post_takeoff_valid_frames": int(np.count_nonzero(valid_mask[takeoff_frame:])),
        },
        "anchors": {
            "frame_index": int(anchors.frame_index),
            "bar_height_m": float(anchors.bar_height_m),
            "upright_separation_m": float(anchors.upright_separation_m),
            "labels": ANCHOR_NAMES,
        },
        "contact_takeoff": {
            "selected_frame": int(selected_frame),
            "used_frame": int(takeoff_frame),
            "fallback_argmax_vy_frame": int(fallback),
            "contact_interval_detected": bool(contact_detected),
            "contact_interval_count": int(contact_count),
            "takeoff_anchor_review_passed": bool(anchor_ok),
            "takeoff_frame_override_used": takeoff_frame_override is not None,
        },
        "projectile_fit": best_fit,
        "focal_sweep": {
            "base_focal_length_px": round(float(base_focal), 3),
            "fits": fits,
            "summary": _summarise_sweep(fits),
        },
        "quality": {
            "accepted_takeoff_window_physics": not quality_failures,
            "failures": quality_failures,
            "note": (
                "Stable takeoff-window physics only. Does not recover panned run-up "
                "or unblock optimiser claims without held-out validation."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--anchor-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bar-height", type=float, default=None)
    parser.add_argument("--mass", type=float, default=65.0)
    parser.add_argument("--height", type=float, default=1.75)
    parser.add_argument("--thigh", type=float, default=0.45)
    parser.add_argument("--shank", type=float, default=0.45)
    parser.add_argument("--roi-crop", choices=("off", "on", "takeoff"), default="off")
    parser.add_argument("--takeoff-frame", type=int, default=None)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument(
        "--focal-sweep",
        default="0.85,1.0,1.15",
        help="Comma-separated focal-length factors for uncertainty/sensitivity.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = build_parser().parse_args()
    report = analyse_stable_takeoff_window(
        args.video,
        anchor_json=args.anchor_json,
        bar_height_m=args.bar_height,
        body_mass_kg=args.mass,
        height_m=args.height,
        thigh_length_m=args.thigh,
        shank_length_m=args.shank,
        roi_crop=args.roi_crop,
        takeoff_frame_override=args.takeoff_frame,
        focal_length_px=args.focal_length_px,
        fov_deg=args.fov_deg,
        focal_sweep=args.focal_sweep,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    quality = report["quality"]
    fit = report["projectile_fit"]
    print(f"Wrote {args.output}")
    print(
        "accepted_takeoff_window_physics="
        f"{quality['accepted_takeoff_window_physics']} failures={quality['failures']}"
    )
    if "takeoff_angle_deg" in fit:
        print(
            f"takeoff: vh={fit['takeoff_horizontal_mps']:.2f} m/s, "
            f"vy={fit['takeoff_vertical_mps']:.2f} m/s, "
            f"angle={fit['takeoff_angle_deg']:.1f} deg"
        )


if __name__ == "__main__":
    main()
