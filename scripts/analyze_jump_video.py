"""Analyse a high jump video end-to-end.

Full pipeline: video → pose → joint angles → CoM → phase metrics
              → PINN GRF estimation → actionable feedback.

Usage:
    python scripts/analyze_jump_video.py path/to/jump.mp4
    python scripts/analyze_jump_video.py path/to/jump.mp4 --mass 67 --height 1.75
    python scripts/analyze_jump_video.py path/to/jump.mp4 \
        --model experiments/results/pretrain_dynamics/best_model.pth
    python scripts/analyze_jump_video.py data/videos/raw/  # process all videos in a folder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.sample import BiomechanicalSample, MovementType, SubjectInfo
from src.kinematics.run_up_analysis import (
    compute_horizontal_velocity,
    detect_ground_contacts,
)
from src.kinematics.takeoff_analysis import estimate_grf_from_com
from src.pose_estimation.egomotion import estimate_camera_motion
from src.pose_estimation.estimators.mediapipe_estimator import MediaPipeEstimator
from src.pose_estimation.opensim_ik import is_opensim_available, run_opensim_ik
from src.pose_estimation.scale_calibration import (
    calibrate_landmarks_to_world,
    calibrate_landmarks_with_scene,
    compute_per_frame_scale_mpp,
)
from src.pose_estimation.scene_calibration import detect_scene_anchors
from src.pose_estimation.skeleton.com_estimation import compute_com_trajectory
from src.pose_estimation.skeleton.joint_angles import compute_joint_angles_sequence
from src.pose_estimation.skeleton.landmark_postprocessor import (
    PostProcessorConfig,
    postprocess_landmarks,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class QualityGateConfig:
    """Clip-level gates for calibration validation and training admission."""

    min_pose_validity_pct: float = 60.0
    min_kinematics_pose_validity_pct: float = 40.0
    max_segment_spread_ratio: float = 1.35
    max_kinematics_segment_spread_ratio: float = 1.60
    min_takeoff_horizontal_mps: float = 2.5
    max_takeoff_horizontal_mps: float = 5.5
    min_takeoff_angle_deg: float = 38.0
    max_takeoff_angle_deg: float = 55.0
    max_peak_com_m: float = 3.0


DEFAULT_QUALITY_GATES = QualityGateConfig()


def parse_bar_height(filename: str) -> float | None:
    """Extract bar height from filename like '06_03_24_one_1.75.mp4' → 1.75."""
    import re
    # Match decimal like _1.75 or _1.88 before the extension
    m = re.search(r'_(\d+\.\d+)(?:\.[a-zA-Z0-9]+)?$', filename)
    return float(m.group(1)) if m else None


def resolve_bar_height(filename: str, override_m: float | None = None) -> float | None:
    """Return explicit bar height when supplied, otherwise parse the filename."""
    if override_m is not None:
        if override_m <= 0:
            raise ValueError("bar height override must be positive")
        return float(override_m)
    return parse_bar_height(filename)


def parse_session_date(path: Path) -> str | None:
    """Extract session date from folder name like '06_03_24' → '2024-03-06'."""
    import re
    for part in path.parts:
        m = re.match(r'^(\d{2})_(\d{2})_(\d{2})$', part)
        if m:
            dd, mm, yy = m.groups()
            year = 2000 + int(yy)
            return f"{year}-{mm}-{dd}"
    return None


def collect_videos(input_path: Path) -> list[Path]:
    """Collect video files from a path, recursing into subdirectories."""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        videos = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
        return videos
    return []


def infer_default_report_dir(samples_dir: Path | None) -> Path:
    """Infer a run-specific report directory from samples_<tag> output paths."""
    default_dir = Path("data") / "results"
    if samples_dir is None:
        return default_dir
    if samples_dir.parent == default_dir and samples_dir.name.startswith("samples_"):
        tag = samples_dir.name.removeprefix("samples_")
        if tag:
            return default_dir / tag
    return default_dir


def _write_json_report(path: Path, report: object) -> None:
    """Write a report, creating an explicitly requested parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


# ── Pose Extraction ───────────────────────────────────────────────────


def extract_poses(
    video_path: Path, use_roi_crop: bool = False
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Run MediaPipe BlazePose on a video and return 2D + 3D landmarks.

    Returns:
        landmarks_2d: (T, 33, 3) normalised 2D landmarks (x, y, visibility).
        landmarks_3d: (T, 33, 4) world landmarks (x, y, z, visibility).
        fps: Video frame rate.
        width: Video frame width in pixels.
        height: Video frame height in pixels.

    When use_roi_crop=True, a two-pass athlete-crop strategy is used to improve
    MediaPipe detection when the athlete occupies a small fraction of the frame.
    Landmarks are remapped to full-frame normalised coordinates so downstream
    scale_calibration (which uses image_width/height projections) is unaffected.
    """
    estimator = MediaPipeEstimator(model_complexity=2)
    sequence = estimator.process_video(video_path, roi_crop=use_roi_crop)

    if not sequence.frames:
        raise ValueError(f"No poses detected in {video_path}")

    valid = sum(1 for f in sequence.frames if f.is_valid)

    # Get video resolution for aspect ratio correction
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info(
        f"  Pose estimation: {len(sequence.frames)} frames, "
        f"{valid} valid ({100 * valid / len(sequence.frames):.0f}%), "
        f"resolution={vid_width}x{vid_height}"
    )

    # Always stack 2D normalised landmarks
    landmarks_2d = np.stack([f.landmarks_2d for f in sequence.frames])  # (T, 33, 3)

    # Stack 3D world landmarks; fall back to 2D if no 3D available
    has_3d = all(f.landmarks_3d is not None for f in sequence.frames)
    if has_3d:
        landmarks_3d = np.stack([f.landmarks_3d for f in sequence.frames])  # (T, 33, 4)
    else:
        logger.warning("  No 3D landmarks available — using 2D (depth will be approximate)")
        # Create (T, 33, 4) with z=0
        landmarks_3d = np.zeros(
            (landmarks_2d.shape[0], landmarks_2d.shape[1], 4), dtype=np.float32
        )
        landmarks_3d[:, :, :2] = landmarks_2d[:, :, :2]
        landmarks_3d[:, :, 3] = landmarks_2d[:, :, 2]

    return landmarks_2d, landmarks_3d, sequence.fps, vid_width, vid_height


# [TIGHTENED — Phase 10 stationary validation, Change 4]
# Previously: frame valid if >=4 of 33 landmarks visible (weak proxy).
# Now: frame valid only when ALL 8 key joints (shoulders/hips/knees/ankles,
# consistent with PoseFrame.is_valid) are visible. The 60 % gate is unchanged.
# This is a deliberate tightening; do NOT revert to the 4-of-33 criterion.
_KEY_JOINT_INDICES = np.array([11, 12, 23, 24, 25, 26, 27, 28])  # BlazePose indices


def pose_validity_pct(landmarks_2d: np.ndarray, min_visibility: float = 0.5) -> float:
    """Fraction (%) of frames where all 8 key joints are visible at min_visibility.

    Key joints: both shoulders (11,12), hips (23,24), knees (25,26), ankles (27,28).
    Matches PoseFrame.is_valid so the same frames are flagged valid in both contexts.
    The 60 % admission threshold is unchanged; only the per-frame criterion tightened.

    NOTE: This is the GLOBAL metric (whole clip). For the admission gate, use
    takeoff_window_pose_validity_pct when available — early approach frames where
    the athlete is too far from camera to detect drag the global number down.
    """
    if landmarks_2d.size == 0:
        return 0.0
    key_vis = landmarks_2d[:, _KEY_JOINT_INDICES, 2]  # (T, 8)
    return float(np.mean(np.all(key_vis >= min_visibility, axis=1)) * 100.0)


def takeoff_window_pose_validity_pct(
    landmarks_2d: np.ndarray,
    takeoff_frame: int,
    half_window: int = 30,
    min_visibility: float = 0.5,
) -> float:
    """Key-joint validity rate in a ±half_window frame window around takeoff.

    This is the gate metric for stationary (and handheld) clips. It measures
    pose coverage in the critical region — final approach strides, takeoff, and
    early flight — excluding the early run-up where the athlete is far from camera
    and undetectable. At 30 fps, half_window=30 covers ±1 second (2 s total),
    which spans the last 2-3 strides and the flight phase.

    The plan gate 'pose_validity_pct >= 60' should be read as coverage in this
    window, not over the whole clip. Using the whole-clip rate penalises clips
    where the athlete starts far away, which is irrelevant to takeoff kinematics.
    """
    n_frames = landmarks_2d.shape[0]
    start = max(0, takeoff_frame - half_window)
    end = min(n_frames, takeoff_frame + half_window)
    window = landmarks_2d[start:end]
    if window.shape[0] == 0:
        return 0.0
    key_vis = window[:, _KEY_JOINT_INDICES, 2]  # (W, 8)
    return float(np.mean(np.all(key_vis >= min_visibility, axis=1)) * 100.0)


def segment_spread_ratio(calibration_info: dict | None) -> float | None:
    """Return max/min anatomical segment scale spread from calibration diagnostics."""
    if not calibration_info:
        return None
    per_segment = calibration_info.get("scale_info", {}).get("per_segment", {})
    scales = [
        float(values["scale_mpp"])
        for values in per_segment.values()
        if values.get("scale_mpp") and np.isfinite(values.get("scale_mpp"))
    ]
    if len(scales) < 2:
        return None
    min_scale = min(scales)
    if min_scale <= 0:
        return None
    return float(max(scales) / min_scale)


def _calibration_source(calibration_info: dict | None, capture_mode: str = "handheld") -> str:
    """Return the horizontal-reference source for the quality gate.

    For panned footage, the source is derived from egomotion or scene_homography.
    For stationary footage (asserted via --capture-mode stationary), the fixed
    camera itself is the scene-fixed horizontal reference, so we credit
    "stationary_camera" when egomotion/scene_homography are not active.
    This must be asserted — never inferred from the footage.
    """
    method = (calibration_info or {}).get("method")
    if method in {"egomotion", "scene_homography"}:
        return method
    if capture_mode == "stationary":
        return "stationary_camera"
    return "none"


def _quality_block(
    *,
    pose_pct: float,
    windowed_pose_pct: float | None = None,
    contact_interval_detected: bool,
    calibration_info: dict | None,
    peak_com_height_m: float,
    takeoff_horizontal_mps: float | None,
    takeoff_angle_deg: float | None,
    capture_mode: str = "handheld",
    takeoff_anchor_review_passed: bool = True,
    gates: QualityGateConfig = DEFAULT_QUALITY_GATES,
) -> dict:
    spread = segment_spread_ratio(calibration_info)
    source = _calibration_source(calibration_info, capture_mode)
    anatomical_scale_converged = spread is None or spread <= gates.max_segment_spread_ratio
    kinematics_scale_ok = spread is None or spread <= gates.max_kinematics_segment_spread_ratio

    # Gate on the windowed metric when available (preferred: measures the critical
    # takeoff/flight window, not the whole clip including far-approach frames).
    # Falls back to global metric for short clips or when no takeoff frame is known.
    pose_pct_for_gate = windowed_pose_pct if windowed_pose_pct is not None else pose_pct

    training_failures: list[str] = []
    if pose_pct_for_gate < gates.min_pose_validity_pct:
        training_failures.append("pose_validity_below_threshold")
    if not contact_interval_detected:
        training_failures.append("no_contact_interval")
    if not anatomical_scale_converged:
        training_failures.append("anatomical_segment_spread_high")
    if source == "none":
        training_failures.append("no_scene_fixed_horizontal_source")
    if not takeoff_anchor_review_passed:
        training_failures.append("takeoff_anchor_review_failed")
    if takeoff_horizontal_mps is None or not (
        gates.min_takeoff_horizontal_mps
        <= takeoff_horizontal_mps
        <= gates.max_takeoff_horizontal_mps
    ):
        training_failures.append("takeoff_horizontal_out_of_range")
    if takeoff_angle_deg is None or not (
        gates.min_takeoff_angle_deg <= takeoff_angle_deg <= gates.max_takeoff_angle_deg
    ):
        training_failures.append("takeoff_angle_out_of_range")
    if peak_com_height_m > gates.max_peak_com_m:
        training_failures.append("peak_com_above_guardrail")

    kinematics_failures: list[str] = []
    if pose_pct_for_gate < gates.min_kinematics_pose_validity_pct:
        kinematics_failures.append("pose_validity_below_kinematics_threshold")
    if not contact_interval_detected:
        kinematics_failures.append("no_contact_interval")
    if not kinematics_scale_ok:
        kinematics_failures.append("anatomical_segment_spread_high")

    return {
        "pose_validity_pct": round(float(pose_pct), 2),
        "takeoff_window_pose_validity_pct": (
            round(float(windowed_pose_pct), 2) if windowed_pose_pct is not None else None
        ),
        "contact_interval_detected": bool(contact_interval_detected),
        "takeoff_anchor_review_passed": bool(takeoff_anchor_review_passed),
        "anatomical_scale_converged": bool(anatomical_scale_converged),
        "segment_spread_ratio": round(spread, 3) if spread is not None else None,
        "scene_fixed_horizontal_source": source,
        "training_grade": not training_failures,
        "kinematics_grade": not kinematics_failures,
        "training_grade_failures": training_failures,
        "kinematics_grade_failures": kinematics_failures,
    }


# ── Takeoff anchor validation ──────────────────────────────────────────


def _validate_takeoff_anchor(
    candidate_frame: int,
    peak_com_frame: int,
    com_vel: np.ndarray,
    fps: float,
) -> bool:
    """Return True when candidate_frame is a plausible toe-off, not an approach stride.

    Two physics-based checks:

    (a) Upward launch condition: CoM vertical velocity must be positive at the
        candidate frame and peak CoM must come after it.  An approach-stride
        contact ends before the athlete is airborne upward.

    (b) Flight-time plausibility: from projectile physics, the elapsed time from
        toe-off to peak CoM is at most t_apex = v_y / g (time to apex).  With a
        2× safety margin the maximum allowable frame lead is:
            max_lead_frames = ceil(2 * v_y / g * fps)
        A contact that precedes peak CoM by more than this is an approach stride,
        not takeoff — the selected frame corresponds to toe-off of an earlier step
        rather than the final ground contact.  The margin accounts for noise in
        the velocity estimate.

    Physiological floor on v_y for the lead calculation: 2.0 m/s.  For
    sub-threshold v_y (noisy near-zero) the check correctly rejects the frame
    via condition (a).
    """
    g = 9.81
    if candidate_frame >= len(com_vel):
        return False
    vy = float(com_vel[candidate_frame, 1])
    # (a) upward at toe-off and peak is later
    if vy <= 0.0 or peak_com_frame <= candidate_frame:
        return False
    # (b) flight-time lead: use physiological floor so noise cannot shrink the window
    vy_for_lead = min(max(vy, 2.0), 5.5)
    t_apex_s = vy_for_lead / g
    max_lead_frames = int(np.ceil(2.0 * t_apex_s * fps))
    return (peak_com_frame - candidate_frame) <= max_lead_frames


# ── Kinematics ─────────────────────────────────────────────────────────


def compute_kinematics(
    landmarks_3d: np.ndarray,
    fps: float,
    body_mass_kg: float,
) -> dict:
    """Compute joint angles, CoM, and estimated GRF from 3D landmarks.

    Returns a dict with all computed arrays ready for BiomechanicalSample.
    """
    n_frames = landmarks_3d.shape[0]
    logger.info(f"  Computing kinematics for {n_frames} frames at {fps:.1f} fps")

    # ── Joint angles (degrees) ──
    joint_angle_dict = compute_joint_angles_sequence(landmarks_3d)
    joint_names = list(joint_angle_dict.keys())
    joint_angles_deg = np.column_stack([joint_angle_dict[k] for k in joint_names])
    joint_angles_rad = np.deg2rad(joint_angles_deg)

    # ── Joint angular velocities (rad/s) ──
    dt = 1.0 / fps
    joint_angular_velocities = np.gradient(joint_angles_rad, dt, axis=0)

    # ── CoM trajectory ──
    com_result = compute_com_trajectory(landmarks_3d, fps)
    com_pos = com_result["position"]      # (T, 3)
    com_vel = com_result["velocity"]      # (T, 3)
    com_acc = com_result["acceleration"]  # (T, 3)

    # ── Estimated GRF from Newton's 2nd law: F = m*(a + g) ──
    estimated_grf = estimate_grf_from_com(com_acc, body_mass_kg)

    # ── Horizontal speed profile ──
    horizontal_speed = compute_horizontal_velocity(com_pos, fps)

    return {
        "joint_angles_rad": joint_angles_rad,
        "joint_angular_velocities": joint_angular_velocities,
        "joint_names": joint_names,
        "com_position": com_pos,
        "com_velocity": com_vel,
        "com_acceleration": com_acc,
        "estimated_grf": estimated_grf,
        "horizontal_speed": horizontal_speed,
    }


# ── Build BiomechanicalSample ─────────────────────────────────────────


def build_sample(
    video_path: Path,
    landmarks_3d: np.ndarray,
    kinematics: dict,
    fps: float,
    body_mass_kg: float,
    height_m: float,
    opensim_joint_angles: np.ndarray | None = None,
    opensim_joint_names: list[str] | None = None,
) -> BiomechanicalSample:
    """Package everything into the canonical data format."""
    subject = SubjectInfo(
        subject_id=video_path.stem,
        body_mass_kg=body_mass_kg,
        height_m=height_m,
    )

    # Prefer OpenSim IK joint angles when available
    if opensim_joint_angles is not None:
        joint_angles = opensim_joint_angles
        joint_names = opensim_joint_names or []
    else:
        joint_angles = kinematics["joint_angles_rad"]
        joint_names = kinematics["joint_names"]

    return BiomechanicalSample(
        dataset_name="personal_video",
        trial_id=video_path.stem,
        subject=subject,
        movement_type=MovementType.HIGH_JUMP,
        fps=fps,
        joint_angles=joint_angles,
        joint_angular_velocities=kinematics["joint_angular_velocities"],
        joint_names=joint_names,
        com_position=kinematics["com_position"],
        com_velocity=kinematics["com_velocity"],
        com_acceleration=kinematics["com_acceleration"],
        grf=kinematics["estimated_grf"],
        pose_3d=landmarks_3d[:, :, :3],
    )


# ── PINN Inference (optional) ─────────────────────────────────────────


def run_pinn_inference(
    sample: BiomechanicalSample,
    model_path: Path,
) -> np.ndarray | None:
    """Load pre-trained PINN and estimate refined GRF.

    Returns refined GRF array (T, 3) or None if model not available.
    """
    if not model_path.exists():
        logger.info(f"  No pre-trained model at {model_path} — using Newton's law GRF only")
        return None

    try:
        import torch
        from src.pinn.physics.inverse_dynamics import InverseDynamicsPINN

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        # Reconstruct the same input format as DynamicsDataset._add_com_windows
        t = np.linspace(0, 1, sample.n_frames, dtype=np.float32)
        input_data = np.column_stack([
            t,
            sample.com_position.astype(np.float32),
            sample.com_velocity.astype(np.float32),
        ])  # (T, 7)

        model = InverseDynamicsPINN(
            input_dim=input_data.shape[1],
            output_dim=6,
            hidden_dim=config.get("hidden_dim", 128),
            n_layers=config.get("n_layers", 5),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            x = torch.from_numpy(input_data)
            pred = model(x).numpy()  # (T, 6): GRF_xyz + torques

        # pred[:, :3] is per-kg GRF (mass-normalized during training)
        pinn_grf = pred[:, :3] * sample.subject.body_mass_kg
        logger.info(
            f"  PINN GRF: peak vertical = {pinn_grf[:, 1].max():.0f} N "
            f"({pinn_grf[:, 1].max() / (sample.subject.body_mass_kg * 9.81):.2f} BW)"
        )
        return pinn_grf.astype(np.float32)

    except Exception as e:
        logger.warning(f"  PINN inference failed: {e}")
        return None


# ── Summary Report ─────────────────────────────────────────────────────


def select_takeoff_frame_details(
    sample: BiomechanicalSample,
    fallback_frame: int,
) -> tuple[int, bool, int, bool]:
    """Select toe-off frame and validate that it is a plausible toe-off.

    Returns:
        (frame, contact_interval_detected, contact_count, takeoff_anchor_review_passed)

    contact_interval_detected: True when at least one ankle-ground contact was
        found.  The argmax(vy) fallback is used when False.
    takeoff_anchor_review_passed: True only when the selected frame passes
        _validate_takeoff_anchor — i.e. vy > 0, peak CoM follows, and the
        frame is not too early to be an approach stride.  Always False when no
        contact interval was detected (argmax fallback is not admission-safe).

    `detect_ground_contacts` works in centimetres; video-pipeline pose landmarks
    are calibrated in metres, so ankle trajectories are converted before use.
    """
    if sample.pose_3d is None or sample.pose_3d.ndim != 3 or sample.pose_3d.shape[1] <= 28:
        return fallback_frame, False, 0, False

    contacts: list[tuple[int, int]] = []
    for ankle_idx in (27, 28):
        ankle_m = sample.pose_3d[:, ankle_idx, :3]
        if ankle_m.shape[0] == 0 or not np.isfinite(ankle_m[:, 1]).any():
            continue
        contacts.extend(detect_ground_contacts(ankle_m * 100.0, sample.fps))

    if not contacts:
        return fallback_frame, False, 0, False

    peak_com_frame = fallback_frame
    if sample.com_position is not None and len(sample.com_position) > 0:
        peak_com_frame = int(np.argmax(sample.com_position[:, 1]))
        pre_peak_contacts = [contact for contact in contacts if contact[1] <= peak_com_frame]
        if pre_peak_contacts:
            contacts = pre_peak_contacts

    contacts.sort(key=lambda contact: (contact[1], contact[0]))
    candidate_frame = int(contacts[-1][1])

    anchor_review_passed = False
    if sample.com_velocity is not None and len(sample.com_velocity) > 0:
        anchor_review_passed = _validate_takeoff_anchor(
            candidate_frame, peak_com_frame, sample.com_velocity, sample.fps
        )

    return candidate_frame, True, len(contacts), anchor_review_passed


def select_takeoff_frame_from_ground_contact(
    sample: BiomechanicalSample,
    fallback_frame: int,
) -> int:
    """Backward-compatible takeoff-frame selector."""
    frame, _detected, _count, _anchor = select_takeoff_frame_details(sample, fallback_frame)
    return frame


def generate_report(
    sample: BiomechanicalSample,
    pinn_grf: np.ndarray | None,
    bar_height_m: float | None = None,
    calibration_info: dict | None = None,
    pose_validity_pct_value: float | None = None,
    windowed_pose_validity_pct_value: float | None = None,
    quality_gates: QualityGateConfig = DEFAULT_QUALITY_GATES,
    capture_mode: str = "handheld",
) -> dict:
    """Generate a summary report of the jump analysis."""
    mass = sample.subject.body_mass_kg
    g = 9.81

    # CoM metrics
    com_pos = sample.com_position
    com_vel = sample.com_velocity
    peak_com_height = float(com_pos[:, 1].max())
    com_height_at_start = float(com_pos[0, 1])
    com_rise = peak_com_height - com_height_at_start

    # Velocity metrics
    horizontal_speed = np.sqrt(com_vel[:, 0] ** 2 + com_vel[:, 2] ** 2)
    peak_horizontal = float(horizontal_speed.max())

    # GRF metrics (from Newton's law estimation)
    grf = sample.grf
    peak_vertical_grf = float(grf[:, 1].max())
    peak_grf_bw = peak_vertical_grf / (mass * g)

    # Takeoff angle estimation.
    # Takeoff is the instant the takeoff foot leaves the ground.  Anchor frame
    # selection to detected ankle-ground contact rather than a single-frame
    # velocity maximum, which is sensitive to finite-difference jitter.
    vy = com_vel[:, 1]
    if len(vy) >= 2:
        fallback_takeoff_frame = int(np.argmax(vy))
        (
            takeoff_frame,
            contact_interval_detected,
            contact_interval_count,
            takeoff_anchor_review_passed,
        ) = select_takeoff_frame_details(
            sample,
            fallback_frame=fallback_takeoff_frame,
        )
        takeoff_vel = com_vel[takeoff_frame]
        takeoff_horiz = float(np.sqrt(takeoff_vel[0] ** 2 + takeoff_vel[2] ** 2))
        takeoff_vert = float(takeoff_vel[1])
        takeoff_angle = float(np.degrees(np.arctan2(takeoff_vert, takeoff_horiz)))
    else:
        takeoff_frame = 0
        contact_interval_detected = False
        contact_interval_count = 0
        takeoff_anchor_review_passed = False
        takeoff_angle = None
        takeoff_horiz = None
        takeoff_vert = None

    quality = _quality_block(
        pose_pct=pose_validity_pct_value if pose_validity_pct_value is not None else 0.0,
        windowed_pose_pct=windowed_pose_validity_pct_value,
        contact_interval_detected=contact_interval_detected,
        calibration_info=calibration_info,
        peak_com_height_m=peak_com_height,
        takeoff_horizontal_mps=takeoff_horiz,
        takeoff_angle_deg=takeoff_angle,
        capture_mode=capture_mode,
        takeoff_anchor_review_passed=takeoff_anchor_review_passed,
        gates=quality_gates,
    )

    # PINN-refined metrics
    pinn_metrics = {}
    if pinn_grf is not None:
        pinn_peak_grf = float(pinn_grf[:, 1].max())
        pinn_metrics = {
            "pinn_peak_vertical_grf_N": round(pinn_peak_grf, 1),
            "pinn_peak_vertical_grf_BW": round(pinn_peak_grf / (mass * g), 2),
        }

    report = {
        "video": sample.trial_id,
        "subject": {
            "mass_kg": mass,
            "height_m": sample.subject.height_m,
        },
        "frames": sample.n_frames,
        "duration_s": round(sample.duration_s, 2),
        "fps": sample.fps,
        "com": {
            "peak_height_m": round(peak_com_height, 3),
            "rise_m": round(com_rise, 3),
        },
        "velocity": {
            "peak_horizontal_mps": round(peak_horizontal, 2),
            "takeoff_horizontal_mps": round(takeoff_horiz, 2) if takeoff_horiz else None,
            "takeoff_vertical_mps": round(takeoff_vert, 2) if takeoff_vert else None,
            "takeoff_angle_deg": round(takeoff_angle, 1) if takeoff_angle else None,
        },
        "grf_newton_law": {
            "peak_vertical_N": round(peak_vertical_grf, 1),
            "peak_vertical_BW": round(peak_grf_bw, 2),
        },
        "calibration": calibration_info or {
            "method": "anatomical",
            "anchor_coverage_pct": 0.0,
            "anchor_valid_counts_histogram": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},
            "anchor_four_point_valid_pct": 0.0,
            "crossbar_confirmed_pct": 0.0,
            "scene_anatomical_scale_ratio": None,
            "egomotion_coverage_pct": 0.0,
            "egomotion_median_confidence": None,
        },
        "quality": quality,
        "takeoff_frame": int(takeoff_frame),
        "takeoff_contact_intervals_detected": int(contact_interval_count),
        **pinn_metrics,
    }

    # Bar-height-relative metrics (when bar height is known)
    if bar_height_m is not None:
        report["bar_height_m"] = bar_height_m
        # CoM clearance: how far below/above bar the peak CoM reached
        # For Fosbury Flop, CoM typically passes 5–20 cm below the bar
        com_bar_diff = peak_com_height - bar_height_m
        report["com"]["com_minus_bar_m"] = round(com_bar_diff, 3)
        report["com"]["com_below_bar"] = com_bar_diff < 0
        logger.info(
            f"  Bar at {bar_height_m:.2f}m, peak CoM at {peak_com_height:.2f}m "
            f"({'below' if com_bar_diff < 0 else 'above'} bar by "
            f"{abs(com_bar_diff):.2f}m)"
        )

    return report


# ── Main ───────────────────────────────────────────────────────────────


def analyze_video(
    video_path: Path,
    body_mass_kg: float = 65.0,
    height_m: float = 1.75,
    thigh_length_m: float | None = None,
    shank_length_m: float | None = None,
    model_path: Path | None = None,
    samples_out_dir: Path | None = None,
    use_scene_anchor: bool = False,
    use_egomotion: bool = False,
    scene_detection_stride: int = 15,
    scene_max_detection_width: int = 640,
    bar_height_m: float | None = None,
    capture_mode: str = "handheld",
    use_roi_crop: bool = False,
) -> dict:
    """Run the full analysis pipeline on a single video.

    If `samples_out_dir` is given, the extracted `BiomechanicalSample` is
    cached there as `<video_stem>.npz` for downstream fine-tuning
    (see scripts/finetune_personal.py).

    `thigh_length_m` and `shank_length_m`, when provided, switch scale
    calibration from the legacy single-frame nose-ankle estimator to the
    Phase 9a multi-segment per-frame estimator (much more accurate).
    """
    logger.info(f"=== Analyzing: {video_path.name} ===")

    # 1. Pose estimation (2D normalised + 3D world)
    landmarks_2d, landmarks_3d_world, fps, vid_w, vid_h = extract_poses(
        video_path, use_roi_crop=use_roi_crop
    )
    pose_pct = pose_validity_pct(landmarks_2d)
    bar_height = resolve_bar_height(video_path.name, bar_height_m)

    # 1a. Scale calibration: prefer multi-segment per-frame (Phase 9a) when
    #     anthropometric segment lengths are supplied; fall back to legacy
    #     single-frame nose-ankle otherwise.
    calibration_info: dict
    if use_scene_anchor or use_egomotion:
        scene_anchors = (
            detect_scene_anchors(
                video_path,
                bar_height_m=bar_height,
                detection_stride=scene_detection_stride,
                max_detection_width=scene_max_detection_width,
            )
            if use_scene_anchor
            else None
        )
        camera_motion = (
            estimate_camera_motion(
                video_path,
                landmarks_2d,
                image_width=vid_w,
                image_height=vid_h,
            )
            if use_egomotion
            else None
        )
        if camera_motion is not None:
            inliers = camera_motion.inlier_counts
            features = camera_motion.feature_counts
            confidence = camera_motion.confidence
            low_inlier_pct = (
                float(np.mean(inliers < camera_motion.min_inlier_count) * 100.0)
                if len(inliers)
                else 0.0
            )
            logger.info(
                "  Egomotion: feature median=%s, inlier median=%s, "
                "confidence median=%s, low-inlier frames=%.1f%%, "
                "frames without mask=%s",
                float(np.nanmedian(features)) if len(features) else None,
                float(np.nanmedian(inliers)) if len(inliers) else None,
                float(np.nanmedian(confidence)) if len(confidence) else None,
                low_inlier_pct,
                camera_motion.frames_without_mask,
            )
        landmarks_3d, calibration_info = calibrate_landmarks_with_scene(
            landmarks_2d,
            landmarks_3d_world,
            height_m,
            image_width=vid_w,
            image_height=vid_h,
            thigh_length_m=thigh_length_m,
            shank_length_m=shank_length_m,
            scene_anchors=scene_anchors,
            camera_motion=camera_motion,
        )
        logger.info(
            "  Calibration: method=%s, egomotion coverage=%.1f%%, "
            "egomotion median confidence=%s, scene anchor coverage=%.1f%%, "
            "scene/anatomical scale=%s, anchor 4-point frames=%.1f%%, "
            "crossbar confirmed=%.1f%%",
            calibration_info.get("method"),
            calibration_info.get("egomotion_coverage_pct", 0.0),
            calibration_info.get("egomotion_median_confidence"),
            calibration_info.get("anchor_coverage_pct", 0.0),
            calibration_info.get("scene_anatomical_scale_ratio"),
            calibration_info.get("anchor_four_point_valid_pct", 0.0),
            calibration_info.get("crossbar_confirmed_pct", 0.0),
        )
    else:
        landmarks_3d = calibrate_landmarks_to_world(
            landmarks_2d, landmarks_3d_world, height_m,
            image_width=vid_w, image_height=vid_h,
            thigh_length_m=thigh_length_m,
            shank_length_m=shank_length_m,
        )
        _mpp, scale_info = compute_per_frame_scale_mpp(
            landmarks_2d,
            image_width=vid_w,
            image_height=vid_h,
            thigh_length_m=thigh_length_m,
            shank_length_m=shank_length_m,
        )
        calibration_info = {
            "method": "anatomical",
            "capture_mode": capture_mode,
            "anchor_coverage_pct": 0.0,
            "anchor_valid_counts_histogram": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},
            "anchor_four_point_valid_pct": 0.0,
            "crossbar_confirmed_pct": 0.0,
            "scene_anatomical_scale_ratio": None,
            "scene_anchor_decision_reason": "unavailable",
            "egomotion_coverage_pct": 0.0,
            "egomotion_median_confidence": None,
            "egomotion_p10_confidence": None,
            "egomotion_min_inliers": None,
            "egomotion_frames_without_mask": 0,
            "egomotion_decision_reason": "unavailable",
            "scale_info": scale_info,
        }
        # Record the horizontal source explicitly when stationary camera is asserted
        if capture_mode == "stationary":
            calibration_info["scene_fixed_horizontal_source"] = "stationary_camera"
    logger.info(
        f"  Scale calibration applied: athlete height = {height_m:.2f}m, "
        f"CoM Y range = {landmarks_3d[:, :, 1].min():.2f}–"
        f"{landmarks_3d[:, :, 1].max():.2f} m"
    )

    # 1b. Post-process landmarks (gap fill → Butterworth filter → segment length)
    pp_config = PostProcessorConfig(
        do_gap_fill=True,
        do_filter=True,
        filter_cutoff_hz=10.0,      # 10 Hz — appropriate for jumping movements
        do_segment_enforce=True,
        height_m=height_m,
        segment_enforce_weight=0.8,
    )
    landmarks_3d = postprocess_landmarks(landmarks_3d, fps, pp_config)
    logger.info(
        f"  Post-processing: gap-fill + {pp_config.filter_cutoff_hz:.0f} Hz "
        f"Butterworth + segment enforcement (h={height_m:.2f}m)"
    )

    # 1c. OpenSim IK (optional — requires conda opensim_ik env + model file)
    opensim_joint_angles = None
    opensim_joint_names = None
    if is_opensim_available():
        try:
            opensim_joint_angles, opensim_joint_names = run_opensim_ik(
                landmarks_3d, fps, height_m=height_m, mass_kg=body_mass_kg,
            )
            logger.info(
                f"  OpenSim IK: {opensim_joint_angles.shape[1]} coordinates, "
                f"{opensim_joint_angles.shape[0]} frames"
            )
        except Exception as e:
            logger.warning(f"  OpenSim IK failed, using geometric angles: {e}")
    else:
        logger.info("  OpenSim IK: not available (using geometric joint angles)")

    # 2. Kinematics
    kinematics = compute_kinematics(landmarks_3d, fps, body_mass_kg)

    # 3. Build BiomechanicalSample
    sample = build_sample(
        video_path, landmarks_3d, kinematics, fps, body_mass_kg, height_m,
        opensim_joint_angles=opensim_joint_angles,
        opensim_joint_names=opensim_joint_names,
    )
    logger.info(
        f"  BiomechanicalSample: {sample.n_frames} frames, "
        f"duration={sample.duration_s:.2f}s, "
        f"movement={sample.movement_type.value}"
    )

    # 4. PINN-refined GRF (optional)
    pinn_grf = None
    if model_path:
        pinn_grf = run_pinn_inference(sample, model_path)

    # 4a. Windowed pose validity: measure coverage only in the critical takeoff
    # window (±30 frames).  This requires the takeoff frame, so compute it here
    # using the same logic that generate_report uses internally.
    windowed_pose_pct: float | None = None
    if sample.com_velocity is not None and len(sample.com_velocity) >= 2:
        _fb = int(np.argmax(sample.com_velocity[:, 1]))
        _tf, _, _, _ = select_takeoff_frame_details(sample, _fb)
        windowed_pose_pct = takeoff_window_pose_validity_pct(landmarks_2d, _tf)

    # 5. Generate report
    report = generate_report(
        sample,
        pinn_grf,
        bar_height_m=bar_height,
        calibration_info=calibration_info,
        pose_validity_pct_value=pose_pct,
        windowed_pose_validity_pct_value=windowed_pose_pct,
        capture_mode=capture_mode,
    )

    # 6. Add session metadata
    session_date = parse_session_date(video_path)
    if session_date is not None:
        report["session_date"] = session_date
    report["session_folder"] = video_path.parent.name

    # 7. Cache the sample for personal-data fine-tuning, if requested
    if samples_out_dir is not None:
        samples_out_dir.mkdir(parents=True, exist_ok=True)
        sample_path = samples_out_dir / f"{video_path.stem}.npz"
        sample.save_npz(sample_path)
        report["sample_npz"] = str(sample_path)

    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Analyse high jump video(s) end-to-end"
    )
    parser.add_argument(
        "input", type=str,
        help="Path to a video file or directory of videos",
    )
    parser.add_argument("--mass", type=float, default=65.0, help="Body mass in kg")
    parser.add_argument("--height", type=float, default=1.75, help="Height in metres")
    parser.add_argument(
        "--bar-height", type=float, default=None,
        help="Bar height in metres. Overrides filename-derived metadata.",
    )
    parser.add_argument(
        "--thigh", type=float, default=0.45,
        help="Measured hip→knee length in metres (Athlete A: 0.45). "
             "Enables Phase 9a multi-segment scale calibration. "
             "Set to 0 to disable and fall back to nose-ankle.",
    )
    parser.add_argument(
        "--shank", type=float, default=0.45,
        help="Measured knee→ankle length in metres (Athlete A: 0.45). "
             "Enables Phase 9a multi-segment scale calibration. "
             "Set to 0 to disable and fall back to nose-ankle.",
    )
    parser.add_argument(
        "--model", type=str,
        default="experiments/results/pretrain_dynamics/best_model.pth",
        help="Path to pre-trained PINN checkpoint",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: data/results/<video_stem>_report.json)",
    )
    parser.add_argument(
        "--save-samples", type=str, default=None,
        nargs="?", const="data/results/samples",
        help="Cache extracted BiomechanicalSamples as .npz under the given "
             "directory (defaults to data/results/samples when flag is bare). "
             "Required input for scripts/finetune_personal.py.",
    )
    parser.add_argument(
        "--scene-anchor", choices=("on", "off"), default="off",
        help="Opt-in Phase 9c crossbar/upright scene calibration. Default off "
             "keeps the approved Phase 9a anatomical scale path.",
    )
    parser.add_argument(
        "--egomotion", choices=("on", "off"), default="off",
        help="Opt-in camera egomotion compensation for panned phone footage.",
    )
    parser.add_argument(
        "--scene-detection-stride", type=int, default=15,
        help="Run scene-anchor Hough detection every N frames before optical-flow fill.",
    )
    parser.add_argument(
        "--scene-max-detection-width", type=int, default=640,
        help="Resize width for scene-anchor detection. Lower is faster; higher is sharper.",
    )
    parser.add_argument(
        "--capture-mode", choices=("handheld", "stationary"), default="handheld",
        help=(
            "Asserted capture mode. 'stationary' credits the fixed camera as a "
            "scene-fixed horizontal source (no_scene_fixed_horizontal_source gate "
            "is lifted). Must be asserted by the operator — never inferred."
        ),
    )
    parser.add_argument(
        "--roi-crop", choices=("on", "off"), default="off",
        help=(
            "Opt-in two-pass athlete ROI crop. Pass 1 detects on the full frame to "
            "locate the athlete; pass 2 re-detects on the cropped region and remaps "
            "landmarks to full-frame normalised coordinates. Use when the athlete "
            "occupies a small fraction of the frame (e.g. far-back stationary camera)."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    samples_dir = Path(args.save_samples) if args.save_samples else None
    thigh_m = args.thigh if args.thigh and args.thigh > 0 else None
    shank_m = args.shank if args.shank and args.shank > 0 else None

    video_files = collect_videos(input_path)
    if not video_files:
        logger.error(f"No video files found at {input_path}")
        sys.exit(1)
    logger.info(f"Found {len(video_files)} video(s) under {input_path}")

    all_reports = []
    for video_path in video_files:
        try:
            report = analyze_video(
                video_path,
                body_mass_kg=args.mass,
                height_m=args.height,
                thigh_length_m=thigh_m,
                shank_length_m=shank_m,
                model_path=model_path,
                samples_out_dir=samples_dir,
                use_scene_anchor=args.scene_anchor == "on",
                use_egomotion=args.egomotion == "on",
                scene_detection_stride=args.scene_detection_stride,
                scene_max_detection_width=args.scene_max_detection_width,
                bar_height_m=args.bar_height,
                capture_mode=args.capture_mode,
                use_roi_crop=args.roi_crop == "on",
            )
            all_reports.append(report)

            # Print summary to console
            print(f"\n{'=' * 60}")
            print(f"  Video: {report['video']}")
            if report.get("session_date"):
                print(f"  Session: {report['session_date']}")
            if report.get("bar_height_m"):
                print(f"  Bar height: {report['bar_height_m']} m")
            print(f"  Duration: {report['duration_s']}s ({report['frames']} frames)")
            print(f"  CoM peak height: {report['com']['peak_height_m']:.3f} m")
            print(f"  CoM rise: {report['com']['rise_m']:.3f} m")
            if report['velocity']['takeoff_angle_deg']:
                print(f"  Takeoff angle: {report['velocity']['takeoff_angle_deg']}°")
                print(f"  Takeoff H-vel: {report['velocity']['takeoff_horizontal_mps']} m/s")
                print(f"  Takeoff V-vel: {report['velocity']['takeoff_vertical_mps']} m/s")
            print(f"  Peak GRF (Newton): {report['grf_newton_law']['peak_vertical_BW']} BW")
            if "pinn_peak_vertical_grf_BW" in report:
                print(f"  Peak GRF (PINN):   {report['pinn_peak_vertical_grf_BW']} BW")
            print(f"{'=' * 60}")

        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            continue

    # Save reports
    if all_reports:
        if args.output:
            out_path = Path(args.output)
        else:
            out_dir = infer_default_report_dir(samples_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if len(all_reports) == 1:
                out_path = out_dir / f"{all_reports[0]['video']}_report.json"
            else:
                out_path = out_dir / "all_sessions_report.json"

        _write_json_report(out_path, all_reports if len(all_reports) > 1 else all_reports[0])
        logger.info(f"Report saved: {out_path}")

        # Also save per-session reports
        if len(all_reports) > 1:
            out_dir = infer_default_report_dir(samples_dir)
            sessions: dict[str, list] = {}
            for r in all_reports:
                folder = r.get("session_folder", "unknown")
                sessions.setdefault(folder, []).append(r)
            for session_name, reports in sessions.items():
                session_path = out_dir / f"{session_name}_report.json"
                _write_json_report(session_path, reports)
            logger.info(f"  Per-session reports saved to {out_dir}")

        # Print summary table
        if len(all_reports) > 1:
            print(f"\n{'=' * 80}")
            print(f"  BATCH SUMMARY: {len(all_reports)} videos processed")
            print(f"{'=' * 80}")
            print(f"  {'Video':<35} {'Bar':>5} {'CoM rise':>9} {'T.Angle':>8} {'PINN GRF':>9}")
            print(f"  {'-'*35} {'-'*5} {'-'*9} {'-'*8} {'-'*9}")
            for r in all_reports:
                name = r['video'][:35]
                bar = f"{r.get('bar_height_m', '')}"
                rise = f"{r['com']['rise_m']:.3f}m"
                angle = (
                    f"{r['velocity']['takeoff_angle_deg']}°"
                    if r['velocity']['takeoff_angle_deg']
                    else "n/a"
                )
                pinn = (
                    f"{r.get('pinn_peak_vertical_grf_BW', 'n/a')} BW"
                    if 'pinn_peak_vertical_grf_BW' in r
                    else "n/a"
                )
                print(f"  {name:<35} {bar:>5} {rise:>9} {angle:>8} {pinn:>9}")
            print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
