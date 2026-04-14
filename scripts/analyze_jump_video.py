"""Analyse a high jump video end-to-end.

Full pipeline: video → pose → joint angles → CoM → phase metrics
              → PINN GRF estimation → actionable feedback.

Usage:
    python scripts/analyze_jump_video.py path/to/jump.mp4
    python scripts/analyze_jump_video.py path/to/jump.mp4 --mass 67 --height 1.75
    python scripts/analyze_jump_video.py path/to/jump.mp4 --model experiments/results/pretrain_dynamics/best_model.pth
    python scripts/analyze_jump_video.py data/videos/raw/  # process all videos in a folder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.sample import BiomechanicalSample, MovementType, SubjectInfo
from src.pose_estimation.estimators.mediapipe_estimator import MediaPipeEstimator
from src.pose_estimation.skeleton.com_estimation import compute_com_trajectory
from src.pose_estimation.skeleton.joint_angles import compute_joint_angles_sequence
from src.kinematics.run_up_analysis import (
    compute_horizontal_velocity,
    detect_ground_contacts,
)
from src.kinematics.takeoff_analysis import (
    compute_takeoff_angle,
    estimate_grf_from_com,
)
from src.pose_estimation.skeleton.landmark_postprocessor import (
    PostProcessorConfig,
    postprocess_landmarks,
)
from src.pose_estimation.opensim_ik import is_opensim_available, run_opensim_ik

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_bar_height(filename: str) -> float | None:
    """Extract bar height from filename like '06_03_24_one_1.75.mp4' → 1.75."""
    import re
    # Match decimal like _1.75 or _1.88 before the extension
    m = re.search(r'_(\d+\.\d+)(?:\.[a-zA-Z]+)?$', filename)
    return float(m.group(1)) if m else None


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


# ── Pose Extraction ───────────────────────────────────────────────────


def extract_poses(video_path: Path) -> tuple[np.ndarray, float]:
    """Run MediaPipe BlazePose on a video and return 3D landmarks.

    Returns:
        landmarks_3d: (T, 33, 4) array (x, y, z, visibility).
        fps: Video frame rate.
    """
    estimator = MediaPipeEstimator(model_complexity=2)
    sequence = estimator.process_video(video_path)

    if not sequence.frames:
        raise ValueError(f"No poses detected in {video_path}")

    valid = sum(1 for f in sequence.frames if f.is_valid)
    logger.info(
        f"  Pose estimation: {len(sequence.frames)} frames, "
        f"{valid} valid ({100 * valid / len(sequence.frames):.0f}%)"
    )

    # Stack 3D world landmarks; fall back to 2D if no 3D available
    has_3d = all(f.landmarks_3d is not None for f in sequence.frames)
    if has_3d:
        landmarks = np.stack([f.landmarks_3d for f in sequence.frames])  # (T, 33, 4)
    else:
        logger.warning("  No 3D landmarks available — using 2D (depth will be approximate)")
        landmarks = np.stack([f.landmarks_2d for f in sequence.frames])  # (T, 33, 3)

    return landmarks, sequence.fps


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

    # ── Ground contacts (from ankle height) ──
    # Use left ankle (27) and right ankle (28) — take minimum y
    left_ankle_y = landmarks_3d[:, 27, 1]
    right_ankle_y = landmarks_3d[:, 28, 1]
    min_ankle_y = np.minimum(left_ankle_y, right_ankle_y)

    # MediaPipe world landmarks have y pointing down, so contacts when y is large
    # Actually, coordinates depend on the model. Just report what we have.
    ankle_positions = np.column_stack([
        np.zeros(n_frames), min_ankle_y, np.zeros(n_frames)
    ])

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
        dt = 1.0 / sample.fps if sample.fps > 0 else 1.0 / 30.0
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


def generate_report(sample: BiomechanicalSample, pinn_grf: np.ndarray | None) -> dict:
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
    vertical_at_peak_horizontal = float(com_vel[np.argmax(horizontal_speed), 1])

    # GRF metrics (from Newton's law estimation)
    grf = sample.grf
    peak_vertical_grf = float(grf[:, 1].max())
    peak_grf_bw = peak_vertical_grf / (mass * g)

    # Takeoff angle estimation
    # Find approximate takeoff: last frame where vertical velocity goes positive
    vy = com_vel[:, 1]
    takeoff_candidates = np.where(np.diff(np.sign(vy)) > 0)[0]
    if len(takeoff_candidates) > 0:
        takeoff_frame = takeoff_candidates[-1]
        takeoff_vel = com_vel[takeoff_frame]
        takeoff_angle = float(np.degrees(np.arctan2(
            takeoff_vel[1],
            np.sqrt(takeoff_vel[0] ** 2 + takeoff_vel[2] ** 2)
        )))
        takeoff_horiz = float(np.sqrt(takeoff_vel[0] ** 2 + takeoff_vel[2] ** 2))
        takeoff_vert = float(takeoff_vel[1])
    else:
        takeoff_frame = len(vy) // 2
        takeoff_angle = None
        takeoff_horiz = None
        takeoff_vert = None

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
        "takeoff_frame": int(takeoff_frame),
        **pinn_metrics,
    }

    return report


# ── Main ───────────────────────────────────────────────────────────────


def analyze_video(
    video_path: Path,
    body_mass_kg: float = 65.0,
    height_m: float = 1.75,
    model_path: Path | None = None,
) -> dict:
    """Run the full analysis pipeline on a single video."""
    logger.info(f"=== Analyzing: {video_path.name} ===")

    # 1. Pose estimation
    landmarks_3d, fps = extract_poses(video_path)

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

    # 5. Generate report
    report = generate_report(sample, pinn_grf)

    # 6. Add session metadata
    bar_height = parse_bar_height(video_path.name)
    session_date = parse_session_date(video_path)
    if bar_height is not None:
        report["bar_height_m"] = bar_height
    if session_date is not None:
        report["session_date"] = session_date
    report["session_folder"] = video_path.parent.name

    return report


def main():
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
        "--model", type=str,
        default="experiments/results/pretrain_dynamics/best_model.pth",
        help="Path to pre-trained PINN checkpoint",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: data/results/<video_stem>_report.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)

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
                model_path=model_path,
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
            out_dir = Path("data") / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            if len(all_reports) == 1:
                out_path = out_dir / f"{all_reports[0]['video']}_report.json"
            else:
                out_path = out_dir / "all_sessions_report.json"

        with open(out_path, "w") as f:
            json.dump(all_reports if len(all_reports) > 1 else all_reports[0], f, indent=2)
        logger.info(f"Report saved: {out_path}")

        # Also save per-session reports
        if len(all_reports) > 1:
            out_dir = Path("data") / "results"
            sessions: dict[str, list] = {}
            for r in all_reports:
                folder = r.get("session_folder", "unknown")
                sessions.setdefault(folder, []).append(r)
            for session_name, reports in sessions.items():
                session_path = out_dir / f"{session_name}_report.json"
                with open(session_path, "w") as f:
                    json.dump(reports, f, indent=2)
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
                angle = f"{r['velocity']['takeoff_angle_deg']}°" if r['velocity']['takeoff_angle_deg'] else "n/a"
                pinn = f"{r.get('pinn_peak_vertical_grf_BW', 'n/a')} BW" if 'pinn_peak_vertical_grf_BW' in r else "n/a"
                print(f"  {name:<35} {bar:>5} {rise:>9} {angle:>8} {pinn:>9}")
            print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
