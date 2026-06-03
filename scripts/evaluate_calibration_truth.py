"""Compare calibration modes against hand-labelled scene anchors.

This script reads private labels from ``data/results/hand_anchors/`` and prints
aggregate velocity-error diagnostics. It is intentionally local-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_jump_video import parse_bar_height  # noqa: E402
from src.pose_estimation.egomotion import estimate_camera_motion  # noqa: E402
from src.pose_estimation.estimators.mediapipe_estimator import MediaPipeEstimator  # noqa: E402
from src.pose_estimation.gravity_calibration import (  # noqa: E402
    calibrate_landmarks_with_gravity_mpp,
)
from src.pose_estimation.scale_calibration import (  # noqa: E402
    calibrate_landmarks_to_world,
    calibrate_landmarks_with_scene,
)
from src.pose_estimation.scene_calibration import (  # noqa: E402
    SceneAnchors,
    detect_scene_anchors,
    fit_per_frame_homography,
    homography_valid_mask,
    warp_landmarks_to_scene,
)
from src.pose_estimation.skeleton.com_estimation import compute_com_trajectory  # noqa: E402


def _extract_poses_with_source_indices(video_path: Path):
    """Run MediaPipe and preserve each frame's source-video index.

    Returns ``(landmarks_2d, landmarks_3d_world, fps, width, height, source_indices)``
    where ``source_indices[i]`` is the index of frame ``i`` in the original video
    file. Undetected poses remain represented as zero-visibility placeholders,
    preserving source-video timing.
    """
    import cv2

    estimator = MediaPipeEstimator(model_complexity=2)
    sequence = estimator.process_video(video_path)
    if not sequence.frames:
        raise ValueError(f"No poses detected in {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    landmarks_2d = np.stack([f.landmarks_2d for f in sequence.frames])
    if all(f.landmarks_3d is not None for f in sequence.frames):
        landmarks_3d = np.stack([f.landmarks_3d for f in sequence.frames])
    else:
        landmarks_3d = np.zeros(
            (landmarks_2d.shape[0], landmarks_2d.shape[1], 4), dtype=np.float32
        )
        landmarks_3d[:, :, :2] = landmarks_2d[:, :, :2]
        landmarks_3d[:, :, 3] = landmarks_2d[:, :, 2]

    source_indices = np.array([f.frame_index for f in sequence.frames], dtype=np.int64)
    return landmarks_2d, landmarks_3d, sequence.fps, width, height, source_indices

POINT_NAMES = ("left_base", "right_base", "left_top", "right_top")


def _point_from_entry(entry: dict, name: str) -> tuple[float, float] | None:
    value = entry.get("points", {}).get(name)
    if value is None or len(value) < 2:
        return None
    x = float(value[0])
    y = float(value[1])
    if not np.isfinite([x, y]).all():
        return None
    return x, y


def _labels_to_scene_anchors(
    payload: dict,
    target_indices: np.ndarray | None = None,
) -> SceneAnchors:
    """Build per-frame scene anchors by interpolating labelled clicks.

    ``target_indices`` lists the source-video frame indices we want anchors at
    (typically the source indices of MediaPipe's detected frames). When
    omitted, anchors are built at every integer index 0..n_frames-1.
    """
    label_entries = [
        entry
        for entry in payload.get("labels", [])
        if isinstance(entry, dict) and isinstance(entry.get("points"), dict)
    ]
    if not label_entries:
        raise ValueError(f"No labelled frames in {payload.get('video_stem', '<unknown>')}")
    if target_indices is None:
        target = np.arange(int(payload["n_frames"]), dtype=float)
    else:
        target = np.asarray(target_indices, dtype=float)

    arrays: dict[str, np.ndarray] = {}
    valid_counts = np.zeros(len(target), dtype=np.int32)
    for name in POINT_NAMES:
        by_frame: dict[int, tuple[float, float]] = {}
        for entry in label_entries:
            point = _point_from_entry(entry, name)
            if point is not None:
                by_frame[int(entry["frame"])] = point

        interpolated = np.full((len(target), 2), np.nan, dtype=np.float64)
        if by_frame:
            label_frames = np.array(sorted(by_frame), dtype=float)
            xy = np.array([by_frame[int(frame)] for frame in label_frames], dtype=float)
            if len(label_frames) == 1:
                # A single partial point is only known at that exact source
                # frame; do not smear it across unlabelled time.
                in_range = np.isclose(target, label_frames[0])
                interpolated[in_range] = xy[0]
            else:
                in_range = (
                    (target >= float(label_frames.min()))
                    & (target <= float(label_frames.max()))
                )
                interpolated[in_range, 0] = np.interp(
                    target[in_range],
                    label_frames,
                    xy[:, 0],
                )
                interpolated[in_range, 1] = np.interp(
                    target[in_range],
                    label_frames,
                    xy[:, 1],
                )
        arrays[name] = interpolated
        valid_counts += np.isfinite(interpolated).all(axis=1)

    if not any(np.isfinite(arr).any() for arr in arrays.values()):
        raise ValueError(f"No usable labelled points in {payload.get('video_stem', '<unknown>')}")

    confidence = valid_counts.astype(np.float64) / float(len(POINT_NAMES))
    return SceneAnchors(
        upright_left_base_px=arrays["left_base"],
        upright_right_base_px=arrays["right_base"],
        upright_left_top_px=arrays["left_top"],
        upright_right_top_px=arrays["right_top"],
        confidence=confidence,
        bar_height_m=payload.get("bar_height_m"),
        # For truth homography, "confirmed" means all four hand-labelled
        # correspondences are available at the target frame. Partial labels are
        # still retained above for pairwise scale evidence and future tooling.
        crossbar_confirmed=valid_counts >= len(POINT_NAMES),
    )


def _x_velocity_from_landmarks(landmarks_3d: np.ndarray, fps: float) -> np.ndarray:
    com = compute_com_trajectory(landmarks_3d, fps)["position"]
    if len(com) < 2:
        return np.zeros(len(com), dtype=float)
    return np.gradient(com[:, 0], 1.0 / fps)


def _takeoff_index(landmarks_3d: np.ndarray, fps: float) -> int | None:
    """Return the output index of the takeoff frame.

    Mirrors the production pipeline: takeoff is the local-max of CoM vertical
    velocity, which is the single instant that matters for Phase 10 metrics.
    Per-frame velocity comparisons are dominated by pose noise, so this gives
    a much cleaner signal than median-over-clip.
    """
    com = compute_com_trajectory(landmarks_3d, fps)
    vy = com.get("velocity")
    if vy is None or len(vy) < 2:
        return None
    return int(np.argmax(vy[:, 1]))


def _vh_window(landmarks_3d: np.ndarray, fps: float, frame: int, window: int = 3) -> float:
    """Median scene-x velocity across a small window around `frame`.

    Reduces sensitivity to MediaPipe single-frame jitter while staying near
    the takeoff instant.
    """
    vh = _x_velocity_from_landmarks(landmarks_3d, fps)
    lo = max(0, frame - window)
    hi = min(len(vh), frame + window + 1)
    if lo >= hi:
        return float("nan")
    return float(np.nanmedian(vh[lo:hi]))


def _landmarks_to_pixel_space(
    landmarks_2d_normalised: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    landmarks_px = np.asarray(landmarks_2d_normalised, dtype=np.float64).copy()
    landmarks_px[:, :, 0] *= float(image_width)
    landmarks_px[:, :, 1] *= float(image_height)
    return landmarks_px


def _error_summary(mode_vx: np.ndarray, truth_vx: np.ndarray) -> tuple[float, float]:
    n = min(len(mode_vx), len(truth_vx))
    if n == 0:
        return float("nan"), float("nan")
    err = np.abs(mode_vx[:n] - truth_vx[:n])
    return float(np.nanmedian(err)), float(np.nanpercentile(err, 95.0))


def _truth_in_takeoff_window(
    truth_landmarks: np.ndarray,
    truth_valid: np.ndarray,
    takeoff_idx: int,
    window: int = 3,
) -> bool:
    """True iff the truth pipeline has valid anchors near the takeoff instant."""
    lo = max(0, takeoff_idx - window)
    hi = min(len(truth_valid), takeoff_idx + window + 1)
    if lo >= hi:
        return False
    return bool(np.any(truth_valid[lo:hi]))


def evaluate_label_file(label_path: Path, thigh_m: float, shank_m: float) -> dict:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    video_path = Path(payload["video_path"])
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    landmarks_2d, landmarks_3d_world, fps, width, height, source_indices = (
        _extract_poses_with_source_indices(video_path)
    )
    bar_height = payload.get("bar_height_m") or parse_bar_height(video_path.name)
    if bar_height is None:
        raise ValueError(f"{label_path} needs bar_height_m or a parseable bar height")
    # Build truth anchors at MediaPipe's source frame indices. Keeping this
    # alignment explicit preserves compatibility with older cached labels.
    aligned_payload = {**payload, "bar_height_m": bar_height}
    truth_anchors = _labels_to_scene_anchors(aligned_payload, target_indices=source_indices)
    if not np.any(truth_anchors.confidence > 0):
        raise ValueError(
            f"{label_path}: no MediaPipe-detected frames fall inside the labelled "
            f"source-frame window ({int(source_indices.min())}..{int(source_indices.max())} "
            f"vs labels at {sorted({int(e['frame']) for e in payload['labels']})})"
        )

    # Hand labels are the evaluation target, so truth must bypass the
    # scene/anatomical mpp ratio gate; otherwise biased anatomical scale can
    # silently turn the reference back into anatomical fallback.
    truth_homographies = fit_per_frame_homography(truth_anchors)
    truth_valid = homography_valid_mask(truth_homographies, truth_anchors.confidence)
    truth_scene_xy = warp_landmarks_to_scene(
        _landmarks_to_pixel_space(landmarks_2d, width, height),
        truth_homographies,
        truth_valid,
    )
    truth_landmarks = np.zeros_like(landmarks_3d_world, dtype=np.float64)
    truth_landmarks[:, :, 0] = truth_scene_xy[:, :, 0]
    truth_landmarks[:, :, 1] = truth_scene_xy[:, :, 1]
    # Truth is an x/y scene-plane reference for horizontal velocity; z is dropped.
    truth_landmarks[:, :, 2] = 0.0
    if truth_landmarks.shape[2] > 3:
        truth_landmarks[:, :, 3] = landmarks_2d[:, :, 2]
    truth_info = {
        "method": "hand_labelled_scene_homography",
        "anchor_coverage_pct": round(float(np.mean(truth_valid) * 100.0), 2)
        if len(truth_valid)
        else 0.0,
    }
    truth_vx = _x_velocity_from_landmarks(truth_landmarks, fps)
    takeoff_idx = _takeoff_index(landmarks_3d_world, fps)

    anatomical = calibrate_landmarks_to_world(
        landmarks_2d,
        landmarks_3d_world,
        1.75,
        image_width=width,
        image_height=height,
        thigh_length_m=thigh_m,
        shank_length_m=shank_m,
    )

    motion = estimate_camera_motion(
        video_path,
        landmarks_2d,
        image_width=width,
        image_height=height,
    )
    egomotion, egomotion_info = calibrate_landmarks_with_scene(
        landmarks_2d,
        landmarks_3d_world,
        1.75,
        image_width=width,
        image_height=height,
        thigh_length_m=thigh_m,
        shank_length_m=shank_m,
        camera_motion=motion,
    )

    gravity, gravity_info = (
        calibrate_landmarks_with_gravity_mpp(
            landmarks_2d,
            landmarks_3d_world,
            fps=fps,
            image_width=width,
            image_height=height,
            takeoff_frame=takeoff_idx,
            camera_motion=motion,
        )
        if takeoff_idx is not None
        else (
            np.full_like(landmarks_3d_world, np.nan, dtype=np.float32),
            {
                "method": "gravity_mpp_unavailable",
                "accepted": False,
                "decision_reason": "missing_takeoff_frame",
            },
        )
    )

    detected_anchors = detect_scene_anchors(video_path, bar_height_m=bar_height)
    detected, detected_info = calibrate_landmarks_with_scene(
        landmarks_2d,
        landmarks_3d_world,
        1.75,
        image_width=width,
        image_height=height,
        thigh_length_m=thigh_m,
        shank_length_m=shank_m,
        scene_anchors=detected_anchors,
    )

    modes = {
        "anatomical": (anatomical, {"method": "anatomical"}),
        "egomotion": (egomotion, egomotion_info),
        "egomotion_gravity_mpp": (gravity, gravity_info),
        "scene_homography": (detected, detected_info),
    }

    truth_covers_takeoff = (
        takeoff_idx is not None
        and _truth_in_takeoff_window(truth_landmarks, truth_valid, takeoff_idx)
    )
    truth_vh_takeoff = (
        _vh_window(truth_landmarks, fps, takeoff_idx) if truth_covers_takeoff else float("nan")
    )

    out = {
        "label_file": str(label_path),
        "truth_method": truth_info.get("method"),
        "takeoff_frame": takeoff_idx,
        "truth_covers_takeoff": truth_covers_takeoff,
        "truth_vh_takeoff_mps": truth_vh_takeoff,
        "modes": {},
    }
    for name, (landmarks, info) in modes.items():
        # Original noisy per-frame median (kept for reference, but takeoff
        # window is the metric we actually trust).
        median_err, p95_err = _error_summary(
            _x_velocity_from_landmarks(landmarks, fps), truth_vx
        )
        mode_vh_takeoff = (
            _vh_window(landmarks, fps, takeoff_idx) if takeoff_idx is not None else float("nan")
        )
        takeoff_err = (
            abs(mode_vh_takeoff - truth_vh_takeoff)
            if truth_covers_takeoff and not np.isnan(mode_vh_takeoff)
            else float("nan")
        )
        out["modes"][name] = {
            "method": info.get("method"),
            "vh_takeoff_mps": mode_vh_takeoff,
            "vh_takeoff_error_vs_truth_mps": takeoff_err,
            "median_vh_error_mps": median_err,
            "p95_vh_error_mps": p95_err,
            "accepted": info.get("accepted"),
            "decision_reason": info.get("decision_reason"),
            "gravity_mpp": info.get("mpp"),
            "gravity_r2": info.get("y_r_squared"),
            "gravity_horizontal_accel_fraction": info.get("horizontal_accel_fraction"),
        }
    return out


def _aggregate_mode_errors(results: list[dict]) -> dict[str, dict[str, float | int]]:
    modes = sorted({mode for result in results for mode in result["modes"]})
    aggregates: dict[str, dict[str, float | int]] = {}
    for mode in modes:
        errors: list[float] = []
        accepted_count = 0
        available_count = 0
        reason_counts: dict[str, int] = {}
        for result in results:
            values = result["modes"].get(mode, {})
            err = values.get("vh_takeoff_error_vs_truth_mps")
            if isinstance(err, (int, float)) and np.isfinite(err):
                errors.append(float(err))
                available_count += 1
            if values.get("accepted") is True:
                accepted_count += 1
            reason = values.get("decision_reason")
            if isinstance(reason, str):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        aggregates[mode] = {
            "n": len(errors),
            "available_count": available_count,
            "accepted_count": accepted_count,
            "median_error_mps": float(np.nanmedian(errors)) if errors else float("nan"),
            "p95_error_mps": float(np.nanpercentile(errors, 95.0)) if errors else float("nan"),
            "reason_counts": reason_counts,
        }
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate modes against hand-labelled anchors.")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/results/hand_anchors"))
    parser.add_argument("--thigh", type=float, default=0.45)
    parser.add_argument("--shank", type=float, default=0.45)
    args = parser.parse_args()

    label_files = sorted(args.labels_dir.glob("*.json"))
    if not label_files:
        raise SystemExit(f"No label JSON files found in {args.labels_dir}")

    results = [evaluate_label_file(path, args.thigh, args.shank) for path in label_files]
    for result in results:
        stem = Path(result["label_file"]).stem
        truth_vh = result["truth_vh_takeoff_mps"]
        truth_str = f"{truth_vh:+.2f}" if not np.isnan(truth_vh) else "nan"
        covers = "yes" if result["truth_covers_takeoff"] else "NO"
        print(
            f"LABEL {stem} takeoff_frame={result['takeoff_frame']} "
            f"truth_vh_takeoff={truth_str} m/s  truth_covers_takeoff={covers}"
        )
        for mode, values in result["modes"].items():
            mode_vh = values["vh_takeoff_mps"]
            mode_str = f"{mode_vh:+.2f}" if not np.isnan(mode_vh) else "nan"
            err = values["vh_takeoff_error_vs_truth_mps"]
            err_str = f"{err:.2f}" if not np.isnan(err) else "n/a"
            print(
                f"  {mode}: method={values['method']:>22} "
                f"vh_takeoff={mode_str} m/s  err_vs_truth={err_str} m/s  "
                f"reason={values.get('decision_reason') or 'n/a'} "
                f"(per-frame median_err={values['median_vh_error_mps']:.2f})"
            )
    print("AGGREGATE takeoff-vh error vs hand-labelled truth")
    for mode, values in _aggregate_mode_errors(results).items():
        median = values["median_error_mps"]
        p95 = values["p95_error_mps"]
        median_str = f"{median:.2f}" if np.isfinite(median) else "nan"
        p95_str = f"{p95:.2f}" if np.isfinite(p95) else "nan"
        print(
            f"  {mode}: n={values['n']} accepted={values['accepted_count']} "
            f"median_err={median_str} m/s p95_err={p95_str} m/s "
            f"reasons={values['reason_counts']}"
        )


if __name__ == "__main__":
    main()
