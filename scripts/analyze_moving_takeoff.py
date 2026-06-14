"""End-to-end takeoff physics from *moving* (panned/zoomed) footage.

Phase C/D of ``memory/plans/moving_footage_physics_plan.md``.  Earlier work
established that the takeoff in these clips happens where the camera is panning
and zooming, and that a short takeoff-centred window can be *stabilized* to an
effectively stationary reference with sub-pixel background residual.  This script
chains that together:

1.  Run the existing pose pipeline → detect the toe-off frame from heel/forefoot
    contact (``select_takeoff_frame_details``) and the 2D CoM pixel track.
2.  Take a takeoff-centred window of raw frames; mask the athlete from the
    landmarks; **stabilize** the window to the toe-off reference frame
    (``stabilize_window``) → per-frame frame→ref homographies + a residual gate.
3.  Remap the CoM pixels of every window frame into the reference view so a
    single fixed camera pose is valid across the window.
4.  Get apparatus anchors in the reference frame — auto (geometry detector) or a
    supplied manual anchor JSON — and ``solvePnP`` the camera.
5.  Fit the gravity-constrained 3D projectile to the *stabilized* CoM pixels
    (reuses ``fit_projectile_to_com_pixels``).

The stabilization is what makes the fixed-camera solver valid on panned footage;
it also removes the vertical-tilt drift that defeated the parked gravity-mpp
approach.  Outputs are explicitly takeoff-window-only and must clear the same
physics gates as the stationary path before any optimiser use.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_jump_video import (  # noqa: E402
    build_sample,
    compute_kinematics,
    extract_poses,
    pose_validity_pct,
    select_takeoff_frame_details,
    takeoff_window_pose_validity_pct,
)
from scripts.analyze_stable_takeoff_window import (  # noqa: E402
    ANCHOR_NAMES,
    DEFAULT_UPRIGHT_SEPARATION_M,
    ProjectileFitConfig,
    StableWindowAnchors,
    com_pixel_trajectory,
    fit_projectile_to_com_pixels,
    focal_length_from_fov_px,
    key_joint_valid_mask,
    load_anchor_json,
    solve_camera_from_anchors,
)
from src.pose_estimation.apparatus_detector import (  # noqa: E402
    ApparatusConfig,
    detect_apparatus_geometry,
    draw_apparatus_debug,
)
from src.pose_estimation.camera_motion import (  # noqa: E402
    MotionConfig,
    remap_points_through_homography,
    stabilize_window,
)
from src.pose_estimation.scale_calibration import calibrate_landmarks_to_world  # noqa: E402
from src.pose_estimation.skeleton.landmark_postprocessor import (  # noqa: E402
    PostProcessorConfig,
    postprocess_landmarks,
)

logger = logging.getLogger(__name__)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for moving-takeoff analysis") from exc
    return cv2


def _detect_takeoff(
    landmarks_2d: np.ndarray,
    landmarks_3d_world: np.ndarray,
    *,
    fps: float,
    width: int,
    height: int,
    body_mass_kg: float,
    height_m: float,
    thigh_length_m: float,
    shank_length_m: float,
    video_path: Path,
) -> tuple[int, bool, bool]:
    """Reuse the production chain to locate the contact-anchored toe-off frame."""
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
            do_segment_enforce=True, height_m=height_m, segment_enforce_weight=0.8
        ),
    )
    kinematics = compute_kinematics(postprocessed, fps, body_mass_kg)
    sample = build_sample(video_path, postprocessed, kinematics, fps, body_mass_kg, height_m)
    fallback = int(np.argmax(sample.com_velocity[:, 1]))
    selected_frame, contact_detected, _count, anchor_ok = select_takeoff_frame_details(
        sample, fallback_frame=fallback
    )
    return int(selected_frame), bool(contact_detected), bool(anchor_ok)


def _athlete_bbox(lm_row: np.ndarray, *, width: int, height: int, pad: float = 0.18) -> list[float] | None:
    """Athlete bbox (px) from one frame's MediaPipe landmarks, or None."""
    vis = lm_row[:, 2] >= 0.5
    if int(np.count_nonzero(vis)) < 6:
        return None
    xs = lm_row[vis, 0] * width
    ys = lm_row[vis, 1] * height
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    pad_x, pad_y = (x1 - x0) * pad, (y1 - y0) * pad
    return [
        max(0.0, x0 - pad_x),
        max(0.0, y0 - pad_y),
        min(float(width), x1 + pad_x),
        min(float(height), y1 + pad_y),
    ]


def _bbox_mask(bbox: list[float] | None, *, width: int, height: int) -> np.ndarray | None:
    if bbox is None:
        return None
    mask = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
    return mask


def _read_window_frames(video_path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    out: dict[int, np.ndarray] = {}
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                out[idx] = frame
    finally:
        cap.release()
    return out


def analyse_moving_takeoff(
    video_path: Path,
    *,
    bar_height_m: float,
    body_mass_kg: float,
    height_m: float,
    thigh_length_m: float,
    shank_length_m: float,
    upright_separation_m: float,
    window_pre: int,
    window_post: int,
    anchor_json: Path | None,
    roi_crop: str,
    fov_deg: float,
    focal_length_px: float | None,
    max_stab_residual_px: float,
    debug_image: Path | None,
) -> dict[str, Any]:
    cv2 = _require_cv2()
    landmarks_2d, landmarks_3d_world, fps, width, height = extract_poses(
        video_path, use_roi_crop=roi_crop
    )
    n_frames = int(len(landmarks_2d))
    takeoff_frame, contact_detected, anchor_review_ok = _detect_takeoff(
        landmarks_2d,
        landmarks_3d_world,
        fps=fps,
        width=width,
        height=height,
        body_mass_kg=body_mass_kg,
        height_m=height_m,
        thigh_length_m=thigh_length_m,
        shank_length_m=shank_length_m,
        video_path=video_path,
    )

    com_px = com_pixel_trajectory(landmarks_2d, width, height)
    valid_mask = key_joint_valid_mask(landmarks_2d)

    # Takeoff-centred window; the post side must cover the projectile fit horizon.
    win_start = max(0, takeoff_frame - int(window_pre))
    win_end = min(n_frames - 1, takeoff_frame + int(window_post))
    window_indices = list(range(win_start, win_end + 1))

    frames = _read_window_frames(video_path, window_indices)
    if takeoff_frame not in frames:
        return {
            "video": video_path.stem,
            "error": "could_not_read_takeoff_frame",
            "takeoff_frame": takeoff_frame,
        }

    masks: dict[int, np.ndarray | None] = {}
    for idx in window_indices:
        bbox = _athlete_bbox(landmarks_2d[idx], width=width, height=height) if idx < n_frames else None
        masks[idx] = _bbox_mask(bbox, width=width, height=height)

    stab = stabilize_window(
        frames,
        list(frames.keys()),
        ref_index=takeoff_frame,
        foreground_masks=masks,
        config=MotionConfig(),
    )
    homographies = {idx: H for idx, H in zip(stab.frame_indices, stab.homographies)}
    residuals = {idx: r for idx, r in zip(stab.frame_indices, stab.residual_px)}

    # Remap CoM pixels of each window frame into the toe-off reference view.
    com_px_stab = com_px.copy()
    for idx in window_indices:
        H = homographies.get(idx)
        if H is None or not np.isfinite(residuals.get(idx, np.inf)) or residuals[idx] > max_stab_residual_px:
            com_px_stab[idx] = np.nan  # drop frames we could not stabilize well
            continue
        com_px_stab[idx] = remap_points_through_homography(com_px[idx][None, :], H)[0]

    # Apparatus anchors in the reference (toe-off) frame.
    ref_frame = frames[takeoff_frame]
    apparatus_source = "manual_anchor_json" if anchor_json is not None else "geometry_auto"
    detection = None
    if anchor_json is not None:
        anchors = load_anchor_json(anchor_json, bar_height_override_m=bar_height_m)
    else:
        ref_bbox = _athlete_bbox(landmarks_2d[takeoff_frame], width=width, height=height)
        detection = detect_apparatus_geometry(
            ref_frame, athlete_bbox_px=ref_bbox, config=ApparatusConfig()
        )
        if detection is None:
            return {
                "video": video_path.stem,
                "error": "apparatus_not_detected_in_reference_frame",
                "takeoff_frame": takeoff_frame,
                "stabilization": {
                    "n_registered": stab.n_registered,
                    "mean_residual_px": round(stab.mean_residual_px, 3),
                    "max_residual_px": round(stab.max_residual_px, 3),
                },
            }
        pts = detection.points_px()
        image_points = np.asarray([pts[name] for name in ANCHOR_NAMES], dtype=np.float64)
        anchors = StableWindowAnchors(
            frame_index=takeoff_frame,
            bar_height_m=float(bar_height_m),
            upright_separation_m=float(upright_separation_m),
            image_points_px=image_points,
        )

    base_focal = focal_length_px or focal_length_from_fov_px(width, fov_deg)
    camera = solve_camera_from_anchors(
        anchors, image_width=width, image_height=height, focal_length_px=base_focal
    )
    fit = fit_projectile_to_com_pixels(
        com_px_stab,
        valid_mask,
        takeoff_frame=takeoff_frame,
        fps=fps,
        camera=camera,
        anchors=anchors,
        athlete_height_m=height_m,
        config=ProjectileFitConfig(max_fit_s=min(0.9, window_post / max(1.0, fps))),
    )

    quality_failures = list(fit.get("decision_reasons", []))
    if not contact_detected:
        quality_failures.append("no_contact_interval")
    if stab.max_residual_px > max_stab_residual_px:
        quality_failures.append("stabilization_residual_high")

    if debug_image is not None:
        _write_debug(debug_image, ref_frame, anchors, com_px_stab, window_indices,
                     takeoff_frame, detection)

    return {
        "schema_version": 1,
        "analysis_mode": "moving_takeoff_window_stabilized",
        "takeoff_window_only": True,
        "video": video_path.stem,
        "frames": n_frames,
        "fps": round(float(fps), 3),
        "image_size_px": [int(width), int(height)],
        "takeoff_frame": int(takeoff_frame),
        "window": {"start": win_start, "end": win_end, "n_frames": len(window_indices)},
        "stabilization": {
            "ref_index": int(takeoff_frame),
            "n_registered": stab.n_registered,
            "n_window": len(window_indices),
            "mean_residual_px": round(stab.mean_residual_px, 3),
            "max_residual_px": round(stab.max_residual_px, 3),
            "max_residual_gate_px": max_stab_residual_px,
        },
        "apparatus": {
            "source": apparatus_source,
            "points_px": {name: [round(float(v), 2) for v in pt]
                          for name, pt in zip(ANCHOR_NAMES, anchors.image_points_px)},
            "bar_height_m": float(anchors.bar_height_m),
            "upright_separation_m": float(anchors.upright_separation_m),
            "confidence": (round(float(detection.confidence), 3) if detection is not None else None),
            "has_crossbar": (bool(detection.has_crossbar) if detection is not None else None),
        },
        "pose": {
            "pose_validity_pct": round(pose_validity_pct(landmarks_2d), 2),
            "takeoff_window_pose_validity_pct": round(
                takeoff_window_pose_validity_pct(landmarks_2d, takeoff_frame=takeoff_frame), 2
            ),
        },
        "contact_takeoff": {
            "contact_interval_detected": bool(contact_detected),
            "takeoff_anchor_review_passed": bool(anchor_review_ok),
        },
        "projectile_fit": fit,
        "quality": {
            "accepted_moving_takeoff_physics": not quality_failures,
            "failures": quality_failures,
            "note": (
                "Stabilized moving-takeoff physics, window-only. Camera solved in "
                "the toe-off reference frame after background stabilization. Does "
                "not unblock optimiser claims without held-out validation."
            ),
        },
    }


def _write_debug(
    debug_image: Path,
    ref_frame: np.ndarray,
    anchors: StableWindowAnchors,
    com_px_stab: np.ndarray,
    window_indices: list[int],
    takeoff_frame: int,
    detection,
) -> None:
    cv2 = _require_cv2()
    if detection is not None:
        out = draw_apparatus_debug(ref_frame, detection, extra_label=f"toe-off ref f{takeoff_frame}")
    else:
        out = ref_frame.copy()
        pts = {name: anchors.image_points_px[i] for i, name in enumerate(ANCHOR_NAMES)}

        def _pt(p):
            return (int(round(float(p[0]))), int(round(float(p[1]))))

        cv2.line(out, _pt(pts["left_base"]), _pt(pts["left_top"]), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(out, _pt(pts["right_base"]), _pt(pts["right_top"]), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(out, _pt(pts["left_top"]), _pt(pts["right_top"]), (0, 0, 255), 3, cv2.LINE_AA)
        for name in ANCHOR_NAMES:
            cv2.circle(out, _pt(pts[name]), 7, (0, 255, 0), -1, cv2.LINE_AA)

    # Stabilized CoM track in the reference view.
    prev = None
    for idx in window_indices:
        p = com_px_stab[idx]
        if not np.isfinite(p).all():
            continue
        xy = (int(round(float(p[0]))), int(round(float(p[1]))))
        colour = (0, 255, 255) if idx >= takeoff_frame else (255, 200, 0)
        cv2.circle(out, xy, 4, colour, -1, cv2.LINE_AA)
        if prev is not None:
            cv2.line(out, prev, xy, colour, 1, cv2.LINE_AA)
        prev = xy
    debug_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_image), out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--bar-height", type=float, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anchor-json", type=Path, default=None,
                        help="Optional manual apparatus anchors (ref-frame px). Else auto-detect.")
    parser.add_argument("--mass", type=float, default=65.0)
    parser.add_argument("--height", type=float, default=1.75)
    parser.add_argument("--thigh", type=float, default=0.45)
    parser.add_argument("--shank", type=float, default=0.45)
    parser.add_argument("--upright-separation", type=float, default=DEFAULT_UPRIGHT_SEPARATION_M)
    parser.add_argument("--window-pre", type=int, default=4)
    parser.add_argument("--window-post", type=int, default=26)
    parser.add_argument("--roi-crop", choices=("off", "on", "takeoff"), default="off")
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--max-stab-residual-px", type=float, default=3.0)
    parser.add_argument("--debug-image", type=Path, default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = build_parser().parse_args()
    report = analyse_moving_takeoff(
        args.video,
        bar_height_m=args.bar_height,
        body_mass_kg=args.mass,
        height_m=args.height,
        thigh_length_m=args.thigh,
        shank_length_m=args.shank,
        upright_separation_m=args.upright_separation,
        window_pre=args.window_pre,
        window_post=args.window_post,
        anchor_json=args.anchor_json,
        roi_crop=args.roi_crop,
        fov_deg=args.fov_deg,
        focal_length_px=args.focal_length_px,
        max_stab_residual_px=args.max_stab_residual_px,
        debug_image=args.debug_image,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return
    fit = report.get("projectile_fit", {})
    print(f"takeoff_frame={report['takeoff_frame']} "
          f"stab_resid(mean/max)={report['stabilization']['mean_residual_px']}/"
          f"{report['stabilization']['max_residual_px']}px "
          f"apparatus={report['apparatus']['source']}")
    if "takeoff_angle_deg" in fit:
        print(f"physics: angle={fit['takeoff_angle_deg']}deg vh={fit['takeoff_horizontal_mps']}m/s "
              f"vv={fit['takeoff_vertical_mps']}m/s reproj_rms={fit['projectile_reprojection_rms_px']}px "
              f"accepted={report['quality']['accepted_moving_takeoff_physics']}")


if __name__ == "__main__":
    main()
