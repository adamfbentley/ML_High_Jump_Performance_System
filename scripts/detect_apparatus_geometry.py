"""Run the geometry-first, colour-agnostic apparatus detector on a frame or clip.

Unlike ``scripts/detect_stable_takeoff_anchors.py`` (which seeds from the *red*
apparatus region and scans for the *red* crossbar), this detector finds the
standards/crossbar/pad from line geometry alone, so it works on both the
floodlit night-red clips and the daylight clips whose standards are pale and
whose crossbar is a dark line against the sky.

Three modes:

* ``--frame-index N``         single-frame detection.
* ``--frame-indices A,B,C``   temporal aggregation across stationary frames,
                              taking the per-anchor median and requiring stable
                              geometry.
* ``--frame-indices … --plate``  synthesize one athlete-free *static plate* from
                              the window (per-pixel median after registering the
                              frames to a reference; the moving jumper dissolves)
                              and detect once on that clean image.  This is the
                              most reliable auto mode on stabilized clips because
                              line detection no longer fights the jumper, limbs,
                              shadow, or motion blur in the frame.

It writes a ``points_px`` apparatus-anchor JSON (left/right base/top in pixels)
plus a labelled debug overlay for visual review.
The athlete is masked with MediaPipe Pose (athlete-avoidance only — pose never
drives apparatus detection), and the athlete's horizontal position biases pair
selection toward the takeoff apparatus.
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

from src.pose_estimation.apparatus_detector import (  # noqa: E402
    ApparatusConfig,
    ApparatusDetection,
    detect_apparatus_geometry,
    detect_apparatus_geometry_stable,
    draw_apparatus_debug,
    _cluster_posts,
    _extract_segments,
)
from src.pose_estimation.camera_motion import build_static_plate  # noqa: E402
from src.pose_estimation.apparatus_pose_prior import (  # noqa: E402
    ApparatusROI,
    compute_apparatus_roi,
)

DEFAULT_UPRIGHT_SEPARATION_M = 4.02


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for apparatus geometry detection") from exc
    return cv2


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def _mediapipe_person_bbox(frame_bgr: np.ndarray, pad_frac: float = 0.12) -> list[float] | None:
    """Athlete bbox from MediaPipe Pose, or None if unavailable.

    Used only to keep the jumper out of the apparatus line search and to bias
    selection toward the takeoff zone — pose never detects the apparatus.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None
    cv2 = _require_cv2()
    height, width = frame_bgr.shape[:2]
    try:
        pose_module = mp.solutions.pose
    except AttributeError:
        return None
    with pose_module.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.35,
    ) as pose:
        result = pose.process(cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2RGB))
    if result.pose_landmarks is None:
        return None
    pts: list[tuple[float, float]] = []
    for lm in result.pose_landmarks.landmark:
        if float(getattr(lm, "visibility", 1.0)) < 0.45:
            continue
        x_px, y_px = float(lm.x) * width, float(lm.y) * height
        if np.isfinite([x_px, y_px]).all():
            pts.append((x_px, y_px))
    if len(pts) < 6:
        return None
    xy = np.asarray(pts, dtype=np.float64)
    x0, y0 = np.percentile(xy, 2, axis=0)
    x1, y1 = np.percentile(xy, 98, axis=0)
    pad_x = (x1 - x0) * pad_frac
    pad_y = (y1 - y0) * pad_frac
    return [
        float(max(0.0, x0 - pad_x)),
        float(max(0.0, y0 - pad_y)),
        float(min(width, x1 + pad_x)),
        float(min(height, y1 + pad_y)),
    ]


def _bbox_to_mask(bbox: list[float] | None, *, height: int, width: int) -> np.ndarray | None:
    """Rectangular athlete mask (True = athlete) from a bbox, for plate building."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    mask = np.zeros((height, width), dtype=bool)
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    mask[y0:y1, x0:x1] = True
    return mask


def _pose_track(video_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Full-clip BlazePose track ``(T, 33, 3)`` plus ``(width, height)``."""
    from src.pose_estimation.estimators.mediapipe_estimator import MediaPipeEstimator

    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    estimator = MediaPipeEstimator()
    sequence = estimator.process_video(video_path, roi_crop="off")
    if not sequence.frames:
        raise RuntimeError(f"No pose frames extracted from {video_path}")
    return sequence.to_numpy(), (width, height)


def _build_plate(frames: list[np.ndarray], bboxes: list, ref_pos: int):
    """Build a static plate from a window, excluding the athlete via bbox masks."""
    height, width = frames[0].shape[:2]
    masks = [_bbox_to_mask(b, height=height, width=width) for b in bboxes]
    local_indices = list(range(len(frames)))
    return build_static_plate(frames, local_indices, ref_index=ref_pos, foreground_masks=masks)


def _draw_roi(overlay: np.ndarray, roi: ApparatusROI) -> None:
    """Draw the pose-ROI rectangle plus the pose-derived bar/ground lines."""
    cv2 = _require_cv2()
    x0, y0, x1, y1 = (int(round(v)) for v in roi.as_bbox())
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2, cv2.LINE_AA)
    by = int(round(roi.bar_y_est_px))
    gy = int(round(roi.ground_y_px))
    cv2.line(overlay, (x0, by), (x1, by), (0, 255, 255), 1, cv2.LINE_AA)  # pose bar-y
    cv2.line(overlay, (x0, gy), (x1, gy), (255, 0, 255), 1, cv2.LINE_AA)  # pose ground-y
    bx = int(round(roi.bar_x_px))
    cv2.circle(overlay, (bx, by), 6, (0, 255, 255), -1, cv2.LINE_AA)
    for colour, thick in (((0, 0, 0), 5), ((255, 255, 0), 1)):
        cv2.putText(overlay, "POSE ROI (bar=cyan ground=magenta)", (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thick, cv2.LINE_AA)


def _detection_payload(detection: ApparatusDetection) -> dict:
    bar = detection.crossbar
    pad = detection.pad_top
    pad_bottom = detection.pad_bottom
    return {
        "separation_px": round(detection.separation_px, 3),
        "score": round(detection.score, 4),
        "confidence": round(detection.confidence, 4),
        "has_crossbar": bool(detection.has_crossbar),
        "has_pad": bool(detection.has_pad),
        "crossbar_y_center_px": round(bar.y_center_px, 3) if bar is not None else None,
        "pad_top_y_center_px": round(pad.y_center_px, 3) if pad is not None else None,
        "pad_bottom_y_center_px": round(pad_bottom.y_center_px, 3) if pad_bottom is not None else None,
        "pad_ratio": round(detection.pad_ratio, 4) if detection.pad_ratio is not None else None,
        "pad_ratio_expected": (
            round(detection.pad_ratio_expected, 4)
            if detection.pad_ratio_expected is not None else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="Comma-separated frames for temporal aggregation, e.g. 78,82,86.",
    )
    parser.add_argument("--bar-height", type=float, required=True)
    parser.add_argument("--upright-separation", type=float, default=DEFAULT_UPRIGHT_SEPARATION_M)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--debug-image", required=True, type=Path)
    parser.add_argument(
        "--mediapipe-person-mask",
        choices=("on", "off"),
        default="on",
        help="Mask the athlete and bias selection toward the takeoff zone.",
    )
    parser.add_argument(
        "--require-crossbar",
        action="store_true",
        help="Reject pairs without a connecting crossbar (no pad-only admission).",
    )
    parser.add_argument(
        "--plate",
        action="store_true",
        help=(
            "With --frame-indices: synthesize one athlete-free static plate from "
            "the window (median after registering to a reference) and detect on "
            "it. Writes the plate alongside the debug image as *_plate.png."
        ),
    )
    parser.add_argument(
        "--pose-roi",
        action="store_true",
        help=(
            "Localize the apparatus from the athlete's pose track (apex over the "
            "bar, plant by the near standard, stature as a metric ruler) and "
            "restrict detection to that ROI. Implies --plate. Kills background "
            "structures the athlete never touches."
        ),
    )
    parser.add_argument(
        "--athlete-height",
        type=float,
        default=1.75,
        help="Athlete standing stature (m), used as the pose-ROI metric ruler.",
    )
    return parser


def _frame_indices(args: argparse.Namespace) -> list[int]:
    if args.frame_indices:
        return [int(v.strip()) for v in args.frame_indices.split(",") if v.strip()]
    if args.frame_index is not None:
        return [int(args.frame_index)]
    raise SystemExit("provide --frame-index or --frame-indices")


def main() -> None:
    cv2 = _require_cv2()
    args = build_parser().parse_args()
    indices = _frame_indices(args)
    use_mask = args.mediapipe_person_mask == "on"
    config = ApparatusConfig(require_crossbar=args.require_crossbar)

    frames = [_read_frame(args.video, idx) for idx in indices]
    bboxes = [_mediapipe_person_bbox(f) if use_mask else None for f in frames]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.debug_image.parent.mkdir(parents=True, exist_ok=True)

    if args.pose_roi:
        if len(frames) < 2:
            raise SystemExit("--pose-roi requires --frame-indices with at least 2 frames")
        track, (iw, ih) = _pose_track(args.video)
        roi = compute_apparatus_roi(
            track,
            image_w=iw,
            image_h=ih,
            athlete_height_m=float(args.athlete_height),
            bar_height_m=float(args.bar_height),
            upright_separation_m=float(args.upright_separation),
        )
        if roi is None:
            raise SystemExit("pose-ROI failed: no usable athlete track (hips never visible)")
        ref_pos = len(frames) // 2
        plate = _build_plate(frames, bboxes, ref_pos)
        x0, y0, x1, y1 = (int(round(v)) for v in roi.as_bbox())
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(plate.plate.shape[1], x1), min(plate.plate.shape[0], y1)
        crop = plate.plate[y0:y1, x0:x1]

        # Seed the two standards from vertical structure inside the ROI, symmetric
        # about the pose bar_x.  (The metric scale is too coarse to place the base
        # posts directly, because the apparatus sits at a different depth than the
        # athlete.)  The bases then drive the proven base-post->upward bar scan.
        base_posts = None
        if crop.shape[0] > 20 and crop.shape[1] > 20:
            ap_cfg = ApparatusConfig()
            verts, _h = _extract_segments(crop, config=ap_cfg, athlete_bbox_px=None)
            posts = _cluster_posts(verts, config=ap_cfg, width=crop.shape[1], height=crop.shape[0])
            bar_x_crop = roi.bar_x_px - x0
            left_posts = [p for p in posts if p.x_base_px < bar_x_crop]
            right_posts = [p for p in posts if p.x_base_px > bar_x_crop]
            if left_posts and right_posts:
                lp = max(left_posts, key=lambda p: p.support_px)
                rp = max(right_posts, key=lambda p: p.support_px)
                base_posts = (
                    np.array([lp.x_base_px + x0, roi.ground_y_px], dtype=np.float64),
                    np.array([rp.x_base_px + x0, roi.ground_y_px], dtype=np.float64),
                )

        # The faint daylight crossbar is not reliably detectable (the landing-mat
        # top edge dominates any edge/dark-line response).  And the geometric bar
        # height (from horizontal standard separation) is corrupted by perspective
        # foreshortening.  The robust bar height is the POSE APEX: the athlete's CoM
        # peaks essentially at the bar, so bar_y_est is the bar line directly.
        points_px = None
        local_scale = None
        bar_y_geom = None
        bar_y = None
        if base_posts is not None:
            lb, rb = base_posts
            sep_px = float(abs(rb[0] - lb[0]))
            local_scale = sep_px / float(args.upright_separation)  # px/m (diagnostic)
            bar_y_geom = roi.ground_y_px - float(args.bar_height) * local_scale  # diagnostic
            bar_y = float(roi.bar_y_est_px)  # pose-apex bar height (used)
            points_px = {
                "left_base": [round(float(lb[0]), 3), round(float(lb[1]), 3)],
                "right_base": [round(float(rb[0]), 3), round(float(rb[1]), 3)],
                "left_top": [round(float(lb[0]), 3), round(bar_y, 3)],
                "right_top": [round(float(rb[0]), 3), round(bar_y, 3)],
            }

        overlay = plate.plate.copy()
        if points_px is not None:
            lb_p = (int(points_px["left_base"][0]), int(points_px["left_base"][1]))
            rb_p = (int(points_px["right_base"][0]), int(points_px["right_base"][1]))
            lt_p = (int(points_px["left_top"][0]), int(points_px["left_top"][1]))
            rt_p = (int(points_px["right_top"][0]), int(points_px["right_top"][1]))
            cv2.line(overlay, lb_p, lt_p, (0, 220, 255), 2, cv2.LINE_AA)   # left standard
            cv2.line(overlay, rb_p, rt_p, (0, 220, 255), 2, cv2.LINE_AA)   # right standard
            cv2.line(overlay, lt_p, rt_p, (0, 0, 255), 3, cv2.LINE_AA)     # computed bar
            cv2.line(overlay, lb_p, rb_p, (255, 0, 200), 2, cv2.LINE_AA)   # ground
            for p, col, name in ((lb_p, (0, 255, 0), "left_base"), (rb_p, (0, 255, 0), "right_base"),
                                 (lt_p, (0, 0, 255), "left_top"), (rt_p, (0, 0, 255), "right_top")):
                cv2.circle(overlay, p, 8, col, -1, cv2.LINE_AA)
                cv2.putText(overlay, f"{name}({p[0]},{p[1]})", (p[0] - 70, p[1] + (24 if "base" in name else -12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(overlay, f"{name}({p[0]},{p[1]})", (p[0] - 70, p[1] + (24 if "base" in name else -12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(overlay, "BAR = POSE APEX HEIGHT; bases = ROI vertical seed",
                            (30, overlay.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
        _draw_roi(overlay, roi)
        plate_path = args.debug_image.with_name(args.debug_image.stem + "_plate.png")
        cv2.imwrite(str(plate_path), plate.plate)
        payload = {
            "video": str(args.video),
            "frame_indices": indices,
            "bar_height_m": float(args.bar_height),
            "athlete_height_m": float(args.athlete_height),
            "upright_separation_m": float(args.upright_separation),
            "method": "pose_roi_seed_plus_pose_apex_bar_v1",
            "bar_y_used_px": round(bar_y, 2) if bar_y is not None else None,
            "local_scale_px_per_m": round(local_scale, 3) if local_scale else None,
            "bar_y_geom_px": round(bar_y_geom, 2) if bar_y_geom is not None else None,
            "pose_roi": {
                "bbox_px": [round(v, 2) for v in roi.as_bbox()],
                "bar_x_px": round(roi.bar_x_px, 2),
                "bar_y_est_px": round(roi.bar_y_est_px, 2),
                "ground_y_px": round(roi.ground_y_px, 2),
                "scale_px_per_m": round(roi.scale_px_per_m, 3),
                "apex_frame": roi.apex_frame,
                "takeoff_frame": roi.takeoff_frame,
            },
            "base_post_seed": (
                {"left": [round(float(v), 2) for v in base_posts[0]],
                 "right": [round(float(v), 2) for v in base_posts[1]]}
                if base_posts is not None else None
            ),
            "detected": points_px is not None,
            "plate_image": str(plate_path),
        }
        if points_px is not None:
            payload["points_px"] = points_px
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not cv2.imwrite(str(args.debug_image), overlay):
            raise RuntimeError(f"Could not write {args.debug_image}")
        print(f"Wrote {plate_path}")
        print(f"Wrote {args.output_json}")
        print(f"Wrote {args.debug_image}")
        print(
            f"pose-ROI: bar_x={roi.bar_x_px:.0f} ground_y={roi.ground_y_px:.0f} "
            f"apex_frame={roi.apex_frame}  base_seed={'yes' if base_posts is not None else 'no'}"
        )
        print("points_px:", json.dumps(points_px) if points_px else "None (no base posts found)")
        return

    if args.plate:
        if len(frames) < 2:
            raise SystemExit("--plate requires --frame-indices with at least 2 frames")
        height, width = frames[0].shape[:2]
        masks = [_bbox_to_mask(b, height=height, width=width) for b in bboxes]
        ref_pos = len(frames) // 2
        ref_index = indices[ref_pos]
        # build_static_plate indexes by absolute frame index; map our window onto
        # a contiguous local index space so `frames[idx]` lines up.
        local_indices = list(range(len(frames)))
        local_masks = masks
        plate = build_static_plate(
            frames,
            local_indices,
            ref_index=ref_pos,
            foreground_masks=local_masks,
        )
        centers = [float((b[0] + b[2]) / 2.0) for b in bboxes if b is not None]
        athlete_center_x = float(np.median(centers)) if centers else None
        detection = detect_apparatus_geometry(
            plate.plate,
            athlete_center_x_px=athlete_center_x,
            bar_height_m=float(args.bar_height),
            config=config,
        )
        plate_path = args.debug_image.with_name(args.debug_image.stem + "_plate.png")
        if not cv2.imwrite(str(plate_path), plate.plate):
            raise RuntimeError(f"Could not write {plate_path}")
        payload = {
            "video": str(args.video),
            "frame_indices": indices,
            "ref_frame_index": ref_index,
            "bar_height_m": float(args.bar_height),
            "upright_separation_m": float(args.upright_separation),
            "method": "geometry_first_colour_agnostic_v1_static_plate",
            "plate_frames_used": plate.n_frames_used,
            "plate_image": str(plate_path),
            "detected": detection is not None,
        }
        if detection is not None:
            payload["points_px"] = detection.points_px()
            payload["detection"] = _detection_payload(detection)
            overlay = draw_apparatus_debug(
                plate.plate,
                detection,
                extra_label=f"STATIC PLATE n={plate.n_frames_used} ref={ref_index}",
            )
        else:
            overlay = plate.plate.copy()
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not cv2.imwrite(str(args.debug_image), overlay):
            raise RuntimeError(f"Could not write {args.debug_image}")
        print(f"Wrote {plate_path}")
        print(f"Wrote {args.output_json}")
        print(f"Wrote {args.debug_image}")
        if detection is not None:
            print("points_px:", json.dumps(detection.points_px()))
        else:
            print("No apparatus detected on the static plate.")
        return

    if len(frames) == 1:
        detection = detect_apparatus_geometry(
            frames[0], athlete_bbox_px=bboxes[0], bar_height_m=float(args.bar_height), config=config
        )
        if detection is None:
            payload = {
                "video": str(args.video),
                "frame_index": indices[0],
                "bar_height_m": float(args.bar_height),
                "upright_separation_m": float(args.upright_separation),
                "detected": False,
                "method": "geometry_first_colour_agnostic_v1",
            }
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"No apparatus detected in frame {indices[0]}; wrote {args.output_json}")
            return
        payload = {
            "video": str(args.video),
            "frame_index": indices[0],
            "bar_height_m": float(args.bar_height),
            "upright_separation_m": float(args.upright_separation),
            "detected": True,
            "method": "geometry_first_colour_agnostic_v1",
            "points_px": detection.points_px(),
            "detection": _detection_payload(detection),
            "athlete_bbox_px": (
                [round(float(v), 2) for v in bboxes[0]] if bboxes[0] is not None else None
            ),
        }
        overlay = draw_apparatus_debug(
            frames[0], detection, extra_label=f"frame {indices[0]}"
        )
    else:
        stable = detect_apparatus_geometry_stable(
            frames, athlete_bboxes=bboxes, bar_height_m=float(args.bar_height), config=config
        )
        payload = {
            "video": str(args.video),
            "frame_indices": indices,
            "bar_height_m": float(args.bar_height),
            "upright_separation_m": float(args.upright_separation),
            "detected": bool(stable.n_frames_used > 0),
            "is_stable": bool(stable.is_stable),
            "method": "geometry_first_colour_agnostic_v1_temporal",
            "n_frames_used": stable.n_frames_used,
            "n_frames_total": stable.n_frames_total,
            "anchor_std_px": (
                round(stable.anchor_std_px, 3) if np.isfinite(stable.anchor_std_px) else None
            ),
            "crossbar_frac": round(stable.crossbar_frac, 3),
            "pad_frac": round(stable.pad_frac, 3),
        }
        if stable.n_frames_used > 0:
            payload["points_px"] = stable.points_px()
        # Overlay the median geometry on the first frame for review.
        if stable.n_frames_used > 0:
            best = max(
                (d for d in stable.per_frame if d is not None),
                key=lambda d: d.confidence,
            )
            overlay = draw_apparatus_debug(
                frames[0],
                best,
                extra_label=(
                    f"TEMPORAL used={stable.n_frames_used}/{stable.n_frames_total} "
                    f"std={stable.anchor_std_px:.1f}px stable={stable.is_stable}"
                ),
            )
        else:
            overlay = frames[0].copy()

    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not cv2.imwrite(str(args.debug_image), overlay):
        raise RuntimeError(f"Could not write {args.debug_image}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.debug_image}")
    if payload.get("detected") and "points_px" in payload:
        print("points_px:", json.dumps(payload["points_px"]))


if __name__ == "__main__":
    main()
