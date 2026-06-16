"""Run the geometry-first, colour-agnostic apparatus detector on a frame or clip.

Unlike ``scripts/detect_stable_takeoff_anchors.py`` (which seeds from the *red*
apparatus region and scans for the *red* crossbar), this detector finds the
standards/crossbar/pad from line geometry alone, so it works on both the
floodlit night-red clips and the daylight clips whose standards are pale and
whose crossbar is a dark line against the sky.

Two modes:

* ``--frame-index N``         single-frame detection.
* ``--frame-indices A,B,C``   temporal aggregation across stationary frames,
                              taking the per-anchor median and requiring stable
                              geometry.

It writes a ``points_px`` JSON in the same schema consumed by
``scripts/analyze_stable_takeoff_window.py`` plus a labelled debug overlay.
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


def _detection_payload(detection: ApparatusDetection) -> dict:
    bar = detection.crossbar
    pad = detection.pad_top
    return {
        "separation_px": round(detection.separation_px, 3),
        "score": round(detection.score, 4),
        "confidence": round(detection.confidence, 4),
        "has_crossbar": bool(detection.has_crossbar),
        "has_pad": bool(detection.has_pad),
        "crossbar_y_center_px": round(bar.y_center_px, 3) if bar is not None else None,
        "pad_top_y_center_px": round(pad.y_center_px, 3) if pad is not None else None,
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

    if len(frames) == 1:
        detection = detect_apparatus_geometry(frames[0], athlete_bbox_px=bboxes[0], config=config)
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
        stable = detect_apparatus_geometry_stable(frames, athlete_bboxes=bboxes, config=config)
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
