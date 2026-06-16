"""Scan high-jump clips for a usable stationary takeoff window.

Phase A of ``memory/plans/moving_footage_physics_plan.md``.  For each clip this
estimates background (camera) motion frame-to-frame and reports the longest
near-stationary run — the window where the existing fixed-camera takeoff solver
can legitimately be applied.  Use it to measure how much of the panned corpus is
actually recoverable before investing in stabilization and apparatus work.

Examples::

    python scripts/scan_stable_windows.py --video "<clip>.mov" --fps 30
    python scripts/scan_stable_windows.py --glob "data/High Jump Videos/25_03_26/*.mov" \
        --athlete-mask on --output-json data/results/stationary_validation/stable_window_scan_v1.json

Background motion is measured at a fixed processing width so the px/frame
threshold is resolution-independent.  ``--athlete-mask on`` excludes the jumper
via MediaPipe Pose (slower, more accurate); the default leaves it off for a fast
first pass, which is usually fine because the athlete is small at takeoff.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.camera_motion import (  # noqa: E402
    MotionConfig,
    analyze_camera_motion,
)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("OpenCV is required for the stable-window scan") from exc
    return cv2


def _decode(video_path: Path, *, proc_width: int, max_frames: int) -> tuple[list[np.ndarray], float]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if proc_width > 0 and w > proc_width:
                scale = proc_width / w
                frame = cv2.resize(frame, (proc_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        cap.release()
    return frames, fps


def _athlete_masks(frames: list[np.ndarray]) -> list[np.ndarray | None]:
    """Per-frame boolean foreground masks (True = athlete) via MediaPipe Pose."""
    try:
        import mediapipe as mp
    except ImportError:
        return [None] * len(frames)
    cv2 = _require_cv2()
    try:
        pose_module = mp.solutions.pose
    except AttributeError:
        return [None] * len(frames)
    masks: list[np.ndarray | None] = []
    with pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.4,
    ) as pose:
        for frame in frames:
            h, w = frame.shape[:2]
            result = pose.process(cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB))
            if result.pose_landmarks is None:
                masks.append(None)
                continue
            pts = []
            for lm in result.pose_landmarks.landmark:
                if float(getattr(lm, "visibility", 1.0)) < 0.4:
                    continue
                pts.append((float(lm.x) * w, float(lm.y) * h))
            if len(pts) < 6:
                masks.append(None)
                continue
            xy = np.asarray(pts, dtype=np.float64)
            x0, y0 = np.percentile(xy, 2, axis=0)
            x1, y1 = np.percentile(xy, 98, axis=0)
            pad_x, pad_y = (x1 - x0) * 0.25, (y1 - y0) * 0.20
            mask = np.zeros((h, w), dtype=bool)
            xi0, yi0 = int(max(0, x0 - pad_x)), int(max(0, y0 - pad_y))
            xi1, yi1 = int(min(w, x1 + pad_x)), int(min(h, y1 + pad_y))
            mask[yi0:yi1, xi0:xi1] = True
            masks.append(mask)
    return masks


def _scan_one(video_path: Path, args: argparse.Namespace) -> dict:
    frames, container_fps = _decode(video_path, proc_width=args.proc_width, max_frames=args.max_frames)
    if len(frames) < 4:
        return {"video": str(video_path), "error": "too_few_frames", "n_frames": len(frames)}
    fps = float(args.fps) if args.fps else (container_fps if container_fps > 1.0 else 30.0)
    masks = _athlete_masks(frames) if args.athlete_mask == "on" else None
    config = MotionConfig(
        max_disp_px_per_frame=args.max_disp_px,
        min_window_s=args.min_window_s,
    )
    result = analyze_camera_motion(frames, fps=fps, foreground_masks=masks, config=config)
    window = result.stable_window
    finite = result.disp_px[np.isfinite(result.disp_px) & (result.disp_px < config.fail_disp_px)]
    payload: dict = {
        "video": str(video_path),
        "n_frames": len(frames),
        "fps": round(fps, 3),
        "proc_width": int(frames[0].shape[1]),
        "athlete_mask": args.athlete_mask,
        "median_disp_px_per_frame": round(float(np.median(finite)), 3) if finite.size else None,
        "n_failed_estimates": int(np.count_nonzero(~result.ok)),
        "has_usable_window": window is not None,
    }
    if window is not None:
        payload["stable_window"] = {
            "start": window.start,
            "end": window.end,
            "ref_index": window.ref_index,
            "n_frames": window.n_frames,
            "duration_s": round(window.duration_s, 3),
            "mean_disp_px": round(window.mean_disp_px, 3),
            "max_disp_px": round(window.max_disp_px, 3),
        }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--glob", type=str, help="Glob of clips to scan.")
    parser.add_argument("--fps", type=float, default=None, help="Override container fps.")
    parser.add_argument("--proc-width", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--max-disp-px", type=float, default=2.5)
    parser.add_argument("--min-window-s", type=float, default=0.25)
    parser.add_argument("--athlete-mask", choices=("on", "off"), default="off")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.video is not None:
        videos = [args.video]
    else:
        videos = [Path(p) for p in sorted(glob.glob(args.glob))]
    if not videos:
        raise SystemExit("no clips matched")

    reports = []
    for video in videos:
        try:
            report = _scan_one(video, args)
        except Exception as exc:  # noqa: BLE001 - scan should not abort on one bad clip
            report = {"video": str(video), "error": f"{type(exc).__name__}: {exc}"}
        reports.append(report)
        if "error" in report:
            print(f"[skip] {video.name}: {report['error']}")
        elif report["has_usable_window"]:
            w = report["stable_window"]
            print(
                f"[ OK ] {video.name}: window {w['start']}-{w['end']} "
                f"({w['n_frames']}f, {w['duration_s']}s, max {w['max_disp_px']}px/f) "
                f"median motion {report['median_disp_px_per_frame']}px/f"
            )
        else:
            print(
                f"[none] {video.name}: no stable window "
                f"(median motion {report['median_disp_px_per_frame']}px/f)"
            )

    usable = sum(1 for r in reports if r.get("has_usable_window"))
    print(f"\nSummary: {usable}/{len(reports)} clips have a usable stable window "
          f"(>= {args.min_window_s}s under {args.max_disp_px}px/frame).")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"config": vars(args) | {"video": str(args.video), "output_json": str(args.output_json)},
                        "reports": reports}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
