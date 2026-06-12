"""Create takeoff-focused derivative clips from stationary high-jump footage.

This is a current-footage rescue tool. It does not invent new evidence; it
creates local, ignored analysis clips that make the athlete larger around the
plant/takeoff/flight window so MediaPipe has an easier detection problem.

The crop is static for each derivative clip, so it preserves the fixed-camera
assumption. The output should be treated as an analysis derivative and kept
under ignored `data/results/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.estimators.mediapipe_estimator import (  # noqa: E402
    MediaPipeEstimator,
    _aggregate_takeoff_crop,
    _estimate_motion_apex_frame,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def collect_videos(input_paths: list[Path]) -> list[Path]:
    """Collect videos recursively from files or directories."""
    videos: list[Path] = []
    seen: set[Path] = set()
    for input_path in input_paths:
        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir():
            candidates = sorted(
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
        else:
            raise FileNotFoundError(input_path)

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(candidate)
                seen.add(resolved)
    return videos


def _bbox_norm_to_px(
    bbox_norm: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1_n, y1_n, x2_n, y2_n = bbox_norm
    x1 = max(0, int(round(x1_n * width)))
    y1 = max(0, int(round(y1_n * height)))
    x2 = min(width, int(round(x2_n * width)))
    y2 = min(height, int(round(y2_n * height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate crop bbox: {bbox_norm}")
    return x1, y1, x2, y2


def _resize_shape(crop_width: int, crop_height: int, resize_height: int | None) -> tuple[int, int]:
    if resize_height is None:
        return crop_width, crop_height
    scale = resize_height / crop_height
    resize_width = int(round(crop_width * scale))
    # Some codecs prefer even dimensions.
    resize_width += resize_width % 2
    resize_height += resize_height % 2
    return max(2, resize_width), max(2, resize_height)


def create_focus_clip(
    video_path: Path,
    *,
    output_dir: Path,
    estimator: MediaPipeEstimator,
    pre_window_s: float,
    post_window_s: float,
    resize_height: int | None,
) -> dict:
    """Write one focused derivative clip and return its local manifest entry."""
    import cv2

    logger.info("Planning focus crop for %s", video_path.name)
    pass1 = estimator.process_video(video_path, roi_crop=False)
    apex_frame = _estimate_motion_apex_frame(pass1)
    if apex_frame is None:
        raise ValueError(f"Could not estimate takeoff/flight window for {video_path.name}")
    bbox_norm = _aggregate_takeoff_crop(
        pass1,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
    )
    if bbox_norm is None:
        raise ValueError(f"Could not estimate crop bbox for {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = pass1.fps if pass1.fps > 0 else float(cap.get(cv2.CAP_PROP_FPS))
    start_frame = max(0, apex_frame - int(round(pre_window_s * fps)))
    end_frame = min(source_frame_count, apex_frame + int(round(post_window_s * fps)) + 1)
    if end_frame <= start_frame:
        raise ValueError(f"Degenerate temporal window for {video_path.name}")

    x1, y1, x2, y2 = _bbox_norm_to_px(
        bbox_norm,
        width=source_width,
        height=source_height,
    )
    crop_width = x2 - x1
    crop_height = y2 - y1
    out_width, out_height = _resize_shape(crop_width, crop_height, resize_height)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_takeoff_focus.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y1:y2, x1:x2]
        if (out_width, out_height) != (crop_width, crop_height):
            crop = cv2.resize(crop, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        writer.write(crop)
        written += 1

    writer.release()
    cap.release()
    if written == 0:
        raise RuntimeError(f"No frames written for {video_path.name}")

    return {
        "source_video": video_path.name,
        "output_video": output_path.name,
        "apex_frame_estimate": int(apex_frame),
        "source_frame_window": [int(start_frame), int(end_frame - 1)],
        "written_frames": int(written),
        "fps": round(float(fps), 3),
        "crop_bbox_norm": [round(float(value), 6) for value in bbox_norm],
        "crop_bbox_px": [int(x1), int(y1), int(x2), int(y2)],
        "source_size_px": [int(source_width), int(source_height)],
        "output_size_px": [int(out_width), int(out_height)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Video files or directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pre-window-s", default=1.2, type=float)
    parser.add_argument("--post-window-s", default=0.7, type=float)
    parser.add_argument(
        "--resize-height",
        default=1080,
        type=int,
        help="Upscaled output height in pixels; use 0 to keep crop size",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = build_parser().parse_args()
    if args.pre_window_s <= 0 or args.post_window_s <= 0:
        raise ValueError("Temporal windows must be positive")
    resize_height = args.resize_height if args.resize_height > 0 else None
    videos = collect_videos(args.inputs)
    if not videos:
        raise ValueError("No video files found")

    estimator = MediaPipeEstimator(model_complexity=2)
    entries = [
        create_focus_clip(
            video,
            output_dir=args.output_dir,
            estimator=estimator,
            pre_window_s=args.pre_window_s,
            post_window_s=args.post_window_s,
            resize_height=resize_height,
        )
        for video in videos
    ]
    manifest = {
        "tool": "create_takeoff_focus_clips.py",
        "pre_window_s": args.pre_window_s,
        "post_window_s": args.post_window_s,
        "resize_height": resize_height,
        "clips": entries,
    }
    manifest_path = args.output_dir / "_takeoff_focus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} focused clips to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
