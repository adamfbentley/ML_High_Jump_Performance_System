"""Render MediaPipe BlazePose overlays on one video or a directory of videos.

Usage:
    python scripts/render_pose_overlay.py "data/High Jump Videos/09_02_26/09_02_26_one.mp4"
    python scripts/render_pose_overlay.py "data/High Jump Videos" \
        --output-dir data/results/pose_overlays
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.pose_overlay import render_mediapipe_pose_overlay

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def collect_videos(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BlazePose overlays on video footage")
    parser.add_argument("input", type=str, help="Video file or directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/pose_overlays",
        help="Where to save overlay videos",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override for pose_landmarker_heavy.task",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    videos = collect_videos(input_path)
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_path}")

    print(f"Found {len(videos)} video(s)")
    for video_path in videos:
        output_path = output_dir / video_path.parent.name / f"{video_path.stem}_overlay.mp4"
        summary = render_mediapipe_pose_overlay(
            video_path,
            output_path,
            model_path=args.model_path,
        )
        print(
            f"{video_path.name}: {summary['frames']} frames, "
            f"valid {summary['valid_frames']} ({summary['valid_ratio']:.0%}) -> {output_path}"
        )


if __name__ == "__main__":
    main()
