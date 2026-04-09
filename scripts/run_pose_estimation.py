"""Quick-start script: run MediaPipe BlazePose on a single video.

Usage:
    python scripts/run_pose_estimation.py path/to/video.mp4

Outputs 2D landmarks as .npy to data/poses/landmarks_2d/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose_estimation.estimators.mediapipe_estimator import MediaPipeEstimator


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pose_estimation.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: {video_path} does not exist")
        sys.exit(1)

    print(f"Processing: {video_path}")
    estimator = MediaPipeEstimator(model_complexity=2)
    sequence = estimator.process_video(video_path)

    print(f"  Detected poses in {len(sequence.frames)} / {sequence.duration_s:.1f}s frames")
    valid = sum(1 for f in sequence.frames if f.is_valid)
    print(f"  Valid frames (key joints visible): {valid}/{len(sequence.frames)}")

    # Save output
    output_dir = Path("data/poses/landmarks_2d")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{video_path.stem}_landmarks_2d.npy"
    np.save(out_file, sequence.to_numpy())
    print(f"  Saved: {out_file}")


if __name__ == "__main__":
    main()
