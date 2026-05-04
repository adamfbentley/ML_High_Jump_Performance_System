"""Hand-label high-jump apparatus anchors for calibration validation.

Labels are private local artifacts written to ``data/results/hand_anchors/``.
For each selected frame, click in this order:

1. left upright base
2. right upright base
3. left upright top / bar end
4. right upright top / bar end

Keys:
- left mouse click: place a point
- r: redo all four clicks for the current frame
- s: skip this frame (apparatus not visible / partly off-screen)
- q or Esc: abort the entire labelling session
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_jump_video import parse_bar_height  # noqa: E402

POINT_NAMES = ("left_base", "right_base", "left_top", "right_top")


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if count <= 0:
        raise RuntimeError(f"Could not read frame count from {video_path}")
    return count


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Return the requested frame, or None if it cannot be read.

    cv2's CAP_PROP_FRAME_COUNT can over-report frames on some encodings, so
    the caller must tolerate a None return rather than crashing.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    return frame


def _collect_clicks(frame: np.ndarray, frame_idx: int) -> list[tuple[float, float]] | None:
    """Collect 4 clicks for one frame. Returns None if the user skipped the frame."""
    clicks: list[tuple[float, float]] = []
    display = frame.copy()
    window = f"anchor labels frame {frame_idx}  (r=redo, s=skip, q=quit)"

    def callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= len(POINT_NAMES):
            return
        clicks.append((float(x), float(y)))
        colour = (0, 0, 255) if len(clicks) <= 2 else (255, 0, 0)
        cv2.circle(display, (x, y), 8, colour, thickness=-1)
        label = POINT_NAMES[len(clicks) - 1]
        cv2.putText(
            display,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, callback)
    while len(clicks) < len(POINT_NAMES):
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            clicks.clear()
            display = frame.copy()
        if key == ord("s"):
            cv2.destroyWindow(window)
            return None
        if key == ord("q") or key == 27:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("anchor labelling cancelled")
    cv2.imshow(window, display)
    cv2.waitKey(250)
    cv2.destroyWindow(window)
    return clicks


def main() -> None:
    parser = argparse.ArgumentParser(description="Label 4 scene anchors on 10 frames.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/hand_anchors"))
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    n_frames = _video_frame_count(args.video)
    frame_indices = sorted(
        {int(round(v)) for v in np.linspace(0, n_frames - 1, max(1, args.frames))}
    )
    labels: list[dict] = []
    skipped: list[int] = []
    unreadable: list[int] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.video.stem}.json"

    def _save() -> None:
        payload = {
            "video_path": str(args.video),
            "video_stem": args.video.stem,
            "bar_height_m": parse_bar_height(args.video.name),
            "n_frames": int(n_frames),
            "point_order": list(POINT_NAMES),
            "labels": labels,
            "skipped_frames": skipped,
            "unreadable_frames": unreadable,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        for frame_idx in frame_indices:
            frame = _read_frame(args.video, frame_idx)
            if frame is None:
                unreadable.append(int(frame_idx))
                print(f"  frame {frame_idx} unreadable (cv2 cannot decode); continuing")
                continue
            print(f"Frame {frame_idx}: click {', '.join(POINT_NAMES)}  (s=skip if no apparatus visible)")
            points = _collect_clicks(frame, frame_idx)
            if points is None:
                skipped.append(int(frame_idx))
                print(f"  frame {frame_idx} skipped (no apparatus)")
                continue
            labels.append(
                {
                    "frame": int(frame_idx),
                    "points": {
                        name: [float(x), float(y)]
                        for name, (x, y) in zip(POINT_NAMES, points)
                    },
                }
            )
            # Persist after every successful label so a later crash or quit
            # doesn't lose work.
            _save()
    except KeyboardInterrupt:
        if labels:
            _save()
            print(f"Aborted after {len(labels)} labelled frames; saved partial to {out_path}")
        raise

    if not labels:
        raise SystemExit(
            f"All {len(skipped) + len(unreadable)} sampled frames had no usable apparatus "
            f"({len(skipped)} skipped, {len(unreadable)} unreadable). "
            f"This clip cannot be ground-truthed via hand labels."
        )

    _save()
    print(
        f"Wrote {out_path}  ({len(labels)} labelled, {len(skipped)} skipped, "
        f"{len(unreadable)} unreadable)"
    )


if __name__ == "__main__":
    main()
