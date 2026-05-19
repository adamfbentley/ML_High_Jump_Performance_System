"""Hand-label high-jump apparatus anchors for calibration validation.

Labels are private local artifacts written to ``data/results/hand_anchors/``.
For each selected frame, label every apparatus point that is visible. Frames do
not need all four points; partial labels are useful scale evidence and are
saved as-is.

Keys:
- 1/2/3/4: choose left base, right base, left top, right top
- left mouse click: place or replace the chosen point
- n or Enter: save this frame with the labelled visible points
- u: undo the most recent point
- r: clear all points for the current frame
- s: skip this frame when no useful apparatus is visible
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
POINT_COLOURS = {
    "left_base": (0, 190, 255),
    "right_base": (0, 255, 120),
    "left_top": (255, 120, 0),
    "right_top": (255, 0, 190),
}


def _point_status(points: dict[str, tuple[float, float]], name: str) -> str:
    return "set" if name in points else "missing"


def _next_point_index(points: dict[str, tuple[float, float]], current: int) -> int:
    for offset in range(1, len(POINT_NAMES) + 1):
        candidate = (current + offset) % len(POINT_NAMES)
        if POINT_NAMES[candidate] not in points:
            return candidate
    return (current + 1) % len(POINT_NAMES)


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


def _draw_overlay(
    frame: np.ndarray,
    frame_idx: int,
    points: dict[str, tuple[float, float]],
    selected_idx: int,
    status: str,
) -> np.ndarray:
    display = frame.copy()

    cv2.rectangle(display, (8, 8), (760, 168), (0, 0, 0), thickness=-1)
    cv2.putText(
        display,
        f"Frame {frame_idx}: label visible apparatus points",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "1-4 select | click place/replace | n save partial | s skip empty | u undo | r clear | q quit",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    for idx, name in enumerate(POINT_NAMES):
        colour = POINT_COLOURS[name]
        prefix = ">" if idx == selected_idx else " "
        y = 86 + idx * 20
        cv2.putText(
            display,
            f"{prefix}{idx + 1} {name}: {_point_status(points, name)}",
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2 if idx == selected_idx else 1,
            cv2.LINE_AA,
        )
        if name not in points:
            continue
        x_px, y_px = points[name]
        xy = (int(round(x_px)), int(round(y_px)))
        cv2.circle(display, xy, 8, colour, thickness=-1)
        cv2.putText(
            display,
            str(idx + 1),
            (xy[0] + 10, xy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA,
        )

    if status:
        cv2.putText(
            display,
            status,
            (18, 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return display


def _collect_points(
    frame: np.ndarray,
    frame_idx: int,
    initial_points: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Collect visible apparatus points for one frame.

    Returns None only when the frame is intentionally skipped. A returned dict
    may contain one to four points.
    """
    points: dict[str, tuple[float, float]] = dict(initial_points or {})
    history: list[str] = []
    selected_idx = [0]
    status = ""
    window = f"anchor labels frame {frame_idx}"

    def callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        name = POINT_NAMES[selected_idx[0]]
        points[name] = (float(x), float(y))
        history.append(name)
        selected_idx[0] = _next_point_index(points, selected_idx[0])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, callback)
    while True:
        display = _draw_overlay(frame, frame_idx, points, selected_idx[0], status)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if ord("1") <= key <= ord("4"):
            selected_idx[0] = key - ord("1")
            status = f"Selected {POINT_NAMES[selected_idx[0]]}"
            continue
        if key in (ord("n"), 13, 10):
            if not points:
                status = "No points set. Press s to skip, or click visible apparatus."
                continue
            cv2.destroyWindow(window)
            return dict(points)
        if key == ord("u"):
            if history:
                last_name = history.pop()
                points.pop(last_name, None)
                selected_idx[0] = POINT_NAMES.index(last_name)
                status = f"Removed {last_name}"
            else:
                status = "Nothing to undo."
            continue
        if key == ord("r"):
            points.clear()
            history.clear()
            selected_idx[0] = 0
            status = "Cleared current frame."
            continue
        if key == ord("s"):
            if points:
                status = "Points are set. Press n to save, or r then s to skip."
                continue
            cv2.destroyWindow(window)
            return None
        if key == ord("q") or key == 27:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("anchor labelling cancelled")


def _frame_indices(
    n_frames: int,
    *,
    requested_frames: int,
    every: int | None,
    start_frame: int | None,
    end_frame: int | None,
) -> list[int]:
    start = max(0, int(start_frame) if start_frame is not None else 0)
    end = min(n_frames - 1, int(end_frame) if end_frame is not None else n_frames - 1)
    if start > end:
        raise ValueError(f"start-frame {start} is after end-frame {end}")
    if every is not None:
        if every <= 0:
            raise ValueError("--every must be positive")
        return list(range(start, end + 1, int(every)))
    return sorted(
        {int(round(v)) for v in np.linspace(start, end, max(1, requested_frames))}
    )


def _complete_label_count(labels: list[dict]) -> int:
    return sum(1 for entry in labels if len(entry.get("points", {})) == len(POINT_NAMES))


def _normalise_points(points: dict) -> dict[str, tuple[float, float]]:
    normalised: dict[str, tuple[float, float]] = {}
    for name in POINT_NAMES:
        value = points.get(name)
        if value is None or len(value) < 2:
            continue
        x = float(value[0])
        y = float(value[1])
        if np.isfinite([x, y]).all():
            normalised[name] = (x, y)
    return normalised


def main() -> None:
    parser = argparse.ArgumentParser(description="Label visible high-jump scene anchors.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/hand_anchors"))
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument(
        "--every",
        type=int,
        default=None,
        help="Label every Nth frame instead of using evenly spaced samples.",
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Discard an existing label file.")
    parser.add_argument(
        "--revisit-skipped",
        action="store_true",
        help="Re-open selected frames even if they were previously skipped.",
    )
    parser.add_argument(
        "--revisit-partial",
        action="store_true",
        help="Re-open selected partial-label frames with their existing points preloaded.",
    )
    parser.add_argument(
        "--revisit-labelled",
        action="store_true",
        help="Re-open selected complete-label frames with their existing points preloaded.",
    )
    args = parser.parse_args()

    n_frames = _video_frame_count(args.video)
    frame_indices = _frame_indices(
        n_frames,
        requested_frames=args.frames,
        every=args.every,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    labels: list[dict] = []
    skipped: list[int] = []
    unreadable: list[int] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.video.stem}.json"
    if out_path.exists() and not args.overwrite:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        labels = list(existing.get("labels", []))
        skipped = [int(v) for v in existing.get("skipped_frames", [])]
        unreadable = [int(v) for v in existing.get("unreadable_frames", [])]
        print(
            f"Resuming {out_path}: {len(labels)} labelled "
            f"({_complete_label_count(labels)} complete), {len(skipped)} skipped, "
            f"{len(unreadable)} unreadable. Use --overwrite to start over."
        )

    def _save() -> None:
        complete = _complete_label_count(labels)
        payload = {
            "label_schema": "scene_anchors_partial_v1",
            "video_path": str(args.video),
            "video_stem": args.video.stem,
            "bar_height_m": parse_bar_height(args.video.name),
            "n_frames": int(n_frames),
            "point_order": list(POINT_NAMES),
            "complete_label_count": int(complete),
            "partial_label_count": int(len(labels) - complete),
            "labels": sorted(labels, key=lambda entry: int(entry["frame"])),
            "skipped_frames": sorted({int(v) for v in skipped}),
            "unreadable_frames": sorted({int(v) for v in unreadable}),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        visited = {
            int(entry["frame"])
            for entry in labels
            if isinstance(entry, dict) and "frame" in entry
        }
        if args.revisit_partial:
            visited = {
                int(entry["frame"])
                for entry in labels
                if isinstance(entry, dict)
                and "frame" in entry
                and len(entry.get("points", {})) == len(POINT_NAMES)
            }
        if args.revisit_labelled:
            visited.clear()
        if not args.revisit_skipped:
            visited.update(int(v) for v in skipped)
        visited.update(int(v) for v in unreadable)
        for frame_idx in frame_indices:
            if int(frame_idx) in visited:
                print(f"Frame {frame_idx}: already labelled/skipped; continuing")
                continue
            frame = _read_frame(args.video, frame_idx)
            if frame is None:
                unreadable.append(int(frame_idx))
                print(f"  frame {frame_idx} unreadable (cv2 cannot decode); continuing")
                _save()
                continue
            print(
                f"Frame {frame_idx}: label any visible points "
                f"({', '.join(POINT_NAMES)}); n saves partial"
            )
            existing_entry = next(
                (
                    entry
                    for entry in labels
                    if isinstance(entry, dict) and int(entry.get("frame", -1)) == int(frame_idx)
                ),
                None,
            )
            existing_points = (
                _normalise_points(existing_entry.get("points", {}))
                if existing_entry is not None
                else None
            )
            points = _collect_points(frame, frame_idx, initial_points=existing_points)
            if points is None:
                labels = [
                    entry
                    for entry in labels
                    if not (
                        isinstance(entry, dict)
                        and int(entry.get("frame", -1)) == int(frame_idx)
                    )
                ]
                skipped.append(int(frame_idx))
                print(f"  frame {frame_idx} skipped (no useful apparatus)")
                _save()
                continue
            point_payload = {
                name: [float(x), float(y)]
                for name, (x, y) in points.items()
                if name in POINT_NAMES
            }
            labels = [
                entry
                for entry in labels
                if not (
                    isinstance(entry, dict)
                    and int(entry.get("frame", -1)) == int(frame_idx)
                )
            ]
            skipped = [int(v) for v in skipped if int(v) != int(frame_idx)]
            labels.append(
                {
                    "frame": int(frame_idx),
                    "points": point_payload,
                    "n_points": int(len(point_payload)),
                    "complete": bool(len(point_payload) == len(POINT_NAMES)),
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
    complete = _complete_label_count(labels)
    print(
        f"Wrote {out_path}  ({len(labels)} labelled frames, {complete} complete, "
        f"{len(labels) - complete} partial, {len(skipped)} skipped, "
        f"{len(unreadable)} unreadable)"
    )


if __name__ == "__main__":
    main()
