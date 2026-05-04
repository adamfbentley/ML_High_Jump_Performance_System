"""Probe high-jump scene-anchor detections on a small local video sample.

This script is for local QA before any private-video reprocess.  It writes
annotated frames under ``data/results/anchor_probe/``; that directory is private
and gitignored through ``data/results/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_jump_video import collect_videos, extract_poses, parse_bar_height  # noqa: E402
from src.pose_estimation.scale_calibration import compute_per_frame_scale_mpp  # noqa: E402
from src.pose_estimation.scene_calibration import (  # noqa: E402
    SceneAnchors,
    detect_scene_anchors,
    estimate_scene_scale_mpp,
    scene_anchor_diagnostics,
)


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    width: int
    height: int
    n_frames: int
    bar_height_m: float | None

    @property
    def area(self) -> int:
        return self.width * self.height


def _video_meta(video_path: Path) -> VideoMeta | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or n_frames <= 0:
        return None
    return VideoMeta(
        path=video_path,
        width=width,
        height=height,
        n_frames=n_frames,
        bar_height_m=parse_bar_height(video_path.name),
    )


def _spread_pick(items: list[VideoMeta], n_items: int) -> list[VideoMeta]:
    if n_items <= 0 or not items:
        return []
    if len(items) <= n_items:
        return items.copy()
    positions = np.linspace(0, len(items) - 1, n_items)
    picked: list[VideoMeta] = []
    used: set[Path] = set()
    for pos in positions:
        item = items[int(round(float(pos)))]
        if item.path not in used:
            picked.append(item)
            used.add(item.path)
    for item in items:
        if len(picked) >= n_items:
            break
        if item.path not in used:
            picked.append(item)
            used.add(item.path)
    return picked


def select_probe_videos(
    input_dir: Path,
    n_clips: int = 5,
    min_bar_tagged: int = 2,
) -> list[VideoMeta]:
    """Select a deterministic small sample with bar-tagged and varied framing proxies."""
    metas = [_video_meta(path) for path in collect_videos(input_dir)]
    valid_metas = sorted((meta for meta in metas if meta is not None), key=lambda m: m.area)
    if not valid_metas:
        raise RuntimeError(f"No readable videos found under {input_dir}")

    selected: list[VideoMeta] = []
    used: set[Path] = set()

    tagged = [meta for meta in valid_metas if meta.bar_height_m is not None]
    for meta in _spread_pick(tagged, min(min_bar_tagged, len(tagged))):
        selected.append(meta)
        used.add(meta.path)

    remaining = [meta for meta in valid_metas if meta.path not in used]
    for meta in _spread_pick(remaining, n_clips - len(selected)):
        selected.append(meta)
        used.add(meta.path)

    if len(selected) < n_clips:
        raise RuntimeError(f"Only selected {len(selected)} readable videos; need {n_clips}")
    return selected[:n_clips]


def _frame_indices(n_frames: int) -> list[int]:
    last = max(0, n_frames - 1)
    return sorted({0, int(round(last * 0.33)), int(round(last * 0.66)), last})


def _anchor_point(arr: np.ndarray | None, frame_idx: int) -> tuple[int, int] | None:
    if arr is None or frame_idx >= len(arr):
        return None
    point = np.asarray(arr[frame_idx], dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return int(round(float(point[0]))), int(round(float(point[1])))


def _draw_probe_frame(frame_bgr: np.ndarray, anchors: SceneAnchors, frame_idx: int) -> np.ndarray:
    annotated = frame_bgr.copy()
    base_colour = (0, 0, 255)
    top_colour = (255, 0, 0)
    base_points = [
        _anchor_point(anchors.upright_left_base_px, frame_idx),
        _anchor_point(anchors.upright_right_base_px, frame_idx),
    ]
    top_points = [
        _anchor_point(anchors.upright_left_top_px, frame_idx),
        _anchor_point(anchors.upright_right_top_px, frame_idx),
    ]

    for point in base_points:
        if point is not None:
            cv2.circle(annotated, point, 12, base_colour, thickness=-1)
    for point in top_points:
        if point is not None:
            cv2.circle(annotated, point, 12, top_colour, thickness=-1)
    if top_points[0] is not None and top_points[1] is not None:
        cv2.line(annotated, top_points[0], top_points[1], top_colour, thickness=2)
    return annotated


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def _valid_anchor_mask(anchors: SceneAnchors) -> np.ndarray:
    n_frames = len(anchors.confidence)
    valid = np.asarray(anchors.confidence, dtype=np.float64) >= 0.5
    for arr in (
        anchors.upright_left_base_px,
        anchors.upright_right_base_px,
        anchors.upright_left_top_px,
        anchors.upright_right_top_px,
    ):
        if arr is None:
            return np.zeros(n_frames, dtype=bool)
        valid &= np.isfinite(arr).all(axis=1)
    return valid


def _scene_anatomical_ratio(
    video_path: Path,
    anchors: SceneAnchors,
    valid: np.ndarray,
    thigh_length_m: float,
    shank_length_m: float,
) -> float | None:
    landmarks_2d, _landmarks_3d, _fps, width, height = extract_poses(video_path)
    anatomical_mpp, _scale_info = compute_per_frame_scale_mpp(
        landmarks_2d,
        image_width=width,
        image_height=height,
        thigh_length_m=thigh_length_m,
        shank_length_m=shank_length_m,
    )
    if anatomical_mpp is None:
        return None
    n_frames = min(len(valid), len(anatomical_mpp))
    scene_mpp = estimate_scene_scale_mpp(anchors)[:n_frames]
    ratio_valid = valid[:n_frames] & np.isfinite(scene_mpp) & np.isfinite(anatomical_mpp[:n_frames])
    ratio_valid &= anatomical_mpp[:n_frames] > 0
    if not np.any(ratio_valid):
        return None
    return float(np.nanmedian(scene_mpp[ratio_valid] / anatomical_mpp[:n_frames][ratio_valid]))


def probe_video(
    meta: VideoMeta,
    output_dir: Path,
    thigh_length_m: float,
    shank_length_m: float,
    detection_stride: int,
    max_detection_width: int,
) -> list[str]:
    anchors = detect_scene_anchors(
        meta.path,
        bar_height_m=meta.bar_height_m,
        detection_stride=detection_stride,
        max_detection_width=max_detection_width,
    )
    valid = _valid_anchor_mask(anchors)
    ratio = _scene_anatomical_ratio(meta.path, anchors, valid, thigh_length_m, shank_length_m)

    valid_frames = int(valid.sum())
    total_frames = int(len(valid))
    coverage_pct = float(valid.mean() * 100.0) if total_frames else 0.0
    mean_confidence = float(np.nanmean(anchors.confidence)) if total_frames else 0.0
    ratio_text = "nan" if ratio is None else f"{ratio:.3f}"
    diagnostics = scene_anchor_diagnostics(anchors)

    lines = [
        (
            f"CLIP {meta.path.stem}: "
            f"anchor_four_point_valid_pct={diagnostics['anchor_four_point_valid_pct']:.2f}% "
            f"coverage={coverage_pct:.2f}% "
            f"crossbar_confirmed_pct={diagnostics['crossbar_confirmed_pct']:.2f}% "
            f"mean_conf={mean_confidence:.3f} valid={valid_frames}/{total_frames} "
            f"scene_anatomical_mpp_ratio={ratio_text}"
        )
    ]

    for slot, frame_idx in enumerate(_frame_indices(meta.n_frames)):
        frame_bgr = _read_frame(meta.path, frame_idx)
        annotated = _draw_probe_frame(frame_bgr, anchors, frame_idx)
        out_path = output_dir / f"{meta.path.stem}_{frame_idx}.png"
        cv2.imwrite(str(out_path), annotated)
        confidence = float(anchors.confidence[frame_idx]) if frame_idx < total_frames else 0.0
        valid_text = bool(valid[frame_idx]) if frame_idx < total_frames else False
        lines.append(
            f"FRAME {meta.path.stem} slot={slot} idx={frame_idx} "
            f"valid={valid_text} confidence={confidence:.3f}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe scene-anchor detection on five local clips."
    )
    parser.add_argument("--input", type=Path, default=Path("data/High Jump Videos"))
    parser.add_argument("--output", type=Path, default=Path("data/results/anchor_probe"))
    parser.add_argument("--thigh", type=float, default=0.43)
    parser.add_argument("--shank", type=float, default=0.47)
    parser.add_argument("--scene-detection-stride", type=int, default=15)
    parser.add_argument("--scene-max-detection-width", type=int, default=640)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    selected = select_probe_videos(args.input)
    for meta in selected:
        for line in probe_video(
            meta,
            args.output,
            args.thigh,
            args.shank,
            args.scene_detection_stride,
            args.scene_max_detection_width,
        ):
            print(line)
    print(f"anchor probe complete, {len(selected)} clips probed, see {args.output}/")


if __name__ == "__main__":
    main()
