"""Evaluate raw stationary-video pose proportions against known measurements.

This diagnostic deliberately runs before anatomical calibration and skeleton
postprocessing. The production pipeline accepts known segment lengths and later
enforces height-derived proportions, so evaluating its final skeleton would be
circular.

Usage:
    python scripts/evaluate_stationary_anthropometry.py data/videos/ \
        --known-leg-m LEG_METRES --known-shank-m SHANK_METRES \
        --known-thigh-m THIGH_METRES --known-arm-m ARM_METRES \
        --output data/results/stationary_validation/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.estimators.mediapipe_estimator import MediaPipeEstimator  # noqa: E402

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# BlazePose chains. Composite lengths are sums of adjacent bone lengths, so
# bent knees and elbows do not artificially shorten the measurement.
SEGMENT_CHAINS = {
    "leg": ((23, 25, 27), (24, 26, 28)),
    "shank": ((25, 27), (26, 28)),
    "thigh": ((23, 25), (24, 26)),
    "arm": ((11, 13, 15), (12, 14, 16)),
}

RATIO_PAIRS = (
    ("thigh", "shank"),
    ("thigh", "leg"),
    ("shank", "leg"),
    ("arm", "leg"),
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def collect_videos(input_paths: list[Path]) -> list[Path]:
    """Collect videos recursively from one or more files or directories."""
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


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _segment_samples(
    landmarks: np.ndarray,
    chains: tuple[tuple[int, ...], ...],
    *,
    dimensions: int,
    coordinate_scales: tuple[float, ...],
    visibility_column: int,
    min_visibility: float,
) -> np.ndarray:
    """Return visible chain lengths from both body sides across all frames."""
    points = np.asarray(landmarks[:, :, :dimensions], dtype=np.float64)
    scales = np.asarray(coordinate_scales, dtype=np.float64)
    if scales.shape != (dimensions,):
        raise ValueError(f"Expected {dimensions} coordinate scales, got {scales.shape}")
    points = points * scales

    visible_lengths: list[np.ndarray] = []
    for chain in chains:
        chain_indices = np.asarray(chain, dtype=np.int64)
        chain_points = points[:, chain_indices, :]
        chain_visibility = landmarks[:, chain_indices, visibility_column]
        lengths = np.linalg.norm(np.diff(chain_points, axis=1), axis=2).sum(axis=1)
        valid = (
            np.all(chain_visibility >= min_visibility, axis=1)
            & np.isfinite(lengths)
            & (lengths > 0)
        )
        visible_lengths.append(lengths[valid])

    if not visible_lengths:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(visible_lengths)


def summarize_segment_lengths(
    landmarks: np.ndarray,
    *,
    dimensions: int,
    coordinate_scales: tuple[float, ...],
    visibility_column: int,
    min_visibility: float = 0.7,
    reducer: str,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize visible raw landmark segment lengths.

    For 2D image projections, use ``reducer="p95"`` to favor frames where a
    segment is closest to the image plane. For raw MediaPipe world landmarks,
    use ``reducer="median"`` because the 3D segment length should be stable.
    """
    if reducer not in {"median", "p95"}:
        raise ValueError(f"Unsupported reducer: {reducer}")

    summaries: dict[str, dict[str, float | int | None]] = {}
    for segment, chains in SEGMENT_CHAINS.items():
        samples = _segment_samples(
            landmarks,
            chains,
            dimensions=dimensions,
            coordinate_scales=coordinate_scales,
            visibility_column=visibility_column,
            min_visibility=min_visibility,
        )
        representative: float | None = None
        iqr: float | None = None
        if samples.size:
            representative = float(
                np.percentile(samples, 95.0) if reducer == "p95" else np.median(samples)
            )
            iqr = float(np.percentile(samples, 75.0) - np.percentile(samples, 25.0))
        summaries[segment] = {
            "representative": _round_or_none(representative),
            "iqr": _round_or_none(iqr),
            "n_observations": int(samples.size),
        }
    return summaries


def estimate_lengths_from_anchor(
    summaries: dict[str, dict[str, float | int | None]],
    *,
    anchor_segment: str,
    anchor_length_m: float,
    known_lengths_m: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    """Scale raw proportions from one known anchor and compare held-out lengths."""
    anchor_observed = summaries[anchor_segment]["representative"]
    estimates: dict[str, dict[str, float | None]] = {}
    for segment in SEGMENT_CHAINS:
        observed = summaries[segment]["representative"]
        estimated: float | None = None
        if isinstance(anchor_observed, (float, int)) and isinstance(observed, (float, int)):
            if anchor_observed > 0:
                estimated = anchor_length_m * observed / anchor_observed
        known = known_lengths_m.get(segment)
        error_pct = None
        if estimated is not None and known is not None:
            error_pct = 100.0 * (estimated - known) / known
        estimates[segment] = {
            "estimated_length_m": _round_or_none(estimated),
            "known_length_m": _round_or_none(known),
            "error_pct": _round_or_none(error_pct, digits=3),
        }
    return estimates


def compare_ratios(
    summaries: dict[str, dict[str, float | int | None]],
    known_lengths_m: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    """Compare raw scale-free ratios with known taped-measurement ratios."""
    comparisons: dict[str, dict[str, float | None]] = {}
    for numerator, denominator in RATIO_PAIRS:
        observed_numerator = summaries[numerator]["representative"]
        observed_denominator = summaries[denominator]["representative"]
        observed_ratio = None
        if (
            isinstance(observed_numerator, (float, int))
            and isinstance(observed_denominator, (float, int))
            and observed_denominator > 0
        ):
            observed_ratio = observed_numerator / observed_denominator
        known_ratio = known_lengths_m[numerator] / known_lengths_m[denominator]
        error_pct = None
        if observed_ratio is not None:
            error_pct = 100.0 * (observed_ratio - known_ratio) / known_ratio
        comparisons[f"{numerator}_to_{denominator}"] = {
            "observed_ratio": _round_or_none(observed_ratio),
            "known_ratio": _round_or_none(known_ratio),
            "error_pct": _round_or_none(error_pct, digits=3),
        }
    return comparisons


def _method_report(
    summaries: dict[str, dict[str, float | int | None]],
    *,
    anchor_segment: str,
    known_lengths_m: dict[str, float],
) -> dict:
    return {
        "segment_summaries": summaries,
        "estimates_from_anchor": estimate_lengths_from_anchor(
            summaries,
            anchor_segment=anchor_segment,
            anchor_length_m=known_lengths_m[anchor_segment],
            known_lengths_m=known_lengths_m,
        ),
        "ratio_checks": compare_ratios(summaries, known_lengths_m),
    }


def evaluate_landmark_arrays(
    landmarks_2d: np.ndarray,
    landmarks_3d: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    anchor_segment: str,
    known_lengths_m: dict[str, float],
    min_visibility: float = 0.7,
) -> dict:
    """Evaluate raw 2D and raw MediaPipe-world proportions for one clip."""
    projected_summaries = summarize_segment_lengths(
        landmarks_2d,
        dimensions=2,
        coordinate_scales=(float(image_width), float(image_height)),
        visibility_column=2,
        min_visibility=min_visibility,
        reducer="p95",
    )
    world_summaries = summarize_segment_lengths(
        landmarks_3d,
        dimensions=3,
        coordinate_scales=(1.0, 1.0, 1.0),
        visibility_column=3,
        min_visibility=min_visibility,
        reducer="median",
    )
    return {
        "projected_2d_p95": _method_report(
            projected_summaries,
            anchor_segment=anchor_segment,
            known_lengths_m=known_lengths_m,
        ),
        "raw_world_3d_median": _method_report(
            world_summaries,
            anchor_segment=anchor_segment,
            known_lengths_m=known_lengths_m,
        ),
    }


def _median_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"median": None, "iqr": None, "n_clips": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": _round_or_none(float(np.median(array))),
        "iqr": _round_or_none(float(np.percentile(array, 75) - np.percentile(array, 25))),
        "n_clips": int(array.size),
    }


def aggregate_clip_reports(clips: list[dict], known_lengths_m: dict[str, float]) -> dict:
    """Aggregate per-clip estimates with equal weight for each source video."""
    aggregate: dict[str, dict] = {}
    for method in ("projected_2d_p95", "raw_world_3d_median"):
        estimates: dict[str, dict] = {}
        for segment in SEGMENT_CHAINS:
            values = [
                clip["anthropometry"][method]["estimates_from_anchor"][segment][
                    "estimated_length_m"
                ]
                for clip in clips
            ]
            finite_values = [float(value) for value in values if value is not None]
            summary = _median_summary(finite_values)
            known = known_lengths_m[segment]
            median = summary["median"]
            summary["known_length_m"] = _round_or_none(known)
            summary["error_pct"] = (
                _round_or_none(100.0 * (float(median) - known) / known, digits=3)
                if median is not None
                else None
            )
            estimates[segment] = summary

        ratios: dict[str, dict] = {}
        for numerator, denominator in RATIO_PAIRS:
            ratio_name = f"{numerator}_to_{denominator}"
            values = [
                clip["anthropometry"][method]["ratio_checks"][ratio_name]["observed_ratio"]
                for clip in clips
            ]
            finite_values = [float(value) for value in values if value is not None]
            summary = _median_summary(finite_values)
            known_ratio = known_lengths_m[numerator] / known_lengths_m[denominator]
            median = summary["median"]
            summary["known_ratio"] = _round_or_none(known_ratio)
            summary["error_pct"] = (
                _round_or_none(
                    100.0 * (float(median) - known_ratio) / known_ratio,
                    digits=3,
                )
                if median is not None
                else None
            )
            ratios[ratio_name] = summary

        aggregate[method] = {
            "estimates_from_anchor": estimates,
            "ratio_checks": ratios,
        }
    return aggregate


def _extract_raw_landmarks(video_path: Path, *, roi_crop: bool) -> tuple:
    import cv2

    estimator = MediaPipeEstimator(model_complexity=2)
    sequence = estimator.process_video(video_path, roi_crop=roi_crop)
    if not sequence.frames:
        raise ValueError(f"No decoded frames in {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    landmarks_2d = np.stack([frame.landmarks_2d for frame in sequence.frames])
    if not all(frame.landmarks_3d is not None for frame in sequence.frames):
        raise ValueError(f"Missing world landmarks in {video_path.name}")
    landmarks_3d = np.stack([frame.landmarks_3d for frame in sequence.frames])
    valid_frames = sum(frame.is_valid for frame in sequence.frames)
    return landmarks_2d, landmarks_3d, sequence.fps, width, height, valid_frames


def evaluate_videos(
    videos: list[Path],
    *,
    anchor_segment: str,
    known_lengths_m: dict[str, float],
    min_visibility: float,
    roi_crop: bool,
) -> dict:
    """Run the raw anthropometry diagnostic across videos."""
    clips: list[dict] = []
    for index, video_path in enumerate(videos, start=1):
        logger.info("Processing clip %d/%d", index, len(videos))
        landmarks_2d, landmarks_3d, fps, width, height, valid_frames = (
            _extract_raw_landmarks(video_path, roi_crop=roi_crop)
        )
        clips.append(
            {
                "clip": f"clip_{index:02d}",
                "video_name": video_path.name,
                "frames": int(landmarks_2d.shape[0]),
                "fps": _round_or_none(fps, digits=3),
                "pose_valid_frames": int(valid_frames),
                "pose_validity_pct": _round_or_none(
                    100.0 * valid_frames / landmarks_2d.shape[0],
                    digits=3,
                ),
                "anthropometry": evaluate_landmark_arrays(
                    landmarks_2d,
                    landmarks_3d,
                    image_width=width,
                    image_height=height,
                    anchor_segment=anchor_segment,
                    known_lengths_m=known_lengths_m,
                    min_visibility=min_visibility,
                ),
            }
        )

    return {
        "diagnostic": "raw_stationary_pose_anthropometry",
        "roi_crop": roi_crop,
        "anchor_segment": anchor_segment,
        "known_anchor_length_m": known_lengths_m[anchor_segment],
        "min_visibility": min_visibility,
        "n_clips": len(clips),
        "clips": clips,
        "aggregate": aggregate_clip_reports(clips, known_lengths_m),
    }


def _print_summary(report: dict) -> None:
    print(
        f"Raw stationary anthropometry: {report['n_clips']} clips, "
        f"ROI crop={'on' if report['roi_crop'] else 'off'}, "
        f"anchor={report['anchor_segment']}"
    )
    for clip in report["clips"]:
        print(
            f"  {clip['clip']}: {clip['frames']} frames, "
            f"pose validity={clip['pose_validity_pct']:.1f}%"
        )

    for method, label in (
        ("raw_world_3d_median", "Raw MediaPipe world 3D"),
        ("projected_2d_p95", "2D projected p95"),
    ):
        print(f"\n{label} estimates from the held-out anchor:")
        estimates = report["aggregate"][method]["estimates_from_anchor"]
        for segment in SEGMENT_CHAINS:
            values = estimates[segment]
            if values["median"] is None:
                print(f"  {segment:>5}: unavailable")
                continue
            print(
                f"  {segment:>5}: {values['median']:.3f} m "
                f"(known {values['known_length_m']:.3f} m, "
                f"error {values['error_pct']:+.1f}%, n={values['n_clips']})"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Video files or directories")
    parser.add_argument("--known-leg-m", required=True, type=_positive_float)
    parser.add_argument("--known-shank-m", required=True, type=_positive_float)
    parser.add_argument("--known-thigh-m", required=True, type=_positive_float)
    parser.add_argument("--known-arm-m", required=True, type=_positive_float)
    parser.add_argument(
        "--anchor-segment",
        choices=tuple(SEGMENT_CHAINS),
        default="leg",
        help="Single known measurement used to scale held-out estimates",
    )
    parser.add_argument("--min-visibility", default=0.7, type=float)
    parser.add_argument(
        "--roi-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run MediaPipe's two-pass stationary-video ROI crop",
    )
    parser.add_argument("--output", type=Path, help="Optional local JSON output path")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = build_parser().parse_args()
    if not 0 <= args.min_visibility <= 1:
        raise ValueError("--min-visibility must be between 0 and 1")
    videos = collect_videos(args.inputs)
    if not videos:
        raise ValueError("No video files found")

    known_lengths_m = {
        "leg": args.known_leg_m,
        "shank": args.known_shank_m,
        "thigh": args.known_thigh_m,
        "arm": args.known_arm_m,
    }
    report = evaluate_videos(
        videos,
        anchor_segment=args.anchor_segment,
        known_lengths_m=known_lengths_m,
        min_visibility=args.min_visibility,
        roi_crop=args.roi_crop,
    )
    _print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote local report: {args.output}")


if __name__ == "__main__":
    main()
