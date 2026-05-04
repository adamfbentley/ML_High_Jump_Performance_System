"""Aggregate anatomical, egomotion, and scene calibration validation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

MODE_DIRS = {
    "anatomical": Path("data/results/anatomical/all_sessions_report.json"),
    "egomotion": Path("data/results/egomotion/all_sessions_report.json"),
    "scene": Path("data/results/scene/all_sessions_report.json"),
}
HAND_LABEL_DIR = Path("data/results/hand_anchors")


def _load_reports(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        return list(data)

    single_clip_reports = sorted(
        p for p in path.parent.glob("*_report.json")
        if p.name != "all_sessions_report.json"
    )
    if single_clip_reports:
        return [
            json.loads(report_path.read_text(encoding="utf-8"))
            for report_path in single_clip_reports
        ]
    raise FileNotFoundError(path)


def _values(reports: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    out: list[float] = []
    for report in reports:
        value: Any = report
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _accepted_scene_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        report for report in reports
        if report.get("calibration", {}).get("scene_anchor_decision_reason") == "accepted"
    ]


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _count_between(values: list[float], lo: float, hi: float) -> int:
    return sum(1 for value in values if lo <= value <= hi)


def _mode_counts(reports: list[dict[str, Any]], field_path: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in reports:
        value: Any = report
        for key in field_path:
            value = value.get(key) if isinstance(value, dict) else None
        label = str(value) if value is not None else "missing"
        counts[label] = counts.get(label, 0) + 1
    return counts


def _summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    h_speed = _values(reports, ("velocity", "takeoff_horizontal_mps"))
    angles = _values(reports, ("velocity", "takeoff_angle_deg"))
    anchor_coverage = _values(reports, ("calibration", "anchor_coverage_pct"))
    anchor_four_point_valid = _values(
        reports, ("calibration", "anchor_four_point_valid_pct")
    )
    egomotion_coverage = _values(reports, ("calibration", "egomotion_coverage_pct"))
    scene_ratio = _values(
        _accepted_scene_reports(reports),
        ("calibration", "scene_anatomical_scale_ratio"),
    )
    com_minus_bar = _values(reports, ("com", "com_minus_bar_m"))

    quality = [report.get("quality", {}) for report in reports]
    training_count = sum(1 for q in quality if q.get("training_grade") is True)
    kinematics_count = sum(1 for q in quality if q.get("kinematics_grade") is True)

    return {
        "n_reports": len(reports),
        "training_grade_count": training_count,
        "kinematics_grade_count": kinematics_count,
        "calibration_methods": _mode_counts(reports, ("calibration", "method")),
        "quality_sources": _mode_counts(reports, ("quality", "scene_fixed_horizontal_source")),
        "anchor_coverage_mean_pct": _mean(anchor_coverage),
        "anchor_coverage_median_pct": _median(anchor_coverage),
        "anchor_four_point_valid_pct_mean": _mean(anchor_four_point_valid),
        "egomotion_coverage_mean_pct": _mean(egomotion_coverage),
        "egomotion_coverage_median_pct": _median(egomotion_coverage),
        "takeoff_horizontal_median_mps": _median(h_speed),
        "takeoff_horizontal_p25_mps": _quantile(h_speed, 0.25),
        "takeoff_horizontal_p75_mps": _quantile(h_speed, 0.75),
        "takeoff_horizontal_count_2p5_5p5": _count_between(h_speed, 2.5, 5.5),
        "takeoff_angle_median_deg": _median(angles),
        "takeoff_angle_count_38_48": _count_between(angles, 38.0, 48.0),
        "takeoff_angle_count_38_55": _count_between(angles, 38.0, 55.0),
        "bar_tagged_count": len(com_minus_bar),
        "com_minus_bar_median_m": _median(com_minus_bar),
        "com_minus_bar_count_neg0p30_0": _count_between(com_minus_bar, -0.30, 0.0),
        "scene_anatomical_ratio_median": _median(scene_ratio),
        "scene_anatomical_ratio_p10": _quantile(scene_ratio, 0.10),
        "scene_anatomical_ratio_p90": _quantile(scene_ratio, 0.90),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ratio(count: int, total: int) -> float:
    return float(count / total) if total > 0 else 0.0


def _shared_pass_fail(summary: dict[str, Any]) -> dict[str, bool]:
    h_median = summary["takeoff_horizontal_median_mps"]
    angle_median = summary["takeoff_angle_median_deg"]
    bar_count = int(summary["bar_tagged_count"])
    com_bar_median = summary["com_minus_bar_median_m"]
    com_bar_ratio = _ratio(int(summary["com_minus_bar_count_neg0p30_0"]), bar_count)

    return {
        "training_grade_count_ge_35": summary["training_grade_count"] >= 35,
        "takeoff_horizontal_median_3p5_7p0": (
            h_median is not None and 3.5 <= h_median <= 7.0
        ),
        "takeoff_angle_median_40_55": (
            angle_median is not None and 40.0 <= angle_median <= 55.0
        ),
        "bar_tagged_subset_30_and_clearance": (
            bar_count >= 30
            and com_bar_median is not None
            and -0.20 <= com_bar_median <= -0.05
            and com_bar_ratio >= 0.60
        ),
    }


def _pass_fail(summary: dict[str, Any], mode: str) -> dict[str, bool]:
    checks = _shared_pass_fail(summary)
    if mode == "scene":
        checks["scene_anchor_four_point_valid_pct_mean_ge_30"] = (
            summary["anchor_four_point_valid_pct_mean"] or 0.0
        ) >= 30.0
    return checks


def _truth_evaluator_has_run() -> bool:
    return HAND_LABEL_DIR.exists() and any(HAND_LABEL_DIR.glob("*.json"))


def _best_mode(mode_summaries: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    ranked = sorted(
        mode_summaries.items(),
        key=lambda item: (
            sum(_shared_pass_fail(item[1]).values()),
            item[1].get("training_grade_count", 0),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else ("missing", {})


def _decision(mode_summaries: dict[str, dict[str, Any]]) -> str:
    if any(summary.get("n_reports", 0) < 30 for summary in mode_summaries.values()):
        return "insufficient sample size, run wider validation"
    best_mode, best = _best_mode(mode_summaries)
    checks = _pass_fail(best, best_mode) if best else {}
    if all(checks.values()):
        return "Phase 10 unblocked"
    scene_anchor_pct = (
        mode_summaries.get("scene", {}).get("anchor_four_point_valid_pct_mean") or 0.0
    )
    if scene_anchor_pct < 30.0 and not _truth_evaluator_has_run():
        return "needs hand-labelled anchors"
    return "escalate to fixed-camera filming"


def write_markdown(
    mode_summaries: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Experiment 002: Scene Calibration Validation",
        "",
        "Purpose: compare anatomical, egomotion, and scene-anchor calibration "
        "runs on aggregate private-video metrics. Raw per-clip values remain "
        "under `data/results/` and are not pasted here.",
        "",
        "## Summary",
        "",
        "| Mode | n | Training-grade | Kinematics-grade | Method counts | Source counts |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for mode, summary in mode_summaries.items():
        lines.append(
            f"| {mode} | {summary['n_reports']} | {summary['training_grade_count']} | "
            f"{summary['kinematics_grade_count']} | {summary['calibration_methods']} | "
            f"{summary['quality_sources']} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance Metrics",
            "",
            "| Mode | Anchor mean % | 4-anchor mean % | Egomotion mean % | H-speed median | H-speed 2.5-5.5 | Angle median | Angle 38-48 | Angle 38-55 | CoM-bar median | CoM-bar -0.30..0 | Scene/anat ratio median |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, summary in mode_summaries.items():
        lines.append(
            f"| {mode} | {_fmt(summary['anchor_coverage_mean_pct'])} | "
            f"{_fmt(summary['anchor_four_point_valid_pct_mean'])} | "
            f"{_fmt(summary['egomotion_coverage_mean_pct'])} | "
            f"{_fmt(summary['takeoff_horizontal_median_mps'])} | "
            f"{summary['takeoff_horizontal_count_2p5_5p5']} | "
            f"{_fmt(summary['takeoff_angle_median_deg'])} | "
            f"{summary['takeoff_angle_count_38_48']} | "
            f"{summary['takeoff_angle_count_38_55']} | "
            f"{_fmt(summary['com_minus_bar_median_m'])} | "
            f"{summary['com_minus_bar_count_neg0p30_0']} | "
            f"{_fmt(summary['scene_anatomical_ratio_median'])} |"
        )

    lines.extend(["", "## Pass/Fail", ""])
    for mode, summary in mode_summaries.items():
        lines.append(f"### {mode}")
        for name, passed in _pass_fail(summary, mode).items():
            lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
        lines.append("")

    lines.extend(["## Decision", "", _decision(mode_summaries), ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three calibration validation runs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("memory/experiments/exp_002_scene_calibration.md"),
    )
    args = parser.parse_args()

    summaries = {
        mode: _summary(_load_reports(path))
        for mode, path in MODE_DIRS.items()
    }
    write_markdown(summaries, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
