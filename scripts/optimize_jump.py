"""Optimise a high jump from video analysis results.

Takes a video analysis report (from analyze_jump_video.py) and produces:
  1. Predicted bar clearance from current technique
  2. Optimised technique parameters
  3. Sensitivity analysis (which changes matter most)
  4. What-if scenarios
  5. Ranked coaching cues

Usage:
    python scripts/optimize_jump.py data/results/all_sessions_report.json
    python scripts/optimize_jump.py data/results/all_sessions_report.json --video "14_02_26_one_1.79"
    python scripts/optimize_jump.py data/results/all_sessions_report.json --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.optimizer import (
    AthleteConstraints,
    TechniqueParameters,
    compute_sensitivity,
    extract_params_from_report,
    optimize_technique,
    predict_bar_clearance,
    what_if_scenario,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IMOGEN_MASS = 67.0
IMOGEN_HEIGHT = 1.78


def optimize_single_video(
    report: dict,
    body_mass_kg: float = IMOGEN_MASS,
    athlete_height_m: float = IMOGEN_HEIGHT,
    n_iterations: int = 200,
) -> dict:
    """Run full optimisation pipeline on one video analysis report.

    Returns a results dict with current prediction, optimal params,
    sensitivity, what-if scenarios, and coaching cues.
    """
    video_name = report.get("video", "unknown")
    bar_height = report.get("bar_height_m")
    session = report.get("session_date", "")

    # 1. Extract current technique from the video report
    current_params = extract_params_from_report(report)

    # 2. Predict current bar clearance from the extracted technique
    current_pred = predict_bar_clearance(current_params, body_mass_kg, athlete_height_m)

    # 3. Run gradient-based optimisation
    result = optimize_technique(
        current_params=current_params,
        body_mass_kg=body_mass_kg,
        athlete_height_m=athlete_height_m,
        n_iterations=n_iterations,
    )

    # 4. Compute sensitivity at current parameters
    sensitivity = compute_sensitivity(
        current_params.to_tensor(), body_mass_kg, athlete_height_m,
    )

    # 5. Run key what-if scenarios
    what_ifs = {}

    # What if approach speed +0.5 m/s?
    what_ifs["speed_plus_0.5"] = what_if_scenario(
        current_params, body_mass_kg, athlete_height_m,
        {"approach_speed_mps": current_params.approach_speed_mps + 0.5},
    )

    # What if approach speed +1.0 m/s?
    what_ifs["speed_plus_1.0"] = what_if_scenario(
        current_params, body_mass_kg, athlete_height_m,
        {"approach_speed_mps": current_params.approach_speed_mps + 1.0},
    )

    # What if knee drive improves by 0.5 m/s?
    what_ifs["knee_drive_plus_0.5"] = what_if_scenario(
        current_params, body_mass_kg, athlete_height_m,
        {"knee_drive_peak_speed_mps": current_params.knee_drive_peak_speed_mps + 0.5},
    )

    # What if contact time decreases by 20 ms?
    new_ct = max(80, current_params.ground_contact_time_takeoff_ms - 20)
    what_ifs["contact_time_minus_20ms"] = what_if_scenario(
        current_params, body_mass_kg, athlete_height_m,
        {"ground_contact_time_takeoff_ms": new_ct},
    )

    # Compile output
    output = {
        "video": video_name,
        "session_date": session,
        "bar_height_actual_m": bar_height,
        "athlete": {
            "mass_kg": body_mass_kg,
            "height_m": athlete_height_m,
        },
        "current_technique": {
            "approach_speed_mps": current_params.approach_speed_mps,
            "plant_angle_deg": current_params.plant_angle_deg,
            "takeoff_knee_angle_deg": current_params.takeoff_knee_angle_deg,
            "takeoff_hip_angle_deg": current_params.takeoff_hip_angle_deg,
            "ground_contact_time_ms": current_params.ground_contact_time_takeoff_ms,
            "knee_drive_peak_speed_mps": current_params.knee_drive_peak_speed_mps,
        },
        "current_prediction": {
            "predicted_bar_m": round(current_pred["predicted_bar_height_m"], 3),
            "v_vertical_mps": round(current_pred["v_vertical_mps"], 2),
            "v_horizontal_mps": round(current_pred["v_horizontal_mps"], 2),
            "takeoff_angle_deg": round(current_pred["takeoff_angle_deg"], 1),
            "h_takeoff_m": round(current_pred["h_takeoff_m"], 3),
            "h_rise_m": round(current_pred["h_rise_m"], 3),
        },
        "optimised": {
            "predicted_bar_m": round(result.predicted_height_m, 3),
            "improvement_cm": round(result.improvement_cm, 1),
            "optimal_approach_speed_mps": round(
                result.optimal_params.approach_speed_mps, 2,
            ),
            "optimal_plant_angle_deg": round(
                result.optimal_params.plant_angle_deg, 1,
            ),
            "optimal_knee_angle_deg": round(
                result.optimal_params.takeoff_knee_angle_deg, 1,
            ),
            "optimal_contact_time_ms": round(
                result.optimal_params.ground_contact_time_takeoff_ms, 0,
            ),
        },
        "sensitivity_cm_per_unit": {
            k: round(v, 2) for k, v in sorted(
                sensitivity.items(), key=lambda x: abs(x[1]), reverse=True,
            )
        },
        "what_if_scenarios": {
            name: {
                "delta_cm": round(s["delta_cm"], 1),
                "modified_height_m": round(s["modified_height_m"], 3),
            }
            for name, s in what_ifs.items()
        },
        "coaching_cues": result.coaching_cues,
    }

    return output


def print_optimization_report(output: dict) -> None:
    """Print a human-readable optimization report to console."""
    print(f"\n{'=' * 70}")
    print(f"  OPTIMISATION REPORT: {output['video']}")
    if output.get("session_date"):
        print(f"  Session: {output['session_date']}")
    if output.get("bar_height_actual_m"):
        print(f"  Actual bar height: {output['bar_height_actual_m']} m")
    print(f"{'=' * 70}")

    cp = output["current_prediction"]
    print(f"\n  CURRENT TECHNIQUE → PREDICTED PERFORMANCE")
    print(f"  {'─' * 50}")
    ct = output["current_technique"]
    print(f"  Approach speed:      {ct['approach_speed_mps']:.1f} m/s")
    print(f"  Plant angle:         {ct['plant_angle_deg']:.0f}°")
    print(f"  Knee angle:          {ct['takeoff_knee_angle_deg']:.0f}°")
    print(f"  Contact time:        {ct['ground_contact_time_ms']:.0f} ms")
    print(f"  Knee drive speed:    {ct['knee_drive_peak_speed_mps']:.1f} m/s")
    print(f"  {'─' * 50}")
    print(f"  Predicted bar:       {cp['predicted_bar_m']:.3f} m")
    print(f"  Takeoff angle:       {cp['takeoff_angle_deg']:.1f}°")
    print(f"  Vertical velocity:   {cp['v_vertical_mps']:.2f} m/s")
    print(f"  Horizontal velocity: {cp['v_horizontal_mps']:.2f} m/s")
    print(f"  CoM at takeoff:      {cp['h_takeoff_m']:.3f} m")
    print(f"  CoM rise:            {cp['h_rise_m']:.3f} m")

    opt = output["optimised"]
    print(f"\n  OPTIMISATION RESULT")
    print(f"  {'─' * 50}")
    print(f"  Optimised bar:       {opt['predicted_bar_m']:.3f} m")
    print(f"  Improvement:         +{opt['improvement_cm']:.1f} cm")
    print(f"  Optimal speed:       {opt['optimal_approach_speed_mps']:.2f} m/s")
    print(f"  Optimal plant angle: {opt['optimal_plant_angle_deg']:.1f}°")
    print(f"  Optimal knee angle:  {opt['optimal_knee_angle_deg']:.1f}°")
    print(f"  Optimal contact:     {opt['optimal_contact_time_ms']:.0f} ms")

    print(f"\n  SENSITIVITY ANALYSIS (cm per unit change)")
    print(f"  {'─' * 50}")
    for name, val in output["sensitivity_cm_per_unit"].items():
        if abs(val) >= 0.1:
            direction = "↑" if val > 0 else "↓"
            print(f"  {name:<35} {direction} {abs(val):>6.2f} cm")

    print(f"\n  WHAT-IF SCENARIOS")
    print(f"  {'─' * 50}")
    labels = {
        "speed_plus_0.5": "Approach speed +0.5 m/s",
        "speed_plus_1.0": "Approach speed +1.0 m/s",
        "knee_drive_plus_0.5": "Knee drive speed +0.5 m/s",
        "contact_time_minus_20ms": "Contact time -20 ms",
    }
    for key, scenario in output["what_if_scenarios"].items():
        label = labels.get(key, key)
        delta = scenario["delta_cm"]
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<35} {sign}{delta:.1f} cm → {scenario['modified_height_m']:.3f} m")

    print(f"\n  COACHING CUES")
    print(f"  {'─' * 50}")
    for cue in output["coaching_cues"]:
        print(f"  • {cue}")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Optimise technique from video analysis results",
    )
    parser.add_argument(
        "report_json", type=str,
        help="Path to video analysis report JSON (single or batch)",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Name of specific video to optimise (from batch report)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Optimise all videos in a batch report",
    )
    parser.add_argument(
        "--mass", type=float, default=IMOGEN_MASS,
        help="Athlete mass in kg",
    )
    parser.add_argument(
        "--height", type=float, default=IMOGEN_HEIGHT,
        help="Athlete height in metres",
    )
    parser.add_argument(
        "--iterations", type=int, default=200,
        help="Number of optimisation iterations",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path for optimisation results",
    )
    args = parser.parse_args()

    report_path = Path(args.report_json)
    if not report_path.exists():
        logger.error(f"Report not found: {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        data = json.load(f)

    # Normalise to list
    if isinstance(data, dict):
        reports = [data]
    else:
        reports = data

    # Filter to requested video
    if args.video:
        reports = [r for r in reports if r.get("video") == args.video]
        if not reports:
            logger.error(f"Video '{args.video}' not found in report")
            logger.info(f"Available videos: {[r.get('video') for r in data]}")
            sys.exit(1)
    elif not args.all and len(reports) > 1:
        # Default: pick the best jump (highest PINN GRF as proxy for quality)
        reports.sort(key=lambda r: r.get("pinn_peak_vertical_grf_BW", 0), reverse=True)
        reports = [reports[0]]
        logger.info(f"Selecting best jump: {reports[0].get('video')}")
        logger.info("Use --all to optimise all videos, or --video <name> for a specific one")

    all_results = []
    for report in reports:
        try:
            result = optimize_single_video(
                report,
                body_mass_kg=args.mass,
                athlete_height_m=args.height,
                n_iterations=args.iterations,
            )
            all_results.append(result)
            print_optimization_report(result)
        except Exception as e:
            logger.error(f"Failed to optimise {report.get('video')}: {e}")
            continue

    # Save results
    if all_results:
        if args.output:
            out_path = Path(args.output)
        else:
            out_dir = Path("data") / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            if len(all_results) == 1:
                out_path = out_dir / f"{all_results[0]['video']}_optimization.json"
            else:
                out_path = out_dir / "all_optimizations.json"

        with open(out_path, "w") as f:
            json.dump(
                all_results if len(all_results) > 1 else all_results[0],
                f, indent=2,
            )
        logger.info(f"Results saved: {out_path}")

    # Print summary table for batch
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print(f"  BATCH SUMMARY: {len(all_results)} videos optimised")
        print(f"{'=' * 80}")
        print(f"  {'Video':<35} {'Current':>8} {'Optimal':>8} {'Gain':>6}")
        print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 6}")
        for r in all_results:
            name = r["video"][:35]
            curr = f"{r['current_prediction']['predicted_bar_m']:.3f}"
            opt_h = f"{r['optimised']['predicted_bar_m']:.3f}"
            gain = f"+{r['optimised']['improvement_cm']:.1f}"
            print(f"  {name:<35} {curr:>8} {opt_h:>8} {gain:>6}")


if __name__ == "__main__":
    main()
