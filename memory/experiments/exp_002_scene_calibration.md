# Experiment 002: Scene Calibration Validation

Purpose: compare anatomical, egomotion, and scene-anchor calibration runs on aggregate private-video metrics. Raw per-clip values remain under `data/results/` and are not pasted here.

## Summary

| Mode | n | Training-grade | Kinematics-grade | Method counts | Source counts |
| --- | ---: | ---: | ---: | --- | --- |
| anatomical | 2 | 0 | 2 | {'anatomical': 2} | {'none': 2} |
| egomotion | 2 | 0 | 2 | {'egomotion': 2} | {'egomotion': 2} |
| scene | 2 | 0 | 2 | {'egomotion': 2} | {'egomotion': 2} |

## Acceptance Metrics

| Mode | Anchor mean % | 4-anchor mean % | Egomotion mean % | H-speed median | H-speed 2.5-5.5 | Angle median | Angle 38-48 | Angle 38-55 | CoM-bar median | CoM-bar -0.30..0 | Scene/anat ratio median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anatomical | 0.00 | 0.00 | 0.00 | 0.59 | 0 | 55.55 | 1 | 1 | 0.51 | 0 | n/a |
| egomotion | 0.00 | 0.00 | 100.00 | 1.48 | 0 | 34.85 | 0 | 1 | 0.50 | 0 | n/a |
| scene | 23.16 | 38.84 | 100.00 | 1.48 | 0 | 34.85 | 0 | 1 | 0.50 | 0 | n/a |

## Pass/Fail

### anatomical
- training_grade_count_ge_35: FAIL
- takeoff_horizontal_median_3p5_7p0: FAIL
- takeoff_angle_median_40_55: FAIL
- bar_tagged_subset_30_and_clearance: FAIL

### egomotion
- training_grade_count_ge_35: FAIL
- takeoff_horizontal_median_3p5_7p0: FAIL
- takeoff_angle_median_40_55: FAIL
- bar_tagged_subset_30_and_clearance: FAIL

### scene
- training_grade_count_ge_35: FAIL
- takeoff_horizontal_median_3p5_7p0: FAIL
- takeoff_angle_median_40_55: FAIL
- bar_tagged_subset_30_and_clearance: FAIL
- scene_anchor_four_point_valid_pct_mean_ge_30: PASS

## Decision

insufficient sample size, run wider validation
