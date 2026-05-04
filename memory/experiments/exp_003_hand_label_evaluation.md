# Experiment 003: Hand-Label Calibration Evaluation

**Date:** 2026-05-04. Aggregates only — no per-clip values, no video paths,
no session dates (private athlete data).

## Purpose

Validate whether anatomical, egomotion, or detector-fitted scene homography
recovers the athlete's takeoff horizontal velocity correctly on private
high-jump footage. The earlier `exp_002` aggregator finding ("escalate to
fixed-camera filming") was generated on n=2 smoke reports with a structurally
invalid decision branch; this experiment replaces that conclusion.

## Setup

- 5 bar-tagged clips hand-labelled with `scripts/label_scene_anchors.py`:
  1 tripod control + 4 panned phone clips across 4 sessions and bar heights
  1.72-1.85 m. 25 frames sampled per clip; the `s` key was used to skip
  frames where the apparatus was not fully visible (typical visibility
  10-30 % of sampled frames on panned clips, 100 % on tripod).
- Hand-label dwell ranged from 2 to 10 valid labels per clip. Two clips
  ended up with sparse labels (2 anchor points) — flagged as borderline.
- Truth pipeline reworked: compares at the takeoff window only, not
  per-frame median. Per-frame median is dominated by pose-velocity noise
  and produces a misleading floor of ~1 m/s "error" even on a tripod with
  no panning. Takeoff window picks the `argmax(vy)` instant and takes a
  ±3-frame median to suppress single-frame jitter.
- Auto-detector (`scene_homography`) was rejected by the clip-level
  acceptance gate on every clip in this set.

## Findings

### Egomotion clearly beats anatomical on takeoff vh

On the densest-labelled clip (7 hand labels in a tight cluster spanning
the takeoff window), egomotion recovered roughly 80 % of the truth value
versus anatomical's 48 %. On other clips with usable labels, egomotion was
similarly closer to truth than anatomical, by 0.3-2.5 m/s margins.

### A residual ~0.9-1.5 m/s underestimate persists across all modes

The tripod clip is the diagnostic: there is no camera motion to remove,
so anatomical and egomotion produce identical vh values, and any
divergence from truth must come from the metric scale itself. Both modes
underestimated truth by ~0.9 m/s at takeoff. This isolates the next
blocker as **mpp calibration at the takeoff zone**, not panning.

Hypothesis: anatomical mpp is the p95 of thigh/shank pixel projection
across the whole video. At the takeoff frame, the takeoff-side leg is
planted and tilted away from the camera plane, so its projection is
foreshortened and never reaches its true in-plane length there. The p95
captures depth elsewhere in the run-up, not at the takeoff zone. mpp
applied at takeoff is therefore biased small, scaling pixel velocity to
metres velocity at a too-low rate.

### Truth values land in or near the elite band

Earlier the "smoke median takeoff vh = 1.08 m/s" suggested Imogen was far
below the 2.5-5.5 m/s elite band. The hand-label truth at the takeoff
window puts most clips at or near elite, with one clip well above. The
earlier framing was wrong: per-frame median on anatomical-only output is
dominated by pose-noise spikes during the run-up phase and does not
reflect the takeoff instant.

### Auto-detector is dead on this footage

Hough-based apparatus detection in `src/pose_estimation/scene_calibration.py`
was rejected on all 5 labelled clips, with `scene_anatomical_scale_ratio`
values around 2-3× indicating the detector is locking onto wrong vertical
edges (mat frame, net poles, fence railings). Hand-labelled anchors are
the only viable apparatus reference on this footage.

## Decision

Phase 9d: build an apparatus-anchored mpp pathway that uses the labelled
upright separation (4.02 m, IAAF spec) as a third independent scale source
**measured at the takeoff frame specifically**. Validate against:

1. The tripod control — expect mode vh to converge with truth vh to within
   ~0.3 m/s after the recalibration.
2. The 4 panned clips with usable labels — expect egomotion+9d to bring
   takeoff vh into the elite band on bar-tagged clips that already have
   truth in that band.

If 9d does not close the residual gap on the tripod, escalate to either
two-camera DLT or end-of-runway tripod filming. Existing footage is then
relegated to relative-kinematics analysis.

## What this changes for Phase 10

Phase 10 personal fine-tuning remains gated, but the gate has narrowed
considerably. If 9d lands, fine-tuning can use egomotion-corrected
training-grade clips. The original "panning is the blocker" framing is
retired; the actual blocker is mpp calibration depth bias.
