# Stable Takeoff-Window Physics Extraction

Created 2026-06-13 after revisiting the panned-run-up clips whose final
plant/takeoff segment is mostly stationary.

## Claim

The final takeoff segment can contain meaningful physics even when the run-up
was panned. The correct target is **takeoff-window physics**, not full run-up
physics. A full run-up sample still requires a fixed camera throughout the
approach or a second calibrated view.

This is solvable only as a constrained inverse problem. It should not be solved
by relaxing the current report gates, and it should not resurrect the old
whole-clip gravity-mpp path as a standalone calibration. The old gravity-mpp
attempt failed because camera motion/tilt corrupted the projectile fit. In the
stable window, gravity can be a constraint inside a camera/pose bundle, not the
only scale source.

## Evidence So Far

- The current foot-contact and windowed-apex fixes rescue one manual
  takeoff-window derivative from the panned-run-up set.
- The other derivative still rejects after trim/ROI sweeps: pose evidence can
  be strong, but anchor/velocity/angle gates do not all agree.
- A standalone post-takeoff projectile mpp diagnostic is not reliable enough on
  these derivatives: the fitted image acceleration often points in the wrong
  direction or has too much horizontal component. This confirms gravity should
  be used jointly with scene/camera constraints, not as a single scalar mpp.

## Recoverable Outputs

High confidence from a stable takeoff window:

- plant/toe-off frame and contact interval;
- contact-side foot/ankle/knee/hip geometry;
- body alignment, free-knee drive, arm drive, and bar-relative pose timing;
- vertical launch velocity from time-to-apex when apex is visible and the
  stable window really covers flight;
- vertical impulse proxy from takeoff-window CoM velocity change.

Potentially recoverable with a camera/scene fit:

- takeoff velocity vector in a bar-centred world frame;
- true horizontal takeoff speed, including runway-depth component;
- takeoff angle and projectile bar-clearance estimate with uncertainty.

Not recoverable from these clips alone:

- full run-up curve and approach-speed build-up during panned frames;
- publication-grade optimiser claims without held-out validation.

## Proposed Method

### 1. Detect The Stable Window

Use background optical flow or feature tracks outside the athlete mask to find
the longest low-motion interval covering final plant, toe-off, and early flight.
The output should be source-frame indices, not a new invented timeline.

Acceptance gates:

- background affine translation/rotation below a small threshold across the
  selected interval;
- toe-off and at least early flight visible;
- no large invalid pose gap around toe-off.

### 2. Add One-Frame Apparatus Anchors

For a stable window, one labelled frame can define the apparatus geometry for
the whole interval. Use known high-jump geometry:

- bar height from session metadata;
- upright separation / crossbar span as the horizontal reference;
- upright bases/tops or crossbar endpoints as image anchors.

This avoids relying on the old Hough detector. Automatic detection can be a
future convenience, but the first version should accept a tiny manual JSON.

### 3. Fit A Camera/Physics Bundle

Coordinate system remains project-wide:

- Y up;
- X along the crossbar / scene horizontal;
- Z runway-depth direction.

Unknowns:

- static camera pose and focal length, initialised from apparatus anchors;
- CoM/root 3D trajectory over the stable interval;
- launch velocity at toe-off;
- optional per-frame skeleton depth corrections bounded by limb lengths.

Observed constraints:

- 2D landmark reprojection error;
- known thigh/shank lengths and left/right limb symmetry;
- takeoff foot/forefoot/heel contact on the ground plane before toe-off;
- post-toe-off CoM projectile motion under `[0, -9.81, 0]`;
- bar/upright anchor reprojection;
- temporal smoothness before toe-off.

Loss should be robust, e.g. Huber or Cauchy, with explicit weights for:

- apparatus anchors;
- high-confidence lower-limb landmarks;
- contact ground-plane constraint;
- projectile residual after toe-off;
- limb-length residuals.

### 4. Report Uncertainty

Use bootstrap or perturbation over:

- landmark detections;
- manually labelled anchor points;
- plausible stable-window start/end frames;
- focal-length prior.

Only admit a takeoff-window sample when the confidence interval stays inside
the physics gates. Otherwise report a bounded estimate and keep it out of
training caches.

## First Implementation Pass

Create a separate experimental script, not a change to production admission.
Implemented 2026-06-13 as `scripts/analyze_stable_takeoff_window.py`.

```powershell
python scripts/analyze_stable_takeoff_window.py `
  --video <clip-or-derivative> `
  --bar-height 1.75 `
  --anchor-json <manual-apparatus-anchors.json> `
  --output data/results/stationary_validation/stable_takeoff_window_v1.json
```

Initial scope:

1. Consume an already trimmed stable-window derivative — implemented.
2. Accept one-frame manual apparatus anchors — implemented.
3. Fit a simplified CoM projectile + camera pose model first — implemented
   with a focal-length sensitivity sweep.
4. Add skeleton/limb constraints only after the CoM fit is stable — pending.
5. Output estimates with uncertainty and an explicit `takeoff_window_only`
   flag.

Synthetic tests in `tests/test_tools/test_stable_takeoff_window.py` verify that
the fitter recovers known launch velocity from projected CoM pixels.

## Validation Order

1. Run on the three admitted fully stationary clips. It should reproduce the
   current takeoff velocity/angle within a tight tolerance and preserve
   training-grade admission.
2. Run on the two earlier stationary controls. It should still reject them.
3. Run on the panned-run-up stable-window derivatives. Admit only windows whose
   fitted physics and uncertainty pass without gate relaxation.

## Decision Boundary

This work can make the stable part of the panned-run-up clips useful. It should
not change the project policy that full run-up physics, personal fine-tuning,
and optimiser claims require a larger dedicated fixed-camera session with a
held-out subset.
