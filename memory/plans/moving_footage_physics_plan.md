# Plan: Extract takeoff physics from moving (panned) footage

Status started: 2026-06-14. Owner: Claude/Codex execution. Reviewer gate: physics
+ phase-gate per AGENTS.md.

## Thesis

Even panned phone clips are **quasi-stationary during the plant→takeoff→early-flight
window** — the biomechanically critical ~0.2–0.5 s. The existing stable-window
solver (`scripts/analyze_stable_takeoff_window.py`) already extracts trustworthy
physics from a *stationary* window via apparatus PnP + gravity-constrained
projectile bundle adjustment. Its correctness rests on exactly two assumptions:

1. **Accurate apparatus anchors** (camera pose is only as good as them).
2. **A stationary camera across the fit window** (one fixed `CameraModel` reused
   for every frame).

This plan makes both assumptions *computed and checkable* on arbitrary footage,
so the large multi-session corpus under `data/High Jump Videos/` becomes usable.

Do **not** touch optimiser claims or run a real fine-tune until the validation
gate (Phase D) passes. Reuse the takeoff-stationary IMG_4829/4830 clips and the
admitted stationary trio as semi-truth anchors.

## Local feasibility (verified 2026-06-14)

- `mediapipe 0.10.35` Tasks API: ObjectDetector / ImageSegmenter /
  InteractiveSegmenter present.
- `cv2 4.13` has `LineSegmentDetector` **and** `ximgproc` (FastLineDetector).
- `torch 2.12 CPU-only`, `onnxruntime 1.24`. **No GPU**, no SAM/ultralytics.
- => Layers A/B/C below are CPU-feasible now. World-grounded HMR (WHAM/TRAM/GVHMR)
  and SAM2 are **offline/cloud spikes only**, not the local pipeline.

## Phases (smallest-first, each independently testable)

### Phase A — Camera-motion estimation + stable-window detection  [IN PROGRESS]
New `src/pose_estimation/camera_motion.py`:
- Mask the athlete (foreground) so only background drives motion estimation.
- ORB/AKAZE features + RANSAC homography between frames (a pure pan/tilt about
  the optical centre is the infinite homography; the planar track is also a
  homography — both handled).
- Per-frame background displacement signal (median motion of an image-point grid).
- `find_stable_window`: longest run under a px/frame threshold, ≥ min duration.
- Output an objective gate: "clip has a usable stable window of N frames at
  [start,end], ref=R, max_disp=… px/frame."
Tests: synthetic warped sequences with known motion; recover disp + window.
First measurement: run across panned sessions to count how many clips have a
usable window before investing further.

### Phase A finding (2026-06-14) — REVISES the selection heuristic
Built `src/pose_estimation/camera_motion.py` (+9 tests) and
`scripts/scan_stable_windows.py`. Scanned takeoff-stationary IMG_4829/4830 and
sessions 14_02_26, 25_03_26, 20_03_25, 09_02_26.

**Result:** almost every clip has *a* long stable window — but on full match
clips it is the **static wide shot before the run-up**, not the takeoff. Visual
checks (`stable_window_scan_v1/clip_25_03_26_Two_frames.jpg`,
`wincheck_14_02_26_two_*.jpg`) show frames 0–80 are the empty runway / athlete
preparing, while the **takeoff happens near the clip end where the camera pans
and zooms to the apparatus**. So "longest stable window" finds the wrong window.

**Consequence — the approach pivots:** do NOT gate on stillness. The takeoff
window is short (~0.3–0.5 s) and *panning*, and that is fine — Phase B
stabilization models pan/zoom directly. The real pipeline must (a) locate the
takeoff frame from athlete kinematics/contact (the main pipeline already does),
then (b) stabilize a takeoff-centred window. The per-frame motion signal from
Phase A is retained as input + as a stabilization-quality check, not as a gate.

### Phase B — Ground-plane stabilization of the window  [CORE DONE]
Real-world validation (2026-06-14): on a genuinely *panning* takeoff window of
`14_02_26_two` (raw background motion ~6 px/frame), `stabilize_window` registered
all 17 frames to the reference with **mean residual 0.24 px, max 0.95 px** —
i.e. a panning takeoff window becomes effectively stationary with sub-pixel
background alignment (`stable_window_scan_v1/phaseB_stabilize_14_02_26_two.jpg`).
This is the key enabler: it does not require the camera to be still.

Extend `camera_motion.py`:
- Estimate each window frame → reference-frame homography from background features.
- Warp to the reference → synthesize a stationary clip; report stabilization
  residual (reject windows that won't stabilize below threshold).
- Remap MediaPipe 2D landmarks through the same homography so CoM lives in the
  stabilized frame. (3D world landmarks pass through, as in the ROI-crop path.)

### Phase C — Robust apparatus anchors on the stabilized window
- Feed the stabilized, athlete-masked window into a temporally-aggregated
  detector (`detect_apparatus_geometry_stable`).
- Add a **known-geometry PnP-RANSAC** acceptance: candidates must be metrically
  consistent with the rigid 4.02 m × bar-height rectangle and share a vertical
  vanishing point; reject background rooflines/poles that imply impossible
  cameras. (This is the clip1/clip2 failure repair.)
- Upgrade line primitives: try `cv2` LSD / FastLineDetector and M-LSD (ONNX on
  onnxruntime) for cleaner standard/bar segments than Hough.

### Phase D — Physics: intrinsics + multi-frame gravity-constrained solve + validation
- Read real focal length from EXIF/QuickTime metadata (replace the assumed 60°
  FOV) or self-calibrate from apparatus vanishing points.
- Extend the projectile solve to use the apparatus tracked across **all** window
  frames (over-determined pose, lower variance, built-in consistency check).
  NB: this is also why parked gravity-mpp failed on handheld — stabilization
  removes the vertical tilt that corrupted the parabola, making it valid again.
- Validation: reprojection RMS, stabilization residual, recover known physics on
  IMG_4829/4830 via the *panned* path, hold-one-frame-out anchor error,
  cross-session takeoff-angle/vh consistency bands.

### Phase C/D progress + KEY physics finding (2026-06-14)
Built `scripts/analyze_moving_takeoff.py`: extract poses → contact toe-off →
takeoff-centred window → `stabilize_window` (athlete masked from landmarks) →
remap CoM into the toe-off reference → apparatus anchors (auto or manual JSON) →
`solvePnP` → gravity projectile fit. Runs end-to-end on real clips.

Fixed a latent bug in `_projectile_residuals` (analyze_stable_takeoff_window):
it conditionally appended the depth penalty, so the residual vector changed
length between iterations and crashed scipy's robust loss. Now always appended
(zero when satisfied). Added a soft hinge on horizontal takeoff speed
(`max_horizontal_speed_mps=9`) to bound the out-of-plane runaway. 128 tests green.

**KEY FINDING — the real blocker is the monocular physics solver, not the
stabilization or apparatus detection.** The existing free-depth 3D projectile fit
(`fit_projectile_to_com_pixels`) is **degenerate in the camera optical-axis
direction**: a single camera barely observes depth, so the optimiser runs the
out-of-plane velocity away. Evidence: the existing v8 apparatus-PnP results on
IMG_4829/4830 were already garbage (vh ≈ 662 m/s, angle ≈ −1°, all rejected);
the moving orchestration reproduces the same degeneracy (angle −60°, vv −16).
This is independent of camera motion — even the *stationary* derivative fails.

**The fix is a bar-plane-constrained fit.** Diagnostic: warp post-toe-off CoM
pixels through the image→apparatus-plane homography (Z=0) and fit a 2D gravity
parabola (X linear, Y = y0 + vy·t − ½g·t²). On IMG_4829 this yields physical
takeoff metrics — e.g. toe-off 35 / 22-frame window → **46.2°, vh 2.1, vv 2.2**;
toe-off 45 / 10-frame → **40.6°, vh 3.6, vv 3.1** — vs the 3D fit's −60°/662 m/s.
Remaining variance is sensitivity to the toe-off frame and window length (longer
windows drift toward vh≈0 as the athlete leaves the plane over the bar), so the
planar solver must pair with a robust toe-off frame and a short (~0.3–0.4 s)
post-toe-off window.

**Revised priority order:** (1) implement the bar-plane projectile solver as the
primary physics path (well-posed; removes depth degeneracy); (2) robust toe-off
frame + short fit window; (3) then return to apparatus-detection robustness and
stabilization (already working) to feed it. The scene homography for the planar
fit needs only the 4 apparatus anchors — no focal-length/FOV assumption.

### Bar-plane solver implemented + validation finding (2026-06-14, later)
Implemented `fit_bar_plane_projectile` in `analyze_stable_takeoff_window.py`,
wired a `--solver {bar_plane,pnp_3d}` flag into both scripts (default
`bar_plane`), with a `--bar-plane-window-s` knob and synthetic physics-law tests
(310 non-PINN tests green). Two solver forms were explored:

- *Fixed-g* (gravity enforced at 9.81 in bar-plane coords): robust but biased —
  on IMG_4829 it gave a plausible angle (~42 deg, matching the anatomical ~46
  deg) yet a large in-plane residual (~0.65 m), because the athlete is offset
  from the bar plane.
- *Gravity-as-scale* (free quadratic -> apparent g -> depth scale k=Zbar/Zath;
  angle is scale-invariant, velocities rescale by 1/k): the principled form and
  the one kept. The synthetic in-plane test recovers k=1 and exact velocities.

**Validation finding — the blocker is now UPSTREAM of the solver.** Dumping the
warped CoM track for IMG_4829 showed: (1) the takeoff-window pose is sparse
(many invalid frames; the detected toe-off frame 35 lands in an invalid run);
(2) the athlete is well off the bar plane — valid airborne frames warp to scene
X ~= -6 m although the standards are at +/-2.01 m, so the bar-plane warp is
geometrically wrong; (3) the warped track is not ballistic (Y jumps 1.75 -> 0.60
m across an invalid gap from CoM misdetections). The gravity-as-scale solver
correctly *rejects* this (negative apparent g, k out of range) rather than
emitting false physics.

So the next blockers are upstream, not the solver: dense/clean CoM in the
takeoff window, a toe-off frame that lands in valid frames, and either athletes
near the bar plane or an explicit depth-offset (parallel-plane) model. Crucially
we lack a clip with BOTH good takeoff-window pose validity AND apparatus anchors:
IMG_4829/4830 have anchors but sparse/off-plane poses; the daylight stationary
trio has good poses (61-73 %) but the apparatus detector does not yet lock on.
**Priority shifts back to (a) apparatus detection on the good-pose daylight clips
and (b) CoM density/toe-off precision in the takeoff window**, so the (working,
well-posed) bar-plane solver finally gets a clean input to prove itself on.

### Parallel spikes (offline/cloud, not blocking)
- Small 4-anchor keypoint regressor for the apparatus (thin-bar-friendly; corpus
  is large enough to label ~100–200 frames × 4 points).
- World-grounded monocular HMR (WHAM/TRAM/GVHMR) as an independent metric CoM
  cross-check, anchored to apparatus scale.

## Conventions
SI units, Y-up/X-forward/Z-lateral, gravity [0,-9.81,0]. No mock data. Keep
physics-law tolerances intact. Private athlete data stays under ignored
`data/results/`.
