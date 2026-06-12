# System Architecture

## Overview

This repository is a research system for high-jump biomechanical analysis and
personalised technique optimisation. The pipeline is:

```text
Video -> Pose Estimation -> Kinematics -> BiomechanicalSample -> PINN -> Optimiser
```

The canonical interchange format is `BiomechanicalSample` in
`src/data_pipeline/sample.py`. Any module that crosses dataset, training, or
video-analysis boundaries should preserve that data model.

Scientific correctness is the priority. Physics conventions are defined in
`.github/copilot-instructions.md`; chronological status and caveats live in
`ROADMAP.md`.

## Repository Structure

```text
src/
  pose_estimation/      MediaPipe, scale calibration, post-processing, OpenSim IK
  kinematics/           Run-up, takeoff, and flight metrics
  pinn/                 Projectile, inverse-dynamics, and joint PINNs
  gnn/                  Skeleton graph neural network
  optimization/         Differentiable jump simulator and technique optimiser
  data_pipeline/        BiomechanicalSample, loaders, PyTorch datasets
  utils/                Physical constants and segment parameters
  visualization/        Pose-overlay rendering

scripts/
  analyze_jump_video.py
  pretrain_dynamics_pinn.py
  finetune_personal.py
  optimize_jump.py
  download_datasets.py
  render_pose_overlay.py

tests/                  280 tests covering data, kinematics, pose calibration,
                        PINN physics, optimisation, and tooling
tools/memory/           Local RAG build/query scripts for agent context
memory/                 Tracked notes/plans/experiments plus ignored vector index
experiments/configs/    Training configuration
experiments/results/    Pre-training outputs and checkpoints
data/                   Private videos/results plus public datasets; mostly gitignored
services_scaffold/      Parked deployment scaffolding, not part of research pipeline
```

## Physics Conventions

- Coordinate system: Y-up, right-handed. X = forward/run-up, Z = lateral.
- Gravity vector: `[0, -9.81, 0]` m/s2.
- Ground reaction force convention:
  `F_GRF = m * (a_CoM - g_vec)` where `g_vec = [0, -9.81, 0]`.
- Joint angles are radians internally; convert to degrees only at output
  boundaries.
- CoM estimation uses de Leva (1996) segment offsets and mass fractions from
  Winter (2009).
- Fosbury Flop takeoff angles around 38-48 degrees are the current literature
  reference range. Extracted values outside that range should be treated as a
  possible pipeline issue before being treated as athlete behaviour.

## Main Modules

### Pose Estimation

`src/pose_estimation/` converts video into calibrated landmark trajectories.

- `estimators/mediapipe_estimator.py`: MediaPipe BlazePose wrapper producing
  2D normalised landmarks and MediaPipe world landmarks. Supports an opt-in
  two-pass ROI crop via `process_video(roi_crop=True)`: pass 1 locates the
  athlete on the full frame; pass 2 re-detects on the cropped region.
  `remap_normalized_to_full_frame(landmarks_in_crop, bbox_norm)` maps
  crop-normalised coords back to full-frame normalised coords (pure function,
  tested independently). `roi_crop="takeoff"` is an audited current-footage
  fallback that estimates the flight/takeoff window from pass 1 and crops more
  tightly around that region. The 3D world landmarks need no remap.
- `scale_calibration.py`: Phase 9a calibration. When measured thigh/shank
  lengths are available, derives a single video-wide metres-per-pixel value
  from the 95th percentile of visible thigh/shank pixel projections, medianed
  across segments. Ground reference is the 5th percentile of visible ankle Y,
  avoiding single-frame outlier contamination.
- `scene_calibration.py`: Phase 9c Hough-based upright/crossbar detection and
  per-frame scene homography. Retained as historical panned-footage rescue
  infrastructure; the automatic detector is unreliable on those clips.
- `egomotion.py`: Phase 9c background optical-flow camera-motion compensation.
  Retained as historical panned-footage rescue infrastructure. It recovers much
  of the panning component but does not close the translational-scale gap.
- `gravity_calibration.py`: Phase 9e experimental metres-per-pixel recovery
  from flight-parabola curvature. It is correct on synthetic projectile data
  but failed on handheld footage and is not part of the stationary workflow.
- `skeleton/landmark_postprocessor.py`: gap filling, low-pass filtering, and
  segment-length enforcement.
- `opensim_ik.py` plus `scripts/opensim_ik_subprocess.py`: optional OpenSim IK
  integration through a separate environment.
- `dlt_triangulation.py`: available path for true 3D reconstruction from
  synchronised multi-camera footage.

Current policy: panned single-camera footage is not admitted to
training-grade translational analysis. Stationary footage is required for
Phase 10 fine-tuning and optimiser claims. The historical panned clips remain
useful for relative technique review and detector development.

### Kinematics

`src/kinematics/` computes coach-facing movement metrics.

- `run_up_analysis.py`: ground-contact detection, stride timing, horizontal
  speed profile, curve radius, curve adherence, foot-under-hip offset, contact
  labels, arm lateral swing, and body-alignment deviation through the run-up.
- `takeoff_analysis.py`: Newton-law GRF estimate, takeoff angle, vertical
  velocity, impulse, foot-to-ground angle, body alignment, arm drive, and free
  knee drive.
- `flight_analysis.py`: flight parabola metrics, peak CoM height, clearance
  profile, vertical extension timing, and arch transition frame.

Phase 9b changed takeoff-frame selection to the final detected ankle-ground
contact before peak CoM, with `argmax(vy)` only as fallback.

June 2026 additions to `scripts/analyze_jump_video.py`:

- `_validate_takeoff_anchor()`: rejects approach-stride contacts by checking
  vy >= 2.0 m/s at the candidate frame and that the frame lead does not exceed
  2·(vy/g)·fps. `select_takeoff_frame_details` returns a 4-tuple including
  `takeoff_anchor_review_passed`.
- `pose_validity_pct` tightened to all-8-key-joints (shoulders/hips/knees/ankles,
  idx 11,12,23,24,25,26,27,28 — matching `PoseFrame.is_valid`).
- `takeoff_window_pose_validity_pct(landmarks_2d, takeoff_frame, half_window=30)`:
  gate metric for admission. Measures key-joint coverage in the ±30 frame
  window around toe-off, excluding early run-up frames where the athlete is too
  far from camera to detect. Global `pose_validity_pct` retained for diagnostics.
- `--capture-mode {handheld,stationary}` plus `--stationary-camera-confirmed`:
  only explicit operator confirmation that the camera did not pan, tilt, zoom,
  or move credits the fixed camera as the scene-fixed horizontal reference and
  removes the `no_scene_fixed_horizontal_source` training gate.
- `--roi-crop {on,off}`: opt-in two-pass ROI crop (default off).
- `--roi-crop takeoff`: opt-in takeoff-window ROI crop. Useful for
  current-footage rescue experiments, not a default admission path.

### Data Pipeline

`src/data_pipeline/sample.py` defines the shared data model:

- `SubjectInfo`: anthropometrics and optional segment lengths.
- `SessionContext`: optional fatigue, physiology, environment, injury, and
  menstrual-cycle metadata.
- `BiomechanicalSample`: joint angles, angular velocities/accelerations, marker
  positions, CoM position/velocity/acceleration, GRF, CoP, torques, pose data,
  and session metadata.
- `save_npz()` / `load_npz()`: compressed sample cache used by video analysis
  and personal fine-tuning.
- `src/data_pipeline/admission.py`: local `_admission_manifest.json` contract.
  The analyser records every decision but saves `.npz` only for
  `training_grade` clips; personal fine-tuning refuses legacy mixed caches
  without a valid manifest.

Movement relevance priority is:

```text
HIGH_JUMP > SINGLE_LEG_DROP_JUMP > DROP_JUMP > CMJ > VERTICAL_JUMP > SQUAT_JUMP
```

This reflects Athlete A's athlete-domain feedback that single-leg box drop jumps
transfer more directly to high-jump takeoff than CMJ.

### PINNs

`src/pinn/` contains the physics-informed models.

- `pinn/physics/projectile.py`: flight-phase projectile PINN.
- `pinn/physics/inverse_dynamics.py`: main inverse-dynamics PINN. Current
  pre-training target uses `input_dim=7` (`t`, CoM position, CoM velocity) and
  `output_dim=6` (GRF plus joint-torque outputs).
- `pinn/models/joint_pinn.py`: per-joint Euler-Lagrange model.
- `pinn/training/`: composite losses and training utilities.

Current pre-training artifacts are under:

```text
experiments/results/pretrain_dynamics/
  best_model.pth
  final_model.pth
  loss_history.npz
```

### GNN

`src/gnn/skeleton_gnn.py` models the body as a skeleton graph. It uses learned
force-message layers to propagate mechanical information between connected
joints and can couple multiple joint-level PINNs.

### Optimisation

`src/optimization/optimizer.py` contains the Phase 5 differentiable optimiser.

- `TechniqueParameters`: 14 controllable technique variables, including
  approach speed, curve radius, step lengths, plant/takeoff angles, arm timing,
  ground contact time, body alignment, foot-to-mat angle, knee drive speed, and
  curve start step.
- `AthleteConstraints`: individual bounds for speed, force, torque, and range
  of motion.
- `DifferentiableJumpSimulator`: PINN-backed height prediction.
- Sensitivity and what-if analysis for ranked coaching interventions.

Do not refresh optimiser outputs until translational metrics from the video
pipeline are validated; stale `data/results/all_optimizations.json` should not
be trusted.

## Entry Points

| Script | Purpose |
| --- | --- |
| `scripts/analyze_jump_video.py` | Video -> calibrated pose -> kinematics -> report; can cache training-grade `.npz` samples and a local admission manifest. |
| `scripts/pretrain_dynamics_pinn.py` | Train inverse-dynamics PINN on public datasets. |
| `scripts/evaluate_dynamics_pinn.py` | Benchmark the pretrained inverse-dynamics PINN against local public force-plate datasets. |
| `scripts/finetune_personal.py` | Fine-tune on manifest-admitted private samples after guardrails pass. Keep real training blocked until a larger session and held-out split are available. |
| `scripts/evaluate_stationary_anthropometry.py` | Compare raw stationary-pose body proportions against local known proportions without publishing private measurements. |
| `scripts/optimize_jump.py` | Generate optimiser/sensitivity outputs from reports. Currently stale pending metric validation. |
| `scripts/download_datasets.py` | Print manual public-dataset download instructions. |
| `scripts/render_pose_overlay.py` | Render pose overlays for inspection. |
| `scripts/create_takeoff_focus_clips.py` | Create ignored, static-crop derivative clips around takeoff/flight for current-footage rescue experiments. |

## Local Agent Memory

`memory/` holds tracked notes for Claude/Codex collaboration. The optional local
RAG tooling under `tools/memory/` remains available but is parked while direct
file reads and `rg` are faster for this small corpus.

- `memory/docs/`: architecture notes, physics notes, equations, decisions, open
  questions.
- `memory/plans/`: current Opus plan and Codex execution notes.
- `memory/experiments/`: aggregate experiment summaries safe to share between
  agents.
- `memory/vector_index/`: generated ChromaDB index, ignored by git.
- `tools/memory/build_index.py`: chunks configured project files and builds the
  local vector index.
- `tools/memory/query_index.py`: retrieves relevant file/line snippets for a
  task.

The index uses deterministic local hashing embeddings. This is intentionally
less powerful than a transformer embedding but avoids external calls and private
code leakage. Private athlete data and generated report folders are excluded by
default in `tools/memory/config.yaml`.

## Current Phase

Phases 9a-9e established the video-validation boundary:

- Phase 9a: multi-segment anatomical scale plus robust ground reference.
- Phase 9b: contact-anchored takeoff-frame selection.
- Phase 9c: scene homography, egomotion, and hand-label truth evaluation.
- Phase 9e: historical experimental gravity-mpp scale recovery.
- Full historical private reprocess completed: 45/45 reports and cached samples.

June 2026 — stationary-footage admission tooling shipped and validated:

- explicit `--stationary-camera-confirmed`, takeoff-anchor review with a
  2.0 m/s upward-launch floor, heel/forefoot contact detection, ROI crop,
  stricter key-joint pose metric, and windowed pose-validity gate all added.
- Analyzer caching is training-grade-only, each decision is recorded in the
  ignored local admission manifest, and fine-tuning refuses legacy mixed
  sample directories.
- Stationary pilot: **3/3 newer clips pass the implemented report gates**
  (training_grade True).
  Takeoff angles 40.5–42.2°, vh 3.65–4.11 m/s, window pose validity 61.7–73.3 %.
- Test suite: **285 total passing**.

Phase 10 personal fine-tuning remains blocked pending a larger dedicated 60 fps
session (8–12 attempts) and a held-out split. Do not refresh optimiser claims
until the fine-tuned model is validated on that held-out subset.

The public-data PINN workstream is independent: `scripts/evaluate_dynamics_pinn.py`
benchmarks the existing checkpoint. AddBiomechanics remains the next high-quality
GRF dataset to add before a publication-grade retrain with a formal held-out
subject split.

## Tests

Run the non-PINN suite with:

```powershell
.venv/Scripts/python.exe -m pytest tests/ --ignore=tests/test_pinn -q
```

Current result: 285 total passing. Test coverage includes data pipeline
roundtrips, scale calibration, kinematics, optimiser behaviour, pose skeleton
utilities, landmark post-processing, parsers, physics-law checks, bbox remap
round-trips, takeoff-anchor validation, windowed pose-validity logic, and
stationary-camera admission gates.

## Private Data

The following are private and gitignored:

```text
data/High Jump Videos/
data/results/
data/models/personal/
```

Do not paste raw athlete video paths, session metadata, private reports, or raw
extracted values into commits, PRs, or external services.

## Services Scaffold

`services_scaffold/` contains deployment-oriented FastAPI scaffolding. It is
parked and intentionally disconnected from `src/`. Do not add deployment code to
the research modules.
