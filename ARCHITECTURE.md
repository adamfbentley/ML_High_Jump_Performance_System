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

tests/                  190 non-PINN tests currently passing
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
  2D normalised landmarks and MediaPipe world landmarks.
- `scale_calibration.py`: Phase 9a calibration. When measured thigh/shank
  lengths are available, derives a single video-wide metres-per-pixel value
  from the 95th percentile of visible thigh/shank pixel projections, medianed
  across segments. Ground reference is the 5th percentile of visible ankle Y,
  avoiding single-frame outlier contamination.
- `skeleton/landmark_postprocessor.py`: gap filling, low-pass filtering, and
  segment-length enforcement.
- `opensim_ik.py` plus `scripts/opensim_ik_subprocess.py`: optional OpenSim IK
  integration through a separate environment.
- `dlt_triangulation.py`: available path for true 3D reconstruction from
  synchronised multi-camera footage.

Current limitation: panned single-camera footage is not scene-fixed, so
horizontal displacement and velocity are not yet training-grade.

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

Phase 9b changed report takeoff-frame selection in `scripts/analyze_jump_video.py`
from `argmax(vy)` to the final frame of the final detected ankle-ground contact
before flight, with `argmax(vy)` only as fallback.

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
| `scripts/analyze_jump_video.py` | Video -> calibrated pose -> kinematics -> report; can cache `.npz` samples. |
| `scripts/pretrain_dynamics_pinn.py` | Train inverse-dynamics PINN on public datasets. |
| `scripts/finetune_personal.py` | Fine-tune on cached private samples after guardrails pass. Currently blocked. |
| `scripts/optimize_jump.py` | Generate optimiser/sensitivity outputs from reports. Currently stale pending metric validation. |
| `scripts/download_datasets.py` | Print manual public-dataset download instructions. |
| `scripts/render_pose_overlay.py` | Render pose overlays for inspection. |

## Local Agent Memory

`memory/` and `tools/memory/` implement the lightweight local RAG workflow for
Claude/Codex collaboration.

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

Phase 9a and 9b are implemented and tested:

- Phase 9a: multi-segment scale calibration plus robust ground reference.
- Phase 9b: contact-anchored takeoff-frame selection.
- Bar-height parsing fixed for numeric extensions such as `.mp4`.
- Full private reprocess completed: 45/45 reports and 45/45 cached samples.
- Full non-PINN suite: 190 passing.

Phase 10 personal fine-tuning is blocked. Aggregate reprocess metrics are not
training-grade:

```text
Peak CoM median:            1.67 m   (9/45 in 2.0-2.7 m)
Takeoff vertical velocity:  3.16 m/s (18/45 in 3.0-4.5 m/s)
Takeoff angle:              66.6 deg (3/45 in 38-48 deg)
Takeoff horizontal speed:   1.08 m/s (2/45 in 2.5-5.5 m/s)
```

The main current blocker is horizontal velocity from panned single-camera
footage. The next architecture-level fix should recover scene-fixed horizontal
motion with a physical reference or homography, or use two-camera DLT for future
data capture.

## Tests

Run the non-PINN suite with:

```powershell
.venv/Scripts/python.exe -m pytest tests/ --ignore=tests/test_pinn -q
```

Current result: 190 passing. Test coverage includes data pipeline roundtrips,
scale calibration, kinematics, optimiser behavior, pose skeleton utilities,
landmark post-processing, parsers, and physics-law checks.

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
