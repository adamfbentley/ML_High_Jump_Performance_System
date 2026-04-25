# Project Roadmap

Records all development work completed since the initial commit on 10 April 2026, ordered chronologically. Each section lists what was built, why it was added, and its current status.

---

## Baseline (10 April 2026) — Initial Commit `ee0d4de`

The initial framework established the full research architecture from scratch.

**Delivered:**
- `src/` monolith with six modules: `pose_estimation`, `kinematics`, `pinn`, `gnn`, `optimization`, `data_pipeline`
- Three PINN variants: `ProjectilePINN`, `InverseDynamicsPINN`, `JointPINN`
- `SkeletonGNN` with `ForceMessageLayer` for inter-joint force propagation
- `BiomechanicalSample` canonical data format with `SubjectInfo`, `MovementType`, `MOVEMENT_RELEVANCE`
- Dataset loaders for AddBiomechanics, BioCV, OpenCap, AthletePose3D
- Pre-training script `scripts/pretrain_dynamics_pinn.py` with YAML config
- `TechniqueParameters` and gradient-based optimiser scaffold
- Kinematics modules: `RunUpMetrics`, `TakeoffMetrics`, `FlightMetrics`
- MediaPipe BlazePose estimator with DLT triangulation
- Joint angle computation for 10 bilateral joints
- CoM estimator (de Leva 1996 segment model)
- BVH skeleton generator
- 41 passing tests
- Microservice deployment scaffolding in `services_scaffold/` (parked)

---

## Phase 1 — Dataset Expansion and Pre-Training Config

**Commits:** `aaf2855` → `d7ee665`

**Work done:**
- Added real Zenodo dataset download support with a `cmj_npz.py` loader for the Zenodo CMJ GRF dataset (`zenodo.org/record/19136480`)
- Simplified `scripts/download_datasets.py` from 195 lines of HTTP automation to a clean 80-line instructions-only script (avoids unreliable download automation on Windows)
- Extended `experiments/configs/pretrain_dynamics.yaml` with a tier-1 (Zenodo, directly downloadable) / tier-2 (manual, require registration) dataset structure covering six datasets
- Added `.github/copilot-instructions.md` as the canonical technical specification for the project

---

## Phase 2 — Athlete Alignment: Imogen's Priority Metrics

**Commits:** `715133d` → `e466fac` (merged from `copilot/movement-metric-agent` branch)

This phase incorporated direct feedback from the athlete (national-champion high jumper, H=1.78 m, W=67 kg) on what the system should actually measure and prioritise.

**Key athlete requirements addressed:**
- Per-stride ground contact time, foot-under-hip offset, foot contact type (toe/flat/heel), and curve deviation added to `RunUpMetrics`
- Takeoff metrics expanded: foot-to-ground contact angle, body alignment score (straight-line hip–shoulder–ankle), arm drive peak speed and timing, free-leg knee drive speed and timing
- Flight metrics expanded: vertical extension time (frames before CoM descent), arch transition frame
- `TechniqueParameters` expanded from 9 to 14 parameters, adding GCT, body alignment, foot-to-mat angle, knee drive speed, and curve start step
- `SessionContext` dataclass added to `sample.py` to record physiological confounders: within-session fatigue, accumulated training load, injury status, HRV, weather, and menstrual cycle phase
- `MOVEMENT_RELEVANCE` updated: `DROP_JUMP` elevated above `CMJ` based on athlete input that single-leg drop jump is a closer transfer to high jump takeoff mechanics than countermovement jump
- Hip abduction and adduction angles added to `joint_angles.py` (frontal-plane kinematics, previously absent)
- `arm_swing_tracker.py` added to track arm path and velocity throughout the jump
- NumPy 2.x compatibility fix: `np.trapz` → `np.trapezoid`

**Tests added:** `test_movement_relevance.py` — verifies DROP_JUMP > CMJ, HIGH_JUMP = 1.0, SessionContext field types

---

## Phase 3 — Pose Pipeline Hardening

**Commits:** `5643464` → `9dd57fc`

**Work done:**

### Landmark Post-Processor (`src/pose_estimation/skeleton/landmark_postprocessor.py`)
- Gap filling via cubic spline interpolation for occluded landmarks
- 4th-order zero-lag Butterworth low-pass filter (configurable cutoff, default 6 Hz)
- Anatomical segment length enforcement (constrains bone lengths to ±15% of calibrated values)
- 14 new tests in `test_landmark_postprocessor.py`

### Scale Calibration (`src/pose_estimation/scale_calibration.py`)
- Estimates physical scale (metres per pixel) from known anthropometrics (height, limb lengths) against detected landmark positions
- Handles partial occlusion by using the subset of visible segments for the estimate
- 4 tests in `test_scale_calibration.py`

### OpenSim IK Integration (`src/pose_estimation/opensim_ik.py`, `scripts/opensim_ik_subprocess.py`)
- Runs OpenSim Inverse Kinematics via a conda subprocess to work around NumPy ABI incompatibility between OpenSim (NumPy 1.x) and the main environment (NumPy 2.x)
- Converts MediaPipe landmark positions to OpenSim `.trc` marker format
- Writes scaled `.osim` model and IK setup XML, invokes `opensim-cmd`, reads result `.mot` back into joint angles
- Falls back gracefully to direct angle computation when OpenSim is unavailable

### Video Analysis Pipeline (`scripts/analyze_jump_video.py`)
- End-to-end orchestration script: video file → MediaPipe → post-processor → scale calibration → OpenSim IK (optional) → kinematics (run-up, takeoff, flight) → console feedback report
- Handles multi-attempt video sessions; segments individual jumps automatically
- 602 lines; accepts `--video`, `--height`, `--mass`, `--limb-lengths`, `--opensim` CLI flags

### Imogen's jump videos processed (`ac4be47`, `fb60f04`)
- Full pipeline run against 45 recorded jump attempts
- Results surfaced data quality issues later addressed by the post-processor

---

## Phase 4 — Additional Dataset Loaders

**Commits:** `00f1e29` → `eb0398f`

**Work done:**
- `src/data_pipeline/loaders/cod_zenodo.py` — loader for Nitschke et al. (2022) Change-of-Direction Zenodo dataset; extracts GRF, joint angles, and CoM from `.c3d` files
- `src/data_pipeline/loaders/dvj_zenodo.py` — loader for Drop Vertical Jump Zenodo dataset; 547 lines covering bilateral force plate data, synchronised kinematics, and movement phase segmentation
- `src/data_pipeline/registry.py` — central dataset registry with priority ordering and `MovementType` relevance filtering; 76 lines
- Pre-training script updated with checkpoint resume (`--resume` flag) and correct routing of each dataset to its loader
- `DynamicsDataset` updated to prefer the CoM-based windowing path when `com_position` is available, falling back to joint-angle windowing
- DVJ dataset disabled by default in config until cache is built (avoids cold-start latency on first run)
- Training config `log_interval` set to 1 for granular loss monitoring during initial runs

---

## Phase 5 — Optimisation Engine Rewrite

**Commits:** `37900c5` → `40c0fad`

**Work done:**

### Differentiable Simulator (`src/optimization/optimizer.py`)
Complete rewrite of the optimisation engine (from scaffold to 664 lines):
- `DifferentiableJumpSimulator` — wraps the trained PINN and exposes `predict_height(params)` as a differentiable function via `torch.autograd`
- `TechniqueParameters` extended to 14 parameters with physical bounds and step sizes
- `AthleteConstraints` — per-athlete bounds on approach speed, GRF, joint torques, and ROM
- L-BFGS and Adam solver backends with configurable tolerance and iteration limits
- Sensitivity analysis: computes `∂height / ∂param_i` at the athlete's current technique for all 14 parameters, producing a ranked table of marginal gains
- What-if simulation: given a proposed parameter delta, returns predicted height change with confidence interval from Monte Carlo dropout
- 380-line test suite (`test_optimizer.py`) covering constraint enforcement, gradient correctness (finite-difference check), sensitivity ranking stability, and what-if accuracy

### Optimisation entry-point script (`scripts/optimize_jump.py`)
- 342-line CLI script: loads extracted kinematics from a prior `analyze_jump_video.py` run, initialises the optimiser, runs sensitivity analysis, prints a ranked intervention table, and optionally runs full optimisation
- Accepts `--session`, `--target-height`, `--max-iterations`, `--backend` flags
- Output example: `"Approach speed +0.3 m/s → predicted +2.8 cm (rank 1 of 14)"`

---

## Phase 6 — Visualisation

**Commit:** `388467e` (partial — included in chore commit)

**Work done:**
- `src/visualization/pose_overlay.py` — renders MediaPipe landmark skeleton overlay onto video frames; 178 lines
- `scripts/render_pose_overlay.py` — CLI wrapper: accepts video + landmarks file, writes annotated output video

---

## Phase 7 — Repository Hygiene and Documentation

**Commit:** `388467e`

**Work done:**
- `.gitignore` updated to exclude agent/prompt files, athlete private data, OpenSim log artifacts, ML checkpoints, and backup scripts
- `ARCHITECTURE.md` fully rewritten in formal technical prose (from informal notes to 300-line structured specification covering all modules, data tables, physics conventions, and research phase status)
- Agent definition files and prompt files removed from git tracking (remain on disk for local AI tooling; not public)
- Athlete response data removed from git tracking (private)

---

## Current State (April 2026)

| Area | Status |
|---|---|
| Pose estimation pipeline | ✅ Complete and tested |
| Landmark post-processing | ✅ Complete and tested |
| Kinematics (run-up, takeoff, flight) | ✅ Complete and tested |
| PINN architecture (all three variants) | ✅ Complete |
| GNN skeleton coupling | ✅ Complete |
| Dataset loaders (CMJ, CoD, DVJ, AddBiomechanics, BioCV, OpenCap) | ✅ Complete |
| Differentiable optimiser + sensitivity analysis | ✅ Complete and tested |
| Video analysis end-to-end pipeline | ✅ Complete |
| Athlete priority metrics (Imogen alignment) | ✅ Implemented |
| Test suite | ✅ 174 tests passing |
| Public datasets downloaded | ⬜ Pending (data/public/ empty) |
| PINN pre-training run | ⬜ Pending (requires datasets) |
| Personal data fine-tuning loop | ⬜ Pending (Phase 5) |

---

## Upcoming Work

### Immediate
1. Download public pre-training datasets (AddBiomechanics, Zenodo CMJ/CoD/DVJ)
2. Run pre-training smoke test: `python scripts/pretrain_dynamics_pinn.py --config experiments/configs/pretrain_dynamics.yaml --max-subjects 5 --epochs 100`
3. Validate physics loss convergence (InverseDynamicsPINN residual should decrease monotonically)

### Phase 5 — Personal Data Loop
- Film Imogen's jump sessions using standard phone protocol
- Run `scripts/analyze_jump_video.py` to extract per-attempt kinematics
- Fine-tune the pre-trained PINN on Imogen's data
- Generate personalised intervention recommendations via `scripts/optimize_jump.py`

### Phase 6 — Validation and Paper
- Compare predicted vs. measured jump heights on held-out attempts
- Sensitivity analysis validation: confirm predicted marginal gains are plausible against coaching intuition
- Write up methods and results
