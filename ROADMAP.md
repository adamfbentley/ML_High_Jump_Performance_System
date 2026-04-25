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

## Phase 2 — Athlete Alignment: Athlete A's Priority Metrics

**Commits:** `715133d` → `e466fac` (merged from `copilot/movement-metric-agent` branch)

This phase incorporated direct feedback from the athlete (national-champion high jumper, H=1.75 m, W=65 kg) on what the system should actually measure and prioritise.

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

### Athlete A's jump videos processed (`ac4be47`, `fb60f04`)
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

## Phase 8 — Pre-Training Execution and Athlete A Video Processing

**Work done (14–15 April 2026):**

### Public dataset pre-training
- Downloaded Zenodo CMJ GRF dataset (`data/public/cmj_grf_zenodo/`), Nitschke et al. CoD dataset (`data/public/cod_ik_id_zenodo/`, 10 participants), and DVJ OpenSim dataset (`data/public/dvj_opensim_zenodo/`)
- Ran `scripts/pretrain_dynamics_pinn.py` for 3000 epochs
- Checkpoints saved every 500 epochs; best model saved by physics loss
- Results:
  - `experiments/results/pretrain_dynamics/final_model.pth` (1.03 MB)
  - `experiments/results/pretrain_dynamics/best_model.pth` (829 KB)
  - `experiments/results/pretrain_dynamics/loss_history.npz`
  - Final losses: `data_loss=0.899`, `physics_loss=1.263`, `total_loss=2.162`
  - ⚠ Physics loss plateaued around 1.25 rather than converging to <0.5 — AddBiomechanics (highest-quality GRF data) not yet included

### Athlete A's jump videos processed
- 45 jump attempts across 8 training sessions run through `scripts/analyze_jump_video.py`
- Per-session reports in `data/results/` (e.g. `13_12_25_report.json`, `14_02_26_one_1.79_report.json`)
- `data/results/all_sessions_report.json` — full multi-session summary
- `data/results/all_optimizations.json` — optimiser output for one session
- ⚠ Takeoff frame detection failures identified: some clips report `takeoff_angle_deg = -85°`, `takeoff_vertical_mps = -2.26 m/s` — physically impossible; frame-selection logic requires fixing

---

## Phase 9 — Onboarding Audit and Takeoff-Detection Fix (25 April 2026)

A new Claude onboarding pass audited the working tree against the committed state, fixed a regression in the optimisation engine, and resolved a systemic takeoff-detection failure that was affecting **all** 45 reports (not just ~5 as previously believed).

### Working-tree drift restored

- `src/optimization/optimizer.py` was found locally reverted to the pre-Phase-5 9-parameter stub (215 lines, old `optimize_technique(pinn_model, ...)` signature). This caused `tests/test_optimization/test_optimizer.py` to fail at collection (missing `GRAVITY`, `predict_bar_clearance`, `extract_params_from_report`, `generate_coaching_cues`, `_estimate_takeoff_com_height`, `_impulse_model_vertical_velocity`, `_evaluate_height_differentiable`) and broke `scripts/optimize_jump.py`.
- Restored to the committed 706-line version via `git checkout HEAD -- src/optimization/optimizer.py`. Test suite now reports the expected **174 passing**; without the restore it was 149 passing + 25 collection errors.

### Local YAML tweaks retained

- `experiments/configs/pretrain_dynamics.yaml` working-tree changes preserved as the active configuration: `dvj_opensim_zenodo` re-enabled (the cache it requires is now built) and `log_interval` changed from 1 → 50 (less verbose; appropriate after initial debugging).

### Takeoff-detection bug fixed

- Root cause located in `scripts/analyze_jump_video.py:329-345`: the frame-selection logic used `np.diff(np.sign(vy)) > 0` and took the *last* upward zero-crossing of vertical CoM velocity. In a jump trajectory this catches the **landing rebound** rather than the takeoff impulse (whose signature is `vy` at its peak just before gravity-driven decline).
- Replaced with `takeoff_frame = int(np.argmax(vy))`. Newton's 2nd law guarantees `vy` decreases monotonically once the foot leaves the ground (only force acting is gravity), so peak `vy` IS the takeoff instant.
- Smoke-tested on `14_02_26_one_1.79.mp4`: previously reported `takeoff_angle_deg=-2.8°, takeoff_vertical_mps=-0.01`; post-fix `takeoff_angle_deg=49.0°, takeoff_vertical_mps=3.72` (takeoff vertical velocity now physically plausible for an elite female jumper attempting 1.79 m).
- All 45 jump videos re-processed end-to-end through `scripts/analyze_jump_video.py "data/High Jump Videos"`; per-session JSONs in `data/results/` and the `all_sessions_report.json` summary regenerated.

### Onboarding-doc inaccuracies surfaced

- `CLAUDE_ONBOARDING.md` claims "5 clips" of takeoff detection failure; actual count was **45/45**.
- `CLAUDE_ONBOARDING.md` lists optimal Fosbury Flop takeoff angle as ~20–24° (Dapena 1980); peer-reviewed values are nearer 38–48°. Not changed in this pass — flagged for review with the BMS PhD student.

---

## Current State (25 April 2026)

| Area | Status |
|---|---|
| Pose estimation pipeline | ✅ Complete and tested |
| Landmark post-processing | ✅ Complete and tested |
| Kinematics (run-up, takeoff, flight) | ✅ Complete and tested |
| PINN architecture (all three variants) | ✅ Complete |
| GNN skeleton coupling | ✅ Complete |
| Dataset loaders (CMJ, CoD, DVJ, AddBiomechanics, BioCV, OpenCap) | ✅ Complete |
| Differentiable optimiser + sensitivity analysis | ✅ Complete and tested |
| Video analysis end-to-end pipeline | ✅ Complete (takeoff-detection bug fixed Phase 9) |
| Athlete priority metrics (Athlete A alignment) | ✅ Implemented |
| Test suite | ✅ 174 tests passing |
| Public datasets downloaded (Zenodo CMJ, CoD, DVJ) | ✅ Done |
| PINN pre-training run (3000 epochs) | ✅ Done — `final_model.pth` saved |
| Athlete A's 45 jump videos processed | ✅ Re-processed Phase 9 with corrected takeoff detection |
| Takeoff detection accuracy | ✅ Fixed — `argmax(vy)` selects the impulse peak |
| AddBiomechanics dataset downloaded | ⬜ Pending (highest-quality GRF pre-training data) |
| Personal data fine-tuning loop | ⬜ Pending (no fine-tuned model yet) |

---

## Upcoming Work

### Immediate

1. Validate physics loss convergence: assess whether 1.263 is acceptable or re-training with AddBiomechanics would meaningfully improve GRF prediction
2. Optionally download AddBiomechanics (requires registration at simtk.org) and re-run pre-training
3. Re-run `scripts/optimize_jump.py` on the corrected reports to refresh `data/results/all_optimizations.json`

### Phase 10 — Personal Data Fine-Tuning

- Create `scripts/finetune_personal.py` — load `best_model.pth`, fine-tune on Athlete A's extracted BiomechanicalSamples at lower learning rate, save to `data/models/personal/`
- Validate fine-tuned model predictions against held-out jump attempts
- Run `scripts/optimize_jump.py` on Athlete A's full dataset to generate personalised intervention rankings

### Phase 11 — Validation and Paper
- Compare predicted vs. measured jump heights on held-out attempts
- Confirm sensitivity analysis marginal gains are plausible against coaching intuition
- Write up methods and results
