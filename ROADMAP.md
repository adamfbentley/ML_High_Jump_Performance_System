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

## Phase 8 — Pre-Training Execution and Imogen Video Processing

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

### Imogen's jump videos processed
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

### Phase 9-bis (29 April 2026) — fine-tune scaffolding and Phase 9a v1

After surfacing the Phase 9a/9b/9c follow-ups, the next session built the
infrastructure needed to actually run personalised fine-tuning, plus a v1
of the scale-calibration fix.

**Sample serialisation (`src/data_pipeline/sample.py`)**
- Added `BiomechanicalSample.save_npz(path)` / `load_npz(path)` — compressed
  arrays + JSON metadata. Tested for full + sparse roundtrips.
- Lets `analyze_jump_video.py` cache extracted kinematics so fine-tuning
  doesn't have to re-run MediaPipe + OpenSim.

**Cache hook (`scripts/analyze_jump_video.py`)**
- New `--save-samples [DIR]` flag (default off; bare flag → `data/results/samples`).
- Smoke-tested: 286 KB `.npz` per video, full roundtrip preserves all fields.

**Personal fine-tune script (`scripts/finetune_personal.py`)**
- Loads `.npz` cache, applies a Phase-9a scale guardrail (rejects clips with
  peak CoM > 3 m, configurable via `--max-peak-com-m`), reuses
  `DynamicsDataset` + Newton-Euler residual identical to pretraining,
  fine-tunes `best_model.pth` at lr=1e-4, saves
  `data/models/personal/imogen_finetuned.pth`. Has `--dry-run`.

**Phase 9a v1 — multi-segment scale calibration**
- `src/pose_estimation/scale_calibration.py`:
  - New `compute_per_frame_scale_mpp(landmarks_2d, image_w, image_h, thigh_length_m, shank_length_m, …)`.
  - For each known-length segment (thigh L/R, shank L/R from
    `SubjectInfo.thigh_length_m` / `shank_length_m`), collects pixel
    projections across all visibility-gated frames, takes the **95th
    percentile** as the in-plane projection (rejects foreshortened frames
    where the limb rotates out of the camera plane), then medians across
    segments → single video-wide metres-per-pixel.
  - Returns a (T,)-replicated constant for caller compatibility.
  - Initial v0 design used a per-frame median across segments and produced
    1.09 mm/px frame-to-frame std (~31 % of median), which inflated
    finite-difference velocities. Replaced by the across-video p95 approach
    above.
  - Ground reference now uses **5th percentile of visible-ankle Y** instead
    of `min(...)`. The old `min` was contaminated by single-frame landmark
    jitter (a lone outlier could push ground 3 m below reality, inflating
    every downstream Y).
- `calibrate_landmarks_to_world` accepts optional `thigh_length_m` and
  `shank_length_m`; falls back to legacy nose-ankle when absent.
- `scripts/analyze_jump_video.py` adds `--thigh` / `--shank` CLI flags
  (defaults 0.43 / 0.47 — Imogen's measured values).
- 8 new tests in `tests/test_pose_estimation/test_scale_calibration.py`
  covering scalar recovery, foreshortening rejection, visibility gating,
  graceful fallback, multi-segment median.
- **Validation results (smoke tests):**
  - `14_02_26_one_1.79.mp4` (previously plausible): peak CoM 2.60 → 2.39 m,
    takeoff angle 49 → 49°, vy 3.72 → 3.43 m/s, vh 3.23 → 2.97 m/s.
    All within elite-female Fosbury-Flop range.
  - `09_02_26_one.mp4` (previously broken, peak CoM 5.04 m): peak CoM
    5.04 → 4.17 → 3.41 m through v1 → v1 + ground-ref fix. Improved but
    still inflated.
  - The remaining inflation on this clip comes from MediaPipe landmark
    jitter being amplified by the larger m/px scale (athlete is far from
    camera, ~63 px thigh vs 151 px in close clip). This is partly
    fundamental — single-camera resolution is the limit. The vy=38 m/s
    artifact is also influenced by Phase 9b (`argmax(vy)` picking
    transient spikes). Re-processing all 45 clips to characterise the
    distribution of remaining error is still pending.
- All tests pass (12/12 scale calibration; full suite was 183 last verified
  before the ground-reference percentile fix — re-run before the next
  commit).

**Phase 9b — ground-contact takeoff anchor**
- `scripts/analyze_jump_video.py` now selects takeoff as the final frame of
  the last detected ankle-ground contact, using `detect_ground_contacts` on
  both ankle trajectories after converting calibrated metres to centimetres.
- Falls back to `argmax(vy)` when no contacts are detected, preserving output
  for short clips or failed pose extraction.
- Ground-contact candidates after peak CoM are ignored when pre-peak contacts
  exist, preventing landing/mat contacts from being selected as takeoff.
- Added regression tests in `tests/test_kinematics/test_takeoff.py` showing the
  report ignores a later one-frame vertical-velocity spike and still falls back
  when no contact interval is available.
- Full non-PINN test suite now reports **186 passing**.
- Full 45-video re-processing with `--save-samples --thigh 0.43 --shank 0.47`
  remains pending; this is the next validation step before personal fine-tune.

**Context files refresh**
- `.github/copilot-instructions.md`: refreshed phase list to reflect Phase 9
  audit + 9a, corrected Fosbury Flop takeoff-angle range to 38–48° from
  Dapena (1980, 1995) — was incorrectly listed as 20–24°.
- `AGENTS.md`: replaced obsolete agent-orchestrator scaffolding section
  with a sources-of-truth table.
- `CLAUDE.md`: created — tight bootstrap auto-loaded each conversation.

### Onboarding-doc inaccuracies surfaced

- `CLAUDE_ONBOARDING.md` claims "5 clips" of takeoff detection failure; actual count was **45/45**.
- `CLAUDE_ONBOARDING.md` lists optimal Fosbury Flop takeoff angle as ~20–24° (Dapena 1980); peer-reviewed values are nearer 38–48°. Not changed in this pass — flagged for review with the BMS PhD student.

### Phase 9 follow-ups (open issues uncovered by re-processing)

After the fix, **0/45** reports have negative takeoff angles or velocities (was 45/45). However, validation of the regenerated `all_sessions_report.json` revealed two upstream issues that bound how trustworthy the absolute-magnitude metrics are. They are **out of scope for the takeoff-frame fix itself** and are tracked here as deferred work.

#### 9a. Single-camera scale calibration is fragile

- **Symptom:** 27 of 45 clips report peak CoM heights between 3.0 m and 8.9 m — physically impossible (an elite female athlete's peak CoM in flight is ~2.2–2.7 m). Inflated CoM positions propagate to inflated takeoff velocities (median vy on the inflated subset is ~10 m/s; ground-truth elite range is 3.5–4.5 m/s).
- **Root cause:** `src/pose_estimation/scale_calibration.py` derives metres-per-pixel from the **nose-ankle landmark span in a single high-confidence frame**. When that frame has the athlete far from camera, partially occluded, or at an oblique angle, the normalised span shrinks and the resulting m/unit balloons (e.g. observed `nose-ankle = 0.141 normalised → 12.01 m/unit` for one clip). Once scale is wrong, every downstream CoM coordinate is wrong by the same factor.
- **Implication:** **Relative kinematics** (joint angles, body alignment, hip arch, free-knee drive direction, foot-to-ground angle) are scale-invariant and remain trustworthy on all 45 clips. **Absolute translational metrics** (CoM peak height, takeoff vy/vh, predicted bar clearance) are only trustworthy on the 18 clips where the calibration happened to land on a clean nose-ankle frame.
- **Proposed fix (preferred — works with existing footage):** add a physical-reference calibration path. The crossbar (3.98–4.02 m horizontal length, IAAF spec) and the standards (uprights at known separation) are visible in most clips. Detect them once per video (Hough line + colour mask, or a small fine-tuned YOLO head) and use the bar's known length as the scale reference instead of nose-ankle. Filenames already encode bar height (e.g. `_1.79`) which gives a vertical reference too.
- **Alternative fix (cleaner, but requires re-filming):** two-camera DLT triangulation. Pipeline support already exists in `src/pose_estimation/dlt_triangulation.py`; only requires synced second-camera setup at training sessions going forward.
- **Hardening regardless of approach:** robustify nose-ankle calibration by taking the **median scale across all frames where both nose and ankles have visibility > 0.7**, rejecting outliers via MAD, instead of trusting a single frame.

#### 9b. `argmax(vy)` was sensitive to noise spikes — fixed

- **Symptom:** on the plausible-scale subset (peak CoM < 3 m), the median takeoff angle came out at 79° — too steep. This is partly a real Fosbury-Flop signature (high vy, low vh at toe-off) but is exaggerated by the takeoff frame sometimes landing on a single-frame velocity spike where vh has been noise-suppressed near zero.
- **Root cause:** `np.gradient(com_position)` amplifies the residual jitter left by the 10 Hz Butterworth filter; `argmax(vy)` then sometimes picks a 1-frame spike rather than the impulse peak.
- **Fix:** `scripts/analyze_jump_video.py` anchors takeoff to ground contact instead of velocity. It runs `src/kinematics/run_up_analysis.py:detect_ground_contacts` on both ankle trajectories and reads `com_velocity` at the final frame of the final pre-peak contact interval. If no contact is detected, it falls back to `argmax(vy)`.
- **Validation:** `tests/test_kinematics/test_takeoff.py` covers the noise-spike case and the no-contact fallback. Full suite: 186 passing with `tests/test_pinn` ignored.

#### 9c. Optimisation results are stale relative to corrected reports

- `data/results/all_optimizations.json` was generated against the pre-fix `all_sessions_report.json` and references negative takeoff angles. After 9b above, re-run `scripts/optimize_jump.py --all data/results/all_sessions_report.json` to refresh.

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
| Athlete priority metrics (Imogen alignment) | ✅ Implemented |
| Test suite | ✅ 174 tests passing |
| Public datasets downloaded (Zenodo CMJ, CoD, DVJ) | ✅ Done |
| PINN pre-training run (3000 epochs) | ✅ Done — `final_model.pth` saved |
| Imogen's 45 jump videos processed | ✅ Re-processed Phase 9 with corrected takeoff detection |
| Takeoff detection (frame selection) | ✅ Fixed — `argmax(vy)` selects the impulse peak |
| Takeoff detection (noise robustness) | ⚠ See Phase 9b — switch to ground-contact anchor |
| Single-camera scale calibration | ⚠ See Phase 9a — 27/45 clips have inflated CoM (need crossbar reference) |
| AddBiomechanics dataset downloaded | ⬜ Pending (highest-quality GRF pre-training data) |
| Personal data fine-tuning loop | ⬜ Pending (no fine-tuned model yet) |

---

## Upcoming Work

### Immediate

1. Validate physics loss convergence: assess whether 1.263 is acceptable or re-training with AddBiomechanics would meaningfully improve GRF prediction
2. Optionally download AddBiomechanics (requires registration at simtk.org) and re-run pre-training
3. Re-run `scripts/optimize_jump.py` on the corrected reports to refresh `data/results/all_optimizations.json`

### Phase 10 — Personal Data Fine-Tuning

- Create `scripts/finetune_personal.py` — load `best_model.pth`, fine-tune on Imogen's extracted BiomechanicalSamples at lower learning rate, save to `data/models/personal/`
- Validate fine-tuned model predictions against held-out jump attempts
- Run `scripts/optimize_jump.py` on Imogen's full dataset to generate personalised intervention rankings

### Phase 11 — Validation and Paper
- Compare predicted vs. measured jump heights on held-out attempts
- Confirm sensitivity analysis marginal gains are plausible against coaching intuition
- Write up methods and results
