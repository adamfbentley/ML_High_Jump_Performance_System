---
mode: agent
description: Process a high jump video through the full pipeline — pose estimation → kinematics → PINN inference → improvement targets
tools:
  - read_file
  - file_search
  - run_in_terminal
  - grep_search
---

Process a high jump video through the full analysis pipeline.

## Required inputs (ask the user if not provided)

- Path to the video file (`.mp4`, `.mov`, or `.avi`)
- Athlete anthropometrics:
  - Body mass (kg)
  - Height (m)
  - Bar height attempted (m)

## Pipeline steps

### Step 1 — Pose estimation
```powershell
& ".venv\Scripts\python.exe" scripts/run_pose_estimation.py <video_path>
```
Read `src/pose_estimation/estimators/mediapipe_estimator.py` to understand the output format.
Output: 2D landmarks → 3D via DLT triangulation (if multi-camera) or single-view heuristic.

### Step 2 — Phase segmentation
Use `src/kinematics/run_up_analysis.py` to detect:
- Approach phases (straight run + curve)
- Penultimate step
- Takeoff (plant foot contact → last ground contact)
- Flight phase (airborne)

### Step 3 — Extract kinematics
From `src/kinematics/takeoff_analysis.py` and `src/kinematics/flight_analysis.py`:
- Takeoff angle (target: 20–24° per Dapena 1980)
- Horizontal velocity at takeoff (target: depends on athlete)
- Ground contact time
- Peak CoM height vs bar height
- Bar clearance profile

### Step 4 — PINN inference (if checkpoint available)
Load the pre-trained `InverseDynamicsPINN` from `data/models/` and compute:
- Estimated GRF time series during takeoff
- Joint torques (ankle, knee, hip)
- Compare against `F_GRF = m * (a_CoM + g)` as a sanity check

### Step 5 — Generate improvement targets
Using `src/optimization/optimizer.py`:
- Sensitivity analysis: which parameter (takeoff angle, approach speed, contact time) has the largest effect on jump height?
- What-if: quantify height gain from a +0.5 m/s approach speed improvement
- Output specific, numerical targets (not qualitative advice)

## Output format

Report these numbers specifically:
| Parameter | Current | Target | Predicted gain |
|---|---|---|---|
| Takeoff angle (°) | X | 22° | +Y cm |
| Horizontal velocity at takeoff (m/s) | X | X+0.3 | +Y cm |
| Ground contact time (ms) | X | X-10 | +Y cm |
| Peak GRF (BW) | X | X+0.2 | +Y cm |

## Physics checks

- Fitted CoM parabola R² must be > 0.95 during flight (otherwise pose estimation is wrong)
- Estimated g from parabola fit must be 9.5–10.1 m/s² (sanity check on scale)
- Takeoff velocity from kinematics must match `√(2g * Δh_CoM)` within 10%
