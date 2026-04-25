# System Architecture

## Overview

This document describes the architecture of the Physics-Informed Machine Learning system for high jump biomechanical analysis. The system ingests video footage of high jump attempts, extracts biomechanical data from those recordings, and produces personalised, quantified technique recommendations grounded in Newtonian mechanics.

The processing pipeline follows a linear chain of transformations:

```
Video → Pose Estimation → Kinematics → PINN → Optimiser → Feedback Report
```

Each stage is implemented as an independent module under `src/`, with defined input and output contracts. The canonical data interchange format is `BiomechanicalSample` (see `src/data_pipeline/sample.py`).

---

## Repository Structure

```
src/                          # All research and ML source code
    pose_estimation/          # Video → 3D joint positions
    kinematics/               # Biomechanical phase segmentation and metrics
    pinn/                     # Physics-Informed Neural Networks
    gnn/                      # Graph Neural Network for skeleton force propagation
    optimization/             # Gradient-based technique optimisation
    data_pipeline/            # Public dataset loaders and BiomechanicalSample format
    utils/                    # Physical constants (constants.py)
    visualization/            # Pose overlay rendering

scripts/                      # Runnable entry points (pre-training, analysis, optimisation)
tests/                        # Automated test suite (174 tests, physics-law validation)
experiments/configs/          # YAML configuration files for training runs
data/public/                  # Downloaded pre-training datasets (excluded from git)
services_scaffold/            # Deployment microservices (parked, not connected to src/)
```

---

## Module Descriptions

### 1. Pose Estimation (`src/pose_estimation/`)

This module accepts a video file and produces time-series 3D joint positions for every frame.

**Processing steps:**
1. **MediaPipe BlazePose** detects 33 anatomical landmarks per frame from a single-camera video stream, yielding 2D pixel coordinates with confidence scores.
2. When multiple synchronised camera angles are available, **DLT triangulation** (`dlt_triangulation.py`) reconstructs true 3D world coordinates via the Direct Linear Transform.
3. Raw landmark trajectories pass through a **post-processor** (`landmark_postprocessor.py`) that fills temporal gaps via cubic interpolation, applies a 4th-order zero-lag Butterworth low-pass filter (cutoff configurable, default 6 Hz), and enforces anatomical segment length constraints.
4. **Joint angles** (`skeleton/joint_angles.py`) are computed from landmark triples using the law of cosines. All angles are stored in radians. Currently defined joints: bilateral knee flexion, hip flexion, hip abduction/adduction, ankle dorsiflexion, elbow flexion, and shoulder flexion.
5. **Centre of mass** (`skeleton/com_estimator.py`) is estimated each frame using the de Leva (1996) segment model: each body segment's mass fraction and proximal CoM offset are drawn from `src/utils/constants.py`, and the whole-body CoM is the mass-weighted sum of segment CoMs.
6. A **BVH skeleton** (`skeleton/bvh_generator.py`) is generated scaled to the athlete's measured anthropometrics for visualisation and downstream export.

**Physics conventions:** Y-up right-handed coordinate system; X = direction of run-up; Z = lateral. Gravity vector `[0, −9.81, 0]` m/s². Joint angles in radians throughout.

---

### 2. Kinematics (`src/kinematics/`)

This module takes 3D landmark positions and CoM trajectories and computes phase-segmented biomechanical metrics.

#### 2.1 Run-Up Analysis (`run_up_analysis.py`)

- Detects ground contact events from vertical ankle position minima.
- Segments the attempt into phases: `APPROACH → CURVE → PENULTIMATE → TAKEOFF → FLIGHT → LANDING`.
- Computes horizontal velocity at each stride, per-stride ground contact time (GCT), foot placement relative to the athlete's hip (foot-under-hip offset), and lateral deviation from the prescribed J-curve.
- Fits the curve radius using a least-squares circle fit to the ground contact positions.
- Output: `RunUpMetrics` dataclass.

#### 2.2 Takeoff Analysis (`takeoff_analysis.py`)

- Estimates ground reaction forces from CoM kinematics via Newton's second law: `F_GRF = m × (a_CoM − g_vec)`, where `g_vec = [0, −9.81, 0]` m/s².
- Computes takeoff angle, vertical velocity at toe-off, total impulse, and the foot-to-ground contact angle at touchdown.
- Scores body alignment (deviation from straight-line hip–shoulder–ankle geometry) and extracts arm drive and free-leg knee drive peak speeds with their timing relative to toe-off.
- Predicts maximum CoM height from takeoff conditions using the projectile equation: `h_max = v_y² / (2g)`.
- Output: `TakeoffMetrics` dataclass.

#### 2.3 Flight Analysis (`flight_analysis.py`)

- Fits a second-order polynomial to the CoM trajectory during the airborne phase; residuals from the fitted parabola quantify non-gravitational perturbations.
- Computes the clearance profile: the vertical distance between each body segment and the bar at each frame.
- Records peak CoM height, bar clearance margin, vertical extension time (frames before the CoM begins descending), and the arch transition frame (where hip extension reverses into flexion for the Fosbury Flop).
- Output: `FlightMetrics` dataclass.

---

### 3. Physics-Informed Neural Networks (`src/pinn/`)

PINNs embed physical laws directly into the training loss, constraining the network to produce outputs consistent with Newtonian mechanics. This allows generalisation from smaller datasets and enables physically meaningful extrapolation.

**Loss structure** (defined in `src/pinn/training/trainer.py`):

```
L_total = λ_data · L_data  +  λ_physics · L_residual  +  λ_boundary · L_boundary
```

Default weights: `λ_data = 1.0`, `λ_physics = 1.0`, `λ_boundary = 10.0`. All physics residuals are computed via `torch.autograd.grad` with `create_graph=True`.

#### 3.1 Projectile PINN (`pinn/physics/projectile.py`)

Models the flight phase. Enforces `ẍ = 0, ÿ = −g, z̈ = 0` (pure gravitational motion). Input: time. Output: 3D CoM position. Used to validate physics loss convergence before training more complex models.

#### 3.2 Inverse Dynamics PINN (`pinn/physics/inverse_dynamics.py`)

The primary pre-training target. Input: time, body mass, body height, joint angles, joint angular velocities. Output: ground reaction forces, joint torques, CoM acceleration. The physics residual enforces the Newton-Euler equation: predicted GRF must satisfy `F_GRF = m · a_CoM − m · g_vec`. Pre-trained on public biomechanics datasets (AddBiomechanics, BioCV, OpenCap) before fine-tuning on athlete-specific data.

#### 3.3 Joint PINN (`pinn/models/joint_pinn.py`)

Per-joint model based on the Euler-Lagrange equations of motion. Input: joint angle, angular velocity, segment inertial properties. Output: angular acceleration, joint torque, contact forces. Intended for high-resolution per-joint analysis in later project phases.

---

### 4. Graph Neural Network (`src/gnn/skeleton_gnn.py`)

Models the musculoskeletal system as a directed graph where joints are nodes and body segments are edges. Force and moment information propagates between connected joints via `ForceMessageLayer`, implementing learned message-passing functions. This captures mechanical coupling across the kinematic chain — for example, the influence of ankle joint stiffness on knee and hip loading.

The `SkeletonGNN` couples multiple `JointPINN` instances: each joint's PINN output is used as a node feature, and inter-joint messages modulate the physics residuals of neighbouring joints.

---

### 5. Optimisation Engine (`src/optimization/optimizer.py`)

Takes a trained PINN and finds technique parameter values that maximise predicted jump height subject to athlete-specific constraints.

**Parameterisation (`TechniqueParameters`):** approach speed, curve radius, penultimate step length, final step length, plant angle (leg-to-horizontal at foot contact), takeoff lean angle, takeoff direction angle, arm swing timing, free-leg angle at takeoff, ground contact time on the takeoff foot, body alignment score, foot-to-mat angle at touchdown, free-leg knee drive speed, and curve start step number.

**Constraint model (`AthleteConstraints`):** maximum approach speed, maximum joint torques, and joint range-of-motion bounds derived from the athlete's anthropometrics and measured movement capacity.

**Method:** The PINN serves as a differentiable physics simulator. Gradients of predicted jump height with respect to each `TechniqueParameter` are computed via automatic differentiation. A gradient-based solver (L-BFGS or Adam) iterates until the height gain converges within constraint bounds.

**Sensitivity analysis:** Partial derivatives of jump height with respect to each parameter, evaluated at the athlete's current technique, quantify the marginal gain from each intervention. This produces ranked, actionable outputs such as: *"increasing approach speed by 0.3 m/s is predicted to add 2.8 cm; adjusting arm swing timing by 40 ms is predicted to add 0.4 cm."*

---

### 6. Data Pipeline (`src/data_pipeline/`)

Loads, normalises, and prepares public biomechanics datasets for PINN pre-training.

#### 6.1 BiomechanicalSample Format (`sample.py`)

The canonical data container. All loaders must produce this type. Key fields:

| Field | Shape | Units | Description |
|---|---|---|---|
| `joint_angles` | (T, J) | rad | Anatomical joint angles |
| `joint_angular_velocities` | (T, J) | rad/s | First temporal derivative |
| `joint_angular_accelerations` | (T, J) | rad/s² | Second temporal derivative |
| `grf` | (T, 3) | N | Ground reaction force, 3-component |
| `joint_torques` | (T, J) | N·m | Inverse dynamics result |
| `com_position` | (T, 3) | m | Centre of mass position |
| `com_velocity` | (T, 3) | m/s | CoM velocity |
| `com_acceleration` | (T, 3) | m/s² | CoM acceleration |
| `subject` | — | — | `SubjectInfo`: `body_mass_kg`, `height_m` required |

#### 6.2 Movement Relevance (`MOVEMENT_RELEVANCE`)

Datasets and movement types are filtered by relevance to high jump mechanics:

| Movement | Score |
|---|---|
| `HIGH_JUMP` | 1.0 |
| `CMJ` | 0.9 |
| `VERTICAL_JUMP` | 0.85 |
| `DROP_JUMP` | 0.8 |

Note: Based on domain-expert input, single-leg drop jump is considered a closer transfer to high jump takeoff mechanics than countermovement jump. The `DROP_JUMP` relevance score is under review.

#### 6.3 Dataset Loaders

| Dataset | Priority | Format | Path |
|---|---|---|---|
| AddBiomechanics | 10 | OpenSim `.mot`/`.sto` (Windows) | `data/public/addbiomechanics/` |
| BioCV | 9 | `.c3d` via `ezc3d` | `data/public/biocv/` |
| OpenCap | 8 | `.trc` + `.mot` | `data/public/opencap/` |
| Zenodo CMJ GRF | 7 | `.npz` | `data/public/cmj_grf_zenodo/` |

On Windows, AddBiomechanics must use the OpenSim text export (`IK/`, `ID/`, `GRF/`, `bodyKinematics/` subdirectories). The `nimblephysics` binary format is Linux/macOS only.

#### 6.4 PyTorch Dataset Wrappers

- `DynamicsDataset` — feeds windowed kinematics into the inverse dynamics PINN.
- `FlightPhaseDataset` — extracts airborne segments for the projectile PINN.
- `PoseLiftingDataset` — 2D landmark → 3D pose lifting supervision.

---

## Entry-Point Scripts

| Script | Purpose |
|---|---|
| `scripts/pretrain_dynamics_pinn.py` | Train the inverse dynamics PINN on public datasets. Reads `experiments/configs/pretrain_dynamics.yaml`. |
| `scripts/analyze_jump_video.py` | Full pipeline: video → pose → kinematics → PINN inference → feedback report. |
| `scripts/optimize_jump.py` | Run the gradient-based technique optimiser on extracted jump data. |
| `scripts/run_pose_estimation.py` | Process a single video through MediaPipe and save landmarks. |
| `scripts/download_datasets.py` | Print instructions for manually downloading each public dataset. |
| `scripts/train_projectile_pinn.py` | Smoke-test: train the projectile PINN on a synthetic parabolic trajectory. |

---

## Test Suite

174 automated tests in `tests/`, organised by module. Tests are run with:

```
pytest tests/ --ignore=tests/test_pinn -q
```

Test categories include joint angle computation (edge cases: parallel, perpendicular, and collinear vectors), CoM estimation (3D validation, convex hull containment, trajectory shape), BVH generation, data pipeline (sample properties, registry queries, all transforms), file parsers (TRC and MOT formats), landmark post-processing, scale calibration, kinematics (phase segmentation, takeoff metrics, flight parabola), GNN (message passing correctness), and optimiser (sensitivity analysis, constraint enforcement).

Physics-law tests are preferred over pure unit tests: where possible, assertions verify energy conservation, Newton's second law, or projectile motion invariants rather than fixed numerical outputs.

---

## Research Phase Status

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ Complete | Repository structure, data pipeline, dataset loaders |
| 1a | ✅ Complete | Pose estimation pipeline (MediaPipe, BVH, joint angles, CoM) |
| 1b | ✅ Complete | Kinematics modules (run-up, takeoff, flight analysis) |
| 2a | ✅ Complete | PINN architecture (ProjectilePINN, InverseDynamicsPINN, JointPINN) |
| 2b | ✅ Complete | GNN skeleton coupling (SkeletonGNN, ForceMessageLayer) |
| 2c | ✅ Complete | Pre-training infrastructure (loaders, training script, configs) |
| 3 | 🔄 Current | Download public datasets → pre-train PINNs → validate physics loss |
| 4 | ⬜ Pending | Optimisation engine: gradient-based search, sensitivity analysis |
| 5 | ⬜ Pending | Personal data loop: film → estimate pose → fine-tune → feedback |

---

## Out of Scope (Currently)

`services_scaffold/` contains 12 FastAPI microservices (API gateway, user profiles, video ingestion, feedback reporting, etc.) intended for a future web deployment. These services are not connected to `src/` and should be disregarded during research phases 3–5.


---

## Folder Structure

```
src/                          ← all the real research code lives here
    pose_estimation/          ← gets body positions from video
    kinematics/               ← analyses the movement phases
    pinn/                     ← physics-informed neural networks
    gnn/                      ← graph neural network for skeleton forces
    optimization/             ← finds what technique changes improve height
    data_pipeline/            ← loads + processes public training datasets

scripts/                      ← runnable entry points
tests/                        ← automated tests (41 passing)
experiments/configs/          ← training configuration files
data/public/                  ← where downloaded training data goes
services_scaffold/            ← web deployment stuff (not needed yet, parked)
```

---

## The Modules (What Each One Does)

### 1. Pose Estimation (`src/pose_estimation/`)

**What it does:** Takes a video of a jump and extracts the 3D positions of every joint in every frame.

**How it works:**
- Uses Google's MediaPipe BlazePose — it detects 33 body landmarks (ankles, knees, hips, shoulders, wrists, etc.) from a single camera view
- If we have mulitple camera angles, there's a triangulation module that combines 2D views into proper 3D coordinates using a technique called DLT (Direct Linear Transform)
- From the 3D landmarks it can generate a full skeleton file (.bvh format) scaled to your actual body proportions

**Key bits:**
- **MediaPipe Estimator** — processes a video file, gives you a time series of 33 landmark positions
- **Joint Angles** — computes anatomical angles (knee flexion, hip flexion, etc.) from the landmark positions. These are the same angles a biomechanist would measure
- **Centre of Mass (CoM)** — estimates where your body's centre of mass is each frame using a segment model from the literature (de Leva 1996). This is crucial becuase the CoM trajectory is what determines jump height
- **BVH Generator** — builds a skeleton rig scaled to ur anthropometrics (height, limb lengths) so we can visualise the movement and export it

**Why it matters:** This is the entry point for your video data. U film a jump → this module extracts the biomechanics from it automatically.

---

### 2. Kinematics (`src/kinematics/`)

**What it does:** Takes the pose data and breaks the jump down into phases, then computes performance metrics for each phase.

**Three sub-modules:**

- **Run-Up Analysis** — detects ground contacts from ankle positions, computes horizontal velocity through the approach, fits the curve radius (the J-curve you run before takeoff). Segments the jump into phases: APPROACH → CURVE → PENULTIMATE → TAKEOFF → FLIGHT → LANDING

- **Takeoff Analysis** — the critical bit. Estimates ground reaction forces from CoM acceleration (Newton's second law — F = ma), computes takeoff angle, vertical velocity at takeoff, and impulse. Also has a function that predicts maximum CoM height from takeoff conditions using projectile equations

- **Flight Analysis** — fits a parabola to the CoM trajectory during flight (should be near-perfect projectile motion in the air), computes clearance profile over the bar for each body part, and measures peak height

**Why it matters:** These are the metrics that connect what you see in the video to what actually determines jump height. Takeoff velocity, takeoff angle, CoM height at takeoff — changing these is how u jump higher. The model needs to learn these reltionships.

---

### 3. Physics-Informed Neural Networks (`src/pinn/`)

**What it does:** These are the core ML models. Unlike normal neural networks that just learn patterns from data, PINNs have physics equations baked into their loss function. So they can't learn something that violates Newton's laws.

**Three model types:**

- **Projectile PINN** — models the flight phase. Enforces ẍ = 0, ÿ = -g, z̈ = 0 (i.e. in the air, the only force is gravity). Input: time → Output: 3D CoM position. This is the simplest one and it's already tested and working

- **Inverse Dynamics PINN** — the main one for pre-training. Input: time, body mass, height, joint angles and angular velocities → Output: ground reaction forces, joint torques, CoM acceleration. The physics constraint is Newton-Euler: the predicted GRF must equal mass × (CoM acceleration + gravity). This is what we'll train on public data first

- **Joint PINN** — per-joint model based on Euler-Lagrange mechanics. Input: joint angle, angular velocity, segment properties → Output: angular acceleration, torque, contact forces. For more detailed per-joint analysis later

**Training (`src/pinn/training/trainer.py`):**
The training loop uses a composite loss = data_loss + physics_loss + boundary_loss. The physics loss is what makes it a PINN rather than just a neural network. It means the model can generalise from small amounts of data because it already "knows" how forces and motion relate.

**Why it matters:** This is basically the brain of the system. Once trained, you can ask it "if I change my takeoff angle by 5°, what happens to my jump height?" and it'll give a physically plausible answer, not just a statistical guess.

---

### 4. Graph Neural Network (`src/gnn/`)

**What it does:** Models the skeleton as a graph where joints are nodes and bones are edges, then passes force/moment messages along the skeleton.

- **SkeletonGNN** — uses message-passing layers where each joint sends information to its neighbours (like how force really propagates through a linked body chain)
- The edges are the actual bone connections (ankle↔knee, knee↔hip, etc.)
- This captures the fact that what ur ankle does affects ur knee which affects your hip — they're not independant

**Why it matters:** It lets the model understand that the body is a connected chain, not just a bag of independent joint angles. This is important for realistic optimisation — you can't change one joint without affecting the others.

---

### 5. Optimisation (`src/optimization/`)

**What it does:** Takes a trained PINN model and uses gradient-based optimisation to find what technique changes would increase jump height, within your physical constraints.

- **TechniqueParameters** — 9 controllable variables: approach speed, curve radius, penultimate step length, last step length, plant angle, takeoff lean angle, takeoff direction angle, arm swing timing, free leg angle
- **AthleteConstraints** — your personal limits: max speed, max joint torques, range of motion bounds
- **Optimiser** — uses the PINN as a differentiable physics simulator. It literally computes the gradient of predicted jump height with respect to each technique parameter, then nudges them in the direction that increases height
- **Sensitivity Analysis** — tells you which parameters matter most (e.g. "increasing ur approach speed by 0.5 m/s would add ~3cm, but changing arm timing would only add ~0.5cm")

**Why it matters:** This is the end product — the thing that actually tells u what to change and by how much. And becuase it's gradient-based through a physics model, it respects your body's actual mechanics rather than just saying "be faster" generically.

---

### 6. Data Pipeline (`src/data_pipeline/`)

**What it does:** Loads, normalises, and prepares public biomechanics datasets for pre-training the PINNs before we have your personal data.

**Sub-components:**

- **Sample format** — a unified data container (`BiomechanicalSample`) that holds joint angles, forces, torques, CoM trajectories etc. regardless of which dataset it came from. Every dataset gets converted into this common format

- **Registry** — a catalogue of the 5 public datasets we're using, ranked by priority:
  1. **AddBiomechanics** (priority 10) — 273 subjects, full inverse dynamics. Best for PINN training
  2. **BioCV** (priority 9) — 15 subjects with synchronised video + motion capture + force plates
  3. **OpenCap** (priority 8) — ~100 subjects, markerless capture from phone video
  4. **AthletePose3D** (priority 7) — 1.3M frames of athletic poses (track & field, gymnastics etc.)
  5. **Vertical Jump IMU** (priority 6) — supplementary IMU data

- **Loaders** — one per dataset, handles the specific file formats:
  - AddBiomechanics: reads OpenSim text files (.mot/.sto) on Windows, or .b3d binary on Linux
  - BioCV: reads .c3d motion capture files
  - OpenCap: reads .trc marker files + .mot motion files
  - AthletePose3D: reads COCO-format JSON annotations

- **Transforms** — normalise by body mass (so forces are in N/kg), normalise by height, slice into fixed-length windows, apply low-pass Butterworth filter to remove noise, compute velocity/acceleration from position data

- **PyTorch Datasets** — three wrappers that feed processed data into the neural network training:
  - `DynamicsDataset` — for inverse dynamics PINN (kinematics → forces)
  - `FlightPhaseDataset` — for projectile PINN (extracts airborne segments)
  - `PoseLiftingDataset` — for 2D→3D pose lifting

**Why it matters:** We need thousands of movement samples to train the PINN before it ever sees your data. This pipeline handles the messy reality of differnt labs using different file formats, different marker sets, different sampling rates etc. and turns it all into one clean format.

---

## Scripts (How to Run Things)

| Script | What it does |
|---|---|
| `scripts/download_datasets.py` | Checks which datasets r downloaded, prints instructions for getting them |
| `scripts/pretrain_dynamics_pinn.py` | Trains the inverse dynamics PINN on public data. Reads a YAML config, loads datasets, runs the training loop, saves checkpoints |
| `scripts/train_projectile_pinn.py` | Quick proof-of-concept: generates a synthetic parabolic trajectory and trains the projectile PINN to recover it |
| `scripts/run_pose_estimation.py` | Processes a single video through MediaPipe and saves the landmarks |

---

## Tests

41 automated tests covering:
- Joint angle computation (parallel, perpendicular, opposite vectors, straight/bent limbs)
- Centre of mass estimation (3D point validation, convex hull check, trajectory shape)
- BVH skeleton generation (correct hierarchy, file output)
- Data pipeline (sample properties, registry queries, all transforms)
- File parsers (TRC and MOT format parsing)
- Projectile PINN (model shape, loss functions, physics convergence)

All passing on the current codebase.

---

## Services Scaffold (Parked)

There's a `services_scaffold/` folder with 12 FastAPI microservices — things like an API gateway, user profiles, video ingestion, feedback reporting. These are for when we eventually deploy this as a web app. They're not connected to the research code yet and you can completely ignore them for now. They'll become useful once the models are trained and we want to build an interface around them.

---

## What's Done vs What's Next

### Done ✓
- Full pose estimation pipeline (video → 3D landmarks → joint angles → CoM)
- Phase segmentation and biomechanical metrics
- Three types of physics-informed neural networks
- Graph neural network for skeleton force propagation
- Gradient-based technique optimiser with sensitivity analysis
- Complete data pipeline for 5 public datasets
- Pre-training script with YAML config
- 41 passing tests

### In Progress
- Downloading AddBiomechanics dataset (need the actual data files)
- Pre-training the inverse dynamics PINN on public data

### Next Steps
1. Pre-train PINN on AddBiomechanics
2. Process your high jump videos through pose estimation
3. Fine-tune the pre-trained model on your personal data
4. Run the optimiser to get personalised recommendations
5. Validate: do the recommendations actually match what a coach would say?

---

## Tech Stack

- **Python 3.11** — everything's Python
- **PyTorch** — neural network framework (PINNs, GNN)
- **MediaPipe** — Google's pose estimation
- **NumPy / SciPy** — numerical computing, signal processing
- **OpenCV** — video processing
- **scikit-learn** — some classical ML utilities

If you have any questions about any of this just ask — happy to go deeper on any bit.
