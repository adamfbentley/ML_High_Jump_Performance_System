# Architecture Overview
### How Everything Fits Together

Hey — this doc walks u through every part of the codebase, what it does, and how it all connects. Ive tried to keep it jargon-light but if anythings unclear just shout.

---

## The Big Picture

The goal is pretty simple: take video of a high jump, extract the biomechanics from it, feed that into physics-aware neural networks, and get back specific personalised advice on what to change to jump higher.

Here's the pipeline at a high level:

```
Video → Pose Estimation → Kinematics → Physics Model (PINN) → Optimiser → Feedback
```

Each of those arrows is a module in the code. Right now were at the stage where the modules all exist and we're about to start training the physics model on public biomechanics datasets before we fine-tune it on your actual jump data.

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
