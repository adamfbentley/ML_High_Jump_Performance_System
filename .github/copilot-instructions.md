# GitHub Copilot Instructions — High Jump Biomechanics System

## Project Goal

Build a system where a high jumper films their attempts on a phone, enters their
anthropometrics (height, mass, limb lengths), and receives:
- Exact physics of their run-up curve, penultimate step, plant, takeoff angle, and bar clearance
- Actionable, quantified improvement targets: *"increase horizontal velocity by 0.3 m/s at takeoff"*
- What-if simulations: *"if I speed up my approach by 0.5 m/s, predicted height gain = +3 cm"*
- Personalised predictions grounded in real biomechanics, not population averages

**This is a research project.** The primary users are an AI PhD student, a Biomedical PhD
student, and a national-champion high jumper. Correctness and scientific validity beat
code elegance. All physics must be explicitly justified. Peer-reviewed biomechanics
literature (Winter 2009, de Leva 1996, Dapena 1980, Rajagopal 2015) is the authority.

---

## Architecture

```
src/
├── pose_estimation/     # MediaPipe BlazePose → 2D landmarks → 3D via DLT
├── kinematics/          # Phase segmentation, run-up metrics, takeoff/flight analysis
├── pinn/                # Physics-Informed Neural Networks (the core innovation)
│   ├── physics/         #   Projectile PINN (flight), InverseDynamics PINN (takeoff)
│   ├── models/          #   Per-joint PINN (JointPINN, Euler-Lagrange)
│   └── training/        #   Composite loss: data + physics residual + boundary conditions
├── gnn/                 # Skeleton GNN — couples JointPINNs via message passing
├── optimization/        # Differentiable simulation → gradient-based technique search
├── data_pipeline/       # BiomechanicalSample format, loaders, PyTorch datasets
└── utils/               # constants.py — GRAVITY, segment mass fractions (de Leva 1996)
```

**Key data format:** `BiomechanicalSample` (`src/data_pipeline/sample.py`) is the unified
container for all cross-dataset data. Every loader must produce this type. Fields:
- `joint_angles`, `joint_angular_velocities`, `joint_angular_accelerations` — kinematics
- `grf` — ground reaction force (N), 3-component
- `joint_torques` — inverse dynamics result (N·m)
- `com_position`, `com_velocity`, `com_acceleration` — centre of mass
- `subject` (SubjectInfo) — `body_mass_kg`, `height_m` required for physics

**Movement type priority** (MOVEMENT_RELEVANCE in sample.py):
- HIGH_JUMP = 1.0, CMJ = 0.9, VERTICAL_JUMP = 0.85, DROP_JUMP = 0.8
- Always filter for the highest-relevance movements first

---

## Physics Conventions

**Always follow these; never change them without a comment:**

- Coordinate system: Y-up, right-handed. X = forward (direction of run-up), Z = lateral
- Gravity vector: `[0, -9.81, 0]` m/s² (from `src/utils/constants.py`)
- `F_GRF = m * (a_CoM - g_vec)` where `g_vec = [0, -9.81, 0]`
  — equivalently `F_GRF = m * (a_CoM + [0, 9.81, 0])`
- Joint angles in **radians** everywhere internally; convert deg→rad on ingestion
- Segment mass fractions from **Winter (2009) Table 4.1** (see `constants.py`)
- CoM estimation uses **de Leva (1996)** proximal segment offsets
- Fosbury Flop optimal takeoff angle: empirically ~20–24° from horizontal (Dapena 1980)

**PINN loss structure** (`src/pinn/training/trainer.py`):
```
L_total = λ_data * L_data + λ_physics * L_residual + λ_boundary * L_boundary
```
Default: `λ_data=1.0, λ_physics=1.0, λ_boundary=10.0` (boundary conditions tightly enforced)
Physics residual uses `torch.autograd.grad` — always called with `create_graph=True`

---

## Datasets (for pre-training)

Public datasets in `data/public/` (all excluded from git):

| Dataset | Format | Win-compatible | Priority | Contents |
|---|---|---|---|---|
| AddBiomechanics | `.b3d` (Linux/mac) or `IK/ID/GRF/*.mot` dirs (Windows) | ✓ (OpenSim export) | 10 | Full inverse dynamics, GRF, CoM |
| BioCV | `.c3d` via `ezc3d` | ✓ | 9 | Markers + force plates |
| OpenCap | `.trc` + `.mot` text files | ✓ | 8 | Markerless kinematics |
| AthletePose3D | COCO JSON | ✓ | 7 | 2D/3D pose pairs |

**On Windows**, AddBiomechanics must use the OpenSim text export (subject folders with
`IK/`, `ID/`, `GRF/`, `bodyKinematics/` subdirs). `nimblephysics` is Linux/macOS only.

---

## Code Conventions

**Python version:** 3.10+ — use `X | Y` union types, `match` statements where appropriate

**Imports:** stdlib → third-party → local `src.*`. Always use `from __future__ import annotations`

**Numpy/PyTorch shapes:** Document array shapes in docstrings as `(T, 3)` notation.
- `T` = time frames, `N` = batch/samples, `J` = joints, `D` = feature dim

**Physical quantities:** Include units in variable names or docstrings. Prefer SI throughout.
- Angles → radians (`angle_rad`) unless a UI/output function where degrees is appropriate
- Forces → Newtons, Torques → N·m, Mass → kg, Length → metres, Time → seconds

**No mock data.** Every function must implement real physics or real ML, or raise
`NotImplementedError` with a comment explaining what's needed.

**Test coverage:** All new functions in `src/` that don't require torch/GPU get a
pytest test in `tests/`. Physics-law tests (e.g. energy conservation, F=ma) are preferred
over pure unit tests. Run with: `pytest tests/ --ignore=tests/test_pinn`

**Line length:** 100 (configured in `pyproject.toml` → ruff)

---

## What "Done" Looks Like

A task is complete when:
1. The physics is correct (check with a sanity-check calculation or test)
2. The function handles the real data format (`BiomechanicalSample` or raw numpy)
3. There is a test (even a minimal one)
4. The function could, in principle, be called from the pre-training script
   `scripts/pretrain_dynamics_pinn.py` or the inference path

Do **not** add deployment infrastructure (FastAPI endpoints, Docker, S3) to `src/`.
That belongs in `services_scaffold/` for later.

---

## Research Phases (current status)

- ✅ Phase 0: Workspace structure, data pipeline, public dataset loaders
- ✅ Phase 1a: Pose estimation pipeline (MediaPipe, BVH, joint angles, CoM)
- ✅ Phase 1b: Kinematics modules (run-up, takeoff, flight analysis)
- ✅ Phase 2a: PINN architecture (ProjectilePINN, InverseDynamicsPINN, JointPINN)
- ✅ Phase 2b: GNN skeleton coupling (SkeletonGNN, ForceMessageLayer)
- ✅ Phase 2c: Pre-training infrastructure (data loaders, training script, configs)
- 🔄 **Current:** Download public data → pre-train PINNs → validate physics loss convergence
- ⬜ Phase 3: Optimisation engine (gradient-based, sensitivity analysis, what-if)
- ⬜ Phase 4: Personal data loop (film → estimate pose → fine-tune → feedback)

---

## Key Files to Read First

When working on any task, orient yourself with:
- `src/data_pipeline/sample.py` — the canonical data format
- `src/utils/constants.py` — all physical constants and segment parameters
- `src/pinn/physics/inverse_dynamics.py` — the main pre-training target
- `experiments/configs/pretrain_dynamics.yaml` — current training config
- `tests/` — running tests confirms nothing is broken
