# High Jump Biomechanical Analysis System

> Personalised high jump technique optimisation using physics-informed neural networks (PINNs), markerless pose estimation, and graph neural networks.



---

## The Vision

A system where you film your high jump attempts on a phone, enter your height/weight/limb lengths, and get back:
- Exact physics of your run-up curve, plant, takeoff angle, and bar clearance
- Actionable improvements: *"increase horizontal velocity by 0.3 m/s"*, *"adjust penultimate step braking"*, *"optimal takeoff angle ≈ 22°"*
- What-if simulations: *"what happens if I speed up my approach by 0.5 m/s?"*
- Predicted jump height gains from each change

## Project Structure

```
├── src/                          # Core research code
│   ├── pose_estimation/          # Markerless CV pipeline
│   │   ├── estimators/           #   MediaPipe, ViTPose, multi-view triangulation
│   │   └── skeleton/             #   BVH generation, joint angles, CoM estimation
│   ├── kinematics/               # High-jump phase analysis
│   │   ├── run_up_analysis.py    #   Approach curve, step metrics, velocity profile
│   │   ├── takeoff_analysis.py   #   Plant, GRF, takeoff angle, impulse
│   │   └── flight_analysis.py    #   Bar clearance, CoM parabola fitting
│   ├── pinn/                     # Physics-Informed Neural Networks
│   │   ├── physics/              #   Projectile PINN, inverse dynamics PINN
│   │   ├── models/               #   Per-joint PINN architecture
│   │   └── training/             #   Training loop with physics-weighted loss
│   ├── gnn/                      # Graph Neural Network
│   │   └── skeleton_gnn.py       #   Joint coupling via message passing
│   ├── optimization/             # Technique optimisation
│   │   └── optimizer.py          #   Gradient-based search, sensitivity, what-if
│   ├── visualization/            # Plotting & overlays
│   ├── data_pipeline/            # Video loading, session management
│   ├── api/                      # Simple FastAPI for athlete/coach
│   └── utils/                    # Constants, shared helpers
│
├── data/                         # All data (gitignored, keep backups)
│   ├── videos/raw/               #   Raw phone recordings
│   ├── videos/processed/         #   Trimmed/stabilised clips
│   ├── poses/                    #   Extracted 2D and 3D landmarks
│   ├── bvh/                      #   Generated BVH skeleton files
│   ├── parameters/               #   Computed biomechanical parameters
│   ├── models/                   #   Trained model checkpoints
│   └── calibration/              #   Camera calibration files
│
├── notebooks/                    # Jupyter experiments
├── experiments/                  # Training configs & results
├── scripts/                      # Quick-start scripts
│   ├── run_pose_estimation.py    #   Process a video with MediaPipe
│   └── train_projectile_pinn.py  #   Train flight-phase PINN on synthetic data
├── tests/                        # Test suite
├── paper/                        # LaTeX for publication
│
├── services_scaffold/            # [REMOVABLE] Microservices deployment scaffolding
│                                 #   12 FastAPI services with simulated ML logic
│                                 #   See services_scaffold/README.md for details
│
├── pyproject.toml                # Python project config & dependencies
├── .gitignore
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- A phone camera (120fps slow-mo recommended)
- GPU optional but recommended for PINN training

### Setup

```bash
git clone <repository-url>
cd high-jump-biomechanics

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install core dependencies
pip install -e .

# Install everything (including GNN, PINN, dev tools)
pip install -e ".[all]"
```

### Run Pose Estimation on a Video

```bash
python scripts/run_pose_estimation.py data/videos/raw/jump_001.mp4
```

### Train the Projectile PINN (Synthetic Data Proof-of-Concept)

```bash
python scripts/train_projectile_pinn.py
```

This generates a noisy parabolic CoM trajectory and trains the PINN to recover it — validates the approach without needing real data.

### Run Tests

```bash
pytest
```

## Research Phases

### Phase 1: Data Foundation
Film jumps, run MediaPipe BlazePose, validate 2D→3D reconstruction against known measurements.

### Phase 2: Physics-Informed Models
Build PINNs for flight phase (projectile), takeoff (inverse dynamics/GRF), and joint-level dynamics. Couple via GNN.

### Phase 3: Optimisation & What-If
Use trained PINNs as differentiable simulators. Gradient-based optimisation of technique parameters. Sensitivity analysis.

### Phase 4: Personalisation
Population model (Transformer on BVH corpus), personal LoRA fine-tuning, longitudinal anomaly detection.

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Pose Estimation | MediaPipe BlazePose, OpenCV, multi-view triangulation |
| Skeleton/BVH | Custom BVH generator with anthropometric scaling (Winter 2009) |
| Kinematics | NumPy/SciPy — CoM estimation (de Leva 1996), joint angles, phase segmentation |
| PINNs | PyTorch — projectile, inverse dynamics, per-joint Euler-Lagrange |
| GNN | PyTorch Geometric — skeletal force/moment propagation |
| Optimisation | PyTorch autograd — gradient-based technique optimisation |
| Visualisation | Matplotlib |

## Deployment Scaffolding

The `services_scaffold/` directory contains 12 FastAPI microservices designed for eventual web deployment. These are separated because all ML logic is currently simulated — they become useful once the research produces working models. See [services_scaffold/README.md](services_scaffold/README.md) for details.

To remove:
```bash
rm -rf services_scaffold/
```

## License

MIT
