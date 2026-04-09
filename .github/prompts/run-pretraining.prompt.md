---
mode: agent
description: Run PINN pre-training and report loss convergence
tools:
  - read_file
  - run_in_terminal
  - grep_search
---

Run the PINN pre-training pipeline and report results.

## Steps

1. Read `experiments/configs/pretrain_dynamics.yaml` to understand current settings
2. Check `data/public/` for available datasets:
   ```powershell
   & ".venv\Scripts\python.exe" scripts/download_datasets.py
   ```
3. Run pre-training with the available data:
   ```powershell
   & ".venv\Scripts\python.exe" scripts/pretrain_dynamics_pinn.py --config experiments/configs/pretrain_dynamics.yaml
   ```
4. If no real data is available, report exactly what needs to be downloaded and from where
5. After training (or if training fails), report:
   - Final `L_data`, `L_physics`, `L_boundary` values
   - Whether physics loss decreased (it must — if it didn't, there is a bug)
   - The ratio `L_physics / L_data` at convergence (should be < 0.1 for a well-trained PINN)
   - Path to the saved checkpoint

## Physics sanity check

After training, verify the Newton-Euler residual:
- `F_GRF = m * (a_CoM + [0, 9.81, 0])` must hold within 5% RMS error on the validation set
- If it doesn't, the physics loss weight `lambda_physics` needs increasing in the config

## What NOT to do

- Do not reduce `lambda_physics` to make training easier — physics correctness is non-negotiable
- Do not use mock/synthetic data and report it as real pre-training
