---
mode: agent
description: Add a new physics constraint or loss term to an existing PINN
tools:
  - read_file
  - grep_search
  - replace_string_in_file
  - run_in_terminal
---

Add a new physics constraint to the PINN system.

## Before starting

1. Read `src/pinn/training/trainer.py` — understand the existing loss structure:
   ```
   L_total = λ_data * L_data + λ_physics * L_physics + λ_boundary * L_boundary
   ```
2. Read the target PINN file (projectile, inverse dynamics, or joint PINN)
3. Read `src/utils/constants.py` for physical constants — do not hardcode values

## Requirements for any new physics constraint

- The constraint must be derivable from Newton's laws, Euler-Lagrange, or a peer-reviewed
  biomechanics source. Cite the source in the docstring.
- Physics residuals must use `torch.autograd.grad(..., create_graph=True)` for
  second derivatives — never finite differences.
- The new loss term must be weighted and added to `TrainingConfig` with a sensible default.
- Add a unit test that verifies the residual is ≈ 0 on analytically known data
  (e.g. for a parabola, the projectile residual must be zero).

## Coordinate system reminder

- Y-up, right-handed: X = forward, Z = lateral
- `g_vec = [0, -9.81, 0]` m/s²
- `F_GRF = m * (a_CoM - g_vec)` = `m * (a_CoM + [0, 9.81, 0])`

## What constraint to add

<!-- Describe the physical law to enforce, the equation, and which PINN it applies to -->
