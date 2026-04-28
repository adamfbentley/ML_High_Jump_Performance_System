# Physics Notes

Authoritative conventions remain in `.github/copilot-instructions.md`.

Key invariants:

- Coordinate frame: Y-up, right-handed. X is forward/run-up, Z is lateral.
- Gravity vector: `[0, -9.81, 0]` m/s2.
- GRF convention: `F_GRF = m * (a_CoM - g_vec)`.
- Joint angles are radians internally.
- Output-facing takeoff angles are degrees.
- CoM estimation uses de Leva segment offsets and Winter segment parameters.

Do not weaken physics-law tests to make reports look plausible.
