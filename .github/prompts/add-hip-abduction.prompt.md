---
mode: agent
description: Add hip abduction/adduction joint angles to joint_angles.py using frontal-plane vectors (Imogen expert input)
tools:
  - read_file
  - replace_string_in_file
  - grep_search
  - run_in_terminal
---

## Source of truth

- Use `.github/copilot-instructions.md` as the technical source of truth for architecture, physics conventions, data structures, datasets, and code conventions.
- Use `imogens_response- to-questions/Highjumpproject.html` as athlete-domain guidance for which movement priorities, metrics, and outputs matter most.
- If they conflict on implementation details, follow `.github/copilot-instructions.md` and flag the mismatch instead of guessing.

## Task

Add bilateral hip abduction/adduction angle computation to `src/pose_estimation/skeleton/joint_angles.py`.

This is requested by Imogen (national-champion high jumper): "In addition to the joint
movements you've already put I would add hip abduction and adduction."

Hip abduction/adduction is essential for high jump because:
- The free-leg drive crosses the midline (adduction) at bar clearance
- Body lean into the curve produces lateral hip displacement
- The arch position requires hip extension AND leg abduction simultaneously

---

## Context — coordinate system

- **Y-up**, right-handed. X = forward (run-up direction), Z = lateral.
- BlazePose landmark indices (33 total, 0-indexed):
  - Left hip = 23, Right hip = 24
  - Left knee = 25, Right knee = 26
  - Left shoulder = 11, Right shoulder = 12
  - Pelvis midpoint is not a BlazePose landmark — approximate as midpoint of hips 23/24.

## How to compute hip abduction/adduction

Hip abduction/adduction is the **frontal-plane** angle between the thigh and the pelvis
midline. Use this approach:

1. **Pelvis midline vector**: `pelvis_mid = (hip_L + hip_R) / 2`
2. **Thigh vector**: `thigh_L = knee_L - hip_L` (pointing distally)
3. **Frontal plane angle**: project both thigh and pelvis-to-hip vectors onto the YZ
   plane (frontal plane), then compute the angle between them.
4. **Sign convention**: positive = abduction (leg moves away from midline),
   negative = adduction (leg crosses midline).

For the angle computation, use the existing `angle_between_vectors()` helper.
Sign is determined by the Z-component of the cross product (positive Z = abduction
for the left leg in right-handed Y-up coordinates).

---

## Change 1 — Add compute_hip_abduction_angle()

Add this new function to `src/pose_estimation/skeleton/joint_angles.py` after the
existing `compute_joint_angle()` function:

```python
def compute_hip_abduction_angle(
    hip: np.ndarray,
    knee: np.ndarray,
    opposite_hip: np.ndarray,
    side: str,
) -> float:
    """Compute hip abduction (+) / adduction (-) angle in the frontal plane.

    Projects the thigh vector and pelvis lateral vector onto the YZ (frontal)
    plane and returns the signed angle between them.

    Positive = abduction (leg away from midline).
    Negative = adduction (leg toward or crossing midline).

    Coordinate system: Y-up, X = forward, Z = lateral.

    Args:
        hip: (3,) 3D position of the hip joint of interest.
        knee: (3,) 3D position of the knee on the same side.
        opposite_hip: (3,) 3D position of the contralateral hip.
        side: "left" or "right" — determines sign convention.

    Returns:
        Hip abduction/adduction angle in degrees.
    """
    # Thigh vector in YZ plane (frontal plane)
    thigh = knee - hip
    thigh_yz = np.array([0.0, thigh[1], thigh[2]])

    # Pelvis lateral vector (from this hip outward)
    pelvis_lateral = hip - opposite_hip
    pelvis_lateral_yz = np.array([0.0, pelvis_lateral[1], pelvis_lateral[2]])

    if np.linalg.norm(thigh_yz) < 1e-8 or np.linalg.norm(pelvis_lateral_yz) < 1e-8:
        return 0.0

    unsigned_angle = angle_between_vectors(thigh_yz, pelvis_lateral_yz)

    # Sign: cross product Z component. For left side, positive Z = abduction.
    cross_z = thigh_yz[1] * pelvis_lateral_yz[2] - thigh_yz[2] * pelvis_lateral_yz[1]
    sign = 1.0 if cross_z >= 0 else -1.0
    if side == "right":
        sign = -sign  # mirror for right side due to lateral axis flip

    return float(sign * unsigned_angle)
```

---

## Change 2 — Add to JOINT_ANGLE_DEFINITIONS

The `JOINT_ANGLE_DEFINITIONS` dict currently maps joint name → triplet for flexion angles.
Hip abduction requires a different signature (needs `opposite_hip`, not a triplet).

Add a **separate dict** `HIP_ABDUCTION_DEFINITIONS` immediately after `JOINT_ANGLE_DEFINITIONS`:

```python
# Hip abduction/adduction requires knowledge of both hips (not a simple triplet).
# Format: side → (hip_idx, knee_idx, opposite_hip_idx)
HIP_ABDUCTION_DEFINITIONS = {
    "left_hip_abduction": ("left", 23, 25, 24),   # left hip, left knee, right hip
    "right_hip_abduction": ("right", 24, 26, 23),  # right hip, right knee, left hip
}
```

---

## Change 3 — Add to compute_all_joint_angles()

In `compute_all_joint_angles()`, after the existing loop that fills `angles`, add:

```python
    pos = landmarks_3d[:, :3]  # already defined above in the function

    for name, (side, hip_idx, knee_idx, opp_hip_idx) in HIP_ABDUCTION_DEFINITIONS.items():
        angles[name] = compute_hip_abduction_angle(
            pos[hip_idx], pos[knee_idx], pos[opp_hip_idx], side
        )
```

Note: `pos` is already defined at the top of `compute_all_joint_angles()`, so just add
this block — do not redefine `pos`.

---

## Constraints

- Do NOT change `compute_joint_angle()` or `angle_between_vectors()`.
- Do NOT change `JOINT_ANGLE_DEFINITIONS` or `compute_joint_angles_sequence()`.
- `compute_joint_angles_sequence()` calls `compute_all_joint_angles()` per frame —
  it will automatically include hip abduction angles in its output.
- Angles returned are in **degrees** (matching ALL other outputs in this file).
- The `side` parameter must be `"left"` or `"right"` (lowercase string).

---

## Verify

Run the test suite:

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ --ignore=tests/test_pinn -v --tb=short 2>&1
```

All tests must pass.

Also add a test to `tests/test_joint_angles.py` (or create if it doesn't exist)
for `compute_hip_abduction_angle`:
- A perfectly vertical thigh (hip directly above knee) with a level pelvis should
  return 0° abduction.
- A thigh pointing 30° laterally from vertical should return ~30° abduction.
Use `np.testing.assert_allclose(..., atol=1.0)`.
