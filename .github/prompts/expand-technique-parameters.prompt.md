---
mode: agent
description: Expand TechniqueParameters in optimizer.py with GCT, body alignment, foot angle, knee drive speed, and curve start step (Imogen expert input)
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

Expand `TechniqueParameters` in `src/optimization/optimizer.py` to include new
controllable variables identified by Imogen (national-champion high jumper).

Read the full file first before making changes.

---

## Background

Imogen's top predictors not yet in `TechniqueParameters`:
- Ground contact time of the takeoff foot — "most important" metric she listed
- Body alignment at takeoff ("body in straight line")
- Foot-to-ground angle at plant (different from `plant_angle_deg` which is the full leg angle)
- Free-leg knee drive speed
- Curve start step (which step of the approach begins the curve)

The current `TechniqueParameters` has 9 fields. The `to_tensor()` and `from_tensor()`
methods encode/decode via a fixed-length tensor — you MUST update both.

---

## Change — Add 5 new fields to TechniqueParameters

After the existing `free_leg_drive_angle_deg: float` field, add:

```python
    # Ground contact time of the takeoff foot at plant (ms)
    # Shorter GCT = more reactive/explosive takeoff (Imogen: "most important")
    # Elite Fosbury Floppers: 150–180 ms (Dapena 1980)
    ground_contact_time_ms: float

    # Whole-body alignment at takeoff: deviation from a straight ankle→hip→shoulder line
    # 0° = perfectly straight (ideal for force transfer), higher = more bend
    body_alignment_dev_deg: float

    # Foot-to-ground angle at plant: angle of foot sole (heel→toe) to the mat surface
    # Distinct from plant_angle_deg (which is the full leg angle hip→foot to horizontal)
    # ~20–30° toe-up is optimal; heel-strike is associated with excessive braking
    foot_to_ground_angle_deg: float

    # Peak speed of the free-leg knee during drive phase at takeoff (m/s)
    # Higher = more upward impulse transferred to CoM
    knee_drive_speed_mps: float

    # Which step in the run-up begins the curve (counting from the first step)
    # Imogen: curve start point matters for approach rhythm and optimal curve radius
    curve_start_step: int
```

---

## Update to_tensor() and from_tensor()

`to_tensor()` currently returns a 9-element tensor. Update it to include the 5 new
fields at the end, making it a 14-element tensor:

```python
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.approach_speed_mps,
            self.curve_radius_m,
            self.penultimate_step_length_cm,
            self.last_step_length_cm,
            self.plant_angle_deg,
            self.takeoff_knee_angle_deg,
            self.takeoff_hip_angle_deg,
            self.arm_swing_timing_ms,
            self.free_leg_drive_angle_deg,
            self.ground_contact_time_ms,
            self.body_alignment_dev_deg,
            self.foot_to_ground_angle_deg,
            self.knee_drive_speed_mps,
            float(self.curve_start_step),
        ], dtype=torch.float32)
```

Update `from_tensor()` to decode all 14 elements:
```python
    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TechniqueParameters:
        v = t.detach().cpu().numpy()
        return cls(
            approach_speed_mps=float(v[0]),
            curve_radius_m=float(v[1]),
            penultimate_step_length_cm=float(v[2]),
            last_step_length_cm=float(v[3]),
            plant_angle_deg=float(v[4]),
            takeoff_knee_angle_deg=float(v[5]),
            takeoff_hip_angle_deg=float(v[6]),
            arm_swing_timing_ms=float(v[7]),
            free_leg_drive_angle_deg=float(v[8]),
            ground_contact_time_ms=float(v[9]),
            body_alignment_dev_deg=float(v[10]),
            foot_to_ground_angle_deg=float(v[11]),
            knee_drive_speed_mps=float(v[12]),
            curve_start_step=int(round(float(v[13]))),
        )
```

---

## Constraints

- Do NOT change the docstring of `TechniqueParameters` (it says "Controllable technique variables").
- Do NOT change any other class or function in the file.
- `curve_start_step` is an `int` field but stored/retrieved as `float` in the tensor
  (neural networks work with floats) — the `from_tensor` must round and cast to int.
- All units in variable names match the field names (ms, deg, mps).

---

## Verify

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ --ignore=tests/test_pinn -v --tb=short 2>&1
```

All tests must pass.

Check if any test constructs `TechniqueParameters(...)` with positional args — if so,
update those tests to include values for the new required fields.

Also add a test in `tests/test_optimizer.py` (or existing test file) that:
- Constructs a `TechniqueParameters` with all 14 fields
- Calls `.to_tensor()` and checks `len(t) == 14`
- Round-trips via `from_tensor(to_tensor())` and checks all fields match.
