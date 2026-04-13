---
mode: agent
description: Add per-stride run-up metrics to RunUpMetrics — stride GCT, foot strike position, curve deviation, acceleration profile, foot contact type (Imogen expert input)
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

Expand the `RunUpMetrics` dataclass in `src/kinematics/run_up_analysis.py` and add
supporting computation functions. Based on expert input from Imogen (national-champion
high jumper), ALL strides matter — not just the last two. The current dataclass only
records `step_lengths_cm`, `step_frequencies_hz`, `penultimate_step_length_cm`, and
`last_step_length_cm`. All of the following need to be added.

---

## Change 1 — Expand RunUpMetrics

Read `src/kinematics/run_up_analysis.py` first to understand the full current structure.

Add the following fields to `RunUpMetrics` after the existing `step_frequencies_hz` line
and before `penultimate_step_length_cm`:

```python
    # Per-stride ground contact times (one entry per detected stride)
    ground_contact_times_ms: list[float] = field(default_factory=list)

    # Foot strike position relative to hip each stride, positive = foot ahead of hip (overstriding)
    foot_strike_under_hip_offset_cm: list[float] = field(default_factory=list)

    # Deviation of each foot contact from the fitted curve line (cm), positive = outside
    curve_deviation_per_stride_cm: list[float] = field(default_factory=list)

    # Per-stride mean horizontal acceleration (m/s^2)
    acceleration_profile_mps2: list[float] = field(default_factory=list)

    # Foot contact type per stride: "toe" | "flat" | "heel"
    foot_contact_type: list[str] = field(default_factory=list)
```

This requires adding `field` to the dataclass import. The file already uses `from dataclasses import dataclass` — change that to `from dataclasses import dataclass, field`.

---

## Change 2 — Add compute_stride_ground_contact_times()

Add this function after the existing `detect_ground_contacts()` function:

```python
def compute_stride_ground_contact_times(
    contact_intervals: list[tuple[int, int]],
    fps: float,
) -> list[float]:
    """Compute ground contact duration for each stride from detected contacts.

    Args:
        contact_intervals: List of (start_frame, end_frame) from detect_ground_contacts().
        fps: Frame rate.

    Returns:
        List of ground contact times in milliseconds, one per contact.
    """
    ms_per_frame = 1000.0 / fps
    return [float((end - start + 1) * ms_per_frame) for start, end in contact_intervals]
```

---

## Change 3 — Add compute_foot_strike_under_hip_offset()

Add this function after `compute_stride_ground_contact_times()`:

```python
def compute_foot_strike_under_hip_offset(
    ankle_positions: np.ndarray,
    hip_positions: np.ndarray,
    contact_start_frames: list[int],
) -> list[float]:
    """Compute how far ahead of the hip each foot strikes (XZ plane).

    An optimal strike has the foot directly under the hip (offset ≈ 0).
    Positive = foot ahead of hip (overstriding, increases braking force).
    Negative = foot behind hip (rare, associated with falling).

    Args:
        ankle_positions: (T, 3) ankle trajectory, Y-up.
        hip_positions: (T, 3) hip trajectory, Y-up.
        contact_start_frames: Frame index of each foot's initial ground contact.

    Returns:
        List of offsets in cm, one per stride.
    """
    offsets_cm = []
    for frame in contact_start_frames:
        if frame >= len(ankle_positions):
            continue
        ankle_xz = ankle_positions[frame, [0, 2]]
        hip_xz = hip_positions[frame, [0, 2]]
        # Project offset onto the direction of travel (X axis in Y-up, X-forward convention)
        offset_m = float(ankle_xz[0] - hip_xz[0])  # forward offset
        offsets_cm.append(offset_m * 100.0)
    return offsets_cm
```

---

## Change 4 — Add compute_curve_deviation_per_stride()

Add after `compute_foot_strike_under_hip_offset()`:

```python
def compute_curve_deviation_per_stride(
    foot_contact_positions_xz: np.ndarray,
    curve_center_xz: np.ndarray,
    curve_radius_m: float,
) -> list[float]:
    """Compute how far each foot lands from the ideal curve arc.

    A jumper running perfectly on the curve has each foot contact on the
    circumference of the fitted circle. Deviation > 0 = outside the curve.

    Args:
        foot_contact_positions_xz: (N, 2) XZ positions at each foot contact.
        curve_center_xz: (2,) center of the fitted curve circle.
        curve_radius_m: Radius of the fitted curve in meters.

    Returns:
        List of deviations in cm. Positive = outside (larger radius).
    """
    deviations_cm = []
    for pos in foot_contact_positions_xz:
        dist_from_center = float(np.linalg.norm(pos - curve_center_xz))
        deviation_m = dist_from_center - curve_radius_m
        deviations_cm.append(deviation_m * 100.0)
    return deviations_cm
```

---

## Constraints

- Do NOT change any existing function signatures.
- Coordinate convention: Y-up, X = forward (direction of run-up), Z = lateral.
- All new functions must have docstrings with Args and Returns, matching the existing style.
- No new imports needed beyond what's already there (`numpy` is already imported).

---

## Verify

Run the test suite:

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ --ignore=tests/test_pinn -v --tb=short 2>&1
```

All tests must pass. If any existing test constructs a `RunUpMetrics` with positional
arguments, it will break — check with `grep_search` for `RunUpMetrics(` and update any
such calls to use keyword args or provide the new defaults.

Also add one test to `tests/test_run_up_analysis.py` (create the file if it doesn't
exist) for `compute_stride_ground_contact_times` — a simple case with 3 contacts at
30 fps should give durations in ms.
