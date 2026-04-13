---
mode: agent
description: Expand TakeoffMetrics with foot-to-ground angle, body alignment, arm drive speed, and free-leg knee drive speed/timing (Imogen expert input)
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

Expand `TakeoffMetrics` in `src/kinematics/takeoff_analysis.py` and add four new
computation functions. Based on expert input from Imogen (national-champion high jumper),
these are among her **top-ranked predictors** of takeoff quality.

Read the full file first before making changes.

---

## Background

Imogen rates these as most important at takeoff (in priority order):
1. Speed at takeoff ✅ already present
2. **Ground contact time of takeoff foot** ✅ already present
3. **Body angle at takeoff** — currently only `trunk_lean_deg` exists, not whole-body alignment
4. **Body in straight line at takeoff** — whole-body alignment (ankle→hip→shoulder)
5. **Takeoff foot angle to the mat** — the angle of the foot sole to the floor surface
6. **Speed and timing of arm drive** — not present
7. **Speed and timing of free-leg knee drive** — not present

---

## Change 1 — Add fields to TakeoffMetrics

After the existing `trunk_lean_deg: float` field, add:

```python
    # Whole-body alignment at takeoff instant
    body_alignment_score: float      # 0–1: how close ankle→hip→shoulder is to a straight line (1=perfect)
    body_alignment_angle_deg: float  # deviation from straight line in degrees (0 = perfectly straight)

    # Foot-to-ground angle: angle of the foot sole (heel→toe vector) to the mat surface
    # Imogen's image shows this as the angle between the foot and the horizontal mat.
    # Optimal for Fosbury Flop: ~20–30° (foot flat to slightly toe-up at plant)
    foot_to_ground_angle_deg: float | None

    # Free-leg (non-takeoff leg) knee drive
    knee_drive_speed_mps: float      # speed of the free knee joint at peak drive
    knee_drive_timing_ms: float      # time from foot contact to peak knee drive speed

    # Arm drive
    arm_drive_speed_mps: float       # resultant speed of wrist landmarks at peak arm drive
    arm_drive_timing_ms: float       # time from foot contact to peak arm drive speed
```

---

## Change 2 — Add compute_body_alignment_score()

Add after `compute_impulse()`:

```python
def compute_body_alignment_score(
    ankle: np.ndarray,
    hip: np.ndarray,
    shoulder: np.ndarray,
) -> tuple[float, float]:
    """Compute whole-body alignment at a single frame (ankle→hip→shoulder).

    A perfectly straight body has ankle, hip, and shoulder collinear.
    Returns a score (1 = perfect) and the angular deviation in degrees.

    For Fosbury Flop takeoff, a straight body maximises force transfer
    from the ground up through the kinetic chain.

    Args:
        ankle: (3,) position of the takeoff ankle.
        hip: (3,) position of the takeoff-side hip.
        shoulder: (3,) position of the takeoff-side shoulder.

    Returns:
        Tuple of (alignment_score 0–1, deviation_angle_degrees).
    """
    lower_seg = hip - ankle        # ankle → hip vector
    upper_seg = shoulder - hip     # hip → shoulder vector

    n_lower = np.linalg.norm(lower_seg)
    n_upper = np.linalg.norm(upper_seg)
    if n_lower < 1e-8 or n_upper < 1e-8:
        return 0.0, 180.0

    cos_angle = np.dot(lower_seg / n_lower, upper_seg / n_upper)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    deviation_deg = float(np.degrees(np.arccos(cos_angle)))

    # 180° = perfectly straight. deviation_from_straight = 180 - angle
    deviation_from_straight = abs(180.0 - deviation_deg)
    score = float(1.0 - deviation_from_straight / 180.0)
    return score, deviation_from_straight
```

---

## Change 3 — Add compute_foot_to_ground_angle()

Add after `compute_body_alignment_score()`:

```python
def compute_foot_to_ground_angle(
    heel: np.ndarray,
    toe: np.ndarray,
) -> float:
    """Compute the angle of the foot sole relative to the ground (horizontal) plane.

    The foot vector runs from heel to toe. Its angle to horizontal gives the
    "foot-to-ground angle" — how the foot meets the mat at plant.

    Imogen: optimal Fosbury Flop plant is ~20–30° toe-up (positive = toe higher than heel).

    Coordinate system: Y-up, X = forward, Z = lateral.

    Args:
        heel: (3,) position of the heel landmark.
        toe: (3,) position of the toe landmark.

    Returns:
        Angle in degrees. Positive = toe above heel (dorsiflexed). 0 = flat.
    """
    foot_vec = toe - heel
    horizontal_dist = float(np.sqrt(foot_vec[0] ** 2 + foot_vec[2] ** 2))
    vertical_rise = float(foot_vec[1])
    if horizontal_dist < 1e-8:
        return 0.0
    angle_rad = float(np.arctan2(vertical_rise, horizontal_dist))
    return float(np.degrees(angle_rad))
```

---

## Change 4 — Add compute_drive_peak_speed_and_timing()

Add after `compute_foot_to_ground_angle()`:

```python
def compute_drive_peak_speed_and_timing(
    landmark_positions: np.ndarray,
    contact_start_frame: int,
    fps: float,
) -> tuple[float, float]:
    """Compute peak speed and its timing for a driven segment (arm or free-leg knee).

    Used for arm drive (wrist landmark) and free-leg knee drive (knee landmark).
    Returns the peak resultant speed and the time elapsed from ground contact.

    Args:
        landmark_positions: (T, 3) position of the landmark over time.
        contact_start_frame: Frame index when takeoff foot contacted the ground.
        fps: Frame rate.

    Returns:
        Tuple of (peak_speed_mps, timing_ms_from_contact).
    """
    dt = 1.0 / fps
    velocities = np.gradient(landmark_positions, dt, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)

    # Only look at the window from contact onwards
    speeds_from_contact = speeds[contact_start_frame:]
    if len(speeds_from_contact) == 0:
        return 0.0, 0.0

    peak_idx = int(np.argmax(speeds_from_contact))
    peak_speed = float(speeds_from_contact[peak_idx])
    timing_ms = float(peak_idx / fps * 1000.0)
    return peak_speed, timing_ms
```

---

## Constraints

- `foot_to_ground_angle_deg` is `float | None` — use `None` if heel/toe landmarks aren't available.
- Do NOT change any existing function signatures.
- All angles in degrees, all times in ms, all speeds in m/s (SI).
- Include docstrings with Args and Returns.

---

## Verify

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ --ignore=tests/test_pinn -v --tb=short 2>&1
```

All tests must pass.

Add a test in `tests/test_takeoff_analysis.py` for `compute_body_alignment_score`:
- Perfect straight line (ankle at [0,0,0], hip at [0,1,0], shoulder at [0,2,0])
  should return score ≈ 1.0 and deviation ≈ 0.0.
- A 90° bend should return deviation ≈ 90.0.
