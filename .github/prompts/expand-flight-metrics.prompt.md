---
mode: agent
description: Add vertical extension phase timing and arch transition detection to FlightMetrics (Imogen expert input)
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

Expand `FlightMetrics` in `src/kinematics/flight_analysis.py` and add two new functions
to detect the sub-phases of the Fosbury Flop flight phase.

Read the full file first before making changes.

---

## Background — Imogen's key insight

**Imogen says:** "The main thing I would work on in the flight is the amount of time
extending vertically off the ground with knee driving up before dropping knee and head
to go over the bar."

The Fosbury Flop flight phase has two distinct sub-phases:
1. **Vertical extension phase** — Jumper drives knee upward, head up, body extended.
   This phase maximises height gain before bar clearance begins.
2. **Arch/clearance phase** — Jumper drops knee, drops head back, lifts hips.
   The "arch" position clears the bar.

The transition between these phases (the "arch transition frame") is a critical
technical variable. Starting the arch too early wastes vertical momentum.
Starting it too late means the hips hit the bar.

---

## Change 1 — Add fields to FlightMetrics

After the existing `estimated_angular_momentum_l: float | None` field, add:

```python
    # Flight sub-phase timing (Fosbury Flop specific)
    vertical_extension_time_ms: float | None   # time spent in "knee up / head up" phase
    arch_transition_frame: int | None          # frame index when jumper begins arch (relative to flight start)
    arch_transition_time_ms: float | None      # time from takeoff to arch transition
```

---

## Change 2 — Add detect_arch_transition()

Add after the existing `compute_clearance_profile()` function:

```python
def detect_arch_transition(
    body_landmarks_3d_seq: np.ndarray,
    fps: float,
    knee_landmark_idx: int = 25,
    head_landmark_idx: int = 0,
) -> dict[str, float | int | None]:
    """Detect the arch transition frame in Fosbury Flop flight.

    The transition from vertical-extension to arch-clearance is identified by
    the reversal of two signals:
    - The free knee begins to descend (knee Y velocity goes from positive to negative)
    - The head begins to drop (head Y velocity goes from positive to negative)

    The arch transition is defined as the frame where the knee starts descending
    (free-leg knee Y velocity first crosses zero after takeoff, going negative).
    Uses the free (non-takeoff) knee, which drives up first then drops over the bar.

    Args:
        body_landmarks_3d_seq: (T, 33, 3) landmark positions during flight phase.
        fps: Frame rate.
        knee_landmark_idx: BlazePose index of the free knee (default 25 = left knee).
            Caller should pass the non-takeoff-leg knee index.
        head_landmark_idx: BlazePose index of the head/nose landmark (default 0 = nose).

    Returns:
        Dict with keys:
            "arch_transition_frame": int or None — frame index within the flight array.
            "arch_transition_time_ms": float or None — time from flight start in ms.
            "vertical_extension_time_ms": float or None — duration of extension phase.
            "peak_knee_height_m": float — maximum knee height during flight.
    """
    if len(body_landmarks_3d_seq) < 3:
        return {
            "arch_transition_frame": None,
            "arch_transition_time_ms": None,
            "vertical_extension_time_ms": None,
            "peak_knee_height_m": 0.0,
        }

    dt = 1.0 / fps
    knee_heights = body_landmarks_3d_seq[:, knee_landmark_idx, 1]

    # Compute vertical velocity of the free knee
    knee_vy = np.gradient(knee_heights, dt)

    # Find the frame where knee vertical velocity first crosses zero (negative after positive)
    transition_frame = None
    for i in range(1, len(knee_vy)):
        if knee_vy[i - 1] >= 0 and knee_vy[i] < 0:
            transition_frame = i
            break

    peak_knee_height = float(np.max(knee_heights))

    if transition_frame is None:
        return {
            "arch_transition_frame": None,
            "arch_transition_time_ms": None,
            "vertical_extension_time_ms": None,
            "peak_knee_height_m": peak_knee_height,
        }

    transition_time_ms = float(transition_frame / fps * 1000.0)

    return {
        "arch_transition_frame": int(transition_frame),
        "arch_transition_time_ms": transition_time_ms,
        "vertical_extension_time_ms": transition_time_ms,  # extension = time 0 → transition
        "peak_knee_height_m": peak_knee_height,
    }
```

---

## Constraints

- `arch_transition_frame` is relative to the start of the **flight phase** array,
  not the global frame index across the full video.
- Do NOT change `fit_com_parabola()` or `compute_clearance_profile()`.
- Use BlazePose landmark index 25 = left knee as the default free-leg knee.
  The caller will pass the correct knee based on which leg is the takeoff leg.
- Y-up coordinate convention — knee height is the Y component.

---

## Verify

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ --ignore=tests/test_pinn -v --tb=short 2>&1
```

All tests must pass.

Add a test in `tests/test_flight_analysis.py` (create if doesn't exist) for
`detect_arch_transition`:
- Construct a synthetic `body_landmarks_3d_seq` of shape (20, 33, 3) where all
  zeros except landmark 25 Y which goes 0→1→0 (rises then falls) over 20 frames.
- At 30 fps, the knee peak is at frame ~10. `arch_transition_frame` should be 10 ± 1.
