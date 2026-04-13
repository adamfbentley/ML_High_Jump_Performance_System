---
mode: agent
description: Fix movement type relevance rankings and add SessionContext for physiological confounders (Imogen expert input)
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

Make two targeted changes to `src/data_pipeline/sample.py` based on expert input from
Imogen, a national-champion high jumper.

---

## Change 1 — Fix movement relevance rankings

**Problem:** The current rankings are wrong according to the domain expert.
CMJ is rated 0.9 (too high). Drop jump should be the closest transfer to high jump
after actual high jump trials — it's what Imogen uses in strength and conditioning testing.

**Current code in `src/data_pipeline/sample.py`:**
```python
MOVEMENT_RELEVANCE = {
    MovementType.HIGH_JUMP: 1.0,
    MovementType.COUNTERMOVEMENT_JUMP: 0.9,
    MovementType.DROP_JUMP: 0.85,
    ...
}
```

**Required change:**
- `DROP_JUMP` → 0.9
- `COUNTERMOVEMENT_JUMP` → 0.8
- Leave all other values unchanged

Also add a new movement type `SINGLE_LEG_DROP_JUMP = "single_leg_drop_jump"` to the
`MovementType` enum immediately after `DROP_JUMP`, and add it to `MOVEMENT_RELEVANCE`
with a value of 0.92 (single-leg drop jump is the most specific transfer exercise —
closer to HJ than bilateral drop jump).

---

## Change 2 — Add SessionContext dataclass

Add a new `SessionContext` dataclass to `sample.py`, after the `SubjectInfo` dataclass
and before `BiomechanicalSample`. This captures physiological and environmental state
that confounds biomechanical measurements across sessions.

```python
@dataclass
class SessionContext:
    """Environmental and physiological state at the time of a session.

    These are confounding variables that affect jump performance independently
    of technique. Collected from athlete self-report and wearables.

    All fields are Optional — not all will be available for every session.
    """

    # Fatigue (within-session)
    jumps_completed_this_session: int | None = None   # jumps done before this trial
    session_number_this_week: int | None = None        # 1 or 2 (2x/week protocol)

    # Accumulated fatigue (prior training load)
    days_since_last_jump_session: float | None = None
    training_load_preceding_48h: str | None = None    # "light" | "moderate" | "heavy"

    # General health/readiness
    hrv_morning: float | None = None                  # morning HRV (ms) from wearable
    sleep_hours_prev_night: float | None = None
    subjective_fatigue_1_10: float | None = None       # 1 = fresh, 10 = exhausted

    # Injury status
    injury_present: bool | None = None
    injury_description: str | None = None             # free text

    # Environmental
    temperature_celsius: float | None = None
    indoor_outdoor: str | None = None                 # "indoor" | "outdoor"

    # Hormonal (menstrual cycle — relevant for female athletes)
    menstrual_cycle_phase: str | None = None          # "follicular" | "ovulatory" | "luteal" | "menstrual"
```

Then add a `session_context` field to `BiomechanicalSample`:
```python
session_context: SessionContext | None = None
```

Add it after the existing `fps: float = 0.0` line in the `BiomechanicalSample` dataclass.

---

## Constraints

- Do NOT change any other code.
- Maintain `from __future__ import annotations` at the top.
- All fields use SI units or documented string literals.
- Do not add imports — `dataclass` and `Optional` are already imported.
  Use `int | None` union syntax (Python 3.10+) in the new dataclass, consistent
  with `SubjectInfo` style.

---

## Verify

Run the test suite and confirm nothing is broken:

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ --ignore=tests/test_pinn -v --tb=short 2>&1
```

All tests must pass. If any test imports `MOVEMENT_RELEVANCE` and checks exact values,
update the test to match the new values.
