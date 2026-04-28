# CLAUDE.md

Bootstrap context for Claude Code sessions. Auto-loaded each conversation — keep tight.

## What this is

Physics-informed ML system for personalised high-jump coaching. Athlete (Athlete A, H=1.75 m, W=65 kg, thigh=45 cm, shank=45 cm, arm=70 cm) films jumps on a phone → MediaPipe BlazePose extracts 3D landmarks → kinematics → InverseDynamics PINN → differentiable optimiser → quantified coaching cues ("approach +0.3 m/s → +2.8 cm").

Research project, not a product. **Scientific correctness beats code elegance.** Every physics decision must trace to peer-reviewed literature (Winter 2009, de Leva 1996, Dapena 1980, Rajagopal 2015).

## Read these before doing anything substantive

1. `.github/copilot-instructions.md` — physics conventions, units, code style, current phase
2. `ROADMAP.md` — full chronological history, recent commits, deferred follow-ups (Phase 9 audit and the 9a/9b/9c open issues live here)
3. `ARCHITECTURE.md` — module-by-module system overview
4. `src/data_pipeline/sample.py` — `BiomechanicalSample` is the canonical cross-module data format; never break its field names
5. `src/utils/constants.py` — gravity, segment mass fractions

For long-form first-time onboarding: `CLAUDE_ONBOARDING.md` (~340 lines).

## Hard rules

- **Coordinate system:** Y-up, right-handed. X = forward (run-up), Z = lateral. Gravity `[0, -9.81, 0]`. Don't change without a flagged comment.
- **Joint angles:** radians internally, degrees only at output boundary.
- **F = m·a:** `F_GRF = m * (a_CoM - g_vec)` where `g_vec = [0, -9.81, 0]`.
- **No mock data anywhere.** Functions implement real physics or raise `NotImplementedError` with what's needed.
- **Don't weaken physics-law test tolerances** to make a test pass — fix the implementation instead.
- **Don't refactor or rename working modules** without a clear correctness reason. The project has had silent regressions from working-tree drift; before assuming something is missing, check `git diff HEAD` and `git log --oneline <file>`.
- **Don't add deployment infra** (FastAPI, Docker, S3) to `src/`. That belongs in `services_scaffold/` and is parked.

## Quick commands

```bash
# Run the full test suite (currently 176 passing)
.venv/Scripts/python.exe -m pytest tests/ --ignore=tests/test_pinn -q

# Re-process Athlete A's videos with sample caching (~35 min for 45 videos)
.venv/Scripts/python.exe scripts/analyze_jump_video.py "data/High Jump Videos" --save-samples data/results/samples --thigh 0.45 --shank 0.45

# Dry-run personal fine-tuning (loads cached samples, applies Phase 9a guardrail)
.venv/Scripts/python.exe scripts/finetune_personal.py --dry-run

# Smoke-test on a single video
.venv/Scripts/python.exe scripts/analyze_jump_video.py "data/High Jump Videos/14_02_26/14_02_26_one_1.79.mp4"
```

## Current phase (April 2026)

**Phase 9a + 9b SHIPPED, but Phase 10 is blocked.**

- 9a: `src/pose_estimation/scale_calibration.py` derives a single video-wide
  metres-per-pixel from the 95th percentile of visible thigh/shank projections,
  median-aggregated across segments. Ground reference uses the 5th percentile
  of visible ankle Y instead of unsafe `min()`.
- 9b: `scripts/analyze_jump_video.py` selects takeoff as the final frame of the
  last detected ankle-ground contact before flight, falling back to `argmax(vy)`
  only when contact detection fails.
- Bar-height metadata parser fixed for numeric extensions such as `.mp4`.
- Full re-process completed: 45/45 reports, 45/45 `.npz` samples cached.

Residual aggregate validation:
- Peak CoM: median 1.67 m; 9/45 in the handoff's 2.0-2.7 m target range.
- Takeoff vy: median 3.16 m/s; 18/45 in 3.0-4.5 m/s.
- Takeoff angle: median 66.6 deg; 3/45 in 38-48 deg.
- Bar-tagged subset: 17/45 filenames exposed bar height; median CoM-minus-bar
  was -0.05 m and 8/17 were within -0.30 to +0.10 m.

Do **not** fine-tune yet. The blocker is no longer only vertical scale: takeoff
horizontal velocity is unreliable from panned single-camera footage (median
takeoff horizontal speed 1.08 m/s, only 2/45 in 2.5-5.5 m/s), so reported
takeoff angles are still not trustworthy.

## Athlete-domain priorities (from Athlete A's brief)

Run-up: per-stride GCT, foot-under-hip offset, curve adherence, foot contact type, arm lateral swing.
Takeoff: foot-to-mat angle, body alignment, arm-drive speed/timing, free-knee drive speed/timing.
Flight: vertical extension time, arch transition frame.
Movement-relevance ordering: `HIGH_JUMP > SINGLE_LEG_DROP_JUMP > DROP_JUMP > CMJ > VERTICAL_JUMP > SQUAT_JUMP`.

## Communication

- Athlete data (`data/results/`, `data/High Jump Videos/`, processed reports) is **gitignored and private**. Don't paste raw values, video paths, or session metadata into commits, PRs, or external services.
- Commit messages: short imperative, no `Co-Authored-By` line.
- When in doubt about destructive actions (force push, `git reset --hard`, deleting cached models), ask first.
