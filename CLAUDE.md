# CLAUDE.md

Bootstrap context for Claude Code sessions. Auto-loaded each conversation — keep tight.

## What this is

Physics-informed ML system for personalised high-jump coaching. Athlete (Imogen, H=1.78 m, W=67 kg, thigh=43 cm, shank=47 cm, arm=68 cm) films jumps on a phone → MediaPipe BlazePose extracts 3D landmarks → kinematics → InverseDynamics PINN → differentiable optimiser → quantified coaching cues ("approach +0.3 m/s → +2.8 cm").

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

# Re-process Imogen's videos with sample caching (~45 min for 45 videos)
.venv/Scripts/python.exe scripts/analyze_jump_video.py "data/High Jump Videos" --save-samples

# Dry-run personal fine-tuning (loads cached samples, applies Phase 9a guardrail)
.venv/Scripts/python.exe scripts/finetune_personal.py --dry-run

# Smoke-test on a single video
.venv/Scripts/python.exe scripts/analyze_jump_video.py "data/High Jump Videos/14_02_26/14_02_26_one_1.79.mp4"
```

## Current phase (April 2026)

**Phase 9a v1 SHIPPED, partial improvement.** `src/pose_estimation/scale_calibration.py` now derives a single video-wide metres-per-pixel from the 95th percentile of pixel projections of Imogen's measured thigh and shank segments (rejects foreshortening), median-aggregated across segments. Ground reference moved from `min(ankle_y)` to `5th-percentile(ankle_y over visible frames)` to eliminate single-frame outlier corruption. CLI takes `--thigh 0.43 --shank 0.47` to enable this path.

Smoke results:
- `14_02_26_one_1.79.mp4` (previously plausible): peak CoM 2.60→2.39 m, takeoff 49°, vy 3.43 m/s — fully plausible elite-female numbers.
- `09_02_26_one.mp4` (previously inflated): peak CoM 5.04→3.41 m — improved but vy=38 m/s artifact remains because the athlete is far from camera (~63 px thigh vs 151 px in the close clip), so MediaPipe jitter × m/px is amplified, AND the takeoff frame selection is still picking velocity spikes (Phase 9b territory).

**Open Phase 9 follow-ups (still ⏳):**
- 9b: switch takeoff frame from `argmax(vy)` to last-frame-of-last-ground-contact using existing `detect_ground_contacts`. Removes single-frame velocity-spike sensitivity.
- Full 45-video re-process with new calibration to characterise the residual error distribution.
- Fine-tune (Phase 10) blocked on the above.

After 9b: re-cache samples (`--save-samples --thigh 0.43 --shank 0.47`), then run `finetune_personal.py`.

## Athlete-domain priorities (from Imogen's brief)

Run-up: per-stride GCT, foot-under-hip offset, curve adherence, foot contact type, arm lateral swing.
Takeoff: foot-to-mat angle, body alignment, arm-drive speed/timing, free-knee drive speed/timing.
Flight: vertical extension time, arch transition frame.
Movement-relevance ordering: `HIGH_JUMP > SINGLE_LEG_DROP_JUMP > DROP_JUMP > CMJ > VERTICAL_JUMP > SQUAT_JUMP`.

## Communication

- Athlete data (`data/results/`, `data/High Jump Videos/`, processed reports) is **gitignored and private**. Don't paste raw values, video paths, or session metadata into commits, PRs, or external services.
- Commit messages: short imperative, no `Co-Authored-By` line.
- When in doubt about destructive actions (force push, `git reset --hard`, deleting cached models), ask first.
