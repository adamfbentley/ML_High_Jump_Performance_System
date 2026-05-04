# CLAUDE.md

Bootstrap context for Claude Code sessions. Auto-loaded each conversation — keep tight.
For agent-routing and expensive-review rules, read `AGENTS.md`.

## What this is

Physics-informed ML system for personalised high-jump coaching. Athlete (Imogen, H=1.78 m, W=67 kg, thigh=43 cm, shank=47 cm, arm=68 cm) films jumps on a phone → MediaPipe BlazePose extracts 3D landmarks → kinematics → InverseDynamics PINN → differentiable optimiser → quantified coaching cues ("approach +0.3 m/s → +2.8 cm").

Research project, not a product. **Scientific correctness beats code elegance.** Every physics decision must trace to peer-reviewed literature (Winter 2009, de Leva 1996, Dapena 1980, Rajagopal 2015).

## Read these before doing anything substantive

1. `.github/copilot-instructions.md` — physics conventions, units, code style, current phase
2. `ROADMAP.md` — full chronological history, recent commits, deferred follow-ups (Phase 9 audit and the 9a/9b/9c open issues live here)
3. `ARCHITECTURE.md` — module-by-module system overview
4. `src/data_pipeline/sample.py` — `BiomechanicalSample` is the canonical cross-module data format; never break its field names
5. `src/utils/constants.py` — gravity, segment mass fractions

For deeper onboarding, prefer local RAG plus the files above instead of a long
static onboarding prompt.

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
# Run the full non-PINN test suite (currently 214 passing)
.venv/Scripts/python.exe -m pytest tests/ --ignore=tests/test_pinn -q

# Build/query local agent memory
.venv/Scripts/python.exe tools/memory/build_index.py
.venv/Scripts/python.exe tools/memory/query_index.py "takeoff angle horizontal velocity"

# Re-process Imogen's videos with sample caching (~35 min for 45 videos)
.venv/Scripts/python.exe scripts/analyze_jump_video.py "data/High Jump Videos" --save-samples data/results/samples --thigh 0.43 --shank 0.47

# Dry-run personal fine-tuning (loads cached samples, applies Phase 9a guardrail)
.venv/Scripts/python.exe scripts/finetune_personal.py --dry-run

# Smoke-test on a single video
.venv/Scripts/python.exe scripts/analyze_jump_video.py "data/High Jump Videos/14_02_26/14_02_26_one_1.79.mp4"
```

## Current phase (May 2026)

**Phases 9a/9b/9c SHIPPED. Egomotion validated. Phase 10 still gated on Phase 9d (apparatus-anchored mpp).**

- 9a: `src/pose_estimation/scale_calibration.py` derives a single video-wide
  metres-per-pixel from the 95th percentile of visible thigh/shank projections.
- 9b: `scripts/analyze_jump_video.py` anchors takeoff to the last ankle-ground
  contact before peak CoM, with `argmax(vy)` fallback.
- 9c: `src/pose_estimation/scene_calibration.py` (Hough-line apparatus
  detector) and `src/pose_estimation/egomotion.py` (background-flow camera
  motion) shipped with clip-level acceptance gates in
  `calibrate_landmarks_with_scene`. Both opt-in via `--scene-anchor on` /
  `--egomotion on`.
- Validation infrastructure shipped: `scripts/probe_scene_anchors.py`,
  `scripts/label_scene_anchors.py`, `scripts/evaluate_calibration_truth.py`,
  `scripts/aggregate_calibration_modes.py`. 72 private videos available
  (45 originals + 27 unknown-date with bar heights in filenames).

What hand-label evaluation showed (5 clips, takeoff-window comparison):

- Auto-detector (Hough scene_homography): rejected on every clip; the detector
  locks onto wrong vertical edges (net poles, mat frame). Dead end on existing
  footage.
- Egomotion: clearly beats anatomical on takeoff vh (e.g. on the densest-label
  clip, egomotion 6.33 m/s vs anatomical 3.76 m/s vs truth 7.84 m/s).
- A residual ~1 m/s underestimate persists *even on the tripod control* where
  there is no panning to remove. This is not a panning failure; it is a depth
  / mpp calibration bias. Anatomical mpp (p95 of thigh projection) is
  systematically biased small at takeoff-zone depth.
- Smoke claim of "median takeoff vh = 1.08 m/s" was misleading: it used
  per-frame median (noise-dominated) on anatomical-only output. Truth values
  at takeoff fall in or near the 2.5-5.5 m/s elite band on most clips.

Do **not** fine-tune yet. The remaining blocker is the apparatus-anchored mpp
work in Phase 9d (see `memory/plans/opus_plan_current.md`): use the labelled
upright separation (4.02 m, IAAF spec) as a third independent scale source at
the takeoff zone, validate against the tripod clip, then revisit Phase 10.

## Agent memory

Use `memory/` for file-mediated Claude/Codex collaboration. The local RAG
tooling under `tools/memory/` is **parked** — for a corpus this small,
direct `Read` and `Grep` are faster and more accurate than the lexical
hashing index. Revisit only if the corpus grows past ~200 docs or sentence-
transformer embeddings are added. Do not index private athlete reports,
videos, raw session metadata, or private emails.

Expensive-review rule: use Opus-style review only for architecture, physics,
data-contract, phase-gate, personal fine-tuning readiness, or repeated-failure
questions. Send a compact packet with the goal, blocker, inspected files,
commands run, observed metrics, constraints, and exact decision needed.

## Athlete-domain priorities (from Imogen's brief)

Run-up: per-stride GCT, foot-under-hip offset, curve adherence, foot contact type, arm lateral swing.
Takeoff: foot-to-mat angle, body alignment, arm-drive speed/timing, free-knee drive speed/timing.
Flight: vertical extension time, arch transition frame.
Movement-relevance ordering: `HIGH_JUMP > SINGLE_LEG_DROP_JUMP > DROP_JUMP > CMJ > VERTICAL_JUMP > SQUAT_JUMP`.

## Communication

- Athlete data (`data/results/`, `data/High Jump Videos/`, processed reports) is **gitignored and private**. Don't paste raw values, video paths, or session metadata into commits, PRs, or external services.
- Commit messages: short imperative, no `Co-Authored-By` line.
- When in doubt about destructive actions (force push, `git reset --hard`, deleting cached models), ask first.
