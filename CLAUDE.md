# CLAUDE.md

Bootstrap context for Claude Code sessions. Auto-loaded each conversation — keep tight.
For agent-routing and expensive-review rules, read `AGENTS.md`.

## What this is

Physics-informed ML system for personalised high-jump coaching. Athlete (Athlete A, H=1.75 m, W=65 kg, thigh=45 cm, shank=45 cm, arm=70 cm) films jumps on a phone → MediaPipe BlazePose extracts 3D landmarks → kinematics → InverseDynamics PINN → differentiable optimiser → quantified coaching cues ("approach +0.3 m/s → +2.8 cm").

Research project, not a product. **Scientific correctness beats code elegance.** Every physics decision must trace to peer-reviewed literature (Winter 2009, de Leva 1996, Dapena 1980, Rajagopal 2015).

## Read these before doing anything substantive

1. `.github/copilot-instructions.md` — physics conventions, units, code style, current phase
2. `ROADMAP.md` — full chronological history, recent commits, and the stationary-validation gate
3. `ARCHITECTURE.md` — module-by-module system overview
4. `src/data_pipeline/sample.py` — `BiomechanicalSample` is the canonical cross-module data format; never break its field names
5. `src/utils/constants.py` — gravity, segment mass fractions

For deeper onboarding, prefer direct file reads and `rg` over a long static
onboarding prompt. The optional local RAG tooling is parked for now.

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
# Run the full non-PINN test suite
.venv/Scripts/python.exe -m pytest tests/ --ignore=tests/test_pinn -q

# Full suite including PINN
.venv/Scripts/python.exe -m pytest tests/ -q

# Search agent memory directly
rg -n "stationary|stationary_camera|Phase 10" memory ROADMAP.md

# Production stationary-footage run (use for any new fixed-camera session)
.venv/Scripts/python.exe scripts/analyze_jump_video.py "<stationary-dir>" \
    --thigh 0.45 --shank 0.45 \
    --capture-mode stationary \
    --stationary-camera-confirmed \
    --roi-crop on \
    --save-samples <ignored-samples-dir>

# Experimental stable takeoff-window physics from a panned-run-up clip
.venv/Scripts/python.exe scripts/analyze_stable_takeoff_window.py \
    --video "<stable-window-derivative.mp4>" \
    --anchor-json "<manual-apparatus-anchors.json>" \
    --bar-height 1.75 \
    --roi-crop on \
    --output data/results/stationary_validation/stable_takeoff_window_v1.json

# Dry-run personal fine-tuning (loads the admitted cache, applies guardrails)
.venv/Scripts/python.exe scripts/finetune_personal.py \
    --samples-dir <ignored-samples-dir> --dry-run

# Smoke-test on a single video
.venv/Scripts/python.exe scripts/analyze_jump_video.py "<video-file>"
```

## Current phase (June 2026)

**Phases 9a-9e SHIPPED. Stationary admission hardening is complete and all three
newer stationary clips pass the implemented report gates. Phase 10 remains
blocked pending a larger fixed-camera session with a held-out subset.**

Key pipeline additions shipped 2026-06-03:

- `--capture-mode stationary --stationary-camera-confirmed`: requires an
  explicit operator review that the camera did not pan, tilt, zoom, or move
  before crediting `stationary_camera` as the scene-fixed horizontal source.
  Default `handheld` behaviour is unchanged.
- `_validate_takeoff_anchor()`: after selecting the last pre-peak ground contact,
  validates (a) vy ≥ 2.0 m/s and peak follows, (b) frame lead ≤ 2·t_apex·fps.
  Rejects approach-stride false detections. Both `kinematics_grade` and
  `training_grade` require the published anchor-review result.
- Foot-ground contact now keys on heel/forefoot landmarks rather than the ankle
  joint. This fixes plantarflexed toe-off cases where the forefoot is planted
  but the malleolus never enters the 5 cm ground band.
- `--roi-crop on` (off by default): two-pass athlete-crop. Pass 1 locates the
  athlete on the full frame; pass 2 re-detects on the crop and remaps 2D landmarks
  back to full-frame normalised coords via `remap_normalized_to_full_frame` (pure,
  tested). 3D world landmarks are pass-through (no remap needed).
- `pose_validity_pct` tightened from ≥4-of-33 to all-8-key-joints
  (shoulders/hips/knees/ankles, matching `PoseFrame.is_valid`).
- `takeoff_window_pose_validity_pct` (±30 frames around toe-off): the gate
  metric. The global clip metric is retained for diagnostics but excluded approach
  frames — where the athlete is far away — drag it below the 60 % threshold even
  when the critical window is well-covered. The windowed metric correctly reflects
  what the overlays show.

Stationary pilot outcome (5 clips, 2 captures, explicitly confirmed fixed camera):

- 3/3 newer landscape clips: **training_grade = True**. Takeoff angles
  40.5–42.2°, vh 3.65–4.11 m/s, window pose validity 61.7–73.3 %, contact +
  anchor review passed.
- Both earlier controls: not path-forward (approach-stride anchor, high spread).
- Raw pre-calibration proportion check: lower-limb shank/thigh estimates are
  credible across the local clips; arm length remains underestimated and must
  not be used as a scale anchor. This does not validate absolute scene scale.
- Current-footage fallback: the takeoff-focused derivative path was tested and
  remains useful for overlay/diagnostic rescue, but it is no longer required to
  admit newer clip 3. Whole-clip ROI plus foot-contact detection admits the
  newer trio directly.
- Panned-run-up clips with stationary final steps: current code rescues one
  manual takeoff-window derivative as `training_grade`; the other remains
  rejected. Treat the admitted derivative as experimental takeoff-window data
  only, not a full run-up sample or independent attempt for optimiser claims.

The analyser now caches only `training_grade` clips when `--save-samples` is
supplied, records every decision in ignored `_admission_manifest.json`, and
fine-tuning refuses legacy mixed caches without that manifest. Do **not** run a
real fine-tune or refresh optimiser claims yet: collect a larger stationary
session and reserve a held-out subset first. Follow
`memory/plans/stationary_footage_validation_plan.md`.

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

## Athlete-domain priorities (from Athlete A's brief)

Run-up: per-stride GCT, foot-under-hip offset, curve adherence, foot contact type, arm lateral swing.
Takeoff: foot-to-mat angle, body alignment, arm-drive speed/timing, free-knee drive speed/timing.
Flight: vertical extension time, arch transition frame.
Movement-relevance ordering: `HIGH_JUMP > SINGLE_LEG_DROP_JUMP > DROP_JUMP > CMJ > VERTICAL_JUMP > SQUAT_JUMP`.

## Communication

- Athlete data (`data/results/`, `data/High Jump Videos/`, processed reports) is **gitignored and private**. Don't paste raw values, video paths, or session metadata into commits, PRs, or external services.
- Commit messages: short imperative, no `Co-Authored-By` line.
- When in doubt about destructive actions (force push, `git reset --hard`, deleting cached models), ask first.
