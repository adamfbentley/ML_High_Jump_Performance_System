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
# Run the full test suite (currently 263 passing non-PINN)
.venv/Scripts/python.exe -m pytest tests/ --ignore=tests/test_pinn -q

# Full suite including PINN
.venv/Scripts/python.exe -m pytest tests/ -q

# Search agent memory directly
rg -n "stationary|stationary_camera|Phase 10" memory ROADMAP.md

# Production stationary-footage run (use for any new fixed-camera session)
.venv/Scripts/python.exe scripts/analyze_jump_video.py "<stationary-dir>" \
    --thigh 0.45 --shank 0.45 \
    --capture-mode stationary \
    --roi-crop on \
    --save-samples <ignored-samples-dir>

# Dry-run personal fine-tuning (loads cached samples, applies Phase 9a guardrail)
.venv/Scripts/python.exe scripts/finetune_personal.py --dry-run

# Smoke-test on a single video
.venv/Scripts/python.exe scripts/analyze_jump_video.py "<video-file>"
```

## Current phase (June 2026)

**Phases 9a-9e SHIPPED. Two newer stationary clips pass the implemented report
gates. Phase 10 remains blocked pending admitted-only caching, explicit
fixed-camera confirmation, anchor-threshold tightening, and a larger session.**

Key pipeline additions shipped 2026-06-03 (all tested, 263 non-PINN passing):

- `--capture-mode stationary` (asserted, never inferred): credits a fixed camera
  as the scene-fixed horizontal source, removing the `no_scene_fixed_horizontal_source`
  training gate. Default `handheld` unchanged.
- `_validate_takeoff_anchor()`: after selecting the last pre-peak ground contact,
  validates (a) vy > 0 and peak follows, (b) frame lead ≤ 2·t_apex·fps. Rejects
  approach-stride false detections. `quality.takeoff_anchor_review_passed` published.
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

Stationary pilot outcome (5 clips, 2 captures, `--capture-mode stationary --roi-crop on`):

- 2/3 newer landscape clips: **training_grade = True**. Takeoff angles 41–43°,
  vh 3.57–3.60 m/s, window pose validity 70–73 %, contact + anchor review passed.
- 1/3 newer clip: fails — ankle contact detection failure only; physics in-range.
- Both earlier controls: not path-forward (approach-stride anchor, high spread).

Do **not** fine-tune or refresh optimiser claims yet. The analyser currently
caches every processed clip when `--save-samples` is supplied, while the
fine-tune loader filters only peak CoM. Add admitted-only caching before using
the two passing clips, then collect a larger stationary session. Follow
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
