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

# Dry-run personal fine-tuning (loads the admitted cache, applies guardrails)
.venv/Scripts/python.exe scripts/finetune_personal.py \
    --samples-dir <ignored-samples-dir> --dry-run

# Smoke-test on a single video
.venv/Scripts/python.exe scripts/analyze_jump_video.py "<video-file>"

# Apparatus anchor QA frame for stable-window experiments
.venv/Scripts/python.exe scripts/detect_stable_takeoff_anchors.py \
    --video "<clip.mp4>" --frame-index <frame> \
    --auto-base-posts --bar-height 1.75 \
    --output-json data/results/stationary_validation/anchors.json \
    --debug-image data/results/stationary_validation/anchors.jpg
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
- Current stationary retest (2026-06-14): the approved production path
  `scripts/analyze_jump_video.py --capture-mode stationary
  --stationary-camera-confirmed --roi-crop on --bar-height 1.75` still admits
  all three newer fully stationary Athlete A clips and caches all three samples.
  This is the reliable path for extracting physics from current data.
- Apparatus detection status (2026-06-14): the red/night stable-window detector
  in `scripts/detect_stable_takeoff_anchors.py` can label the red apparatus and
  now fits the landing-pad top edge independently, but it is not a general
  daylight detector. A geometry-first, colour-agnostic detector now exists
  (`src/pose_estimation/apparatus_detector.py`, CLI
  `scripts/detect_apparatus_geometry.py`): it reproduces the night-red anchors
  *without* requiring red and correctly rejects the floodlight masts (no paired
  crossbar), but is still only ~1/3 reliable on the daylight stationary clips
  (one clip clean, one wrong substructure, one missed). Not yet admission-grade.
- Apparatus detection update (2026-06-17): added a **static-plate** path —
  `build_static_plate` (in `camera_motion.py`) registers a window to a reference
  and takes the per-pixel temporal median, so the moving athlete dissolves into a
  clean apparatus-only image; `detect_apparatus_geometry.py --plate` detects on
  it. On the daylight trio the **plate is excellent on all 3** (Athlete A fully
  removed), but pure-geometry auto-detect is still only ~1/3 right (clips 2–3
  latch the background shed roofline, which is geometrically two corners + a
  crossbar). The reliable anchor route is now the manual click tool
  `scripts/annotate_apparatus.py` (clean plate → click 4 points → exact
  `points_px`); its clicks also seed a future learned 4-point detector. NB:
  per the parked-physics finding, better anchors give scene **scale**, they do
  not by themselves revive the degenerate monocular projectile solve.
- Apparatus detection update (2026-06-17, pose-ROI): the generalisable approach is
  **pose-localized**. `src/pose_estimation/apparatus_pose_prior.py` uses the
  athlete's CoM apex (over the bar), takeoff plant, and stature ruler to bracket
  the apparatus and give `bar_x`, a ground line, and a bar-height prior;
  `scripts/detect_apparatus_geometry.py --pose-roi` then seeds the two standards
  from vertical structure inside the ROI and sets the **bar line from the pose
  apex** (the faint daylight crossbar is not directly detectable — the landing-mat
  top edge dominates any edge response — and a geometric bar from horizontal
  standard separation is corrupted by perspective). Status on the daylight trio:
  ROI brackets the apparatus reliably; base posts are sometimes correct (clip 2
  both correct); the remaining error is the **left/right top** under perspective
  (flat-bar assumption) and occasional wrong standard seed. The prior red detector
  `detect_stable_takeoff_anchors.py` gained an opt-in `--bar-mode edge` (drops the
  red requirement) but the faint daylight bar is still not reliably found, so this
  is experimental. Reliable anchors today: hand-label via
  `scripts/label_scene_anchors.py`.
- Moving-footage CV tooling (2026-06-14): `src/pose_estimation/camera_motion.py`
  measures background camera motion and **stabilizes** a takeoff-centred window
  to a reference (a ~6 px/frame panning takeoff window registers to sub-pixel
  residual). Corpus scan finding: "longest still window" is the pre-run-up
  standstill, **not** the takeoff — the takeoff is where the camera pans, so a
  window must be centred on the detected toe-off. Retained as active geometry
  tooling (CLI `scripts/scan_stable_windows.py`).
- **Gravity/PnP/bar-plane takeoff physics is PARKED (2026-06-17).** Root cause of
  the long-standing impossible velocities: the monocular projectile fit is
  degenerate along the camera optical axis (free-depth ⇒ vh ≈ 662 m/s, even on
  stationary clips). A bar-plane gravity-as-scale variant is well-posed and gives
  the right *angle*, but is starved by upstream data (sparse takeoff-window pose +
  athlete off the bar plane). The whole gravity path —
  `analyze_stable_takeoff_window.py`, `analyze_moving_takeoff.py`,
  `gravity_calibration.py` — is preserved on branch
  `parked/moving-footage-physics` (history `a0e7ca7`), not deleted. Do not revive
  it for optimiser claims. The reliable physics route remains the stationary +
  anatomical production path.

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
