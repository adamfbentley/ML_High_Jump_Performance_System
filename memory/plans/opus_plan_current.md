# Phase 9d Plan — Apparatus-Anchored Mpp At Takeoff Zone

> **Historical plan, superseded 2026-05-05.** Gravity-mpp was tested first and
> showed that vertical phone tilt corrupts handheld-flight calibration. The
> panned-footage rescue path is closed. Follow `memory/docs/decisions_log.md`
> and `ROADMAP.md`: validate the imported stationary pilot before Phase 10.
> Gravity-mpp, egomotion, scene homography, and hand-labelled apparatus truth
> are historical only; do not carry them into stationary admission.

Author: Claude Opus (architecture/physics review). 2026-05-04.

Phase 9c shipped. Hand-label evaluation
(`memory/experiments/exp_003_hand_label_evaluation.md`) showed:

- Egomotion clearly beats anatomical on takeoff vh.
- Auto-detector (Hough scene_homography) is rejected on every labelled clip.
- A residual ~0.9-1.5 m/s underestimate persists *even on the tripod control*.
  This isolates the next blocker as mpp calibration at the takeoff zone, not
  panning.

This plan addresses that residual.

## 1. The bias, stated precisely

`compute_per_frame_scale_mpp` in `src/pose_estimation/scale_calibration.py`
returns a single video-wide `mpp = thigh_or_shank_length_m / p95(pixel_projection)`.
The p95 picks the closest-depth instance of the segment across the whole clip,
giving the smallest valid mpp at the closest depth. That mpp is then applied
uniformly to every frame, including the takeoff frame.

Two depth-related sources of bias at the takeoff frame:

1. **Takeoff-zone depth ≠ p95 depth.** The athlete is at the takeoff line,
   typically 1-2 m in front of the bar. The p95 may have come from elsewhere
   in the run-up where she passed closer to the camera. Mpp-at-takeoff differs
   from mpp-at-p95-frame by the depth ratio.
2. **Takeoff-leg foreshortening.** At toe-off the planted leg is tilted
   forward into the bar plane and never presents a fully in-plane projection.
   p95 captures the swing leg or earlier in-plane instances; the takeoff-side
   leg's projection at takeoff is biased short and would, if anchored locally,
   give too-large mpp.

Both push the same direction: the `mpp` we apply at takeoff is too small,
scaling pixel-x velocity to metres-x velocity at a too-low rate. Recovered
takeoff vh is biased low, which matches the tripod control's 0.9 m/s
underestimate.

## 2. Goal

Add a third mpp source that is independently anchored to the high-jump
apparatus at the takeoff frame, then validate it against:

- The tripod control (where mode vh should match truth vh to within ≤0.3 m/s
  after recalibration).
- The 4 panned clips with usable labels (where egomotion + 9d should pull
  takeoff vh into the elite band on bar-tagged clips that already have truth
  in that band).

## 3. Architectural shape

Add `compute_apparatus_anchored_mpp` in
`src/pose_estimation/scale_calibration.py`:

```python
def compute_apparatus_anchored_mpp(
    scene_anchors: SceneAnchors,
    *,
    target_frame: int,
    window: int = 3,
) -> tuple[float | None, dict]:
    """Mpp from upright separation, measured at a specific frame window.

    Uses the labelled (or detected) upright_left_base_px and
    upright_right_base_px positions across `target_frame ± window` to compute
    pixel-distance between the upright bases, then divides
    `upright_separation_m` (default 4.02, IAAF spec) by the median pixel
    distance. Reports per-frame distances in `info` for diagnostics.

    Returns (mpp, info). mpp is None if no anchor pair is valid in the window.
    """
```

Wire it into `calibrate_landmarks_with_scene`:

- New optional kwarg `target_frame: int | None = None`. When set and
  `scene_anchors` is provided, prefer apparatus-anchored mpp at the takeoff
  window over anatomical p95 mpp. Anatomical remains the fallback when
  apparatus anchors are absent or invalid in the takeoff window.
- The `calibration_info` dict gains `apparatus_mpp`, `anatomical_mpp`,
  `mpp_source` (`"apparatus" | "anatomical"`), and
  `apparatus_anatomical_mpp_ratio` so we can see the size of the
  recalibration on each clip.

`scripts/analyze_jump_video.py` passes the takeoff frame index into
`calibrate_landmarks_with_scene` after takeoff selection (currently happens
later in the pipeline; will need to reorder: pose extract → takeoff frame
selection → calibration → kinematics).

## 4. Why upright separation is the right anchor

- IAAF specifies a fixed 4.00-4.04 m upright separation for both men's and
  women's high jump.
- Both upright bases are typically visible together in side-view footage
  during the takeoff window (when the apparatus is in the camera's field of
  view at all).
- The horizontal pixel distance between upright bases is invariant to camera
  pan: it is a fixed scene length viewed at the bar plane's depth.
- Critically, this measures mpp at the **bar plane's depth**, which is
  closer to the takeoff line's depth than the run-up's varying depth.

This is independent of:

- Anatomical mpp (which uses athlete segments).
- Egomotion (which removes pan but inherits whatever mpp is in use).
- Bar-height parsing (which is filename-derived).

So a discrepancy between apparatus mpp and anatomical mpp at the takeoff
window directly diagnoses the depth/foreshortening bias.

## 5. Files to edit

1. **EDIT** `src/pose_estimation/scale_calibration.py`:
   - Add `compute_apparatus_anchored_mpp(scene_anchors, *, target_frame, window=3) -> tuple[float | None, dict]`.
   - Add `target_frame: int | None = None` kwarg to
     `calibrate_landmarks_with_scene`.
   - When `target_frame` is set and apparatus mpp is computable, use it as
     the primary mpp instead of anatomical. Fall back to anatomical when
     apparatus mpp is None.
   - Surface `apparatus_mpp`, `anatomical_mpp`, `mpp_source`,
     `apparatus_anatomical_mpp_ratio` in the returned info dict.

2. **EDIT** `scripts/analyze_jump_video.py`:
   - Reorder so that takeoff frame selection runs before
     `calibrate_landmarks_with_scene` (or pass the takeoff frame in via a
     two-pass: a quick anatomical pass to pick takeoff frame, then the
     apparatus-anchored recalibration).
   - Pass `target_frame=takeoff_frame` to `calibrate_landmarks_with_scene`
     when scene anchors are available.
   - Surface the new mpp diagnostics in `report["calibration"]`.

3. **EDIT** `scripts/evaluate_calibration_truth.py`:
   - Use the labelled apparatus anchors (already present in
     `data/results/hand_anchors/*.json`) to compute a truth-side
     `apparatus_mpp` at the takeoff frame.
   - Add a fourth comparison row alongside anatomical / egomotion /
     scene_homography: `egomotion + apparatus_mpp` (the Phase 9d candidate).
   - Report per-mode `vh_takeoff_mps` and `err_vs_truth_mps` so we can see
     whether 9d closes the gap.

4. **NEW** `tests/test_pose_estimation/test_scale_calibration.py` (extend):
   - `test_apparatus_anchored_mpp_recovers_known_separation` — synthetic
     anchors at known pixel separations on synthetic frames; expect
     `compute_apparatus_anchored_mpp` to recover `4.02 / pixel_separation`.
   - `test_apparatus_mpp_window_robust_to_single_frame_jitter` — pass a
     window where one frame's right_base is corrupted; median across the
     window should still be correct.
   - `test_apparatus_mpp_falls_back_when_window_has_no_valid_anchors` —
     window outside any valid anchors; expect `None` and clear `info`.
   - `test_calibrate_landmarks_with_scene_prefers_apparatus_mpp_at_takeoff`
     — synthetic scene with known apparatus mpp ≠ anatomical mpp; with
     `target_frame` set, calibrated CoM-x velocity should reflect apparatus
     mpp scaling.

5. **EDIT** `memory/experiments/exp_003_hand_label_evaluation.md`:
   - After the 9d run, append a "Phase 9d evaluation" section with
     aggregate-only metrics (no per-clip private values).

## 6. Acceptance criteria

After Codex implements and re-runs `evaluate_calibration_truth.py`:

- **Tripod control:** apparatus-anchored egomotion vh-at-takeoff converges
  to truth vh within **≤0.3 m/s**. (Currently ~0.9 m/s gap.)
- **Bar-tagged labelled clips with usable labels:** apparatus-anchored
  egomotion vh-at-takeoff agrees with truth vh within **≤0.5 m/s** on
  ≥3 of 4 clips. (Borderline 2-label clips excluded.)
- **Apparatus / anatomical mpp ratio** on the tripod is reported and
  consistent with the observed vh recovery (e.g. ratio ~1.7 if anatomical
  was ~50 % of true).

If acceptance hits, the path forward is:

- Re-process the 72-clip corpus with `--egomotion on` and the new
  apparatus-anchored mpp where scene anchors are available (detector or
  hand-labelled).
- Update `scripts/aggregate_calibration_modes.py` to weight apparatus-mpp
  clips higher in the training-grade gate.
- Re-evaluate Phase 10 stop/go.

If acceptance misses (residual >0.5 m/s on tripod after 9d), escalate to
two-camera DLT or end-of-runway tripod filming.

## 7. What Codex must NOT do in 9d

- Do not run Phase 10 fine-tuning. Do not refresh
  `data/results/all_optimizations.json`.
- Do not modify physics conventions in `.github/copilot-instructions.md`.
- Do not modify `BiomechanicalSample` or the anatomical fallback path.
- Do not commit hand-label JSON files (already gitignored under
  `data/results/`).
- Do not paste per-clip metrics or clip stems into commits or external
  output.
- Do not add a YOLO/ML detector for apparatus — hand labels are the
  oracle; auto-detector is a separate dead path.

---

## Appendix: Phase 9c Plan (Historical) — Crossbar Homography

This is the original Phase 9c plan. The work below shipped (commit
`327fb00`); the auto-detector portion was found to be unviable on this
footage during the hand-label evaluation. Egomotion and the validation
infrastructure shipped successfully. Kept here for context on why we
ended up where we are.

---

## 1. Review of Codex's Phase 9 work

### 1a — Scale calibration (`src/pose_estimation/scale_calibration.py`)

**Approved with caveats.**

The premise is physically sound: rigid bone-length segments (thigh, shank) are
invariant across postures, so their projected pixel length depends only on the
camera distance and the limb's angle to the camera plane. Foreshortening can
only shorten a segment in the image, never lengthen it, so the *upper tail* of
the projection distribution corresponds to the in-plane (unforeshortened)
geometry. Using the 95th percentile (rather than the max) is correct — robust
to single-frame landmark noise. Median across left/right thigh and left/right
shank further dampens individual landmark errors.

Three caveats Codex should keep in mind:

1. **Surface-anatomical vs. joint-center mismatch.** Imogen's reported
   thigh = 0.43 m is almost certainly a tape-measure greater-trochanter to
   lateral-femoral-condyle distance (palpable surface landmarks). MediaPipe's
   `hip` landmark sits closer to the joint center (medial of greater
   trochanter), so the MediaPipe "thigh" measured in image space is slightly
   longer than 0.43 m and our derived mpp is therefore biased *down* by a
   few percent — peak CoM heights will read slightly low. The diagnostic in
   the close-clip log (thigh ≈ 2.84 mm/px vs shank ≈ 3.40 mm/px, ~20 %
   spread) is consistent with a thigh-length underestimate of similar order.
   Acceptable for now; flag for biomechanics review with the BMS PhD student.
2. **Single-scalar mpp assumes constant camera distance.** True at the
   takeoff zone but biased in the approach phase, where the athlete is
   farther from the camera and true mpp is larger. This is acceptable for
   takeoff metrics (the only metrics that matter for fine-tuning) but means
   approach-phase horizontal speeds will read about 1.5–2× too large in
   metric terms, independently of the panning issue addressed below.
3. **The 95th percentile assumes there are in-plane frames.** If the J-curve
   geometry means the legs are never perpendicular to the camera, p95 still
   underestimates true segment length. Imogen's J-curve does pass through
   roughly camera-perpendicular orientation near plant, so this is mostly
   fine, but worth noting.

Verdict: **ship it as the Phase 9a baseline.** Improvements above are
nice-to-haves, not blockers.

### 1b — Takeoff anchor (`scripts/analyze_jump_video.py:select_takeoff_frame_from_ground_contact`)

**Approved.**

"Takeoff = final frame of the final ground contact before the CoM apex" is
the physically correct definition: by F = m·a, once GRF returns to zero the
body is a projectile and vy decreases monotonically under gravity. Picking
the end of the last sustained contact interval is more robust than `argmax(vy)`
on a finite-difference vy series, which can fix on a single-frame landmark
spike. Pre-peak filtering correctly rejects landing rebounds.

Two minor concerns Codex should check during the next reprocess:

1. The 5 cm height threshold inside `detect_ground_contacts` is fixed in the
   ankle Y reference frame established by Phase 9a's 5th-percentile-of-visible
   ground reference. On far-camera clips with mpp ≈ 8 mm/px, a 5 cm threshold
   corresponds to ~6 px of ankle Y noise tolerance. If MediaPipe ankle
   visibility/jitter exceeds that on the relevant frames, contacts can be
   missed and the code will silently fall back to `argmax(vy)`. A diagnostic
   log line indicating which path was taken (and how many contacts were
   detected) would make this visible without changing behaviour.
2. The `* 100.0` scaling to centimetres on entry to `detect_ground_contacts`
   is correct (the detector takes cm as documented), but it leaves a unit
   mismatch latent across modules. A small `detect_ground_contacts(...,
   units="m" | "cm")` keyword would remove a future foot-gun.

Verdict: **ship it; the two concerns above are tracked-and-deferred.**

### 1c — Validation results

The aggregate distribution of the 45-video reprocess is consistent with what
the physics predicts:

- **Peak CoM median 1.67 m, 9/45 in 2.0–2.7 m.** Below the elite-female
  expected range, but the bar-tagged subset (median CoM-minus-bar = −0.05 m)
  is biomechanically plausible — Fosbury Floppers do clear the bar with the
  CoM passing slightly *below* the bar. Reading −0.05 m is on the cautious
  side of the expected −0.10 to −0.20 m. The most likely cause is the
  thigh-length-vs-joint-center mismatch above, which biases vertical scale
  down by a few percent.
- **Takeoff vy median 3.16 m/s, 18/45 in 3.0–4.5 m/s.** Defensible. Many
  videos do not capture the full stance phase or have the head landmark
  drift out of frame at apex.
- **Takeoff angle median 66.6°, only 3/45 in 38–48°.** This is the
  diagnostic finding. Together with takeoff horizontal speed median
  1.08 m/s (only 2/45 in 2.5–5.5 m/s), it tells us the issue is in the
  **horizontal-velocity numerator**, not the vertical denominator. The
  takeoff angle equation is `arctan2(vy, sqrt(vx² + vz²))`; if vy is
  roughly correct and vh is artefactually small, takeoff angle pegs near
  90°. That matches the observed median 66.6° (high) and the distribution.

Codex's diagnosis — that **panned single-camera footage does not yield
scene-fixed horizontal displacement** — is therefore correct. The fix is
not another scale tweak; it is to recover scene-fixed coordinates.

### 1d — Codex's stop decision

**Endorsed.** Fine-tuning Phase 10 on data with reliable vy but unreliable
vh would teach the PINN that vertical-only takeoff impulse is what matters,
which is wrong physically and would degrade the optimiser's
sensitivity-analysis output. The Phase 9 guardrails are saving us from a
silent training failure.

---

## 2. The blocker, stated precisely

The CoM positions emitted by `calibrate_landmarks_to_world` express the
athlete's location in the **camera image frame**, scaled to metres but not
re-anchored to the **scene** (the ground-fixed coordinate system anchored at
the bar/uprights). When the camera pans to follow the athlete:

- The athlete's pixel-x stays approximately centred in the frame.
- The scene moves through the frame at the panning rate.
- Finite-difference d(pixel_x)/dt × mpp returns approximately
  (athlete velocity − camera velocity) — a relative-to-camera quantity, not a
  ground-truth horizontal velocity.

For a steady-cam approach the bias would be small; for tracking-shot phone
footage of an athlete running at 6–8 m/s, the residual horizontal velocity
collapses to ~1 m/s (matching the validation distribution).

Recovering scene-fixed coordinates requires either:

- a **scene anchor** in the image (the crossbar / standards), used to back
  out camera motion frame-by-frame, or
- a **second synchronised camera** to triangulate via DLT.

The first is achievable on existing footage; the second is achievable on
future footage only. We pursue the first.

---

## 3. Recommended next phase: 9c — Crossbar homography

### 3a — Why the crossbar

The high jump apparatus is the cleanest scene reference available:

- **Two uprights** at IAAF-spec horizontal separation of 4.00–4.04 m
  (women's and men's competition use the same separation).
- **Crossbar** rests horizontally between them — visible in most jump
  attempts, less reliably visible during the run-up itself.
- **High contrast**: matte-coloured (often white or yellow) against either
  the dark mat behind or the sky / wall above, with sharp metallic edges.
- **Geometrically rigid**: the upright bases sit on the ground at known
  spacing.

Three observable anchor points (two upright bases, midpoint of crossbar — or
better, the two upright tops if the crossbar can be detected as a line)
suffice to fit a frame-by-frame homography from image space to scene space,
provided the ground plane assumption holds (camera roughly perpendicular to
the run-up direction).

This gives us **two independent scene-fixed scales**:

1. Horizontal X axis: upright-to-upright ≈ 4.02 m.
2. Vertical Y axis: bar height parsed from filename (e.g. `1.79`).

These can cross-check the Phase-9a thigh/shank scale and provide
independently-derived ground truth for vertical metrics on bar-tagged clips.

### 3b — Architectural shape

Add a new module `src/pose_estimation/scene_calibration.py` exposing:

```python
@dataclass
class SceneAnchors:
    upright_left_base_px:  np.ndarray | None  # (T, 2)  pixel x,y, NaN if absent
    upright_right_base_px: np.ndarray | None  # (T, 2)
    upright_left_top_px:   np.ndarray | None  # (T, 2)
    upright_right_top_px:  np.ndarray | None  # (T, 2)
    confidence:            np.ndarray         # (T,)   per-frame anchor reliability
    upright_separation_m:  float = 4.02
    bar_height_m:          float | None = None  # parsed from filename when known


def detect_scene_anchors(
    frames: Sequence[np.ndarray] | Path,        # video path or RGB frames
    *,
    bar_height_m: float | None = None,
    upright_separation_m: float = 4.02,
) -> SceneAnchors: ...


def fit_per_frame_homography(
    anchors: SceneAnchors,
) -> np.ndarray:                                 # (T, 3, 3) homographies
    """Image-px → scene-metres. Scene origin: midpoint of upright bases.
    Scene X: along the crossbar (right-positive). Scene Y: vertical, up-positive."""


def warp_landmarks_to_scene(
    landmarks_image_px: np.ndarray,              # (T, 33, 3) — px x, px y, vis
    homographies:       np.ndarray,              # (T, 3, 3)
    valid_mask:         np.ndarray,              # (T,) bool — fitted homography reliable
) -> np.ndarray:                                 # (T, 33, 3) scene metres + vis
    ...
```

The detector should emit per-frame confidence so downstream code can fall
back to the existing Phase-9a calibration on frames where the bar is
occluded by the athlete (typical: arch over the bar) or the camera angle
hides the standards (typical: extreme tight panning during run-up).

### 3c — Detection approach

Build the detector in three layers, in order of robustness:

1. **Geometric prior + colour mask** (first cut, low risk):
   - Detect long, near-vertical edges on the ground via probabilistic Hough
     transform after thresholding the image to extract the bar's distinctive
     colour band. Filter candidate segments by length > min_pixels (depth-
     dependent) and orientation within ±10° of vertical.
   - Pair candidate uprights by horizontal separation in pixel space:
     accept the pair whose pixel separation is most consistent across
     adjacent frames.
   - Detect the crossbar as the longest near-horizontal Hough segment that
     intersects both candidate upright tops.
2. **Optical-flow tracking between detections** (smoothing):
   - Once anchors are detected confidently in any frame, propagate them
     through neighbouring frames with sparse Lucas-Kanade flow on the
     anchor pixel positions. Cheap, robust to pan/zoom.
3. **Optional fine-tuned YOLO head** (later, if robustness is insufficient):
   - Hand-label 50–100 frames across the 45 clips, fine-tune a small
     detection model. Only worthwhile if 1 + 2 fail on >25 % of frames.

Expected anchor-detection coverage on our 45 clips, based on the
biomechanics of the task: **~75–90 % of frames during run-up + plant; lower
during the arch**. That is fine — homography is most needed during run-up,
plant, and takeoff (where horizontal velocity matters most), not at apex.

### 3d — Pipeline integration

`scripts/analyze_jump_video.py` should change minimally:

1. After pose extraction and BEFORE `calibrate_landmarks_to_world`:
   - Run `detect_scene_anchors(video_path, bar_height_m=parse_bar_height(...))`.
   - Run `fit_per_frame_homography(...)`.
2. In `calibrate_landmarks_to_world`, when scene homographies exist for a
   frame *and* their fit residual is below a threshold, use the homography
   to map the athlete's landmarks to scene coordinates. Use Phase-9a's
   thigh/shank scale only as a fallback (and as a cross-check).
3. Emit the homography-vs-anatomical scale residual into the report under
   `report["calibration"] = {"method": "scene_homography" | "anatomical",
   "scene_anatomical_scale_ratio": float, "anchor_coverage_pct": float}`.

The new code path must be **opt-in via CLI** for the first reprocess
(`--scene-anchor on|off`) so the v1 method stays runnable while the new
path is validated. Default off until the validation run proves out.

### 3e — Files Codex should edit

1. **NEW** `src/pose_estimation/scene_calibration.py` — `SceneAnchors`
   dataclass, `detect_scene_anchors`, `fit_per_frame_homography`,
   `warp_landmarks_to_scene`. ~250–350 lines.
2. **EDIT** `src/pose_estimation/scale_calibration.py` — add a
   `calibrate_landmarks_with_scene(landmarks_2d_px, scene_anchors,
   fallback_kwargs)` entry point that prefers homography per-frame and
   falls back to the existing anatomical scale on low-confidence frames.
   Keep the existing `calibrate_landmarks_to_world` behaviour intact.
3. **EDIT** `scripts/analyze_jump_video.py` — wire in scene-anchor
   detection, pass results into calibration, add `--scene-anchor` flag
   (default off), expand the report's `calibration` field.
4. **EDIT** `memory/docs/equations.md` — add the
   pixel-to-scene homography invariants and the upright-separation /
   bar-height scene constants.
5. **NEW** `tests/test_pose_estimation/test_scene_calibration.py` —
   synthetic anchor cases; see §4.

### 3f — Acceptance criteria

After Codex implements and reprocesses, expect:

- **Anchor coverage ≥ 70 % of frames** averaged across the 45 clips, with
  ≥ 50 % coverage on run-up frames specifically.
- **Takeoff horizontal speed median ≥ 4.0 m/s** across the 45 clips (was
  1.08 m/s).
- **Takeoff angle median in 38–55°**, with **≥ 25 of 45 in 38–48°** (was
  3/45). The wider acceptance band reflects single-camera homography
  residuals; the narrow band is the gold standard.
- **Bar-tagged peak-CoM clearance** (CoM minus bar height) median in
  −0.20 to −0.05 m, with the bulk inside −0.30 to 0.0 m (was median
  −0.05 m, plausible but uncertain because vertical scale was
  thigh-derived).
- **Cross-check residual**: scene-derived mpp vs anatomical-derived mpp
  agreement within ±10 % on clips where the scene path succeeds, on
  average. A persistent systematic offset will diagnose the
  thigh-vs-joint-center anatomical bias.

If 9c fails to lift takeoff angle into 38–55° range, the next escalation is
two-camera DLT for future filming (Codex retains the existing
`src/pose_estimation/dlt_triangulation.py`); existing footage is then
relegated to relative-kinematics analysis only.

---

## 4. Tests Codex should add

In `tests/test_pose_estimation/test_scene_calibration.py`:

1. **Synthetic perfect homography.** Place known pixel anchors of two
   uprights 4.02 m apart in scene space at fixed image coordinates. Confirm
   `fit_per_frame_homography` returns a homography whose application to the
   pixel anchors recovers the scene anchors to within 1 mm.

2. **Synthetic camera pan.** Generate a sequence where the same scene
   anchors are displaced in pixel space frame-to-frame by a pure horizontal
   translation. Confirm that warping a fixed scene-X point gives a constant
   scene-X across all frames (i.e. camera pan is removed).

3. **Robustness to one missing anchor.** Hide one upright base on a single
   frame; confirm the homography still fits using the remaining three
   anchors (or, in the worst case, marks the frame as low-confidence and
   the pipeline falls back to anatomical scale).

4. **Bar-height ground truth.** Construct a synthetic case with the bar
   at scene Y = 1.78 m, run a trivial detector that returns the bar's
   midpoint pixels, fit a homography that uses bar height as a vertical
   scale anchor, confirm the recovered Y coordinate of the bar in scene
   space is 1.78 ± 0.005 m.

5. **Coverage threshold integration.** End-to-end synthetic: known scene,
   known camera pan, known athlete kinematics. Confirm that the
   `analyze_jump_video.py` integration emits the correct scene-X velocity
   for the synthetic athlete and falls back to anatomical scale on a
   bar-occluded frame range without producing NaNs.

In `tests/test_kinematics/test_takeoff.py` (extend, do not replace):

1. **Anchor-aware takeoff angle.** Synthetic athlete with vy = 3.5,
   vh_scene = 4.5, simulated under camera pan that produces vh_image ≈ 0.5
   m/s. Confirm that with `--scene-anchor on` the takeoff angle reads
   ≈ arctan2(3.5, 4.5) = 38°, not arctan2(3.5, 0.5) = 82°.

---

## 5. Stop / go criteria for Phase 10

Hold Phase 10 fine-tuning until **both** of the following are true:

- 9c reprocess yields the acceptance numbers in §3f.
- A bar-tagged subset of ≥ 8 clips shows predicted bar clearance (rise
  from CoM-at-takeoff using vy²/(2g)) within ±5 cm of bar height when the
  athlete cleared, and within ±5 cm of the lowest body part trajectory
  when the athlete failed.

When both pass, fine-tune with:

```bash
.venv/Scripts/python.exe scripts/finetune_personal.py --dry-run
.venv/Scripts/python.exe scripts/finetune_personal.py --epochs 200 --lr 1e-4
```

Use `--max-peak-com-m 0` (no scale guardrail) only after 9c lands. Until
then keep the guardrail at 3.0 m so a regression in 9a/9b is caught.

---

## 6. Memory pipeline notes

The local RAG at `tools/memory/` is correct in design and privacy posture.
Hashing embeddings are deliberately lexical and avoid network calls, which
is the right tradeoff for this project's threat model. No structural change
is necessary right now, but three small improvements will make the index
more useful as Codex iterates:

1. **Add `AGENTS.md` and `tools/memory/config.yaml` to `include_globs`.**
   `AGENTS.md` is a documented source of truth; the config itself is
   useful when reasoning about what's in the index. Excludes are correct.
2. **Add a `mtime` field to `chunk_metadata`** so future incremental
   indexing can skip unchanged files. Requires no immediate behaviour
   change, just an extra field in the dict returned by `chunk_metadata`
   and read in `chunks_from_files`. Cheap forward-compat.
3. **Optional: pluggable embeddings via a config-selected backend.** Keep
   `hashing` as default. Add a `sentence-transformers` (e.g.
   `all-MiniLM-L6-v2`, ~80 MB, no network at runtime once cached) backend
   that activates only when `embedding.type: sentence-transformers` is
   set in the config. Gives semantic search ("homography" matches "scene
   anchor") when needed without breaking the deterministic-default
   workflow. Defer until 9c is in flight; lexical search is sufficient
   for the current vocabulary.

None of these are blockers. (1) is a one-line change worth doing now; (2)
and (3) are nice-to-haves.

---

## 7. Summary for Codex

Do, in order:

1. Implement 9c per §3 (scene_calibration.py module, pipeline integration,
   CLI flag).
2. Add the §4 tests; suite must remain green (≥ 195 passing).
3. Reprocess all 45 clips with `--scene-anchor on --thigh 0.43 --shank 0.47
   --save-samples data/results/samples`.
4. Compute the §3f acceptance metrics, write summary to a new
   `memory/experiments/exp_002_scene_calibration.md`.
5. Apply the small RAG improvements in §6.1 and §6.2 only.
6. Stop. Hand back to Opus to verify acceptance criteria.

Do not run Phase 10 fine-tuning, do not refresh `data/results/all_optimizations.json`,
do not modify physics conventions in `.github/copilot-instructions.md`.
