# Decisions Log

## 2026-04-29

- Use a file-mediated architect -> builder -> reviewer workflow.
- Claude Opus acts as physics/architecture reviewer.
- Codex acts as execution agent: implement, run tests, integrate, update logs.
- Use lightweight local RAG with ChromaDB as the local vector store.
- Use deterministic local hashing embeddings initially to avoid external model
  calls and private-code leakage.
- Do not fine-tune Phase 10 until horizontal/translational metrics from private
  videos are validated.

## 2026-04-29 (later)

- Phase 9 takeoff-anchor work and 9a scale calibration are accepted as the
  baseline. The Codex stop on Phase 10 fine-tuning is endorsed.
- The remaining blocker is the absence of scene-fixed horizontal coordinates
  in panned single-camera footage (takeoff vh median 1.08 m/s; takeoff angle
  median 66.6°). Scale calibration alone cannot recover this.
- Phase 9c will be **crossbar / upright homography**: detect the apparatus
  (uprights ~4.02 m apart, crossbar at the bar height parsed from filename),
  fit a per-frame image→scene homography, and warp athlete landmarks to
  scene coordinates before kinematics. Phase 9a anatomical scale is kept as
  fallback for frames where anchors are unreliable. Two-camera DLT remains
  the long-term fix for future filming, not a retrofit for existing footage.
- Detailed Phase 9c plan, file list, acceptance criteria, and tests are in
  `memory/plans/opus_plan_current.md`.
- Memory-pipeline change: `AGENTS.md` and `tools/memory/config.yaml` added
  to RAG `include_globs` so future queries can retrieve them. No structural
  change to the embedding or chunking pipeline; both remain appropriate for
  the current vocabulary and privacy posture. Two further improvements
  (mtime metadata for incremental indexing; optional sentence-transformers
  backend) are scoped but deferred until 9c is in flight.

## 2026-05-03

- ChromaDB / local RAG: parked. The corpus is small enough that direct
  `Read` + `Grep` over `memory/`, `src/`, and the root docs is faster and
  more accurate than the lexical-hashing index. `MEMORY.md`, `CLAUDE.md`,
  and `AGENTS.md` already serve the cross-session context role. Revisit
  only if the corpus grows past ~200 docs/files, or if we swap in
  `sentence-transformers` for true semantic search. Tooling under
  `tools/memory/` and the `[memory]` extra are left in place but unused.
- Workspace tidy: removed stale Phase 9a/9b Codex handoff log
  (`memory/plans/codex_execution_notes.md`) and the GitHub-Copilot-style
  agent / prompt scaffolding under `.github/agents/` and `.github/prompts/`.
  ROADMAP.md, `opus_plan_current.md`, `AGENTS.md`, and `CLAUDE.md` are the
  canonical agent-facing docs.

## 2026-05-04

- Hand-label evaluation completed on 5 bar-tagged clips (1 tripod control,
  4 panned phone clips spanning 4 sessions and bar heights 1.72-1.85 m).
  Truth pipeline reworked to compare at the takeoff window only, not
  per-frame median (the original median-over-clip metric was dominated by
  pose-velocity noise and produced misleading "all modes broken" verdicts).
- Earlier "median takeoff vh = 1.08 m/s" smoke finding was an artefact:
  per-frame median on anatomical-only output, with pose noise dominating
  the gradient. At-takeoff comparison shows truth values mostly in or near
  the 2.5-5.5 m/s elite band — Imogen's takeoff vh is closer to elite than
  the smoke implied.
- Egomotion is the better mode by a clear margin: on the densely-labelled
  bar-tagged clip it recovered ~80 % of truth vh vs anatomical's ~48 %.
  Auto-detector (`scene_homography`) was rejected on every labelled clip;
  it locks onto wrong vertical edges (mat frame, net poles) — dead path on
  this footage without per-clip hand labels.
- A residual ~0.9-1.5 m/s underestimate persists *even on the tripod clip*
  where there is no panning to remove. This isolates the next blocker:
  anatomical mpp (p95 of thigh projection) is biased small at the takeoff
  zone, probably because the takeoff-side leg is planted/tilted and never
  presents a fully in-plane projection there.
- Decision: pursue Phase 9d apparatus-anchored mpp recalibration before
  considering tripod re-filming. Use the labelled upright separation
  (4.02 m, IAAF spec) as a third independent scale source measured at the
  takeoff frame. Validate against the tripod clip (where we expect mode
  vh to match truth vh) and the 4 bar-tagged clips. Only re-film if 9d
  cannot close the gap. Plan in `memory/plans/opus_plan_current.md`.
- The `scripts/aggregate_calibration_modes.py` decision logic remains
  smoke-only — its "escalate to fixed-camera filming" output from n=2
  smoke reports was structurally invalid and has been overruled by the
  hand-label evaluation. The file's gates and ranking are correct for a
  full reprocess but should not be trusted on small samples.
- Hand-label tool hardened: `s` key skips frames without visible apparatus,
  partial work autosaves after each successful label, unreadable frames
  (cv2 quirk on some MP4 encodings) are tolerated. Truth evaluator aligns
  labels to MediaPipe's source-frame indices (MediaPipe drops undetected
  frames, so output index ≠ source frame index).

## 2026-05-05

- Future capture policy changed: stationary footage is now required for
  training-grade physics, Phase 10 personal fine-tuning, and optimiser claims.
  Handheld/panned footage is retained for exploratory analysis, detector
  development, and relative technique review, but should not be admitted to
  personalised training unless it passes strict calibration gates.
- Next data-collection step: ask Imogen for a small stationary validation set
  before spending further effort rescuing panned footage. Minimum useful setup:
  tripod/fixed phone, no panning or zoom, 60 fps preferred, landscape, full
  body visible from final approach through landing, bar height in filename, and
  bar/uprights visible where possible. Two fixed cameras with sync clap/flash
  is the preferred gold-standard progression.
