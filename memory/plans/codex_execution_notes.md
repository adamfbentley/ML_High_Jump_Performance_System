# Codex Execution Notes

Use this file for concise implementation notes that should be visible to Claude
Opus during review.

## Codex Session Summary For Claude Review

This session picked up after Claude identified Phase 9a/9b follow-ups.

### Commits pushed by Codex

- `10745e4` — `Complete Phase 9 calibration and takeoff anchor`
- `7c05c23` — `Fix bar height parsing`
- `3fa18c2` — `Update Phase 9 validation status`
- `a0d2e41` — `Refresh context docs`
- `e3e1e85` — `Refresh architecture status`
- `01ea659` — `Add local RAG memory tooling`

### Code and pipeline work completed

- Verified and committed Phase 9a work:
  - `BiomechanicalSample.save_npz()` / `load_npz()`.
  - `scripts/analyze_jump_video.py --save-samples`.
  - `scripts/finetune_personal.py`.
  - Phase 9a multi-segment scale calibration and robust ankle-ground reference.
- Implemented Phase 9b:
  - Added contact-anchored takeoff selection in `scripts/analyze_jump_video.py`.
  - Takeoff frame is the final frame of the final detected ankle-ground contact.
  - Uses both ankle trajectories after converting calibrated metres to
    centimetres for `detect_ground_contacts`.
  - Falls back to `argmax(vy)` if no contact interval is available.
  - Ignores post-peak contact intervals when pre-peak contacts exist, reducing
    landing/mat contact selection risk.
- Fixed bar-height parsing for filenames ending in numeric video extensions
  such as `.mp4`.
- Reprocessed all 45 private high-jump videos twice:
  - Once after 9a/9b.
  - Once again after bar-height parser fix.
  - Final local result: 45/45 reports and 45/45 cached `.npz` samples under
    `data/results/samples/`.
- Did not run Phase 10 fine-tuning because aggregate metrics failed the
  handoff plausibility threshold.

### Validation results

Final full non-PINN test command:

```powershell
.venv/Scripts/python.exe -m pytest tests/ --ignore=tests/test_pinn -q
```

Result after local RAG tests were added:

```text
190 passed
```

Aggregate private-video reprocess results, using only summary statistics:

```text
n_reports = 45
samples = 45
bar-tagged = 17/45
peak CoM median = 1.67 m, 9/45 in 2.0-2.7 m
takeoff vy median = 3.16 m/s, 18/45 in 3.0-4.5 m/s
takeoff angle median = 66.6 deg, 3/45 in 38-48 deg
CoM-minus-bar median = -0.05 m on bar-tagged subset, 8/17 in -0.30 to +0.10 m
takeoff horizontal speed median = 1.08 m/s, 2/45 in 2.5-5.5 m/s
```

### Current scientific decision

Do not fine-tune yet.

Phase 9a improved vertical scale and Phase 9b removed the obvious takeoff-frame
velocity-spike failure mode. The remaining blocker appears to be scene-fixed
horizontal motion: panned single-camera footage does not yield trustworthy
horizontal displacement/velocity, so takeoff angle remains unreliable.

### Local RAG memory system added

- Added `memory/` for file-mediated Claude/Codex collaboration.
- Added `tools/memory/build_index.py` and `tools/memory/query_index.py`.
- Added optional dependency group `memory = ["chromadb>=0.5"]`.
- Installed ChromaDB into the local venv with:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[memory]"
```

- Built local index:

```powershell
.venv/Scripts/python.exe tools/memory/build_index.py --reset
```

Latest build result:

```text
Indexed 272 chunks from 84 files.
```

- Query example verified:

```powershell
.venv/Scripts/python.exe tools/memory/query_index.py "how do I use local RAG memory"
```

### Context docs updated

Tracked and pushed:

- `CLAUDE.md`
- `ROADMAP.md`
- `ARCHITECTURE.md`

Updated locally but ignored/untracked:

- `.github/copilot-instructions.md`
- `HANDOFF.md`

### Remaining local-only state

These are expected:

```text
 M .gitignore
?? HANDOFF.md
!! .github/copilot-instructions.md
!! memory/vector_index/
!! memory/logs/rag_builds.jsonl
```

The unstaged `.gitignore` change is the user's private business-note ignore
block and was intentionally not committed. The committed `.gitignore` change
only added local RAG artifact ignores.

## Requested Claude Role Now

Claude should assess the Codex work above for:

- physics correctness,
- biomechanical validity,
- risk of false confidence in the Phase 9 metrics,
- whether the contact-anchored takeoff definition is acceptable,
- whether the current conclusion about panned-camera horizontal velocity is
  sound,
- the best next architecture plan for resolving the Phase 10 blocker.

Claude should not implement code during that review. It should write its
critique and next implementation plan to `memory/plans/opus_plan_current.md`.
