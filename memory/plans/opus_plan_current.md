# Opus Plan Current

## Task For Claude Opus

You are returning to this repo after Codex completed Phase 9a/9b implementation,
full reprocessing, context-document refresh, and local RAG setup.

Before answering, read:

1. `CLAUDE.md`
2. `ROADMAP.md`
3. `ARCHITECTURE.md`
4. `.github/copilot-instructions.md`
5. `memory/README.md`
6. `memory/docs/physics_notes.md`
7. `memory/docs/equations.md`
8. `memory/docs/decisions_log.md`
9. `memory/experiments/exp_001_phase9_validation.md`
10. `memory/plans/codex_execution_notes.md`

Optional retrieval command:

```powershell
.venv/Scripts/python.exe tools/memory/query_index.py "Phase 10 blocker horizontal velocity takeoff angle"
```

## Your Role

Act as architecture and physics authority, not implementation agent.

Assess the work Codex has done since your last session:

- Phase 9a scale calibration and sample caching.
- Phase 9b contact-anchored takeoff frame.
- Bar-height parser fix.
- Full 45-video reprocess and aggregate residual distribution.
- Decision to block Phase 10 fine-tuning.
- New local RAG memory workflow.

## Questions To Answer

1. Is the contact-anchored takeoff definition physically defensible for this
   pipeline?
2. Do the aggregate validation numbers support Codex's decision to block
   fine-tuning?
3. Is the diagnosis of panned single-camera horizontal velocity as the current
   blocker sound?
4. What architecture should be tried next: crossbar/upright reference,
   homography, camera-motion compensation, two-camera DLT, or another approach?
5. What tests and acceptance criteria should Codex implement next?

## Write Output Here

Replace this section with:

- concise review of Codex work,
- physics/biomechanics concerns,
- recommended next implementation plan,
- specific files/modules Codex should edit,
- validation tests Codex should add,
- clear stop/go criteria for Phase 10 fine-tuning.

Do not include private athlete raw data, source video paths, private emails, or
per-session metadata. Aggregate numbers are acceptable.
