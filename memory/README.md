# Local RAG Memory

This folder holds lightweight shared memory for the Claude/Codex workflow.

Tracked files here should contain only project-level notes, design decisions,
plans, and aggregate experiment summaries. Do not store raw private athlete
data, source video paths, or per-session private report dumps.

Generated vector indexes and JSONL logs are local artifacts and are ignored by
git.

For the current small corpus, prefer direct reads and `rg`:

```powershell
rg -n "stationary|stationary_camera|Phase 10" memory ROADMAP.md
```

The optional local RAG workflow remains available if the corpus grows:

```powershell
.venv/Scripts/python.exe tools/memory/build_index.py
.venv/Scripts/python.exe tools/memory/query_index.py "why are takeoff angles inflated"
```

If ChromaDB is missing:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[memory]"
```
