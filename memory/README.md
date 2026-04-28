# Local RAG Memory

This folder holds lightweight shared memory for the Claude/Codex workflow.

Tracked files here should contain only project-level notes, design decisions,
plans, and aggregate experiment summaries. Do not store raw private athlete
data, source video paths, or per-session private report dumps.

Generated vector indexes and JSONL logs are local artifacts and are ignored by
git.

Recommended workflow:

```powershell
.venv/Scripts/python.exe tools/memory/build_index.py
.venv/Scripts/python.exe tools/memory/query_index.py "why are takeoff angles inflated"
```

If ChromaDB is missing:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[memory]"
```
