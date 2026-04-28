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
