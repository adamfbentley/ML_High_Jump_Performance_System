from __future__ import annotations

import numpy as np

from tools.memory.rag_core import HashingEmbeddingFunction, chunk_text, is_excluded


def test_hashing_embedding_is_deterministic_and_normalized():
    embedder = HashingEmbeddingFunction(dimensions=128)

    first = embedder.embed_one("takeoff vertical velocity")
    second = embedder.embed_one("takeoff vertical velocity")

    np.testing.assert_allclose(first, second)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-6


def test_chunk_text_tracks_source_lines():
    text = "\n".join(f"line {i}" for i in range(1, 21))

    chunks = chunk_text(text, "notes.md", chunk_chars=35, overlap_chars=10)

    assert len(chunks) > 1
    assert chunks[0].rel_path == "notes.md"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line >= chunks[0].start_line


def test_private_data_exclusion_globs():
    excludes = ["data/**", "EMAIL_FOR_IMOGEN.md", "memory/vector_index/**"]

    assert is_excluded("data/results/all_sessions_report.json", excludes)
    assert is_excluded("EMAIL_FOR_IMOGEN.md", excludes)
    assert is_excluded("memory/vector_index/chroma/index.bin", excludes)
    assert not is_excluded("src/kinematics/run_up_analysis.py", excludes)
