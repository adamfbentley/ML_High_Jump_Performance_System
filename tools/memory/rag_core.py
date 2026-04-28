"""Shared helpers for lightweight local RAG tooling.

The embedding is deliberately local and deterministic. It is not as semantically
rich as a transformer embedding, but it avoids network calls, private-code
leakage, and heavyweight model downloads while still giving useful lexical
retrieval over code and project notes.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CONFIG = Path("tools/memory/config.yaml")

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Chunk:
    """A text chunk with source metadata."""

    chunk_id: str
    document: str
    rel_path: str
    start_line: int
    end_line: int
    chunk_index: int


class HashingEmbeddingFunction:
    """Chroma-compatible deterministic hashing embedding function."""

    def __init__(self, dimensions: int = 2048) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def name(self) -> str:
        return f"local_hashing_{self.dimensions}"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name
        return [self.embed_one(text).tolist() for text in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Chroma 1.x document embedding hook."""
        return self(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Chroma 1.x query embedding hook."""
        return self(input)

    def embed_one(self, text: str) -> np.ndarray:
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        features = tokens + [f"{a}::{b}" for a, b in zip(tokens, tokens[1:])]
        vec = np.zeros(self.dimensions, dtype=np.float32)

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little", signed=False)
            index = raw % self.dimensions
            sign = 1.0 if (raw >> 63) == 0 else -1.0
            vec[index] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML config."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root_from_config(config_path: Path) -> Path:
    """Infer repo root from tools/memory/config.yaml by default."""
    return config_path.resolve().parents[2]


def normalise_rel_path(path: Path, root: Path) -> str:
    """Return a stable POSIX-style repo-relative path."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_excluded(rel_path: str, exclude_globs: list[str]) -> bool:
    """Check whether a repo-relative path matches any exclude glob."""
    rel = rel_path.replace("\\", "/")
    return any(fnmatch.fnmatch(rel, pattern) for pattern in exclude_globs)


def collect_source_files(root: Path, config: dict[str, Any]) -> list[Path]:
    """Collect files from include globs after applying excludes."""
    include_globs = config.get("include_globs", [])
    exclude_globs = config.get("exclude_globs", [])
    seen: set[Path] = set()

    for pattern in include_globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            rel_path = normalise_rel_path(path, root)
            if is_excluded(rel_path, exclude_globs):
                continue
            seen.add(path.resolve())

    return sorted(seen, key=lambda p: normalise_rel_path(p, root))


def read_text(path: Path) -> str | None:
    """Read text, returning None for binary or unsupported files."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def make_chunk_id(rel_path: str, chunk_index: int, text: str) -> str:
    """Stable chunk id from path, index, and content digest."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    safe_path = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel_path)
    return f"{safe_path}:{chunk_index}:{digest}"


def chunk_text(
    text: str,
    rel_path: str,
    chunk_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Chunk text on line boundaries with rough character limits."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")

    lines = text.splitlines()
    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    def emit() -> None:
        nonlocal current, current_chars
        if not current:
            return
        start_line = current[0][0]
        end_line = current[-1][0]
        document = "\n".join(line for _, line in current).strip()
        if not document:
            current = []
            current_chars = 0
            return
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(rel_path, chunk_index, document),
                document=document,
                rel_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                chunk_index=chunk_index,
            )
        )

        if overlap_chars == 0:
            current = []
            current_chars = 0
            return

        overlap: list[tuple[int, str]] = []
        count = 0
        for item in reversed(current):
            line_len = len(item[1]) + 1
            if overlap and count + line_len > overlap_chars:
                break
            overlap.append(item)
            count += line_len
        current = list(reversed(overlap))
        current_chars = count

    for line_no, line in enumerate(lines, start=1):
        current.append((line_no, line))
        current_chars += len(line) + 1
        if current_chars >= chunk_chars:
            emit()

    emit()
    return chunks


def chunks_from_files(root: Path, config: dict[str, Any]) -> list[Chunk]:
    """Read and chunk all configured source files."""
    chunk_chars = int(config.get("chunk_chars", 2600))
    overlap_chars = int(config.get("chunk_overlap_chars", 300))
    chunks: list[Chunk] = []

    for path in collect_source_files(root, config):
        text = read_text(path)
        if text is None:
            continue
        rel_path = normalise_rel_path(path, root)
        chunks.extend(chunk_text(text, rel_path, chunk_chars, overlap_chars))

    return chunks


def chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Metadata payload for Chroma."""
    return {
        "path": chunk.rel_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "chunk_index": chunk.chunk_index,
    }


def require_chromadb():
    """Import ChromaDB with a helpful install message."""
    try:
        import chromadb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "ChromaDB is not installed. Install local RAG dependencies with:\n"
            '  .venv/Scripts/python.exe -m pip install -e ".[memory]"'
        ) from exc
    return chromadb


def make_embedding_function(config: dict[str, Any]) -> HashingEmbeddingFunction:
    """Create the configured embedding function."""
    embedding = config.get("embedding", {})
    kind = embedding.get("type", "hashing")
    if kind != "hashing":
        raise ValueError(f"Unsupported embedding type: {kind}")
    dimensions = int(embedding.get("dimensions", 2048))
    return HashingEmbeddingFunction(dimensions=dimensions)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a small JSONL log record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def cosine_distance_to_score(distance: float | None) -> float | None:
    """Convert Chroma cosine distance to a simple similarity score."""
    if distance is None or math.isnan(distance):
        return None
    return 1.0 - distance
