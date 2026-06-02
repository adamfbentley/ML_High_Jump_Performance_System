"""Build the local Chroma RAG index for project docs and code."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.memory.rag_core import (
    DEFAULT_CONFIG,
    append_jsonl,
    chunk_metadata,
    chunks_from_files,
    load_config,
    make_embedding_function,
    repo_root_from_config,
    require_chromadb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local RAG vector index")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reset", action="store_true", help="Delete existing collection first")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def stale_chunk_ids(indexed_ids: list[str], current_ids: list[str]) -> list[str]:
    """Return obsolete chunk IDs left behind by earlier document versions."""
    return sorted(set(indexed_ids) - set(current_ids))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    root = repo_root_from_config(config_path)
    chromadb = require_chromadb()
    embedding_function = make_embedding_function(config)

    chunks = chunks_from_files(root, config)
    if not chunks:
        raise SystemExit("No chunks found. Check tools/memory/config.yaml include/exclude globs.")

    index_path = root / config["index_path"]
    collection_name = config["collection_name"]
    index_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_path))
    if args.reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.document for chunk in chunks]
    metadatas = [chunk_metadata(chunk) for chunk in chunks]
    obsolete_ids = stale_chunk_ids(collection.get(include=[])["ids"], ids)
    if obsolete_ids:
        collection.delete(ids=obsolete_ids)

    for start in range(0, len(chunks), args.batch_size):
        end = start + args.batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    source_paths = sorted({chunk.rel_path for chunk in chunks})
    append_jsonl(
        root / "memory/logs/rag_builds.jsonl",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "collection": collection_name,
            "chunks": len(chunks),
            "source_files": len(source_paths),
            "index_path": config["index_path"],
            "embedding": embedding_function.name(),
        },
    )

    print(f"Indexed {len(chunks)} chunks from {len(source_paths)} files.")
    print(f"Collection: {collection_name}")
    print(f"Index path: {index_path}")


if __name__ == "__main__":
    main()
