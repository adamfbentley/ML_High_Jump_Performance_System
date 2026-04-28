"""Query the local RAG index and print file/line snippets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import shorten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.memory.rag_core import (
    DEFAULT_CONFIG,
    cosine_distance_to_score,
    load_config,
    make_embedding_function,
    repo_root_from_config,
    require_chromadb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local RAG vector index")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=8)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--max-chars", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    root = repo_root_from_config(config_path)
    chromadb = require_chromadb()
    embedding_function = make_embedding_function(config)

    client = chromadb.PersistentClient(path=str(root / config["index_path"]))
    try:
        collection = client.get_collection(
            name=config["collection_name"],
            embedding_function=embedding_function,
        )
    except Exception as exc:
        raise SystemExit(
            "No local RAG collection found. Build it first with:\n"
            "  .venv/Scripts/python.exe tools/memory/build_index.py"
        ) from exc

    result = collection.query(
        query_texts=[args.query],
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )

    rows = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        score = cosine_distance_to_score(distance)
        rows.append(
            {
                "path": metadata["path"],
                "start_line": metadata["start_line"],
                "end_line": metadata["end_line"],
                "score": score,
                "text": document,
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    for i, row in enumerate(rows, start=1):
        location = f"{row['path']}:{row['start_line']}"
        score_text = "n/a" if row["score"] is None else f"{row['score']:.3f}"
        snippet = shorten(
            " ".join(str(row["text"]).split()),
            width=args.max_chars,
            placeholder=" ...",
        )
        print(f"\n[{i}] {location} score={score_text}")
        print(snippet)


if __name__ == "__main__":
    main()
