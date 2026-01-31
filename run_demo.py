#!/usr/bin/env python3
"""
One-command demo: validate ENDEE_TOKEN, ingest sample documents into Endee, then run one retrieval query.
No OpenAI key required for retrieval-only mode.
"""

import os
import sys

# Load .env before importing config
from pathlib import Path
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

from config import ENDEE_TOKEN, INDEX_NAME, SAMPLE_DOCS_PATH


def main() -> None:
    if not ENDEE_TOKEN or ENDEE_TOKEN == "your-endee-token-here":
        print("ERROR: ENDEE_TOKEN is not set.", file=sys.stderr)
        print("  1. Copy .env.example to .env", file=sys.stderr)
        print("  2. Edit .env and set ENDEE_TOKEN to your Endee API token (from https://endee.io)", file=sys.stderr)
        sys.exit(1)

    if not SAMPLE_DOCS_PATH.exists():
        print(f"ERROR: Sample documents not found at {SAMPLE_DOCS_PATH}", file=sys.stderr)
        sys.exit(1)

    print("Ingesting sample documents into Endee...")
    from ingest import run_ingest
    n = run_ingest(documents_path=SAMPLE_DOCS_PATH, index_name=INDEX_NAME)
    print(f"  Ingested {n} chunks into index '{INDEX_NAME}'.\n")

    query = "What is Endee used for?"
    print(f"Query: {query}\n")
    from query import answer
    result = answer(query, index_name=INDEX_NAME, top_k=5, use_llm=False)
    print("--- Retrieved context ---")
    print(result["context"])
    if result.get("answer"):
        print("\n--- Answer ---")
        print(result["answer"])
    print("\nDone. Run 'python -m query' for interactive mode or 'python -m query \"Your question\"' for a single query.")


if __name__ == "__main__":
    main()
