"""Ingest documents: chunk text, embed with sentence-transformers, and upsert to Endee."""

import hashlib
import re
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from config import (
    DATA_DIR,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    ENDEE_TOKEN,
    INDEX_NAME,
    SAMPLE_DOCS_PATH,
)
from embeddings import embed_texts, get_embedding_model
from store import ensure_index, get_client, upsert_vectors


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ", " "),
) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    Tries to break on paragraph, then line, then sentence, then word.
    """
    if not text.strip():
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Prefer splitting at a natural boundary
            segment = text[start:end]
            best = -1
            for sep in separators:
                idx = segment.rfind(sep)
                if idx > best:
                    best = idx
            if best > chunk_size // 2:
                end = start + best + len(separators[0] if separators else "")
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < text_len else text_len
    return chunks


def load_documents(path: Path) -> List[str]:
    """Load documents from a text file (one document per blank-line-separated block)."""
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    docs = re.split(r"\n\s*\n", content)
    return [d.strip() for d in docs if d.strip()]


def stable_id(text: str, index: int) -> str:
    """Generate a stable unique id for a chunk."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"chunk_{index}_{h}"


def _chunk_documents(
    docs: List[str], chunk_size: int, overlap: int
) -> Tuple[List[str], List[str], List[dict]]:
    """Chunk docs and return (ids, vectors not yet computed, meta). Caller embeds and upserts."""
    all_chunks: List[str] = []
    chunk_to_doc: List[int] = []
    for doc_idx, doc in enumerate(docs):
        chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        chunk_to_doc.extend([doc_idx] * len(chunks))
    ids = [stable_id(c, i) for i, c in enumerate(all_chunks)]
    meta = [
        {"text": c, "doc_index": chunk_to_doc[i]}
        for i, c in enumerate(all_chunks)
    ]
    return ids, all_chunks, meta


def run_ingest(
    documents_path: Path = SAMPLE_DOCS_PATH,
    index_name: str = INDEX_NAME,
    chunk_size: int = 512,
    overlap: int = 64,
) -> int:
    """
    Load documents, chunk, embed, and upsert to Endee.
    Returns the number of chunks upserted.
    """
    if not ENDEE_TOKEN:
        raise ValueError(
            "ENDEE_TOKEN is not set. Copy .env.example to .env and add your Endee token."
        )

    docs = load_documents(documents_path)
    if not docs:
        raise FileNotFoundError(
            f"No documents found at {documents_path}. "
            "Add a sample_documents.txt in the data/ folder or pass a custom path."
        )

    ids, all_chunks, meta = _chunk_documents(docs, chunk_size, overlap)
    model = get_embedding_model(EMBEDDING_MODEL)
    vectors = embed_texts(model, all_chunks)

    client = get_client(ENDEE_TOKEN)
    ensure_index(client, index_name, dimension=EMBEDDING_DIMENSION, space_type="cosine")
    upsert_vectors(client, index_name, ids, vectors, meta)

    return len(ids)


def main() -> None:
    """CLI entry: ingest from default or given path."""
    import argparse
    parser = argparse.ArgumentParser(description="Ingest documents into Endee")
    parser.add_argument(
        "--path",
        type=Path,
        default=SAMPLE_DOCS_PATH,
        help="Path to text file with documents (blank-line separated)",
    )
    parser.add_argument("--index", type=str, default=INDEX_NAME, help="Endee index name")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in chars")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between chunks")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = run_ingest(
        documents_path=args.path,
        index_name=args.index,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Ingested {n} chunks into index '{args.index}'.")


if __name__ == "__main__":
    main()
