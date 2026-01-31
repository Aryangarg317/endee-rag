"""Endee vector store: create index, upsert chunks, and query."""

from typing import Any, Dict, List, Optional

from endee import Endee


def get_client(token: str) -> Endee:
    """Create Endee client."""
    return Endee(token=token)


def ensure_index(
    client: Endee,
    name: str,
    dimension: int,
    space_type: str = "cosine",
    precision: str = "medium",
) -> None:
    """Create index if it does not exist (Endee may create on first upsert; this makes config explicit)."""
    try:
        client.create_index(
            name=name,
            dimension=dimension,
            space_type=space_type,
            precision=precision,
        )
    except Exception as e:
        # Index might already exist
        if "already exists" in str(e).lower() or "exist" in str(e).lower():
            return
        raise e


def upsert_vectors(
    client: Endee,
    index_name: str,
    ids: List[str],
    vectors: List[List[float]],
    meta: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Upsert document chunks into the Endee index."""
    index = client.get_index(name=index_name)
    records = []
    for i, (doc_id, vec) in enumerate(zip(ids, vectors)):
        record = {"id": doc_id, "vector": vec}
        if meta and i < len(meta):
            record["meta"] = meta[i]
        records.append(record)
    index.upsert(records)


def query_vectors(
    client: Endee,
    index_name: str,
    vector: List[float],
    top_k: int = 5,
    filter_conditions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run similarity search on Endee and return results with id, similarity, and meta."""
    index = client.get_index(name=index_name)
    results = index.query(
        vector=vector,
        top_k=top_k,
        filter=filter_conditions if filter_conditions else [],
    )
    return list(results)
