"""Embedding model wrapper using sentence-transformers (local, no API key)."""

from typing import List

from sentence_transformers import SentenceTransformer


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load the sentence-transformers model (cached after first run)."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    """Encode a list of texts into vectors."""
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def embed_single(model: SentenceTransformer, text: str) -> List[float]:
    """Encode a single query text into a vector."""
    return model.encode([text], convert_to_numpy=True).tolist()[0]
