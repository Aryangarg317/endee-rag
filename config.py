"""Configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Endee vector database (required for ingest and query)
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")

# Optional: OpenAI API for RAG answer generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Index name used in Endee
INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "rag_documents")

# Embedding model: sentence-transformers (384 dimensions)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_PATH = DATA_DIR / "sample_documents.txt"
