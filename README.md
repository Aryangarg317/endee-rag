# Endee RAG: Retrieval-Augmented Generation with Endee Vector Database

A project-based evaluation submission for the **Endee Machine Learning Engineer Internship**. This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline using [Endee](https://endee.io) as the vector database, demonstrating semantic search, embeddings, and optional LLM-based answer generation.

---

## Quick Start

After cloning and installing dependencies (see [Setup & Execution](#setup--execution)):

1. Copy `.env.example` to `.env` and set your `ENDEE_TOKEN`.
2. Run the demo (ingests sample documents into Endee, then runs one retrieval query):

   ```bash
   python run_demo.py
   ```

   This validates your token, ingests `data/sample_documents.txt`, and prints the top retrieved chunks for the question *"What is Endee used for?"*. No OpenAI key is required for retrieval-only mode.

3. Ask your own questions (retrieval only): `python -m query "Your question here" --no-llm`  
   Or with LLM answers: set `OPENAI_API_KEY` in `.env`, then `python -m query "Your question here"`.

---

## Project Overview & Problem Statement

**Problem:** Building AI applications that answer questions over private or domain-specific documents requires (1) storing document chunks as vectors, (2) retrieving the most relevant chunks for a user query, and (3) optionally generating a natural-language answer grounded in that context. Doing this at scale demands a fast, reliable vector store and a clear pipeline from ingestion to query.

**Solution:** This project implements an end-to-end RAG pipeline:

- **Ingest:** Load documents from a text file, chunk them with configurable size and overlap, embed chunks using [sentence-transformers](https://www.sbert.net/) (local, no API key), and upsert vectors and metadata into **Endee**.
- **Query:** Embed the user question, run similarity search on Endee, and return the top-k relevant chunks. Optionally, pass the retrieved context to an LLM (e.g. OpenAI) to produce a concise answer.

The pipeline showcases **semantic search**, **vector retrieval**, and **RAG**—core use cases for Endee—while keeping setup minimal (embedding model runs locally; only Endee and optionally OpenAI require API keys).

---

## System Design & Technical Approach

### Architecture

```
┌─────────────────┐     chunk      ┌──────────────────┐     embed      ┌─────────────┐
│  Documents      │ ──────────────►│  Text Chunks     │ ──────────────►│  Vectors    │
│  (e.g. .txt)    │                │  (overlap)       │  sentence-     │  (384-dim)   │
└─────────────────┘                └──────────────────┘  transformers  └──────┬──────┘
                                                                              │
                                                                              ▼
┌─────────────────┐     query     ┌──────────────────┐     ANN        ┌─────────────┐
│  User Question   │ ──────────────►│  Query Vector     │ ──────────────►│  Endee      │
└─────────────────┘                └──────────────────┘  similarity    │  Index      │
                                          │                  search    └──────┬──────┘
                                          │                                   │
                                          ▼                                   ▼
                                   ┌──────────────────┐                ┌─────────────┐
                                   │  Top-k Chunks    │◄───────────────│  Results    │
                                   │  (context)       │                │  + meta     │
                                   └────────┬────────┘                └─────────────┘
                                            │
                                            ▼ (optional)
                                   ┌──────────────────┐
                                   │  LLM (OpenAI)    │ ──► Answer
                                   └──────────────────┘
```

### Components

| Component | Role |
|-----------|------|
| **Chunking** | Splits documents into overlapping segments (default 512 chars, 64 overlap) to improve retrieval granularity. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), runs locally; no external API. |
| **Endee** | Vector database: stores chunk vectors and metadata, performs cosine-similarity ANN search. |
| **Retrieval** | Query is embedded; Endee returns top-k chunks by similarity. |
| **RAG (optional)** | Retrieved chunks are formatted as context and sent to OpenAI (or compatible API) to generate an answer. |

### How Endee Is Used

- **Index:** A single Endee index (default name: `rag_documents`) is created with dimension `384` and `space_type="cosine"` to match the embedding model.
- **Upsert:** Each chunk is upserted with a unique id, its embedding vector, and metadata (e.g. `text`, `doc_index`) for display and filtering.
- **Query:** At query time, the question is embedded and passed to Endee’s `query()` with `top_k` (default 5). Results include id, similarity score, and metadata; the `text` field is used as context for the LLM or direct display.

Endee provides the scalable vector store and fast ANN search required for production-style RAG; the rest of the pipeline is standard Python (sentence-transformers, optional OpenAI).

---

## Setup & Execution

### Prerequisites

- Python 3.8+
- [Endee](https://endee.io) account (free tier available) to obtain an API token

### 1. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/endee-rag.git
cd endee-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set:

- `ENDEE_TOKEN` — your Endee API token (required for ingest and query)
- `OPENAI_API_KEY` — (optional) for LLM-generated answers; leave unset for retrieval-only mode

### 3. Ingest Documents

Sample documents are provided in `data/sample_documents.txt`. To ingest them into Endee:

```bash
python -m ingest
```

Or with custom path and index:

```bash
python -m ingest --path data/sample_documents.txt --index rag_documents --chunk-size 512 --overlap 64
```

This chunks the text, embeds with sentence-transformers, and upserts into the Endee index.

### 4. Query (Retrieval and Optional RAG)

**Single question (retrieval only):**

```bash
python -m query "What is Endee used for?" --no-llm
```

**Single question with LLM answer (requires `OPENAI_API_KEY`):**

```bash
python -m query "What is Endee used for?"
```

**Interactive mode:**

```bash
python -m query
```

Then type questions and press Enter; empty line to exit.

---

## Project Structure

```
endee-rag/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
├── .env.example             # Template for ENDEE_TOKEN and OPENAI_API_KEY
├── .gitignore
├── run_demo.py               # One-command demo: ingest + sample query
├── config.py                 # Environment and constants (index name, embedding dim)
├── embeddings.py             # sentence-transformers wrapper
├── store.py                  # Endee client, index creation, upsert, query
├── ingest.py                 # Document loading, chunking, embed, upsert (CLI)
├── query.py                  # Retrieve + optional LLM answer (CLI)
└── data/
    └── sample_documents.txt  # Sample corpus (vector DBs, Endee, RAG, embeddings)
```

---


---

## License

MIT.
