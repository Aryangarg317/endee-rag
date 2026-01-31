"""Query the RAG index: embed query, search Endee, optionally generate answer with LLM."""

from typing import List, Optional

from config import (
    EMBEDDING_MODEL,
    ENDEE_TOKEN,
    INDEX_NAME,
    OPENAI_API_KEY,
)
from embeddings import embed_single, get_embedding_model
from store import get_client, query_vectors


def retrieve(
    query: str,
    index_name: str = INDEX_NAME,
    top_k: int = 5,
    filter_conditions: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Embed the query and run similarity search on Endee.
    Returns list of hits with id, similarity, and meta (including 'text').
    """
    if not ENDEE_TOKEN:
        raise ValueError(
            "ENDEE_TOKEN is not set. Copy .env.example to .env and add your Endee token."
        )

    model = get_embedding_model(EMBEDDING_MODEL)
    vector = embed_single(model, query)
    client = get_client(ENDEE_TOKEN)
    results = query_vectors(
        client, index_name, vector, top_k=top_k, filter_conditions=filter_conditions
    )
    return results


def format_context(results: List[dict]) -> str:
    """Format retrieved chunks as context string for an LLM."""
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("meta") or {}
        text = meta.get("text", "")
        if text:
            parts.append(f"[{i}] {text}")
    return "\n\n".join(parts) if parts else "No relevant context found."


def generate_answer_openai(query: str, context: str) -> str:
    """Generate an answer using OpenAI API given query and retrieved context."""
    try:
        from openai import OpenAI
    except ImportError:
        return "Install openai: pip install openai. Set OPENAI_API_KEY for LLM answers."
    if not OPENAI_API_KEY:
        return "Set OPENAI_API_KEY in .env for LLM-generated answers."

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer the user's question using only the provided context. "
                "If the context does not contain enough information, say so. "
                "Keep answers concise and cite the context when relevant.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def answer(
    query: str,
    index_name: str = INDEX_NAME,
    top_k: int = 5,
    use_llm: bool = True,
) -> dict:
    """
    Run RAG: retrieve relevant chunks from Endee, then optionally generate answer with LLM.
    Returns dict with keys: query, results (raw hits), context (formatted), answer (if use_llm).
    """
    results = retrieve(query, index_name=index_name, top_k=top_k)
    context = format_context(results)
    out = {"query": query, "results": results, "context": context}

    if use_llm and OPENAI_API_KEY:
        out["answer"] = generate_answer_openai(query, context)
    else:
        out["answer"] = None  # Retrieval-only mode

    return out


def main() -> None:
    """CLI entry: run a single query or interactive loop."""
    import argparse
    parser = argparse.ArgumentParser(description="Query the RAG index (Endee)")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question to ask (optional; if omitted, enter interactive mode)",
    )
    parser.add_argument("--index", type=str, default=INDEX_NAME, help="Endee index name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only show retrieved context, do not call OpenAI",
    )
    args = parser.parse_args()

    if args.query:
        q = args.query
        result = answer(q, index_name=args.index, top_k=args.top_k, use_llm=not args.no_llm)
        print("\n--- Retrieved context ---")
        print(result["context"])
        if result.get("answer"):
            print("\n--- Answer ---")
            print(result["answer"])
        return

    # Interactive mode
    print("RAG query (Endee). Type your question and press Enter. Empty line to exit.")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        result = answer(q, index_name=args.index, top_k=args.top_k, use_llm=not args.no_llm)
        print("\n--- Retrieved context ---")
        print(result["context"])
        if result.get("answer"):
            print("\n--- Answer ---")
            print(result["answer"])


if __name__ == "__main__":
    main()
