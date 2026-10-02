"""
IAN 620 - Module 7
Complete RAG workflow using one small student-support knowledge base.

Install:
    pip install pandas numpy sentence-transformers scikit-learn openai python-dotenv
Optional vector DB:
    pip install chromadb
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/module7_rag_support.csv")
EMBED_MODEL = "all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    """Remove repeated whitespace while preserving the words."""
    return re.sub(r"\s+", " ", str(text)).strip()


def chunk_text(text: str, chunk_size: int = 55, overlap: int = 15) -> list[str]:
    """Simple word-based chunker for teaching."""
    words = clean_text(text).split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per chunk and keep the source metadata."""
    rows = []

    for _, row in df.iterrows():
        pieces = chunk_text(row["text"])

        for chunk_number, chunk in enumerate(pieces, start=1):
            rows.append(
                {
                    "chunk_id": f'{row["doc_id"]}_c{chunk_number:02d}',
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "topic": row["topic"],
                    "updated_at": row["updated_at"],
                    "source": row["source"],
                    "text": chunk,
                }
            )

    return pd.DataFrame(rows)


def create_embeddings(
    chunks: pd.DataFrame,
    model: SentenceTransformer,
) -> np.ndarray:
    """Create one embedding vector for each chunk."""
    return model.encode(
        chunks["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def retrieve(
    question: str,
    chunks: pd.DataFrame,
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
    topic: str | None = None,
) -> pd.DataFrame:
    """Retrieve the most similar chunks, with an optional metadata filter."""
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

    results = chunks.copy()
    results["score"] = scores

    if topic:
        results = results[results["topic"] == topic]

    return (
        results.sort_values("score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def build_prompt(question: str, retrieved: pd.DataFrame) -> str:
    """Create a grounded prompt from retrieved chunks."""
    context_parts = []

    for i, row in retrieved.iterrows():
        context_parts.append(
            f"[Source {i + 1}: {row['title']} | {row['source']}]\n{row['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""You are a student-support assistant.

Use only the context below to answer the question.
If the context does not contain enough information, say:
"I do not have enough information in the provided sources."

Cite the source number(s) you used.

CONTEXT
{context}

QUESTION
{question}

ANSWER
"""


def generate_answer(prompt: str) -> str:
    """
    Optional LLM step using an OpenAI-compatible API.

    Environment variables:
      OPENAI_API_KEY
      RAG_MODEL           default: gpt-4.1-mini
      OPENAI_BASE_URL     optional for compatible local/hosted servers
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "No OPENAI_API_KEY was found. Retrieval and prompt construction worked, "
            "but the LLM call was skipped."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("RAG_MODEL", "gpt-4.1-mini")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content


def answer_question(
    question: str,
    chunks: pd.DataFrame,
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> dict:
    """Complete minimal RAG workflow."""
    retrieved = retrieve(
        question,
        chunks,
        chunk_embeddings,
        model,
        top_k=top_k,
    )

    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt)

    return {
        "question": question,
        "retrieved": retrieved,
        "prompt": prompt,
        "answer": answer,
    }


def main() -> None:
    docs = pd.read_csv(DATA_PATH)
    docs["text"] = docs["text"].apply(clean_text)

    chunks = build_chunk_table(docs)

    print("\nDOCUMENTS")
    print(docs[["doc_id", "title", "topic"]])

    print("\nCHUNKS")
    print(chunks[["chunk_id", "title", "topic", "text"]].head())

    model = SentenceTransformer(EMBED_MODEL)
    chunk_embeddings = create_embeddings(chunks, model)

    question = "What should I do before registering for an internship?"

    result = answer_question(
        question,
        chunks,
        chunk_embeddings,
        model,
        top_k=3,
    )

    print("\nRETRIEVED CHUNKS")
    print(
        result["retrieved"][
            ["score", "chunk_id", "title", "source", "text"]
        ].to_string(index=False)
    )

    print("\nPROMPT")
    print(result["prompt"])

    print("\nANSWER")
    print(result["answer"])


if __name__ == "__main__":
    main()
