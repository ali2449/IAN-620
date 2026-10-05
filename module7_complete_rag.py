"""
IAN 620 - Module 7
Complete baseline RAG workflow using the course student-support dataset.

Install:
    pip install pandas numpy sentence-transformers scikit-learn openai python-dotenv

This file intentionally uses the same variable and function names used in the slides.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/module7_rag_support.csv")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    """Remove repeated whitespace while keeping the content intact."""
    return re.sub(r"\s+", " ", str(text)).strip()


def chunk_text(text: str, chunk_size: int = 55, overlap: int = 15) -> list[str]:
    """Simple word-based overlapping chunker used for teaching."""
    words = clean_text(text).split()
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
    """Create one row per chunk and preserve source metadata."""
    chunk_rows = []

    for _, row in df.iterrows():
        pieces = chunk_text(row["clean_text"])

        for i, chunk in enumerate(pieces, start=1):
            chunk_rows.append(
                {
                    "chunk_id": f'{row["doc_id"]}_chunk_{i}',
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "topic": row["topic"],
                    "permission": row["permission"],
                    "updated_at": row["updated_at"],
                    "source": row["source"],
                    "chunk_text": chunk,
                }
            )

    return pd.DataFrame(chunk_rows)


def dense_retrieve(
    question: str,
    chunks_df: pd.DataFrame,
    chunk_vectors: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> pd.DataFrame:
    """Retrieve top chunks with normalized dense embeddings."""
    q_vec = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores = chunk_vectors @ q_vec[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = chunks_df.iloc[top_idx].copy()
    results["score"] = scores[top_idx]

    return results[
        [
            "chunk_id",
            "doc_id",
            "title",
            "topic",
            "permission",
            "updated_at",
            "source",
            "score",
            "chunk_text",
        ]
    ]


def build_prompt(question: str, retrieved: pd.DataFrame) -> str:
    """Build a grounded, source-aware prompt."""
    context_blocks = []

    for i, row in retrieved.reset_index(drop=True).iterrows():
        context_blocks.append(
            f"[{i + 1}] {row['title']} | source: {row['source']}\n"
            f"{row['chunk_text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""You are a student-support assistant.

Use only the retrieved context below.
If the context does not contain enough information, say:
"I do not have enough information in the retrieved sources."

Cite the source number(s) you used, such as [1] or [2].

CONTEXT
{context}

QUESTION
{question}

ANSWER
""".strip()


def generate_answer(prompt: str) -> str:
    """
    Optional LLM step using the OpenAI Python client.

    Environment variables:
      OPENAI_API_KEY
      RAG_MODEL           default: gpt-4.1-mini
      OPENAI_BASE_URL     optional for an OpenAI-compatible endpoint
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "No OPENAI_API_KEY was found. Retrieval and prompt construction "
            "worked, but the LLM call was skipped."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("RAG_MODEL", "gpt-4.1-mini")

    client = (
        OpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else OpenAI(api_key=api_key)
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content


def answer_question(
    question: str,
    chunks_df: pd.DataFrame,
    chunk_vectors: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> dict:
    """Run the complete baseline RAG workflow."""
    retrieved = dense_retrieve(
        question,
        chunks_df,
        chunk_vectors,
        model,
        top_k=top_k,
    )

    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved[
            ["chunk_id", "title", "source", "score"]
        ].to_dict("records"),
        "retrieved_context": retrieved,
        "prompt": prompt,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(clean_text)

    chunks_df = build_chunk_table(df)

    print("\nDOCUMENTS")
    print(df[["doc_id", "title", "topic", "permission"]])

    print("\nCHUNKS")
    print(
        chunks_df[
            ["chunk_id", "doc_id", "title", "topic", "chunk_text"]
        ].head()
    )

    model = SentenceTransformer(EMBED_MODEL)

    chunk_vectors = model.encode(
        chunks_df["chunk_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    question = "What should I do before registering for an internship?"

    result = answer_question(
        question,
        chunks_df,
        chunk_vectors,
        model,
        top_k=3,
    )

    print("\nRETRIEVED CHUNKS")
    print(
        result["retrieved_context"][
            ["score", "doc_id", "title", "source", "chunk_text"]
        ].to_string(index=False)
    )

    print("\nPROMPT")
    print(result["prompt"])

    print("\nANSWER")
    print(result["answer"])


if __name__ == "__main__":
    main()
