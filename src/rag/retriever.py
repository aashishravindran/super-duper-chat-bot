"""
Hybrid retrieval: BM25 + ChromaDB vector search fused via Reciprocal Rank Fusion (RRF).
"""
from collections import defaultdict

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (rank + k)


def hybrid_retrieve(
    query: str,
    vectorstore: Chroma,
    bm25: BM25Okapi,
    texts: list[str],
    k: int = 10,
) -> list[dict]:
    """
    1. BM25 keyword search
    2. ChromaDB semantic search
    3. Merge with Reciprocal Rank Fusion → top-k chunks

    Interview explanation: RRF rewards chunks that appear high in *both* lists,
    combining lexical precision with semantic recall.
    """
    # --- BM25 ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

    # --- ChromaDB vector search ---
    chroma_results = vectorstore.similarity_search(query, k=k)
    chroma_texts = [doc.page_content for doc in chroma_results]

    # --- RRF fusion ---
    rrf: dict[str, float] = defaultdict(float)

    for rank, idx in enumerate(bm25_ranked):
        text = texts[idx]
        rrf[text] += _rrf_score(rank)

    for rank, text in enumerate(chroma_texts):
        rrf[text] += _rrf_score(rank)

    # Sort by fused score descending, return top-k as dicts
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"text": text, "rrf_score": score} for text, score in ranked]
