"""
Cross-encoder reranker: reads (query, chunk) pairs jointly to produce fine-grained scores.
Much more accurate than embedding similarity for final ranking.
"""
import asyncio
from functools import partial
from sentence_transformers import CrossEncoder

# Loaded once at module level — ~90MB, cached in /tmp/models on Lambda
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_MODEL_NAME)
    return _cross_encoder


def _sync_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    model = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [{"text": c["text"], "score": float(s)} for c, s in scored[:top_k]]


async def rerank(query: str, candidates: list[dict], top_k: int = 4) -> list[dict]:
    """Score each (query, chunk) pair with the cross-encoder, return top_k.
    Runs in a thread pool to avoid blocking the event loop on CPU-bound inference.
    """
    if not candidates:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_sync_rerank, query, candidates, top_k))
