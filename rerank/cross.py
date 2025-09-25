"""Cross-encoder reranking helpers with graceful offline fallback."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from sentence_transformers import CrossEncoder

from llm.central_model import get_st_model

_model = None
_attempted = False

_LOCAL_DIR = os.getenv("CROSS_LOCAL_MODEL_DIR", "./models/bge-reranker-base")
_MODEL_NAME = os.getenv("CROSS_MODEL_NAME", "BAAI/bge-reranker-base")
_FALLBACK_LOCAL = os.getenv("CROSS_FALLBACK_LOCAL_DIR", "./models/bge-small")


def _load_crossencoder() -> CrossEncoder | None:
    global _model, _attempted
    if _attempted:
        return _model
    _attempted = True

    for path in (_LOCAL_DIR, _FALLBACK_LOCAL):
        if path and os.path.exists(path):
            try:
                _model = CrossEncoder(path, local_files_only=True)
                return _model
            except Exception:
                _model = None

    try:
        _model = CrossEncoder(_MODEL_NAME)
    except Exception:
        _model = None
    return _model


def get_model() -> CrossEncoder | None:
    return _load_crossencoder()


def _fallback_scores(query: str, metas) -> Dict[str, float]:
    """Cosine similarity fallback using the shared sentence encoder."""
    if not metas:
        return {}

    encoder = get_st_model()
    texts = [_pair_text(m) for m in metas]
    q_vec = encoder.encode([query], normalize_embeddings=True)[0]
    m_vecs = encoder.encode(texts, normalize_embeddings=True)
    sims = np.asarray(m_vecs, dtype=np.float32) @ np.asarray(q_vec, dtype=np.float32)
    return {m["id"]: float(score) for m, score in zip(metas, sims)}


def _pair_text(meta: dict) -> str:
    name = meta.get("name") or ""
    city = meta.get("city") or ""
    desc = meta.get("description") or ""
    pieces = [name]
    if city:
        pieces.append(f"({city})")
    if desc:
        pieces.append(desc)
    return " ".join(pieces)


def score_candidates(query: str, metas):
    """Return raw cross-encoder scores keyed by place id."""
    if not metas:
        return {}
    model = get_model()
    if model is None:
        return _fallback_scores(query, metas)
    pairs = [(query, _pair_text(m)) for m in metas]
    try:
        scores = model.predict(pairs).tolist()
    except Exception:
        return _fallback_scores(query, metas)
    return {m["id"]: float(score) for m, score in zip(metas, scores)}


def rerank(query, metas, topn: int = 15):
    scores = score_candidates(query, metas)
    ranked = sorted(((scores.get(m["id"], 0.0), m) for m in metas), reverse=True)[:topn]
    return [m for _, m in ranked]
