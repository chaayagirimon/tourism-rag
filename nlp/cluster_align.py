# nlp/cluster_align.py
import os
from functools import lru_cache
from typing import Dict, List
import numpy as np

from llm.central_model import get_st_model

ART_DIR = "artifacts/clusters"
CENT_PATH = os.path.join(ART_DIR, "centroids.npy")

# ---------- utilities ----------
def _l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype="float32")
    if X.ndim == 1: X = X[None, :]
    X[~np.isfinite(X)] = 0.0                      # guard
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / n).astype("float32")


def _safe_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matmul guardrail to dodge sporadic BLAS warnings on macOS."""
    A = np.asarray(A, dtype="float32", order="C")
    B = np.asarray(B, dtype="float32", order="C")
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    if A.ndim == 1:
        A = A[None, :]
    if B.ndim == 1:
        B = B[None, :]
    # einsum avoids the BLAS kernel that occasionally emits divide/overflow warnings
    sims = np.einsum("ij,kj->ik", A, B, optimize=True)
    return np.clip(sims, -1.0, 1.0).astype("float32", copy=False)

def _softmax(x: np.ndarray, temp: float = 0.15) -> np.ndarray:
    x = x / max(1e-12, float(temp))
    x = x - x.max(axis=-1, keepdims=True)
    x = np.exp(x)                                # single exp
    s = x.sum(axis=-1, keepdims=True) + 1e-12
    return x / s


# def _cos(a: np.ndarray, b: np.ndarray) -> float:
#     a = a.ravel(); b = b.ravel()
#     na = np.linalg.norm(a) + 1e-12
#     nb = np.linalg.norm(b) + 1e-12
#     return float((a @ b) / (na * nb))

# ---------- model & centroids ----------
_ENCODER = None
def _encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = get_st_model()
    return _ENCODER

@lru_cache(maxsize=1)
def _centroids() -> np.ndarray:
    if not os.path.exists(CENT_PATH):
        raise RuntimeError(f"Missing centroids at {CENT_PATH}. Run: python -m nlp.build_cluster_artifacts")
    C = np.load(CENT_PATH).astype("float32")
    C[~np.isfinite(C)] = 0.0                      # guard
    return _l2_normalize(C)

def _embed_texts(texts: List[str]) -> np.ndarray:
    X = _encoder().encode(list(texts), normalize_embeddings=True).astype("float32")
    X[~np.isfinite(X)] = 0.0                      # guard
    return _l2_normalize(X)


# ---------- public API ----------
def query_cluster_dist(query: str, temp: float = 0.07) -> np.ndarray:
    C = _centroids()
    qv = _embed_texts([query])       # (1,d)
    sims = _safe_dot(qv, C)
    return _softmax(sims, temp=temp).ravel()

def place_cluster_dist(reviews: List[str], temp: float = 0.07) -> np.ndarray:
    C = _centroids()
    if not reviews:
        return np.full((C.shape[0],), 1.0 / C.shape[0], dtype="float32")
    X = _embed_texts(reviews)        # (n,d)
    sims = _safe_dot(X, C)           # (n,C)
    P = _softmax(sims, temp=temp)    # sharpened per-review
    return P.mean(axis=0).astype("float32")

def cluster_alignment(qdist: np.ndarray, pdist: np.ndarray) -> float:
    q = np.sqrt(np.clip(qdist, 1e-12, 1))
    p = np.sqrt(np.clip(pdist, 1e-12, 1))
    return float(np.dot(q, p) / (np.linalg.norm(q) * np.linalg.norm(p) + 1e-12))
