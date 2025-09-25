# llm/central_model.py
# Shared SentenceTransformer loader to avoid repeated downloads & rate limits.

from __future__ import annotations
import os
from typing import Iterable, List, Optional

# Keep tokenizers/BLAS from oversubscribing (important for joblib/multiprocessing)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---- Optional programmatic login (guarded) ----
# We only login if there's NO HF_TOKEN env and NO cached token.
try:
    from huggingface_hub import login, HfFolder
    _env_tok = os.getenv("HF_TOKEN")
    _cached_tok = None
    try:
        _cached_tok = HfFolder.get_token()
    except Exception:
        _cached_tok = None

    if not _env_tok and not _cached_tok:
        # If you prefer storing a token in a file path, point HF_TOKEN_FILE to it.
        tok_file = os.getenv("HF_TOKEN_FILE")
        if tok_file and os.path.exists(tok_file):
            with open(tok_file, "r", encoding="utf-8") as f:
                tok = f.read().strip()
            if tok:
                # no new session, do not add to git credentials
                login(token=tok, add_to_git_credential=False, new_session=False)
except Exception:
    # If huggingface_hub isn't installed or login fails, we still proceed.
    pass

from sentence_transformers import SentenceTransformer

# -------- Config knobs (env-driven) --------
# Prefer a fully local model directory (download once via `hf snapshot download ...`)
_LOCAL_DIR = os.getenv("ST_LOCAL_MODEL_DIR", "./models/all-MiniLM-L6-v2")

# Hub model name (only used if local dir not present)
_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Where to cache hub downloads (if we must go online)
_CACHE_DIR = os.getenv("HF_HOME", "./.hf_cache")

# Optional smaller local fallback if main model can't load online
_FALLBACK_LOCAL = os.getenv("ST_FALLBACK_LOCAL_DIR", "./models/paraphrase-MiniLM-L3-v2")

# Global singleton
_model_singleton: Optional[SentenceTransformer] = None


def _load_sentence_transformer() -> SentenceTransformer:
    """
    Load a SentenceTransformer with preference for local files; fall back to Hub with cache.
    Respects HF_HUB_OFFLINE: if set and local files are missing, this will raise.
    """
    # 1) Use a fully local directory if present (works with HF_HUB_OFFLINE=1)
    if os.path.exists(_LOCAL_DIR):
        return SentenceTransformer(_LOCAL_DIR, local_files_only=True)

    # 2) Attempt hub with a pinned cache directory
    try:
        return SentenceTransformer(_MODEL_NAME, cache_folder=_CACHE_DIR)
    except Exception:
        # 3) Last resort: try a smaller local fallback (if pre-downloaded)
        if os.path.exists(_FALLBACK_LOCAL):
            return SentenceTransformer(_FALLBACK_LOCAL, local_files_only=True)
        raise  # bubble up the real error for visibility


def get_st_model() -> SentenceTransformer:
    """
    Returns a process-wide singleton SentenceTransformer instance.
    Call this everywhere instead of instantiating models directly.
    """
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = _load_sentence_transformer()
    return _model_singleton


def preload_model() -> None:
    """
    Eagerly load the model at process start (useful before spawning workers).
    """
    _ = get_st_model()


def release_model() -> None:
    """
    Drop the singleton reference (rarely needed; mostly for tests).
    """
    global _model_singleton
    _model_singleton = None


def embed_texts(texts: Iterable[str], normalize: bool = True, batch_size: int = 64) -> List[List[float]]:
    """
    Convenience wrapper to embed a list of texts.
    Returns a list of float vectors (convert to numpy as needed).
    """
    model = get_st_model()
    embs = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    # Return as plain Python lists to avoid serialization headaches
    return embs.tolist()
