# server/app.py
# Run: uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

import os
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body

from pydantic import BaseModel, Field

# CPU/thread stability
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Core project deps
from retrieval.search import Retriever
from kg.neo4j_client import KG
from rerank.sar import apply_sar

# Use your shared pipeline helper (tries both module layouts)
from ranking.pipeline import build_aspects_and_gates
from rerank.cross import score_candidates


# -------------------- Config --------------------

_DEFAULT_CFG: Dict[str, Any] = {
    "beta": 0.08,
    "eta": 0.08,
    "tau": 0.20,
    "kappa": 0.25,
    "delta": 0.45,

    "hard_penalty": -0.70,
    "soft_penalty": -0.25,
    "min_reviews_hard": 5,
    "min_reviews_soft": 10,

    "type_gate_scale": 1.5,
    "type_bonus_scale": 0.6,

    "retrieval_topk": 60,
    "alpha": 0.70,
    "use_crossencoder": True,
    "cross_topn": 20,

    "cluster_weight": 0.75,
    "cluster_temp": 0.18,
    "cluster_baseline": 0.65,
}

def load_config(path: str = "./eval/best_rerank_config.json") -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                user = json.load(f)
            cfg = _DEFAULT_CFG.copy()
            cfg.update(user or {})
            return cfg
        except Exception:
            pass
    return _DEFAULT_CFG.copy()

CFG = load_config()

# -------------------- FastAPI --------------------

app = FastAPI(title="Tourism RAG Recommender", version="1.0.0")

# Singletons
_RETRIEVER = Retriever()
_KG = KG()

# -------------------- Schemas --------------------

class RecommendRequest(BaseModel):
    query: str = Field(..., description="e.g., 'food hall in amsterdam'")
    city: Optional[str] = Field(None, description="e.g., 'amsterdam'")
    k: int = Field(5, ge=1, le=50, description="top-K results")
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="hybrid blend weight if supported")
    diagnostics: bool = Field(True, description="include diagnostics per item")

class RecommendItem(BaseModel):
    id: str
    name: Optional[str] = None
    city: Optional[str] = None
    score: float
    diagnostics: Optional[Dict[str, Any]] = None

class RecommendResponse(BaseModel):
    results: List[RecommendItem]
    config_used: Dict[str, Any]
    intent: str

# -------------------- Helpers --------------------

def _hybrid_or_plain_search(ret: Retriever, query: str, k: int, city: Optional[str], alpha: float):
    if hasattr(ret, "hybrid_search"):
        return ret.hybrid_search(query, k=k, city=city, alpha=alpha)
    return ret.search(query, k=k)

# -------------------- Endpoints --------------------

@app.get("/healthz")
def healthz():
    return {"ok": True, "config": CFG}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest = Body(...)):
    # 1) Retrieve
    retr_k = max(req.k, int(CFG.get("retrieval_topk", req.k)))
    alpha = float(CFG.get("alpha", req.alpha))
    cands = _hybrid_or_plain_search(_RETRIEVER, req.query, k=retr_k, city=req.city, alpha=alpha)
    metas = [m for _,_,m in cands]
    ids   = [m["id"] for m in metas]

    # 2) KG signals
    sigs = _KG.popularity(ids)

    # 3) Aspects + gates (from your shared pipeline)
    aspects, type_gate, revs, intent = build_aspects_and_gates(req.query, ids, metas, _KG, CFG)

    type_bonus: Dict[str, float] = {}
    if CFG.get("use_crossencoder", False):
        cross_topn = min(int(CFG.get("cross_topn", 20)), len(cands))
        subset = [m for _, _, m in cands[:cross_topn]]
        raw_scores = score_candidates(req.query, subset)
        if raw_scores:
            vals = list(raw_scores.values())
            hi, lo = max(vals), min(vals)
            denom = hi - lo
            if denom > 1e-9:
                scale = float(CFG.get("type_bonus_scale", 1.0))
                type_bonus = {pid: ((score - lo) / denom - 0.5) * 2.0 * scale for pid, score in raw_scores.items()}
            else:
                type_bonus = {pid: 0.0 for pid in raw_scores}

    # 4) SAR rerank (no type bonus)
    top = apply_sar(
        cands, sigs,
        aspects=aspects,
        type_bonus=type_bonus,
        type_gate=type_gate,
        k=req.k,
        beta=float(CFG["beta"]), eta=float(CFG["eta"]), tau=float(CFG["tau"]),
        delta=float(CFG["delta"]), kappa=float(CFG.get("kappa", 0.0))
    )

    # 5) Shape response
    results: List[RecommendItem] = []
    for m in top:
        pid = m["id"]
        diag = None
        if req.diagnostics:
            diag = {
                "aspect": round(float(aspects.get(pid, 0.0)), 3),
                "gate": round(float(type_gate.get(pid, 0.0)), 3),
                "type_bonus": round(float(type_bonus.get(pid, 0.0)), 3),
                "pop_z": float((sigs.get(pid, {}) or {}).get("pop_z", 0.0)),
                "listings_nearby": int((sigs.get(pid, {}) or {}).get("ln", 0)),
                "flag": (sigs.get(pid, {}) or {}).get("flag"),
                "reviews_count": len(revs.get(pid, [])),
            }
        results.append(RecommendItem(
            id=pid,
            name=m.get("name"),
            city=m.get("city"),
            score=0.0,  # apply_sar returns ordering; expose a score later if desired
            diagnostics=diag
        ))

    return RecommendResponse(results=results, config_used=CFG, intent=intent)

# -------------------- Uvicorn entry --------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
