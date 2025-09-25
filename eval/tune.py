"""Fast-ish tuner for SAR + cluster knobs.

We keep the dataset small by precomputing dense/BM25 retrieval scores and
caching KG signals/reviews once. Each trial reuses those caches so we can
sample many configs without hammering the index/KG.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from retrieval.search import Retriever
from kg.neo4j_client import KG
from nlp.cluster_align import query_cluster_dist, place_cluster_dist, cluster_alignment
from rerank.sar import apply_sar
from rerank.cross import score_candidates

# --------- cached per-query bundle ---------


@dataclass
class Candidate:
    rowid: int
    place_id: str
    meta: Dict
    dense: float
    bm25: float


@dataclass
class QueryBundle:
    gold: Dict
    query: str
    city: str | None
    candidates: List[Candidate]
    signals: Dict[str, Dict]
    reviews: Dict[str, List[str]]
    review_counts: Dict[str, int]


def _normalize_dense(dense: Sequence[Tuple[int, float, Dict]]) -> Dict[int, float]:
    if not dense:
        return {}
    vals = [float(score) for _, score, _ in dense]
    dmax, dmin = max(vals), min(vals)
    if dmax == dmin:
        return {int(rid): 0.0 for rid, _, _ in dense}
    scale = dmax - dmin
    return {int(rid): (float(score) - dmin) / scale for rid, score, _ in dense}


def prepare_dataset(golds: List[Dict], *, base_alpha: float, max_topk: int, pool_multiple: int = 4) -> List[QueryBundle]:
    """Collect retrieval + KG artefacts once per query."""
    ret, kg = Retriever(), KG()
    bundles: List[QueryBundle] = []

    dense_k = max(max_topk * 2, 80)
    fts_k = max(max_topk * 3, 120)

    for g in golds:
        query, city = g["q"], g.get("city")

        dense = ret.search(query, k=dense_k, city=None)
        dense_norm = _normalize_dense(dense)

        fts = ret._fts_search(query, limit=fts_k)

        merged: Dict[int, Candidate] = {}
        for rid, score, meta in dense:
            merged[int(rid)] = Candidate(
                rowid=int(rid),
                place_id=meta["id"],
                meta=dict(meta),
                dense=dense_norm.get(int(rid), 0.0),
                bm25=0.0,
            )

        for rid, bm in fts:
            irid = int(rid)
            if irid not in merged:
                meta_rows = ret._fetch_meta([irid])
                if not meta_rows:
                    continue
                meta = dict(meta_rows[0])
                merged[irid] = Candidate(
                    rowid=irid,
                    place_id=meta["id"],
                    meta=meta,
                    dense=0.0,
                    bm25=0.0,
                )
            merged[irid].bm25 = math.exp(-0.1 * max(0.0, float(bm)))

        # coarse blend to decide which candidates we cache downstream
        scored = []
        for cand in merged.values():
            score = base_alpha * cand.dense + (1.0 - base_alpha) * cand.bm25
            scored.append((score, cand))
        scored.sort(reverse=True, key=lambda x: x[0])

        pool = [cand for _, cand in scored[: max_topk * pool_multiple]]
        ids = [cand.place_id for cand in pool]

        signals = kg.popularity(ids) if ids else {}
        raw_reviews = kg.reviews_for(ids, max_total_per_place=80) if ids else {}
        review_counts = kg.reviews_count(ids) if ids else {}
        reviews = {
            pid: [r["text"].strip() for r in (rev_list or []) if (r.get("text") or "").strip()]
            for pid, rev_list in raw_reviews.items()
        }

        bundles.append(
            QueryBundle(
                gold=g,
                query=query,
                city=city,
                candidates=pool,
                signals=signals,
                reviews=reviews,
                review_counts=review_counts,
            )
        )

    ret.close()
    kg.close()
    return bundles


# --------- scoring with cached bundle ---------


def _select_candidates(bundle: QueryBundle, cfg: Dict) -> List[Tuple[float, Candidate]]:
    alpha = float(cfg.get("alpha", 0.7))
    topk = int(cfg.get("retrieval_topk", 60))

    scored = []
    for cand in bundle.candidates:
        blended = alpha * cand.dense + (1.0 - alpha) * cand.bm25
        scored.append((blended, cand))
    scored.sort(reverse=True, key=lambda x: x[0])

    if bundle.city:
        want = (bundle.city or "").lower()
        filtered = [item for item in scored if (item[1].meta.get("city") or "").lower() == want]
        if not filtered:
            filtered = scored[: max(topk * 2, 50)]
    else:
        filtered = scored

    return filtered[:topk]


def _saturation(n_reviews: int) -> float:
    if n_reviews <= 0:
        return 0.0
    return min(1.0, math.log1p(n_reviews) / math.log(61.0))


def _aspects_and_gates(bundle: QueryBundle, ids: List[str], cfg: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    temp = float(cfg.get("cluster_temp", 0.18))
    baseline = float(cfg.get("cluster_baseline", 0.65))
    weight = float(cfg.get("cluster_weight", 0.75))
    min_soft = int(cfg.get("min_reviews_soft", 10))
    min_hard = int(cfg.get("min_reviews_hard", 5))
    soft_penalty = float(cfg.get("soft_penalty", -0.25))
    hard_penalty = float(cfg.get("hard_penalty", -0.7))
    gate_scale = float(cfg.get("type_gate_scale", 1.0))

    qdist = query_cluster_dist(bundle.query, temp=temp)

    aspects: Dict[str, float] = {}
    gates: Dict[str, float] = {}
    alignments: Dict[str, float] = {}
    for pid in ids:
        texts = bundle.reviews.get(pid, [])
        pdist = place_cluster_dist(texts, temp=temp)
        align = cluster_alignment(qdist, pdist)
        alignments[pid] = align

        sat = _saturation(len(texts))
        aspect_raw = max(0.0, min(1.0, align * sat))
        aspects[pid] = weight * aspect_raw

    if alignments:
        vals = list(alignments.values())
        baseline_eff = max(baseline, float(np.percentile(vals, 70.0)))
    else:
        baseline_eff = baseline

    for pid in ids:
        texts = bundle.reviews.get(pid, [])
        gate = 0.0
        if len(texts) < min_soft:
            gate += soft_penalty
        total_reviews = bundle.review_counts.get(pid, len(texts))
        if total_reviews < min_hard:
            gate += hard_penalty
        align = alignments.get(pid, baseline_eff)
        gate += gate_scale * (align - baseline_eff)
        gates[pid] = gate

    return aspects, gates


def _hit_at_5(after: List[Dict], gold: Dict) -> int:
    if "must" in gold:
        needle = (gold["must"].get("name_contains") or "").lower()
        for meta in after[:5]:
            name = (meta.get("name") or "").lower()
            if needle in name:
                return 1
        return 0
    if "avoid" in gold:
        needle = (gold["avoid"].get("name_contains") or "").lower()
        for meta in after[:5]:
            name = (meta.get("name") or "").lower()
            if needle in name:
                return 0
        return 1
    return 0


def eval_with_cfg(dataset: List[QueryBundle], cfg: Dict) -> float:
    hits = []
    for bundle in dataset:
        scored = _select_candidates(bundle, cfg)
        if not scored:
            continue

        triples = []
        ids = []
        for score, cand in scored:
            meta = dict(cand.meta)
            triples.append((float(score), cand.rowid, meta))
            ids.append(cand.place_id)

        aspects, gates = _aspects_and_gates(bundle, ids, cfg)

        bonus_scale = float(cfg.get("type_bonus_scale", 1.0))
        type_bonus: Dict[str, float] = {}
        if cfg.get("use_crossencoder", False):
            cross_topn = min(int(cfg.get("cross_topn", 15)), len(scored))
            subset = [cand.meta for _, cand in scored[:cross_topn]]
            raw_scores = score_candidates(bundle.query, subset)
            if raw_scores:
                vals = list(raw_scores.values())
                hi, lo = max(vals), min(vals)
                denom = hi - lo
                if denom > 1e-9:
                    type_bonus = {pid: ((score - lo) / denom - 0.5) * 2.0 * bonus_scale for pid, score in raw_scores.items()}
                else:
                    type_bonus = {pid: 0.0 for pid in raw_scores}

        ranked = apply_sar(
            triples,
            bundle.signals,
            aspects=aspects,
            type_gate=gates,
            type_bonus=type_bonus,
            k=5,
            beta=float(cfg["beta"]),
            eta=float(cfg["eta"]),
            tau=float(cfg["tau"]),
            delta=float(cfg["delta"]),
            kappa=float(cfg.get("kappa", 0.0)),
        )
        hits.append(_hit_at_5(ranked, bundle.gold))

    return sum(hits) / len(hits) if hits else 0.0


# --------- public tuner ---------

SEARCH_SPACE = {
    "beta": [0.05, 0.07, 0.09],
    "eta": [0.05, 0.08, 0.11],
    "tau": [0.15, 0.20, 0.25],
    "kappa": [0.0, 0.10, 0.20, 0.30],
    "hard_penalty": [-0.5, -0.7, -0.9, -1.1],
    "soft_penalty": [-0.15, -0.25, -0.35, -0.45],
    "min_reviews_hard": [3, 5, 8],
    "min_reviews_soft": [6, 8, 12],
    "type_gate_scale": [1.0, 1.5, 2.0],
    "type_bonus_scale": [0.4, 0.6, 0.8],
    "retrieval_topk": [50, 60, 70],
    "alpha": [0.65, 0.70, 0.75],
    "cluster_weight": [0.6, 0.75, 0.9],
    "cluster_temp": [0.16, 0.18, 0.20],
    "delta": [0.40, 0.45, 0.50],
}


def _sample_cfg(base: Dict, rng: random.Random) -> Dict:
    cfg = base.copy()
    for key, values in SEARCH_SPACE.items():
        cfg[key] = rng.choice(values)
    return cfg


def tune_fast(golds: List[Dict], base_cfg: Dict, *, n_trials: int = 60, seed: int = 0) -> Dict:
    rng = random.Random(seed)

    max_topk = max(SEARCH_SPACE["retrieval_topk"])
    base_alpha = float(base_cfg.get("alpha", 0.7))
    dataset = prepare_dataset(golds, base_alpha=base_alpha, max_topk=max_topk)
    print("dataset prepared")
    best_score = -1.0
    best_cfg = base_cfg.copy()
    trace: List[str] = []

    # Always evaluate the starting config first
    base_score = eval_with_cfg(dataset, base_cfg)
    best_score = base_score
    trace.append(f"base -> {base_score:.3f}")
    print(f"[tune] base config score={base_score:.3f}")
    
    for trial in range(n_trials):
        cfg = _sample_cfg(base_cfg, rng)
        score = eval_with_cfg(dataset, cfg)
        trace.append(f"trial {trial}: score={score:.3f}")
        print(f"[tune] trial {trial+1}/{n_trials} score={score:.3f}")
        if score > best_score:
            best_score = score
            best_cfg = cfg

    return {"best_score": best_score, "best_cfg": best_cfg, "trace": trace}
