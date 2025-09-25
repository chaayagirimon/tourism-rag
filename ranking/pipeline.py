# ranking/pipeline.py
from typing import Dict, List, Tuple
import numpy as np

from nlp.cluster_aspects import ClusterAspects

_CLUSTERS = ClusterAspects()

def build_aspects_and_gates(query: str,
                            ids: List[str],
                            metas: List[Dict],
                            kg,
                            cfg: Dict):
    TEMP   = float(cfg.get("cluster_temp", 0.15))
    TYPE_GATE_SCALE = float(cfg.get("type_gate_scale", 0.0))
    REV_SAMPLE = int(cfg.get("cluster_reviews_per_place", 12))    # small, fast sample
    BASELINE = float(cfg.get("cluster_baseline", 0.65))           # center so avg gate ~ 0
    LAMBDA = float(cfg.get("cluster_weight", 1.0))  # scale aspects from alignment

    min_reviews_soft = int(cfg.get("min_reviews_soft", 2))
    min_reviews_hard = int(cfg.get("min_reviews_hard", 3))
    soft_penalty = float(cfg.get("soft_penalty", -0.10))
    hard_penalty = float(cfg.get("hard_penalty", -0.60))

    # 1) reviews
    revs = kg.reviews_for(ids, max_total_per_place=80) if hasattr(kg, "reviews_for") else {pid: [] for pid in ids}

    # 2) aspects via clusters
    raw_aspects = _CLUSTERS.aspects_for_places(ids, query, temp=TEMP, reviews_by_id=revs)
    aspects = {pid: float(LAMBDA) * float(v) for pid, v in raw_aspects.items()}

    meta_by_id = {m.get("id"): m for m in metas}

    stay_keywords = cfg.get("stay_keywords") or [
        "airbnb", "apartment", "flat", "loft", "room", "stay",
        "lodging", "hotel", "penthouse", "bnb", "studio"
    ]
    q_lower = (query or "").lower()
    is_stay_query = any(term in q_lower for term in stay_keywords)
    airbnb_penalty = float(cfg.get("airbnb_penalty", -2.5))

    alignments: Dict[str, float] = {}
    for pid in ids:
        n_local = len(revs.get(pid, []))
        sat = float(min(1.0, np.log1p(n_local) / np.log(61.0))) if n_local else 0.0
        if sat > 0.0:
            align = float(raw_aspects.get(pid, 0.0) / sat)
        else:
            align = 0.0
        alignments[pid] = max(0.0, min(1.0, align))

    baseline_eff = float(BASELINE)
    if alignments:
        vals = list(alignments.values())
        perc = float(np.percentile(vals, 70.0))
        baseline_eff = max(BASELINE, perc)

    # 3) unsupervised evidence gates (no keyword hacks)
    review_counts = kg.reviews_count(ids) if hasattr(kg, "reviews_count") else {pid: len(revs.get(pid, [])) for pid in ids}

    type_gate: Dict[str, float] = {}
    for pid in ids:
        g = 0.0
        n_local = len(revs.get(pid, []))
        if n_local < min_reviews_soft:
            g += soft_penalty
        total_reviews = review_counts.get(pid, n_local)
        if total_reviews < min_reviews_hard:
            g += hard_penalty

        align = alignments.get(pid, baseline_eff)
        g += TYPE_GATE_SCALE * (align - baseline_eff)

        meta = meta_by_id.get(pid) or {}
        if not is_stay_query and meta.get("kind") == "airbnb":
            g += airbnb_penalty

        type_gate[pid] = g

    dbg = {
        "cluster_temp": TEMP,
        "cluster_weight": LAMBDA,
        "min_reviews_soft": min_reviews_soft,
        "min_reviews_hard": min_reviews_hard,
        "review_counts": review_counts,
        "type_gate_scale": TYPE_GATE_SCALE,
        "cluster_baseline": BASELINE,
        "cluster_baseline_effective": baseline_eff,
        "alignments": alignments,
        "cluster_reviews_per_place": REV_SAMPLE,
        "is_stay_query": is_stay_query,
        "airbnb_penalty": airbnb_penalty,
    }
    return aspects, type_gate, revs, dbg
