# nlp/cluster_aspects.py
from typing import Dict, List
import numpy as np
from .cluster_align import query_cluster_dist, place_cluster_dist, cluster_alignment

def _safe_len(x) -> int:
    try:
        return len(x or [])
    except Exception:
        return 0

class ClusterAspects:
    """
    Unsupervised aspect scorer:
      aspect(pid) = alignment( query_cluster_dist, place_cluster_dist(pid) ) * saturation(n_reviews)
    """
    def aspects_for_places(self,
                           place_ids: List[str],
                           query: str,
                           temp: float = 0.15,
                           reviews_by_id: Dict[str, List[str]] | None = None) -> Dict[str, float]:
        qdist = query_cluster_dist(query, temp=temp)           # (C,)
        out: Dict[str, float] = {}
        for pid in place_ids:
            revs = (reviews_by_id or {}).get(pid) or []
            pdist = place_cluster_dist(revs, temp=temp)        # (C,)
            align = cluster_alignment(qdist, pdist)            # [0,1]
            # evidence saturation (log curve; 60 reviews ~ ~1.0)
            n = _safe_len(revs)
            sat = float(min(1.0, np.log1p(n) / np.log(61.0)))
            score = float(align * sat)
            # final safety clamp
            if not np.isfinite(score):
                score = 0.0
            out[pid] = max(0.0, min(1.0, score))
        return out
