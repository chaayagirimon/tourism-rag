# retrieval/search.py
import os
import re
import sqlite3
import numpy as np
import faiss
from dotenv import load_dotenv; load_dotenv()

from llm.central_model import get_st_model
# --- helpers (put once in search.py or a shared util) ---
def _norm_city(s):
    return (s or "").strip().lower()

def filter_by_city(cands, city):
    if not city:
        return cands
    want = _norm_city(city)
    out = []
    for score, rid, meta in cands:
        if _norm_city(meta.get("city")) == want:
            out.append((score, rid, meta))
    return out

# Optional: very light geo backoff if strict city filter yields nothing.
def _geo_backoff(cands, query_city_latlon=None, max_results=50):
    # no-op placeholder unless you have lat/lon; otherwise just return a few.
    return cands[:max_results]

class Retriever:
    def __init__(self, store_dir="./retrieval/store"):
        self.model = get_st_model()
        self.index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
        self.db = sqlite3.connect(os.path.join(store_dir, "meta.db"))
        self.db.row_factory = sqlite3.Row

        # --- Consistency check: FAISS index size vs SQLite 'cards' rows ---
        cur = self.db.execute("SELECT COUNT(*) AS n FROM cards")
        n_meta = cur.fetchone()["n"]
        if self.index.ntotal != n_meta:
            raise SystemExit(
                f"[retriever] index.ntotal={self.index.ntotal} != cards={n_meta}. "
                "Rebuild with: python -m retrieval.build_index"
            )

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass

    def _fetch_meta(self, rowids):
        if not rowids:
            return []
        qmarks = ",".join("?" * len(rowids))
        cur = self.db.execute(
            f"SELECT rowid,id,name,city,city_slug,pop_z,flag,kind,room_type,accommodates,bedrooms,bathrooms,lodging_price,review_count "
            f"FROM cards WHERE rowid IN ({qmarks})",
            rowids
        )
        rows = {int(r["rowid"]): dict(r) for r in cur.fetchall()}
        # preserve incoming order
        return [rows[i] for i in rowids if i in rows]

    def search(self, query, k=30, city=None):
        # sentence-transformers can normalize; we add a safety normalizer too
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv, k)
        rowids = [int(i) for i in I[0]]
        metas  = self._fetch_meta(rowids)

        triples = list(zip(rowids, D[0].tolist(), metas))
        if city:
            city_l = city.lower()
            filt = [(rid, d, m) for rid, d, m in triples
                    if (m.get("city") or "").lower() == city_l]
            return filt or triples
        return triples

    def _fts_search(self, query, limit=50):
        """
        Return list of (rowid, score) from FTS5:
          - If bm25() exists: lower is better → score is raw bm25
          - If not: fall back to MATCH order with score=0.0
        """
        # Prefer bm25(cards_fts) when available
        clean = _normalize_for_fts(query)
        if not clean:
            return []
        try:
            cur = self.db.execute(
                "SELECT rowid, bm25(cards_fts) AS score "
                "FROM cards_fts WHERE cards_fts MATCH ? "
                "ORDER BY score LIMIT ?",
                (clean, limit)
            )
            rows = cur.fetchall()
            return [(int(r["rowid"]), float(r["score"])) for r in rows]
        except sqlite3.OperationalError:
            pass

        # Fallback: MATCH without ranking
        cur = self.db.execute(
            "SELECT rowid FROM cards_fts WHERE cards_fts MATCH ? LIMIT ?",
            (clean, limit)
        )
        return [(int(r["rowid"]), 0.0) for r in cur.fetchall()]

    def hybrid_search(self, query, k=30, city=None, alpha=0.65):
        """
        Blend dense cosine (normalized to [0,1]) with inverted BM25 (also [0,1]):
            blended = alpha * dense + (1 - alpha) * bm25_inv
        Then HARD-filter by city and return top-k.
        """
        # Get generous pools first (no pre-filter by city)
        dense = self.search(query, k=max(k * 2, 50), city=None)   # -> [(rowid, dense_score, meta), ...]
        fts   = self._fts_search(query, limit=max(k * 3, 100))    # -> [(rowid, bm25_raw), ...]

        # Normalize dense to [0,1]
        if dense:
            dvals = [float(d) for _, d, _ in dense]
            dmax, dmin = max(dvals), min(dvals)
            def nz_dense(x): 
                return 0.0 if dmax == dmin else (float(x) - dmin) / (dmax - dmin)
        else:
            def nz_dense(_): 
                return 0.0

        # Merge keyed by rowid
        import math
        merged = {}  # rid -> {"meta":..., "d":..., "bm25":...}
        for rid, d, meta in dense:
            merged[rid] = {"meta": meta, "d": nz_dense(d), "bm25": 0.0}

        # Invert BM25 smoothly to [0,1] (lower raw score is better)
        # You can tune the 0.1 slope; smaller = flatter.
        for rid, s in fts:
            if rid not in merged:
                fm = self._fetch_meta([rid])
                if not fm:
                    continue
                merged[rid] = {"meta": fm[0], "d": 0.0, "bm25": 0.0}
            merged[rid]["bm25"] = math.exp(-0.1 * max(0.0, float(s)))

        # Compute blended scores BEFORE filtering so we can log/debug deltas
        blended = []
        for rid, obj in merged.items():
            score = alpha * obj["d"] + (1.0 - alpha) * obj["bm25"]
            blended.append((score, rid, obj["meta"]))

        # Strict city filter (hard)
        filtered = filter_by_city(blended, city)

        # Optional backoff if strict filter yields nothing
        if not filtered:
            filtered = _geo_backoff(sorted(blended, reverse=True), None, max_results=max(k * 2, 50))

        filtered.sort(reverse=True)
        return filtered[:k]
_FTS_PUNCT = re.compile(r"[^0-9a-zA-Z]+")


def _normalize_for_fts(query: str) -> str:
    stripped = _FTS_PUNCT.sub(" ", query or "").strip()
    if not stripped:
        return ""
    return " ".join(stripped.split())
