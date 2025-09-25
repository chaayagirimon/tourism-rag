# nlp/build_cluster_artifacts.py
import os, json, math, numpy as np
from typing import Dict, List, Iterable, Tuple
from collections import Counter

from llm.central_model import get_st_model
from kg.neo4j_client import KG

# ---------- CONFIG ----------
REVIEW_SOURCES = ["yelp", "reddit", "wikivoyage", "airbnb", "airbnb_host", "airbnb_guest"]
MAX_REVIEWS_PER_PLACE = int(os.getenv("CLUSTER_MAX_REVIEWS", "60"))
MAX_SENTENCES_PER_PLACE = int(os.getenv("CLUSTER_MAX_SENTENCES", "60"))
BATCH_SIZE = int(os.getenv("CLUSTER_REVIEW_BATCH", "40"))
ARTIFACT_DIR = "artifacts/clusters"

USE_HDBSCAN = os.getenv("CLUSTER_USE_HDBSCAN", "1") not in ("0", "false", "False")
MIN_CLUSTER_SIZE = 60       # smaller → more clusters on modest data
N_CLUSTERS = 64             # KMeans fallback/forced target
MIN_GOOD_CLUSTERS = 24      # if < this, we declare HDBSCAN too coarse and redo with KMeans
# Optional sampling knobs
SAMPLE_FRACTION = float(os.getenv("CLUSTER_SAMPLE_FRACTION", "1.0"))
MAX_PLACES = int(os.getenv("CLUSTER_MAX_PLACES", "0"))
# ----------------------------

def _sanitize_embeddings(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Force finite dtype and clip outliers
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # L2 normalize safely row-wise
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    X = X / norms
    # Optional: clip to tame any encoder spikes
    np.clip(X, -6.0, 6.0, out=X)
    return X

def _drop_bad_rows(X: np.ndarray, ids: list, texts: list):
    # Remove rows that became all-zeros or non-finite after sanitize
    finite = np.isfinite(X).all(axis=1)
    nonzero = (np.linalg.norm(X, axis=1) > 0)
    keep = finite & nonzero
    return X[keep], [i for i,k in zip(ids, keep) if k], [t for t,k in zip(texts, keep) if k]


_encoder = None
def _embed(texts: List[str]) -> np.ndarray:
    global _encoder
    if _encoder is None:
        _encoder = get_st_model()
    X = _encoder.encode(list(texts), normalize_embeddings=True).astype("float32")
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / n).astype("float32")

def iter_all_reviews(batch_size: int = BATCH_SIZE) -> Iterable[Tuple[str, List[str]]]:
    kg = KG()
    try:
        q_ids = """
        MATCH (cp:CanonicalPlace)
        WHERE EXISTS {
            MATCH (cp)-[:HAS_REVIEW]->(r:Review)
            WHERE r.source IN $sources AND coalesce(r.text,'') <> ''
        }
        OR EXISTS {
            MATCH (cp)<-[:NEAR]-(lst:Listing)-[:HAS_REVIEW]->(r2:Review)
            WHERE r2.source IN $sources AND coalesce(r2.text,'') <> ''
        }
        RETURN cp.id AS id
        ORDER BY cp.id
        """

        with kg._driver.session() as s:
            chunk = []
            for record in s.run(q_ids, sources=REVIEW_SOURCES):
                chunk.append(record["id"])
                if len(chunk) >= batch_size:
                    rev_map = kg.reviews_for(chunk, max_total_per_place=MAX_REVIEWS_PER_PLACE)
                    for pid in chunk:
                        texts = [r["text"].strip() for r in (rev_map.get(pid) or []) if (r.get("text") or "").strip()]
                        if not texts:
                            continue
                        if len(texts) > MAX_SENTENCES_PER_PLACE:
                            texts = texts[:MAX_SENTENCES_PER_PLACE]
                        yield pid, texts
                    chunk = []

            if chunk:
                rev_map = kg.reviews_for(chunk, max_total_per_place=MAX_REVIEWS_PER_PLACE)
                for pid in chunk:
                    texts = [r["text"].strip() for r in (rev_map.get(pid) or []) if (r.get("text") or "").strip()]
                    if not texts:
                        continue
                    if len(texts) > MAX_SENTENCES_PER_PLACE:
                        texts = texts[:MAX_SENTENCES_PER_PLACE]
                    yield pid, texts
    finally:
        try: kg.close()
        except Exception: pass

def _ctfidf_top_terms(docs: List[List[str]], topk: int = 15) -> List[List[str]]:
    import re
    token = re.compile(r"[A-Za-zÀ-ÿ]+")
    df = Counter()
    c_counts = []
    for txts in docs:
        joined = " ".join(txts[:50])
        toks = [t.lower() for t in token.findall(joined)]
        df.update(set(toks))
        from collections import Counter as C
        c_counts.append(C(toks))
    N = len(docs) + 1e-9
    idf = {t: math.log(N / (1 + df[t])) for t in df}
    per = []
    for cnt in c_counts:
        scores = {t: cnt[t] * idf.get(t, 0.0) for t in cnt}
        per.append([w for w,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:topk]])
    return per

def main(out_dir: str = ARTIFACT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # 1) collect review sentences
    texts, pids = [], []
    iterator = iter_all_reviews()
    try:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Collecting reviews", unit="place")
    except ImportError:
        pass

    for pid, revs in iterator:
        take = revs[:MAX_SENTENCES_PER_PLACE]
        texts.extend(take)
        pids.extend([pid] * len(take))

    N = len(texts)
    if N == 0:
        raise RuntimeError("No reviews available for clustering. Verify KG connectivity and sources.")

    unique_places = len({pid for pid in pids})
    print(f"[cluster] Collected {N} review snippets across {unique_places} places")

    if SAMPLE_FRACTION < 1.0 or MAX_PLACES > 0:
        import random
        rng = random.Random(0)
        place_order = []
        seen = set()
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                place_order.append(pid)

        keep = place_order
        if SAMPLE_FRACTION < 1.0:
            keep_count = max(1, int(len(place_order) * SAMPLE_FRACTION))
            keep = rng.sample(place_order, keep_count)
        if MAX_PLACES > 0 and len(keep) > MAX_PLACES:
            keep = keep[:MAX_PLACES]

        keep_set = set(keep)
        filtered_texts, filtered_pids = [], []
        for pid, text in zip(pids, texts):
            if pid in keep_set:
                filtered_pids.append(pid)
                filtered_texts.append(text)

        if filtered_texts:
            texts, pids = filtered_texts, filtered_pids
            N = len(texts)
            print(f"[cluster] Downsampled to {N} snippets across {len(set(pids))} places")

    X = _embed(texts)  # (N, d)
    X = _sanitize_embeddings(X)
    X, pids, texts = _drop_bad_rows(X, pids, texts)

    # 2) optional reduction
    Xr = X
    try:
        if X.shape[0] >= 2000:
            from sklearn.decomposition import TruncatedSVD
            X64 = X.astype("float64", copy=False)
            svd = TruncatedSVD(
                n_components=min(64, X.shape[1]-1),
                algorithm="arpack",   # <- avoids randomized power-iterations
                tol=1e-7,
                random_state=0
            )
            Xr = svd.fit_transform(X64).astype("float32") # M4 numpy issue
            # Defensive: sanitize reduced space too
            Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        Xr = X

    # 3) cluster
    labels = None
    used = "kmeans"
    if USE_HDBSCAN:
        try:
            import hdbscan
            clus = hdbscan.HDBSCAN(min_cluster_size=min(MIN_CLUSTER_SIZE, max(20, N//250)),
                                   min_samples=20, metric='euclidean', cluster_selection_epsilon=0.05,).fit(Xr)
            # --- after clus = hdbscan.HDBSCAN(...).fit(Xr)
            labels = clus.labels_.copy()
            # Count number of clusters, excluding noise label -1
            num_clusters = len(set(labels.tolist()) - {-1})

            if num_clusters < MIN_GOOD_CLUSTERS:
                labels = None      # force KMeans fallback
                used = "kmeans"    # (will be set below when we actually run it)
            else:
                used = "hdbscan"
        except Exception:
            labels = None

    if labels is None:
        from sklearn.cluster import KMeans
        k = min(N_CLUSTERS, max(8, N // 150))
        clus = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xr)
        labels = clus.labels_
        used = "kmeans"

    # after computing `labels` (either HDBSCAN or KMeans)
    # if used == "hdbscan":
    #     num_clusters = len(set(labels.tolist()) - {-1})
    # else:
    #     num_clusters = int(labels.max() + 1)

    # 4) centroids in original space (exclude noise = -1)
    import numpy as np

    valid_mask = (labels >= 0)
    labels_v = labels[valid_mask]
    Xv = X[valid_mask]
    finite_mask = np.isfinite(Xv).all(axis=1)
    labels_v = labels_v[finite_mask]
    Xv = Xv[finite_mask]
    textsv = [t for t, ok in zip(texts, valid_mask) if ok]
    textsv = [t for t, ok in zip(textsv, finite_mask) if ok]

    # remap arbitrary cluster ids to 0..C-1
    uniq = sorted(set(labels_v.tolist()))
    lab2idx = {lab: i for i, lab in enumerate(uniq)}
    C = len(uniq)

    print(f"[cluster] Algorithm used: {used}, clusters: {C}")

    centroids = np.zeros((C, X.shape[1]), dtype="float32")
    counts = np.zeros((C,), dtype="int32")

    for i, lab in enumerate(labels_v):
        ci = lab2idx[int(lab)]
        centroids[ci] += Xv[i]
        counts[ci] += 1

    for ci in range(C):
        if counts[ci] > 0:
            centroids[ci] /= counts[ci]

    # Guard against pathological rows before persisting
    centroids = np.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0)

    # 5) debug top terms (again, exclude noise and use remapped ids)
    cluster_docs = [[] for _ in range(C)]
    for i, lab in enumerate(labels_v):
        ci = lab2idx[int(lab)]
        cluster_docs[ci].append(textsv[i])

    tops = _ctfidf_top_terms(cluster_docs, topk=15)

    # 6) save
    np.save(os.path.join(out_dir, "centroids.npy"), centroids.astype("float32"))
    with open(os.path.join(out_dir, "components.json"), "w") as f:
        json.dump({"top_terms": tops, "counts": counts.tolist(), "algo": used, "n_samples": N}, f)
    with open(os.path.join(out_dir, "version.json"), "w") as f:
        json.dump({"model": "same-as-retrieval-encoder", "dims": int(X.shape[1])}, f)

    # NEW: also write the legacy bundle the cluster readers expect
    os.makedirs("nlp", exist_ok=True)
    np.savez(
        "nlp/review_clusters.npz",
        centroids=centroids.astype("float32"),
        top_terms=np.array(tops, dtype=object),
        counts=counts.astype("int32")
    )

if __name__ == "__main__":
    main()
