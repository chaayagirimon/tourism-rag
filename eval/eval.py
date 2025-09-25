from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as stats
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import multiprocessing as mp

try:  # load .env for API keys when available
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from retrieval.search import Retriever
from rerank.sar import apply_sar
from kg.neo4j_client import KG

from ranking.pipeline import build_aspects_and_gates
from rerank.cross import score_candidates
from llm.prompts import (
    PromptGenerationError,
    generate_benchmark_queries,
    list_cities,
)

# ======= Small gold set; extend this list! =======
GOLDS = [
  # AMSTERDAM
  {"q":"traditional dutch food near canal district","city":"amsterdam","must":{"name_contains":"moeders"}}, 
  {"q":"food hall in amsterdam","city":"amsterdam","must":{"name_contains":"de foodhallen"}}, # museum/attraction 
  {"q":"modern art museum in amsterdam","city":"amsterdam","must":{"name_contains":"stedelijk"}}, 
  {"q":"science museum for kids in amsterdam","city":"amsterdam","must":{"name_contains":"nemo"}}, 
  {"q":"history museum in amsterdam","city":"amsterdam","must":{"name_contains":"anne frank"}},

  # LONDON
  {"q":"traditional british food near covent garden","city":"london","must":{"name_contains":"rules"}},
  {"q":"food hall in london","city":"london","must":{"name_contains":"borough market"}},
  {"q":"modern art museum in london","city":"london","must":{"name_contains":"tate modern"}},
  {"q":"science museum for kids in london","city":"london","must":{"name_contains":"science museum"}},
  {"q":"history museum in london","city":"london","must":{"name_contains":"british museum"}},

  # PARIS
  {"q":"traditional french bistro near le marais","city":"paris","must":{"name_contains":"bistrot"}},
  {"q":"food hall in paris","city":"paris","must":{"name_contains":"marché des enfants rouges"}},
  {"q":"modern art museum in paris","city":"paris","must":{"name_contains":"centre pompidou"}},
  {"q":"science museum for kids in paris","city":"paris","must":{"name_contains":"cité des sciences"}},
  {"q":"history museum in paris","city":"paris","must":{"name_contains":"musée carnavalet"}},

  # BERLIN
  {"q":"traditional german food near kreuzberg","city":"berlin","must":{"name_contains":"max und moritz"}},
  {"q":"food hall in berlin","city":"berlin","must":{"name_contains":"markthalle neun"}},
  {"q":"modern art museum in berlin","city":"berlin","must":{"name_contains":"hamburger bahnhof"}},
  {"q":"science museum for kids in berlin","city":"berlin","must":{"name_contains":"deutsches technikmuseum"}},
  {"q":"history museum in berlin","city":"berlin","must":{"name_contains":"deutsches historisches museum"}},

  # BARCELONA
  {"q":"traditional catalan food near gothic quarter","city":"barcelona","must":{"name_contains":"can culleretes"}},
  {"q":"food hall in barcelona","city":"barcelona","must":{"name_contains":"la boqueria"}},
  {"q":"modern art museum in barcelona","city":"barcelona","must":{"name_contains":"macba"}},
  {"q":"science museum for kids in barcelona","city":"barcelona","must":{"name_contains":"cosmocaixa"}},
  {"q":"history museum in barcelona","city":"barcelona","must":{"name_contains":"muhba"}},

  # VIENNA
  {"q":"traditional austrian food near inner stadt","city":"vienna","must":{"name_contains":"figlmüller"}},
  {"q":"food hall in vienna","city":"vienna","must":{"name_contains":"naschmarkt"}},
  {"q":"modern art museum in vienna","city":"vienna","must":{"name_contains":"mumok"}},
  {"q":"science museum for kids in vienna","city":"vienna","must":{"name_contains":"technisches museum"}},
  {"q":"history museum in vienna","city":"vienna","must":{"name_contains":"kunsthistorisches museum"}},

  # PRAGUE
  {"q":"traditional czech food near old town","city":"prague","must":{"name_contains":"lokál"}},
  {"q":"food hall in prague","city":"prague","must":{"name_contains":"manifesto market"}},
  {"q":"modern art museum in prague","city":"prague","must":{"name_contains":"dox"}},
  {"q":"science museum for kids in prague","city":"prague","must":{"name_contains":"national technical museum"}},
  {"q":"history museum in prague","city":"prague","must":{"name_contains":"national museum"}},

  # COPENHAGEN
  {"q":"traditional smørrebrød near nyhavn","city":"copenhagen","must":{"name_contains":"schønnemann"}},
  {"q":"food hall in copenhagen","city":"copenhagen","must":{"name_contains":"torvehallerne"}},
  {"q":"modern art museum near copenhagen","city":"copenhagen","must":{"name_contains":"louisiana museum"}},
  {"q":"science museum for kids in copenhagen","city":"copenhagen","must":{"name_contains":"experimentarium"}},
  {"q":"history museum in copenhagen","city":"copenhagen","must":{"name_contains":"nationalmuseet"}},

  # AIRBNB STAYS
  {"q":"airbnb with balcony in vienna","city":"vienna","must":{"name_contains":"balcony"}},
  {"q":"loft airbnb in amsterdam with bathtub","city":"amsterdam","must":{"name_contains":"bathtub"}},
  {"q":"canary wharf airbnb room","city":"london","must":{"name_contains":"canary wharf room"}},
  {"q":"studio airbnb near eiffel tower","city":"paris","must":{"name_contains":"eiffel"}},
  {"q":"penthouse airbnb in copenhagen","city":"copenhagen","must":{"name_contains":"penthouse"}},
  {"q":"kreuzberg design airbnb","city":"berlin","must":{"name_contains":"kreuzberg design"}}
]

DEFAULT_LLM_CITIES = sorted({g.get("city") for g in GOLDS if g.get("city")})

# ======= Default knobs (good starting point) =======
DEFAULT_CFG = {
  "beta": 0.08,
  "eta": 0.08,
  "tau": 0.20,
  "kappa": 0.25,

  "hard_penalty": -0.70,
  "soft_penalty": -0.25,
  "min_reviews_hard": 5,
  "min_reviews_soft": 10,

  "type_gate_scale": 1.5,
  "type_bonus_scale": 0.6,

  "retrieval_topk": 60,
  "use_crossencoder": True,
  "cross_topn": 8,
  "alpha": 0.70,

  "cluster_weight": 0.75,
  "cluster_temp": 0.18,
  "delta": 0.45,

  "cluster_baseline": 0.65
}


def _print_delta(before_items, after_items, sigs, revs_by_id=None):
    def _stats(items, sigs, revs_by_id):
        import numpy as np
        pop, ln, low = [], [], []
        for it in items:
            pid = it["id"]
            s = sigs.get(pid, {})
            pz = s.get("pop_z", None)
            if pz is not None:
                pop.append(pz)
            ln.append(s.get("listings_nearby", 0))
            nrev = it.get("reviews")
            if nrev is None and revs_by_id is not None:
                nrev = len(revs_by_id.get(pid, []))
            low.append(1 if (nrev or 0) < 5 else 0)
        return {
            "avg_pop_z": (float(np.mean(pop)) if pop else None),
            "avg_listings_nearby": (float(np.mean(ln)) if ln else None),
            "pct_low_reviews": (sum(low)/len(items) if items else None),
        }

    b, a = _stats(before_items, sigs, revs_by_id), _stats(after_items, sigs, revs_by_id)
    print("Δpop_z:", "n/a" if (b["avg_pop_z"] is None or a["avg_pop_z"] is None) else f"{a['avg_pop_z']-b['avg_pop_z']:.3f}")
    print("Δlistings_nearby:", "n/a" if (b["avg_listings_nearby"] is None or a["avg_listings_nearby"] is None) else f"{a['avg_listings_nearby']-b['avg_listings_nearby']:.3f}")
    if a["pct_low_reviews"] is not None:
        print("low_reviews_after%:", f"{a['pct_low_reviews']*100:.1f}%")


def _name_contains(metas, needle: str) -> int:
    needle = (needle or "").lower()
    for m in metas:
        n = (m.get("name") or "").lower()
        if needle in n:
            return 1
    return 0


def _get_candidates(ret: Retriever,
                    query: str,
                    *,
                    city: str | None,
                    mode: str,
                    k: int,
                    alpha: float) -> List[Tuple[float, int, Dict]]:
    mode = (mode or "hybrid").lower()

    if mode == "hybrid" and hasattr(ret, "hybrid_search"):
        return list(ret.hybrid_search(query, k=k, city=city, alpha=alpha))

    if mode == "dense":
        triples = ret.search(query, k=k, city=city)
        out: List[Tuple[float, int, Dict]] = []
        for rowid, score, meta in triples:
            if rowid is None or rowid < 0:
                continue
            out.append((float(score), int(rowid), meta))
        out.sort(reverse=True, key=lambda x: x[0])
        return out

    if mode == "bm25":
        raw = ret._fts_search(query, limit=max(k * 3, 100))
        if not raw:
            return []
        rowids = [rid for rid, _ in raw]
        metas = ret._fetch_meta(rowids)
        by_rowid = {int(meta["rowid"]): meta for meta in metas if meta.get("rowid") is not None}

        def _city_matches(meta_city: str | None, want: str | None) -> bool:
            if not want:
                return True
            return (meta_city or "").strip().lower() == want.strip().lower()

        filtered: List[Tuple[float, int, Dict]] = []
        fallback: List[Tuple[float, int, Dict]] = []
        for rid, raw_score in raw:
            meta = by_rowid.get(int(rid))
            if not meta:
                continue
            score = math.exp(-0.1 * max(0.0, float(raw_score)))
            triple = (score, int(rid), meta)
            if _city_matches(meta.get("city"), city):
                filtered.append(triple)
            fallback.append(triple)
            if len(filtered) >= k:
                break

        candidates = filtered if filtered else fallback
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[:k]

    return list(ret.hybrid_search(query, k=k, city=city, alpha=alpha))


def _compute_signal_stats(metas: List[Dict], sigs: Dict[str, Dict]) -> Dict[str, float | None]:
    pops, listings, flags = [], [], []
    for meta in metas:
        pid = meta.get("id")
        if not pid:
            continue
        sig = sigs.get(pid) or {}
        pop = sig.get("pop_z")
        if pop is not None:
            pops.append(float(pop))
        listings.append(float(sig.get("ln", 0.0)))
        flags.append(sig.get("flag"))

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return stats.mean(vals) if vals else None

    count = len(metas) if metas else 0
    high = sum(1 for f in flags if f == "high")
    low = sum(1 for f in flags if f == "low")
    return {
        "avg_pop_z": _mean(pops),
        "avg_listings_nearby": _mean(listings),
        "pct_flag_high": (high / count) if count else None,
        "pct_flag_low": (low / count) if count else None,
    }


def _mean_or_none(values):
    vals = [v for v in values if v is not None]
    return stats.mean(vals) if vals else None


def evaluate_queries(cfg: Dict,
                     *,
                     retrieval_mode: str = "hybrid",
                     use_sar: bool = True,
                     verbose: bool = False,
                     golds: List[Dict] | None = None) -> Tuple[Dict, List[Dict]]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    ret, kg = Retriever(), KG()
    results: List[Dict] = []

    topk_retr = int(cfg.get("retrieval_topk", 60))
    alpha = float(cfg.get("alpha", 0.7))

    gold_queries = golds if golds is not None else GOLDS

    try:
        for g in gold_queries:
            q, city = g["q"], g.get("city")
            cands = _get_candidates(ret, q, city=city, mode=retrieval_mode, k=topk_retr, alpha=alpha)
            if not cands:
                continue

            metas = [meta for _, _, meta in cands]
            ids = [m.get("id") for m in metas if m.get("id")]
            sigs = kg.popularity(ids)

            before_top = cands[:5]
            before_metas = [meta for _, _, meta in before_top]
            before_stats = _compute_signal_stats(before_metas, sigs)

            revs = {}
            type_gate = {}
            aspects = {}
            type_bonus: Dict[str, float] = {}

            if use_sar:
                aspects, type_gate, revs, _ = build_aspects_and_gates(q, ids, metas, kg, cfg)

                if cfg.get("use_crossencoder", False):
                    cross_topn = min(int(cfg.get("cross_topn", 15)), len(cands))
                    subset = [meta for _, _, meta in cands[:cross_topn]]
                    raw_scores = score_candidates(q, subset)
                    if raw_scores:
                        vals = list(raw_scores.values())
                        hi, lo = max(vals), min(vals)
                        denom = hi - lo
                        if denom > 1e-9:
                            scale = float(cfg.get("type_bonus_scale", 1.0))
                            type_bonus = {
                                pid: ((score - lo) / denom - 0.5) * 2.0 * scale
                                for pid, score in raw_scores.items()
                            }
                        else:
                            type_bonus = {pid: 0.0 for pid in raw_scores}

                final_metas = apply_sar(
                    cands,
                    sigs,
                    aspects=aspects,
                    type_gate=type_gate,
                    type_bonus=type_bonus,
                    k=5,
                    beta=float(cfg["beta"]),
                    eta=float(cfg["eta"]),
                    tau=float(cfg["tau"]),
                    delta=float(cfg["delta"]),
                    kappa=float(cfg.get("kappa", 0.0)),
                )
            else:
                final_metas = [meta for _, _, meta in before_top]

            final_stats = _compute_signal_stats(final_metas, sigs)

            hit = 0
            if "must" in g:
                hit = _name_contains(final_metas, g["must"]["name_contains"])
            elif "avoid" in g:
                hit = 1 if _name_contains(final_metas, g["avoid"]["name_contains"]) == 0 else 0

            if verbose:
                print(f"\nQuery: {q} | City: {city}")
                print("Top-5 before SAR:" if use_sar else "Top-5 baseline:")
                for (_, _, meta) in before_top[:5]:
                    sig = sigs.get(meta.get("id"), {})
                    print("  -", meta.get("name"), f"(pop_z={sig.get('pop_z')})")
                if use_sar:
                    print("Top-5 after SAR:")
                    for meta in final_metas:
                        pid = meta.get("id")
                        sig = sigs.get(pid, {})
                        print(
                            "  -",
                            meta.get("name"),
                            f"(pop_z={sig.get('pop_z')})"
                        )
                    _print_delta(
                        [{"id": m.get("id"), "name": m.get("name"), "reviews": len(revs.get(m.get("id"), []))}
                         for _, _, m in before_top[:5]],
                        [{"id": meta.get("id")} for meta in final_metas],
                        sigs,
                        revs,
                    )

            results.append({
                "retrieval_mode": retrieval_mode,
                "use_sar": int(use_sar),
                "query": q,
                "city": city or "",
                "hit_at_5": hit,
                "before_avg_pop_z": before_stats.get("avg_pop_z"),
                "final_avg_pop_z": final_stats.get("avg_pop_z"),
                "before_avg_listings": before_stats.get("avg_listings_nearby"),
                "final_avg_listings": final_stats.get("avg_listings_nearby"),
                "before_pct_flag_high": before_stats.get("pct_flag_high"),
                "final_pct_flag_high": final_stats.get("pct_flag_high"),
                "before_pct_flag_low": before_stats.get("pct_flag_low"),
                "final_pct_flag_low": final_stats.get("pct_flag_low"),
                "top_ids": "|".join([meta.get("id", "") or "" for meta in final_metas]),
                "top_names": "|".join([(meta.get("name") or "").replace("|", "/") for meta in final_metas]),
                "gold_needle": g.get("must", {}).get("name_contains") or g.get("avoid", {}).get("name_contains"),
            })

        hit_rate = _mean_or_none([row["hit_at_5"] for row in results])
        summary = {
            "retrieval_mode": retrieval_mode,
            "use_sar": int(use_sar),
            "hit_at_5": hit_rate,
            "mean_final_pop_z": _mean_or_none([row["final_avg_pop_z"] for row in results]),
            "mean_before_pop_z": _mean_or_none([row["before_avg_pop_z"] for row in results]),
            "mean_final_listings": _mean_or_none([row["final_avg_listings"] for row in results]),
            "mean_before_listings": _mean_or_none([row["before_avg_listings"] for row in results]),
            "mean_final_flag_high": _mean_or_none([row["final_pct_flag_high"] for row in results]),
            "mean_before_flag_high": _mean_or_none([row["before_pct_flag_high"] for row in results]),
            "num_queries": len(results),
        }
        return summary, results
    finally:
        try:
            ret.close()
        except Exception:
            pass
        try:
            kg.close()
        except Exception:
            pass


def run_eval(cfg: Dict, verbose: bool = True, golds: List[Dict] | None = None) -> Tuple[float, float, float]:
    summary, _ = evaluate_queries(
        cfg,
        retrieval_mode="hybrid",
        use_sar=True,
        verbose=verbose,
        golds=golds,
    )

    hit = float(summary.get("hit_at_5") or 0.0)
    pop_before = float(summary.get("mean_before_pop_z") or 0.0)
    pop_after = float(summary.get("mean_final_pop_z") or 0.0)

    if verbose:
        print("\nHit@5:", hit)
        print("Avg pop_z before:", pop_before)
        print("Avg pop_z after :", pop_after)

    return hit, pop_before, pop_after


def load_cfg(path: str | None):
    if path and os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        cfg = DEFAULT_CFG.copy()
        cfg.update(user or {})
        return cfg
    return DEFAULT_CFG.copy()


def save_cfg(path: str, cfg: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote best config to {path}")


def _load_golds_file(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golds file not found: {path}")

    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: List[Dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        with p.open("r", encoding="utf-8") as f:
            rows = json.load(f)

    out: List[Dict] = []
    for obj in rows:
        if isinstance(obj, dict) and "q" in obj and "city" in obj:
            out.append(obj)
    return out


def _merge_golds(*segments: Iterable[Dict]) -> List[Dict]:
    seen = set()
    merged: List[Dict] = []
    for segment in segments:
        for obj in segment or []:
            key = (obj.get("q"), obj.get("city"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(obj)
    return merged


def _write_benchmark_csv(path: str, rows: List[Dict]):
    if not path:
        return
    if not rows:
        print("No benchmark rows collected; CSV skipped.")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "retrieval_mode",
        "use_sar",
        "query",
        "city",
        "hit_at_5",
        "before_avg_pop_z",
        "final_avg_pop_z",
        "before_avg_listings",
        "final_avg_listings",
        "before_pct_flag_high",
        "final_pct_flag_high",
        "before_pct_flag_low",
        "final_pct_flag_low",
        "top_ids",
        "top_names",
        "gold_needle",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {}
            for field in fieldnames:
                val = row.get(field)
                out_row[field] = "" if val is None else val
            writer.writerow(out_row)
    print(f"Wrote benchmark CSV: {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-fast", action="store_true", help="Run cached tuner")
    ap.add_argument("--config", type=str, default=None, help="Load knob config JSON")
    ap.add_argument("--save-config", type=str, default="./eval/best_rerank_config.json",
                    help="Where to save tuned knobs JSON")
    ap.add_argument("--alpha", type=float, default=0.7, help="hybrid alpha used during tuning dataset prep")
    ap.add_argument("--n-jobs", type=int, default=-1, help="(unused) kept for compatibility")
    ap.add_argument("--tune-trials", type=int, default=60, help="Number of random configs to sample")
    ap.add_argument("--tune-seed", type=int, default=0, help="Random seed for tuner")
    ap.add_argument("--benchmark-csv", type=str, default=None,
                    help="If set, write per-query benchmark rows for retrieval modes/SAR states to CSV")
    ap.add_argument("--golds-file", type=str, default=None,
                    help="Optional JSON/JSONL file with gold queries to use during evaluation")
    ap.add_argument("--include-default-golds", action="store_true",
                    help="Include the built-in gold list alongside any loaded/generated queries")
    ap.add_argument("--generate-golds-out", type=str, default=None,
                    help="If set, synthesize template-based gold queries and write them to this JSON file")
    ap.add_argument("--template-queries-per-city", type=int, default=5,
                    help="Number of generated queries per city (default 5)")
    ap.add_argument("--template-cities", type=str, default=None,
                    help="Comma-separated list of city slugs for template generation; defaults to built-in cities")
    ap.add_argument("--template-seed", type=int, default=0,
                    help="Random seed for template-based generation")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg["alpha"] = args.alpha

    gold_segments: List[List[Dict]] = []
    if args.include_default_golds:
        gold_segments.append(GOLDS)

    if args.golds_file:
        try:
            gold_segments.append(_load_golds_file(args.golds_file))
            print(f"Loaded gold queries from {args.golds_file}")
        except Exception as exc:
            print(f"Failed to load golds from {args.golds_file}: {exc}")
            sys.exit(1)

    generated_golds: List[Dict] = []
    if args.generate_golds_out:
        raw_cities = (
            [c.strip().lower() for c in args.template_cities.split(",") if c.strip()]
            if args.template_cities else list_cities()
        )
        if not raw_cities:
            print("No cities specified for template generation; aborting.")
            sys.exit(1)

        try:
            generated_golds = generate_benchmark_queries(
                cities=raw_cities,
                per_city=max(1, args.template_queries_per_city),
                seed=args.template_seed,
            )
        except PromptGenerationError as exc:
            print(f"Prompt generation failed: {exc}")
            sys.exit(1)

        out_path = Path(args.generate_golds_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(generated_golds, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(generated_golds)} generated gold queries to {out_path}")
        gold_segments.append(generated_golds)

    golds_for_run = _merge_golds(*gold_segments) if gold_segments else GOLDS

    if args.tune_fast:
        from eval.tune import tune_fast
        result = tune_fast(
            golds_for_run,
            cfg,
            n_trials=args.tune_trials,
            seed=args.tune_seed,
        )
        best_cfg = result["best_cfg"]
        print("FAST TUNE:", result["trace"])
        if args.save_config:
            save_cfg(args.save_config, best_cfg)
        print("\n=== Evaluating with tuned best knobs ===")
        run_eval(best_cfg, verbose=True, golds=golds_for_run)
        return

    if args.benchmark_csv:
        combos = [
            ("hybrid", True),
            ("hybrid", False),
            ("dense", True),
            ("bm25", True),
        ]
        all_rows: List[Dict] = []
        for mode, sar_on in combos:
            summary, rows = evaluate_queries(
                cfg,
                retrieval_mode=mode,
                use_sar=sar_on,
                verbose=False,
                golds=golds_for_run,
            )
            all_rows.extend(rows)
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            print(
                f"[benchmark] mode={mode} sar={'on' if sar_on else 'off'} "
                f"queries={summary.get('num_queries', 0)} "
                f"hit@5={fmt(summary.get('hit_at_5'))} "
                f"final_pop_z={fmt(summary.get('mean_final_pop_z'))} "
                f"final_listings={fmt(summary.get('mean_final_listings'))}"
            )
        _write_benchmark_csv(args.benchmark_csv, all_rows)
        return

    run_eval(cfg, verbose=True, golds=golds_for_run)


if __name__ == "__main__":
    main()
