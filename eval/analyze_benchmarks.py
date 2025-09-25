"""Utility to summarize benchmark CSV outputs for RQ2 and RQ3 analysis.

Usage
-----

    python3 -m eval.analyze_benchmarks --csv eval/benchmarks.csv

Outputs aggregated metrics to stdout so the results can be pasted into the
thesis or lab notebook.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple, List


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None or math.isnan(value):
        return "   nan"
    if signed:
        return f"{value:+.3f}"
    return f"{value:.3f}"


def load_rows(path: str):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_copy = dict(row)
            row_copy["hit_at_5"] = _safe_float(row_copy.get("hit_at_5")) or 0.0
            row_copy["final_avg_pop_z"] = _safe_float(row_copy.get("final_avg_pop_z"))
            row_copy["final_avg_listings"] = _safe_float(row_copy.get("final_avg_listings"))
            row_copy["before_avg_pop_z"] = _safe_float(row_copy.get("before_avg_pop_z"))
            row_copy["before_avg_listings"] = _safe_float(row_copy.get("before_avg_listings"))
            row_copy["use_sar"] = int(float(row_copy.get("use_sar", 0)))
            yield row_copy


def summarize_rq2(rows):
    """Return metrics grouped by retrieval mode using SAR."""
    grouped: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["use_sar"] != 1:
            continue
        mode = row["retrieval_mode"]
        grouped[mode]["hit"].append(row["hit_at_5"])
        grouped[mode]["pop"].append(row.get("final_avg_pop_z"))
        grouped[mode]["listings"].append(row.get("final_avg_listings"))

    lines = ["RQ2 — Retrieval mode comparison (SAR on)"]
    lines.append("mode        queries  hit@5   final_pop_z  final_listings")
    table: List[Dict[str, object]] = []
    for mode, metrics in sorted(grouped.items()):
        hit = _mean(metrics["hit"]) or 0.0
        pop = _mean(metrics["pop"])
        listings = _mean(metrics["listings"])
        q = len(metrics["hit"])
        lines.append(f"{mode:<11}{q:>7}  {hit:.3f}  {_fmt(pop)}  {_fmt(listings)}")
        table.append({
            "retrieval_mode": mode,
            "queries": q,
            "hit_at_5": round(hit, 6),
            "final_pop_z": None if pop is None else round(pop, 6),
            "final_listings": None if listings is None else round(listings, 6),
        })
    return "\n".join(lines), table


def summarize_rq3(rows):
    """Return SAR vs. no-SAR deltas per retrieval mode."""
    grouped: Dict[Tuple[str, int], Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["retrieval_mode"], row["use_sar"])
        grouped[key]["hit"].append(row["hit_at_5"])
        grouped[key]["pop"].append(row.get("final_avg_pop_z"))
        grouped[key]["listings"].append(row.get("final_avg_listings"))

    lines = ["\nRQ3 — SAR effect per retrieval mode"]
    header = "mode        sar  queries  hit@5   final_pop_z  final_listings"
    lines.append(header)
    summary_per_mode = {}
    table: List[Dict[str, object]] = []
    for (mode, sar_flag), metrics in sorted(grouped.items()):
        hit = _mean(metrics["hit"]) or 0.0
        pop = _mean(metrics["pop"])
        listings = _mean(metrics["listings"])
        q = len(metrics["hit"])
        lines.append(
            f"{mode:<11}{sar_flag:>4}  {q:>7}  {hit:.3f}  {_fmt(pop)}  {_fmt(listings)}"
        )
        summary_per_mode.setdefault(mode, {})[sar_flag] = {
            "hit": hit,
            "pop": pop,
            "listings": listings,
        }
        table.append({
            "retrieval_mode": mode,
            "use_sar": sar_flag,
            "queries": q,
            "hit_at_5": round(hit, 6),
            "final_pop_z": None if pop is None else round(pop, 6),
            "final_listings": None if listings is None else round(listings, 6),
        })

    lines.append("\nDeltas (sar_on - sar_off):")
    lines.append("mode        Δhit@5  Δpop_z  Δlistings")
    deltas: List[Dict[str, object]] = []
    for mode, stats in sorted(summary_per_mode.items()):
        on = stats.get(1)
        off = stats.get(0)
        if not on or not off:
            continue
        d_hit = on["hit"] - off["hit"]
        pop_on, pop_off = on["pop"], off["pop"]
        list_on, list_off = on["listings"], off["listings"]
        d_pop = pop_on - pop_off if pop_on is not None and pop_off is not None else None
        d_list = list_on - list_off if list_on is not None and list_off is not None else None
        lines.append(f"{mode:<11}{_fmt(d_hit, signed=True)}  {_fmt(d_pop, signed=True)}  {_fmt(d_list, signed=True)}")
        deltas.append({
            "retrieval_mode": mode,
            "delta_hit_at_5": round(d_hit, 6),
            "delta_pop_z": None if d_pop is None else round(d_pop, 6),
            "delta_listings": None if d_list is None else round(d_list, 6),
        })
    return "\n".join(lines), table, deltas


def _write_csv(path: str | None, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    if not path or not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="./eval/benchmarks.csv", help="Path to benchmark CSV")
    ap.add_argument("--rq2-csv", type=str, default=None, help="Optional path to write RQ2 summary CSV")
    ap.add_argument("--rq3-csv", type=str, default=None, help="Optional path to write RQ3 summary CSV")
    ap.add_argument("--rq3-deltas-csv", type=str, default=None, help="Optional path to write RQ3 delta CSV")
    args = ap.parse_args()

    rows = list(load_rows(args.csv))
    rq2_text, rq2_table = summarize_rq2(rows)
    rq3_text, rq3_table, rq3_deltas = summarize_rq3(rows)

    print(rq2_text)
    print(rq3_text)

    _write_csv(args.rq2_csv, rq2_table, ["retrieval_mode", "queries", "hit_at_5", "final_pop_z", "final_listings"])
    if args.rq2_csv:
        print(f"Wrote RQ2 summary CSV to {args.rq2_csv}")

    _write_csv(args.rq3_csv, rq3_table, ["retrieval_mode", "use_sar", "queries", "hit_at_5", "final_pop_z", "final_listings"])
    if args.rq3_csv:
        print(f"Wrote RQ3 summary CSV to {args.rq3_csv}")

    _write_csv(args.rq3_deltas_csv, rq3_deltas, ["retrieval_mode", "delta_hit_at_5", "delta_pop_z", "delta_listings"])
    if args.rq3_deltas_csv:
        print(f"Wrote RQ3 delta CSV to {args.rq3_deltas_csv}")


if __name__ == "__main__":
    main()
