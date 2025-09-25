# nlp/inspect_clusters.py
import os, json, numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/clusters")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    cents = np.load(os.path.join(args.dir, "centroids.npy"))
    with open(os.path.join(args.dir, "components.json")) as f:
        comp = json.load(f)
    counts = comp.get("counts", [0]*len(cents))
    tops   = comp.get("top_terms", [[]]*len(cents))

    print(f"centroids: {cents.shape}  (clusters x dims)")
    order = np.argsort(counts)[::-1]
    for rank, c in enumerate(order[:args.top]):
        print(f"[{rank+1:02d}] cluster={c} size={counts[c]}")
        print("     terms:", ", ".join(tops[c][:15]))
    if len(order) > args.top:
        tail = [counts[c] for c in order[args.top:args.top+10]]
        print("... tail counts:", tail)

if __name__ == "__main__":
    main()
