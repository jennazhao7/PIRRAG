#!/usr/bin/env python3
"""
Naive clustering padded to a uniform width -- the alternative to forced-equal-size.

## The hypothesis

Forcing every cluster to hold exactly n vectors costs recall efficiency: it needs
~3.5x more candidates than plain k-means to reach the same recall, because plain
k-means is implicitly *adaptive* (dense regions get bigger clusters, so a query
landing in a crowd automatically pulls a bigger candidate pool, and equal sizes
throw that away).

So: keep the naive clustering, and make it merely *look* uniform to the PIR layer
by padding every inverted list out to a common width W. PIR then fetches exactly
p*W records per query regardless of occupancy -- same uniformity property,
possibly at lower total cost.

## Why it might not work

Padding is dead weight. The shipped distribution is min 1, max 103, mean 15.87,
so padding to the max wastes ~85% of every fetch. The question is whether the
adaptivity is worth more than the padding costs, and that is an empirical
question, not one to settle by argument. Hence this trial.

## What is swept

Three overflow policies, because W < max forces a choice about the long tail:

    pad-max    W = max list length. Lossless, no splitting, maximum waste.
    split      Lists longer than W are split into ceil(L/W) chunks, each with a
               duplicated centroid. Lossless, but inflates the cluster count and
               spends probes on chunks of one original cluster.
    truncate   Lists longer than W keep only their W nearest members; the rest
               are dropped from the index entirely. Cheapest fetch, but it is
               *lossy* -- dropped vectors become unretrievable, which caps recall.

Cost model, per query, matched against the equal-size arm:

    PIR records fetched  = p * W          (uniform, no occupancy leak)
    FHE centroid compares = nlist_effective
    total distance evals  = nlist_effective + p * W

Usage:
    python -u pad_naive.py --vectors-npy vectors.npy --work-dir out

Use `-u` (or redirect through `tee`) when logging to a file: each configuration
takes minutes, and a buffered run shows nothing until it finishes.
"""

import argparse
import json
import sys
import time
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_WIKI_RAG_DIR = Path(__file__).resolve().parents[1] / "wiki-rag"
if str(_WIKI_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_WIKI_RAG_DIR))

import balanced_ivf
import ivf_eval

PADDING_SENTINEL = -1
POLICIES = ("pad-max", "split", "truncate")


def train_naive_centroids(
    vectors: np.ndarray, nlist: int, niter: int = 20, seed: int = 1234, verbose: bool = True
):
    """
    Plain k-means centroids plus nearest-centroid assignment -- no capacity limit.

    This is deliberately the *unmodified* baseline behaviour: whatever list-length
    distribution Lloyd's algorithm produces is what we then pad.

    Args:
        vectors: (N, dim) float32.
        nlist: Number of clusters.
        niter: k-means iterations.
        seed: k-means RNG seed.
        verbose: Print progress.

    Returns:
        (centroids, assign, dist_nearest) -- (nlist, dim) float32, (N,) int64,
        (N,) float32 squared L2.
    """
    import faiss

    dim = vectors.shape[1]
    km = faiss.Kmeans(dim, nlist, niter=niter, seed=seed, verbose=False)
    # Match the training regime of the equal-size arm exactly, or the comparison
    # is confounded by clustering quality rather than the padding question.
    km.cp.min_points_per_centroid = 1
    t0 = time.time()
    km.train(vectors)
    centroids = km.centroids.reshape(nlist, dim).astype(np.float32)

    quantizer = faiss.IndexFlatL2(dim)
    quantizer.add(centroids)
    D, I = quantizer.search(vectors, 1)
    if verbose:
        print(f"    k-means nlist={nlist} in {time.time() - t0:.1f}s")
    return centroids, I[:, 0].astype(np.int64), D[:, 0].astype(np.float32)


def build_padded_layout(
    assign: np.ndarray,
    dist: np.ndarray,
    centroids: np.ndarray,
    width: Optional[int],
    policy: str,
) -> dict:
    """
    Turn a variable-size assignment into fixed-width padded lists.

    Args:
        assign: (N,) cluster id per vector.
        dist: (N,) squared L2 to the assigned centroid; orders members so the
            nearest are kept first (matters for `truncate`).
        centroids: (nlist, dim) float32.
        width: Uniform list width W. None means "use the max list length".
        policy: One of POLICIES.

    Returns:
        Dict with slots (n_rows, W) int32, centroids (possibly duplicated for
        `split`), n_rows, width, n_dropped, and waste_frac.

    Raises:
        ValueError: On an unknown policy.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")

    nlist = len(centroids)
    order = np.lexsort((dist, assign))
    sorted_assign = assign[order]
    bounds = np.searchsorted(sorted_assign, np.arange(nlist + 1))
    members = [order[bounds[c]:bounds[c + 1]] for c in range(nlist)]
    lengths = np.array([len(m) for m in members])

    if policy == "pad-max":
        W = int(lengths.max())
    else:
        if width is None:
            raise ValueError(f"policy {policy!r} requires an explicit --widths value")
        W = int(width)

    rows: List[np.ndarray] = []
    row_centroids: List[int] = []
    n_dropped = 0

    for c in range(nlist):
        mem = members[c]
        if policy == "split":
            n_chunks = max(1, ceil(len(mem) / W))
            for j in range(n_chunks):
                rows.append(mem[j * W:(j + 1) * W])
                row_centroids.append(c)
        else:
            # pad-max cannot overflow by construction; truncate drops the tail,
            # which is the farthest members of the cluster.
            keep = mem[:W]
            n_dropped += len(mem) - len(keep)
            rows.append(keep)
            row_centroids.append(c)

    slots = np.full((len(rows), W), PADDING_SENTINEL, dtype=np.int32)
    for i, r in enumerate(rows):
        slots[i, :len(r)] = r

    n_real = int((slots != PADDING_SENTINEL).sum())
    return {
        "slots": slots,
        "centroids": centroids[np.array(row_centroids)],
        "row_to_orig_cluster": np.array(row_centroids),
        "n_rows": len(rows),
        "width": W,
        "n_dropped": int(n_dropped),
        "n_real": n_real,
        "waste_frac": 1.0 - n_real / slots.size,
        "orig_lengths": lengths,
    }


def find_min_nprobe(
    query_set: dict, centroids: np.ndarray, slots: np.ndarray,
    target_recall: float, verbose: bool = False,
) -> dict:
    """
    Smallest p reaching the target recall, by binary search.

    Monotonic in p for the same reason as the equal-size arm: probing p+1 rows
    gives a superset of candidates and the rerank is exact.

    Note for `split`, probing p *rows* may cover fewer than p distinct original
    clusters, since chunks of one cluster carry near-identical centroids and tend
    to be selected together. That is the cost of the policy, and it is reflected
    here automatically because the search ranks rows, not clusters.
    """
    curve: Dict[int, float] = {}
    latency: Dict[int, float] = {}
    max_p = len(centroids)

    def recall_at(p: int) -> float:
        if p not in curve:
            idx, _, lat = ivf_eval.ivf_search_from_slots(
                query_set["queries"], query_set["db_vectors"], centroids, slots,
                nprobe=p, k=query_set["k"],
            )
            curve[p] = ivf_eval.recall_at_k(idx, query_set["ground_truth"])
            latency[p] = float(np.mean(lat))
            if verbose:
                print(f"        p={p:<5d} recall={curve[p]:.4f}")
        return curve[p]

    if recall_at(max_p) < target_recall:
        return {"nprobe": None, "recall": curve[max_p], "ceiling": curve[max_p],
                "curve": dict(curve)}

    lo, hi = 1, max_p
    while lo < hi:
        mid = (lo + hi) // 2
        if recall_at(mid) >= target_recall:
            hi = mid
        else:
            lo = mid + 1
    recall_at(lo)
    return {"nprobe": int(lo), "recall": curve[lo], "ceiling": curve[max_p],
            "latency_ms": latency[lo], "curve": dict(curve)}


def run_trial(args) -> dict:
    """Sweep nlist x width x policy and rank by total distance evaluations."""
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    vectors = np.ascontiguousarray(np.load(args.vectors_npy), dtype=np.float32)
    print(f"Corpus: {vectors.shape[0]:,} vectors, dim {vectors.shape[1]}")

    # Identical query-set construction to the equal-size arm: same mode, count and
    # seed, so the two trials are scored on exactly the same queries.
    query_set = ivf_eval.build_query_set(
        vectors, mode=args.recall_mode, n_test=args.n_test_queries,
        k=args.recall_k, seed=args.seed,
    )
    db = query_set["db_vectors"]
    n_db = len(db)

    # ---- Reference: unpadded naive clustering at the baseline setting --------
    print(f"\n{'#' * 70}")
    print(f"# REFERENCE  naive nlist={args.baseline_nlist} nprobe={args.baseline_nprobe}"
          f"  (variable sizes, NOT padded)")
    print(f"{'#' * 70}")
    ref_c, ref_a, ref_d = train_naive_centroids(
        db, args.baseline_nlist, niter=args.niter, seed=args.seed
    )
    ref_lists = {c: [] for c in range(args.baseline_nlist)}
    for i, c in enumerate(ref_a):
        ref_lists[int(c)].append(i)
    ref_sizes = np.array([len(ref_lists[c]) for c in range(args.baseline_nlist)])
    ref_eval = ivf_eval.evaluate_clustering(
        query_set, ref_c, ref_lists, [args.baseline_nprobe], verbose=False
    )[args.baseline_nprobe]
    target = ref_eval["recall"] if args.target_recall is None else args.target_recall
    ref_candidates = float(args.baseline_nprobe * ref_sizes.mean())

    print(f"  sizes min={ref_sizes.min()} max={ref_sizes.max()} mean={ref_sizes.mean():.2f}")
    print(f"  {query_set['metric_name']} = {ref_eval['recall']:.4f}")
    print(f"  candidates ~{ref_candidates:,.0f} (variable per query -- leaks occupancy)")
    print(f"  target recall for every padded config: {target:.4f}")

    rows: List[dict] = []
    for nlist in args.nlists:
        centroids, assign, dist = train_naive_centroids(
            db, nlist, niter=args.niter, seed=args.seed
        )
        lengths = np.bincount(assign, minlength=nlist)
        print(f"\n  nlist={nlist}: sizes min={lengths.min()} max={lengths.max()} "
              f"mean={lengths.mean():.2f} p95={np.percentile(lengths, 95):.0f}")

        for policy in args.policies:
            widths = [None] if policy == "pad-max" else args.widths
            for W in widths:
                if W is not None and W > int(lengths.max()):
                    continue    # wider than the widest list: same as pad-max
                layout = build_padded_layout(assign, dist, centroids, W, policy)
                found = find_min_nprobe(
                    query_set, layout["centroids"], layout["slots"], target,
                    verbose=args.verbose,
                )
                fetched = found["nprobe"] * layout["width"] if found["nprobe"] else None
                row = {
                    "policy": policy,
                    "nlist": nlist,
                    "width": layout["width"],
                    "n_rows": layout["n_rows"],
                    "n_dropped": layout["n_dropped"],
                    "waste_frac": layout["waste_frac"],
                    "min_nprobe": found["nprobe"],
                    "recall": found["recall"],
                    "recall_ceiling": found["ceiling"],
                    "pir_records": fetched,
                    "centroid_dists": layout["n_rows"],
                    "total_dists": (layout["n_rows"] + fetched) if fetched else None,
                    "latency_ms": found.get("latency_ms"),
                }
                rows.append(row)
                tag = f"{policy} nlist={nlist} W={layout['width']}"
                # flush: each config takes minutes (one full-database scan to test
                # reachability), so a redirected run must not sit in a stdio buffer
                # for an hour with nothing to show.
                if fetched is None:
                    print(f"    {tag:<34s} UNREACHABLE (ceiling {found['ceiling']:.4f}"
                          + (f", dropped {layout['n_dropped']:,}" if layout['n_dropped'] else "")
                          + ")", flush=True)
                else:
                    print(f"    {tag:<34s} p={found['nprobe']:<5d} "
                          f"PIR={fetched:<7,d} total={row['total_dists']:<7,d} "
                          f"waste={layout['waste_frac'] * 100:4.1f}% "
                          f"recall={found['recall']:.4f}", flush=True)

    feasible = [r for r in rows if r["total_dists"] is not None]
    best_total = min(feasible, key=lambda r: r["total_dists"]) if feasible else None
    best_pir = min(feasible, key=lambda r: r["pir_records"]) if feasible else None

    record = {
        "n_vectors_db": n_db,
        "dim": int(vectors.shape[1]),
        "recall_mode": query_set["mode"],
        "recall_metric_name": query_set["metric_name"],
        "n_test_queries": query_set["n_test"],
        "recall_k": query_set["k"],
        "seed": args.seed,
        "niter": args.niter,
        "target_recall": target,
        "reference_unpadded": {
            "nlist": args.baseline_nlist,
            "nprobe": args.baseline_nprobe,
            "recall": ref_eval["recall"],
            "size_min": int(ref_sizes.min()),
            "size_max": int(ref_sizes.max()),
            "size_mean": float(ref_sizes.mean()),
            "candidates_mean": ref_candidates,
            "note": "variable fetch size; not PIR-uniform",
        },
        "rows": rows,
        "best_by_total_dists": best_total,
        "best_by_pir_records": best_pir,
    }
    (work_dir / "padded_naive_trial.json").write_text(json.dumps(record, indent=2))
    print(f"\nWrote {work_dir / 'padded_naive_trial.json'}")
    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trial: naive clustering padded to uniform width vs forced-equal-size"
    )
    p.add_argument("--vectors-npy", type=str, required=True)
    p.add_argument("--work-dir", type=str, default="./padded_naive_out")
    p.add_argument("--nlists", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--widths", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128])
    p.add_argument("--policies", type=str, nargs="+", default=list(POLICIES),
                   choices=list(POLICIES))
    p.add_argument("--baseline-nlist", type=int, default=4096)
    p.add_argument("--baseline-nprobe", type=int, default=100)
    p.add_argument("--target-recall", type=float, default=None)
    p.add_argument("--recall-mode", type=str, default="holdout",
                   choices=list(ivf_eval.RECALL_MODES))
    p.add_argument("--n-test-queries", type=int, default=1000)
    p.add_argument("--recall-k", type=int, default=10)
    p.add_argument("--niter", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    run_trial(build_parser().parse_args())


if __name__ == "__main__":
    main()
