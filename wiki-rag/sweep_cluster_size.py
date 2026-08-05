#!/usr/bin/env python3
"""
Find the constant cluster size equivalent to the baseline "top-100 of 4096".

The baseline probes a fixed fraction of a fixed number of clusters whose sizes
vary from 1 to 103, so the number of candidates it actually scans swings per
query. With a constant cluster size n, probing p clusters scans exactly p*n
candidates every time -- which is what makes the ANN stage FHE-friendly, and also
collapses "fraction of the database scanned" and "candidate budget" into a single
knob.

So the equivalence question is: **for each n, what is the smallest p that matches
the baseline's recall, and which n achieves that at the smallest p*n?**

The baseline is re-measured here rather than taken from recall_analysis.txt,
because that file's numbers come from queries sampled out of the database itself
(every query its own guaranteed top-1 hit). Comparing a balanced clustering
against a self-retrieval number would systematically overstate the cost of
balancing, since balancing is exactly what moves a vector off its own nearest
centroid. Baseline and every cluster size are measured in one process, in the
same recall mode, against one shared query set.

Usage:
    # Fast path: vectors already extracted to .npy
    python sweep_cluster_size.py --vectors-npy vectors.npy --work-dir sweep_out

    # From a FAISS index directory (no embedding model needed)
    python sweep_cluster_size.py --faiss-index-file INDEX/index.faiss --work-dir sweep_out
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import balanced_ivf
import ivf_eval
import ivf_io
import train_ivf

DEFAULT_CLUSTER_SIZES = (16, 32, 64, 128)
BASELINE_NLIST = 4096
BASELINE_NPROBE = 100


def load_vectors(
    vectors_npy: Optional[str] = None,
    faiss_index_file: Optional[str] = None,
    faiss_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> np.ndarray:
    """
    Load the database embeddings.

    Reading ``index.faiss`` directly with faiss.read_index avoids langchain and
    the BGE model entirely -- the embedding model is only needed to satisfy
    FAISS.load_local's constructor, never to read stored vectors.

    Args:
        vectors_npy: Path to a pre-extracted (N, dim) float32 .npy.
        faiss_index_file: Path to a raw index.faiss.
        faiss_path: Path to a langchain FAISS directory (slow: loads the model).
        limit: Keep only the first N vectors.

    Returns:
        (N, dim) float32 C-contiguous array.
    """
    if vectors_npy:
        vectors = np.load(vectors_npy)
    elif faiss_index_file:
        import faiss

        print(f"Reading FAISS index from {faiss_index_file}...")
        index = faiss.read_index(str(faiss_index_file))
        print(f"  ntotal={index.ntotal:,} d={index.d}")
        vectors = index.reconstruct_n(0, index.ntotal)
    elif faiss_path:
        vectors, _ = train_ivf.load_embeddings_from_faiss(Path(faiss_path))
    else:
        raise ValueError("one of --vectors-npy, --faiss-index-file, --faiss-path is required")

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if limit and limit < len(vectors):
        print(f"  truncating {len(vectors):,} -> {limit:,} vectors")
        vectors = np.ascontiguousarray(vectors[:limit])
    return vectors


def find_min_nprobe(
    query_set: dict,
    centroids: np.ndarray,
    slots,
    target_recall: float,
    max_nprobe: int,
    transform: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """
    Smallest nprobe whose recall reaches ``target_recall``, by binary search.

    Binary search is valid because recall is monotonically non-decreasing in
    nprobe: the top-(p+1) centroid set is a superset of the top-p set, so the
    candidate pool only grows, and an exact rerank over a superset can only match
    or improve the result. (This does not hold for approximate reranking, which is
    why the rerank in ivf_search_from_slots is exact.)

    Args:
        query_set: Shared query set from ivf_eval.build_query_set.
        centroids: (nlist, dim) float32.
        slots: Slot array or ragged mapping.
        target_recall: Recall to match or beat.
        max_nprobe: Upper bound, normally nlist.
        transform: Optional OPQ rotation.
        verbose: Print each probe evaluated.

    Returns:
        Dict with nprobe (None if unreachable), recall, latency_ms, and the
        curve of every (p, recall) pair evaluated.
    """
    curve: Dict[int, float] = {}
    latency: Dict[int, float] = {}

    def recall_at(p: int) -> float:
        if p not in curve:
            idx, _, lat = ivf_eval.ivf_search_from_slots(
                query_set["queries"], query_set["db_vectors"], centroids, slots,
                nprobe=p, k=query_set["k"], transform=transform,
            )
            curve[p] = ivf_eval.recall_at_k(idx, query_set["ground_truth"])
            latency[p] = float(np.mean(lat))
            if verbose:
                print(f"      p={p:<5d} recall={curve[p]:.4f}")
        return curve[p]

    # Is the target reachable at all? A full scan is the ceiling.
    if recall_at(max_nprobe) < target_recall:
        return {"nprobe": None, "recall": curve[max_nprobe], "curve": dict(curve)}

    lo, hi = 1, max_nprobe
    while lo < hi:
        mid = (lo + hi) // 2
        if recall_at(mid) >= target_recall:
            hi = mid
        else:
            lo = mid + 1

    recall_at(lo)
    return {
        "nprobe": int(lo),
        "recall": curve[lo],
        "latency_ms": latency[lo],
        "curve": dict(curve),
    }


def _config(base_args, **overrides):
    """Clone the argparse namespace so run_pipeline cannot leak state between runs."""
    args = copy.deepcopy(base_args)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _base_namespace(args) -> argparse.Namespace:
    """Build the train_ivf argument namespace shared by every sweep row."""
    ns = train_ivf.build_parser().parse_args(["--vectors-npy", "unused"])
    ns.vectors_npy = None
    ns.faiss_path = None
    ns.index_type = args.index_type
    ns.niter = args.niter
    ns.seed = args.seed
    ns.recall_k = args.recall_k
    ns.n_test_queries = args.n_test_queries
    ns.recall_mode = args.recall_mode
    ns.query_noise = args.query_noise
    ns.assignment = args.assignment
    ns.assign_topm = args.assign_topm
    ns.price_iters = args.price_iters
    ns.dual_bound = args.dual_bound
    ns.lists_json = args.lists_json
    ns.use_gpu = args.use_gpu
    ns.test_recall = False   # the sweep drives evaluation itself
    ns.nprobe_grid = None
    return ns


def run_sweep(args) -> dict:
    """
    Measure the baseline and every cluster size against one shared query set.

    Args:
        args: Parsed sweep arguments.

    Returns:
        The sweep record, as written to cluster_size_sweep.json.
    """
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    vectors = load_vectors(
        vectors_npy=args.vectors_npy,
        faiss_index_file=args.faiss_index_file,
        faiss_path=args.faiss_path,
        limit=args.limit,
    )
    print(f"Corpus: {vectors.shape[0]:,} vectors, dim {vectors.shape[1]}")

    # One query set, shared by the baseline and every cluster size. This is what
    # makes the comparison valid -- the split depends only on (N, mode, n_test,
    # seed), never on the cluster size.
    query_set = ivf_eval.build_query_set(
        vectors, mode=args.recall_mode, n_test=args.n_test_queries,
        k=args.recall_k, seed=args.seed, query_noise=args.query_noise,
    )
    n_db = len(query_set["db_vectors"])
    base_ns = _base_namespace(args)

    # ---- Baseline: fixed nlist, variable sizes --------------------------
    print(f"\n{'#' * 70}")
    print(f"# BASELINE  nlist={args.baseline_nlist} nprobe={args.baseline_nprobe}"
          f" (variable cluster sizes)")
    print(f"{'#' * 70}")

    base_meta = train_ivf.run_pipeline(
        _config(base_ns, k=args.baseline_nlist, cluster_size=None,
                output_dir=str(work_dir / "baseline")),
        vectors=vectors, dim=int(vectors.shape[1]), query_set=query_set,
    )
    baseline_clustering = ivf_io.load_clustering(
        work_dir / "baseline", load_centroids=True
    )
    base_sizes = np.array(
        [len(baseline_clustering.members(c)) for c in range(baseline_clustering.nlist)]
    )
    base_eval = ivf_eval.evaluate_clustering(
        query_set, baseline_clustering.centroids, baseline_clustering.lists,
        [args.baseline_nprobe],
    )[args.baseline_nprobe]

    baseline_candidates = float(args.baseline_nprobe * base_sizes.mean())
    target = base_eval["recall"] if args.target_recall is None else args.target_recall
    print(f"\n  BASELINE {query_set['metric_name']} = {base_eval['recall']:.4f}")
    print(f"  candidates scanned ~ {baseline_candidates:,.0f} "
          f"({100.0 * baseline_candidates / n_db:.2f}% of DB, variable per query)")
    print(f"  cluster sizes: min={base_sizes.min()} max={base_sizes.max()} "
          f"mean={base_sizes.mean():.2f}")
    print(f"  target recall for the constant-size runs: {target:.4f}")

    rows: List[dict] = []
    for n in args.cluster_sizes:
        print(f"\n{'#' * 70}")
        print(f"# CONSTANT SIZE  n={n}")
        print(f"{'#' * 70}")
        t0 = time.time()
        out_dir = work_dir / f"n{n}"
        meta = train_ivf.run_pipeline(
            _config(base_ns, k=None, cluster_size=n, output_dir=str(out_dir)),
            vectors=vectors, dim=int(vectors.shape[1]), query_set=query_set,
        )
        clustering = ivf_io.load_clustering(out_dir, load_centroids=True)

        print(f"    searching for the smallest p reaching recall {target:.4f}")
        found = find_min_nprobe(
            query_set, clustering.centroids, clustering.slots,
            target_recall=target, max_nprobe=clustering.nlist,
        )

        diag = meta["diagnostics"] or {}
        candidates = (found["nprobe"] * n) if found["nprobe"] else None
        row = {
            "cluster_size": n,
            "nlist": meta["nlist"],
            "total_slots": meta["total_slots"],
            "n_padded_slots": meta["n_padded_slots"],
            "min_nprobe": found["nprobe"],
            "recall_at_min_nprobe": found["recall"],
            "candidates": candidates,
            # Both stages are homomorphic distance evaluations, and the centroid
            # stage shrinks as n grows (nlist = ceil(N/n)). Optimizing the
            # candidate stage alone would pick the wrong n.
            "centroid_dists": meta["nlist"],
            "total_dists": (meta["nlist"] + candidates) if candidates else None,
            "pct_db_scanned": (
                100.0 * found["nprobe"] * n / n_db if found["nprobe"] else None
            ),
            "latency_ms_numpy": found.get("latency_ms"),
            "balance_penalty_ratio": diag.get("balance_penalty_ratio"),
            "frac_forced": diag.get("frac_forced"),
            "n_outside_topm": diag.get("n_outside_topm"),
            "duality_gap_pct": diag.get("duality_gap_pct"),
            "retrievability": diag.get("retrievability_at_nprobe"),
            "recall_curve": found["curve"],
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)

        if found["nprobe"] is None:
            print(f"    n={n}: target recall {target:.4f} UNREACHABLE "
                  f"(best {found['recall']:.4f} at full scan)")
        else:
            print(f"    n={n}: p={found['nprobe']} -> {found['nprobe'] * n:,} candidates "
                  f"({row['pct_db_scanned']:.2f}% of DB), recall {found['recall']:.4f}")

    # Rank by total distance evaluations, not by candidates alone: the ANN stage
    # pays for nlist centroid comparisons *plus* p*n candidate comparisons, and
    # constant-size clustering trades the first against the second.
    feasible = [r for r in rows if r["total_dists"] is not None]
    winner = min(feasible, key=lambda r: r["total_dists"]) if feasible else None
    baseline_total = args.baseline_nlist + baseline_candidates
    for r in rows:
        r["total_dists_vs_baseline"] = (
            r["total_dists"] / baseline_total if r["total_dists"] else None
        )
        r["candidates_vs_baseline"] = (
            r["candidates"] / baseline_candidates if r["candidates"] else None
        )

    record = {
        "n_vectors_corpus": int(len(vectors)),
        "n_vectors_db": int(n_db),
        "dim": int(vectors.shape[1]),
        "recall_mode": query_set["mode"],
        "recall_metric_name": query_set["metric_name"],
        "n_test_queries": query_set["n_test"],
        "recall_k": query_set["k"],
        "seed": args.seed,
        "index_type": args.index_type,
        "niter": args.niter,
        "assignment": args.assignment if hasattr(args, "assignment") else "balanced",
        "baseline": {
            "nlist": args.baseline_nlist,
            "nprobe": args.baseline_nprobe,
            "recall": base_eval["recall"],
            "latency_ms_numpy": base_eval["latency_ms_mean"],
            "cluster_size_min": int(base_sizes.min()),
            "cluster_size_max": int(base_sizes.max()),
            "cluster_size_mean": float(base_sizes.mean()),
            "candidates_mean": baseline_candidates,
            "centroid_dists": args.baseline_nlist,
            "total_dists": baseline_total,
            "pct_db_scanned": 100.0 * baseline_candidates / n_db,
        },
        "target_recall": target,
        "rows": rows,
        "winner": winner,
    }

    (work_dir / "cluster_size_sweep.json").write_text(json.dumps(record, indent=2))
    write_markdown(work_dir / "cluster_size_sweep.md", record)
    write_recall_analysis(work_dir / "recall_analysis_constant_size.txt", record)

    print(f"\n{'=' * 70}")
    print("SWEEP RESULT")
    print(f"{'=' * 70}")
    print(f"Baseline: nlist={args.baseline_nlist} p={args.baseline_nprobe} "
          f"-> {baseline_candidates:,.0f} candidates, "
          f"{query_set['metric_name']}={base_eval['recall']:.4f}")
    print(f"  {'':10s} {'x':>6s} {'p':>5s} {'centroid':>9s} {'cand':>8s} "
          f"{'TOTAL':>8s} {'vs base':>8s} {'penalty':>8s}")
    print(f"  {'baseline':10s} {args.baseline_nlist:>6d} {args.baseline_nprobe:>5d} "
          f"{args.baseline_nlist:>9,d} {baseline_candidates:>8,.0f} "
          f"{baseline_total:>8,.0f} {'1.00x':>8s} {'1.0000':>8s}")
    for r in rows:
        if r["total_dists"] is None:
            print(f"  n={r['cluster_size']:<8d} {r['nlist']:>6d} {'--':>5s} "
                  f"target recall unreachable")
        else:
            print(f"  n={r['cluster_size']:<8d} {r['nlist']:>6d} {r['min_nprobe']:>5d} "
                  f"{r['centroid_dists']:>9,d} {r['candidates']:>8,d} "
                  f"{r['total_dists']:>8,d} {r['total_dists_vs_baseline']:>7.2f}x "
                  f"{r['balance_penalty_ratio']:>8.4f}")
    if winner:
        print(f"\nBest by total distance evaluations: n={winner['cluster_size']} "
              f"at p={winner['min_nprobe']} -- {winner['total_dists']:,} vs "
              f"{baseline_total:,.0f} baseline "
              f"({winner['total_dists_vs_baseline']:.2f}x), with a static "
              f"{winner['cluster_size']}-slot circuit shape.")
        print(f"  Its candidate stage alone is {winner['candidates_vs_baseline']:.2f}x "
              f"baseline; the centroid stage is "
              f"{args.baseline_nlist / winner['nlist']:.1f}x cheaper, which is what "
              f"pays for it.")
    print(f"Artifacts in {work_dir}")
    return record


def write_markdown(path: Path, record: dict) -> None:
    """Write the sweep summary as a markdown table."""
    b = record["baseline"]
    metric = record["recall_metric_name"]
    lines = [
        "# Constant cluster size sweep",
        "",
        f"- Database: **{record['n_vectors_db']:,}** vectors, dim {record['dim']}",
        f"- Metric: **{metric}** over {record['n_test_queries']} queries "
        f"(seed {record['seed']}, shared across every row)",
        f"- Index: {record['index_type']}, niter={record['niter']}",
        "",
        "## Baseline (fixed cluster count, variable sizes)",
        "",
        f"| nlist | nprobe | cluster sizes | centroid dists | candidates | "
        f"total dists | % DB | {metric} |",
        "|---|---|---|---|---|---|---|---|",
        f"| {b['nlist']} | {b['nprobe']} | min {b['cluster_size_min']}, "
        f"max {b['cluster_size_max']}, mean {b['cluster_size_mean']:.2f} | "
        f"{b['nlist']:,} | ~{b['candidates_mean']:,.0f} (varies) | "
        f"{b['total_dists']:,.0f} | {b['pct_db_scanned']:.2f}% | {b['recall']:.4f} |",
        "",
        f"## Constant size: smallest p reaching recall {record['target_recall']:.4f}",
        "",
        "Ranked by **total distance evaluations** (centroid stage + candidate stage),",
        "since both are homomorphic comparisons and constant-size clustering trades",
        "one against the other.",
        "",
        f"| n | x = ceil(N/n) | pad | p | centroid | candidates (p*n) | total | "
        f"vs base | % DB | {metric} | balance penalty | forced | ms (numpy) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in record["rows"]:
        if r["total_dists"] is None:
            lines.append(
                f"| {r['cluster_size']} | {r['nlist']} | {r['n_padded_slots']} | "
                f"— | {r['nlist']:,} | unreachable | — | — | — | "
                f"{r['recall_at_min_nprobe']:.4f} | "
                f"{r['balance_penalty_ratio']:.4f} | "
                f"{100 * (r['frac_forced'] or 0):.1f}% | — |"
            )
        else:
            lines.append(
                f"| {r['cluster_size']} | {r['nlist']} | {r['n_padded_slots']} | "
                f"{r['min_nprobe']} | {r['centroid_dists']:,} | {r['candidates']:,} | "
                f"{r['total_dists']:,} | {r['total_dists_vs_baseline']:.2f}x | "
                f"{r['pct_db_scanned']:.2f}% | "
                f"{r['recall_at_min_nprobe']:.4f} | {r['balance_penalty_ratio']:.4f} | "
                f"{100 * (r['frac_forced'] or 0):.1f}% | "
                f"{r['latency_ms_numpy']:.2f} |"
            )

    w = record["winner"]
    if w:
        lines += [
            "",
            f"**Equivalent operating point: n = {w['cluster_size']}, p = {w['min_nprobe']}.** "
            f"{w['total_dists']:,} total distance evaluations versus "
            f"{b['total_dists']:,.0f} for the baseline "
            f"({w['total_dists_vs_baseline']:.2f}x), at the same recall, with every "
            f"cluster exactly {w['cluster_size']} wide.",
            "",
            f"The candidate stage alone costs {w['candidates_vs_baseline']:.2f}x the "
            f"baseline -- equal sizes give up the baseline's implicit adaptivity, where "
            f"a query landing in a dense region automatically pulls larger clusters. "
            f"What pays for it is the centroid stage: "
            f"{b['nlist']:,} -> {w['nlist']:,} centroids, "
            f"{b['nlist'] / w['nlist']:.1f}x fewer encrypted comparisons.",
        ]
    lines += [
        "",
        "## Notes",
        "",
        "- `total dists` counts nlist centroid comparisons plus p*n candidate",
        "  comparisons. Ranking on candidates alone would pick a different (worse) n.",
        "- Latency is numpy, **not** comparable with `recall_analysis.txt`, which was",
        "  measured through FAISS's SIMD search path.",
        f"- The baseline recall here is re-measured in `{record['recall_mode']}` mode,",
        "  not taken from `recall_analysis.txt` (whose queries came from the database",
        "  itself, making every query its own guaranteed top-1 hit).",
        "- With constant size, candidates scanned is exactly `p*n` for every query.",
        "  The baseline's candidate count varies per query with the sizes of whichever",
        "  clusters were probed.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_recall_analysis(path: Path, record: dict) -> None:
    """Write a fixed-width table in the style of the existing recall_analysis.txt."""
    b = record["baseline"]
    lines = [
        "Constant-cluster-size recall analysis",
        "=" * 78,
        f"Database      : {record['n_vectors_db']:,} vectors, dim {record['dim']}",
        f"Metric        : {record['recall_metric_name']} "
        f"({record['n_test_queries']} queries, seed {record['seed']})",
        f"Index         : {record['index_type']}, niter={record['niter']}",
        "Latency       : numpy reference search, NOT comparable to recall_analysis.txt",
        "",
        f"Baseline: nlist={b['nlist']} nprobe={b['nprobe']} "
        f"sizes[{b['cluster_size_min']}..{b['cluster_size_max']}] "
        f"mean={b['cluster_size_mean']:.2f}",
        f"  centroid dists {b['nlist']:,} + candidates ~{b['candidates_mean']:,.0f} "
        f"= {b['total_dists']:,.0f} total ({b['pct_db_scanned']:.2f}% of DB, varies per query)",
        f"  {record['recall_metric_name']} = {b['recall']:.4f}",
        "",
        f"Smallest p reaching recall {record['target_recall']:.4f}, ranked by total",
        "distance evaluations (centroid stage + candidate stage):",
        "",
        f"{'n':>5} {'x':>6} {'pad':>4} {'p':>5} {'centr':>7} {'cands':>7} {'total':>8} "
        f"{'vs base':>8} {'%DB':>6} {'recall':>8} {'penalty':>8} {'forced%':>8}",
        "-" * 92,
    ]
    for r in record["rows"]:
        if r["total_dists"] is None:
            lines.append(
                f"{r['cluster_size']:>5} {r['nlist']:>6} {r['n_padded_slots']:>4} "
                f"{'--':>5} {r['nlist']:>7,} {'unreach':>7} {'--':>8} {'--':>8} {'--':>6} "
                f"{r['recall_at_min_nprobe']:>8.4f} {r['balance_penalty_ratio']:>8.4f} "
                f"{100 * (r['frac_forced'] or 0):>8.1f}"
            )
        else:
            lines.append(
                f"{r['cluster_size']:>5} {r['nlist']:>6} {r['n_padded_slots']:>4} "
                f"{r['min_nprobe']:>5} {r['centroid_dists']:>7,} {r['candidates']:>7,} "
                f"{r['total_dists']:>8,} {r['total_dists_vs_baseline']:>7.2f}x "
                f"{r['pct_db_scanned']:>6.2f} "
                f"{r['recall_at_min_nprobe']:>8.4f} {r['balance_penalty_ratio']:>8.4f} "
                f"{100 * (r['frac_forced'] or 0):>8.1f}"
            )
    lines.append("-" * 92)

    w = record["winner"]
    if w:
        lines += [
            "",
            f"Equivalent operating point: n={w['cluster_size']}, p={w['min_nprobe']}",
            f"  {w['total_dists']:,} total distance evaluations vs {b['total_dists']:,.0f} "
            f"baseline ({w['total_dists_vs_baseline']:.2f}x), same recall,",
            f"  every cluster exactly {w['cluster_size']} wide.",
            f"  Candidate stage {w['candidates_vs_baseline']:.2f}x baseline (equal sizes",
            f"  give up the baseline's implicit adaptivity in dense regions); paid for by",
            f"  the centroid stage dropping {b['nlist']:,} -> {w['nlist']:,} "
            f"({b['nlist'] / w['nlist']:.1f}x fewer encrypted comparisons).",
        ]

    for r in record["rows"]:
        if r["recall_curve"]:
            # int() because a JSON round-trip turns the keys into strings, which
            # would otherwise sort lexicographically (p=1000 before p=250).
            pts = " ".join(
                f"p={p}:{v:.4f}"
                for p, v in sorted(r["recall_curve"].items(), key=lambda kv: int(kv[0]))
            )
            lines += ["", f"n={r['cluster_size']} recall curve (points evaluated):", f"  {pts}"]

    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find the constant cluster size equivalent to top-100 of 4096"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--report-only", action="store_true",
                     help="Rebuild the reports from an existing "
                          "cluster_size_sweep.json in --work-dir, without re-measuring")
    src.add_argument("--vectors-npy", type=str, help="Pre-extracted (N, dim) float32 .npy")
    src.add_argument("--faiss-index-file", type=str,
                     help="Raw index.faiss; read directly, no embedding model needed")
    src.add_argument("--faiss-path", type=str,
                     help="langchain FAISS directory (slow: loads the BGE model)")

    p.add_argument("--work-dir", type=str, default="./cluster_size_sweep")
    p.add_argument("--cluster-sizes", type=int, nargs="+", default=list(DEFAULT_CLUSTER_SIZES))
    p.add_argument("--baseline-nlist", type=int, default=BASELINE_NLIST)
    p.add_argument("--baseline-nprobe", type=int, default=BASELINE_NPROBE)
    p.add_argument("--target-recall", type=float, default=None,
                   help="Override the measured baseline recall as the target")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N vectors of the corpus")

    p.add_argument("--index-type", type=str, default="ivf-flat",
                   choices=["ivf-flat", "ivf-pq", "ivf-opq"])
    p.add_argument("--assignment", type=str, default="balanced",
                   choices=list(balanced_ivf.CONSTANT_SIZE_STRATEGIES))
    p.add_argument("--assign-topm", type=int, default=balanced_ivf.DEFAULT_TOP_M)
    p.add_argument("--price-iters", type=int, default=balanced_ivf.DEFAULT_PRICE_ITERS)
    p.add_argument("--dual-bound", action="store_true")
    p.add_argument("--lists-json", type=str, default="padded",
                   choices=["padded", "trimmed", "none"])

    p.add_argument("--recall-mode", type=str, default="holdout",
                   choices=list(ivf_eval.RECALL_MODES))
    p.add_argument("--query-noise", type=float, default=0.05)
    p.add_argument("--n-test-queries", type=int, default=1000)
    p.add_argument("--recall-k", type=int, default=10)
    p.add_argument("--niter", type=int, default=20)
    p.add_argument("--seed", type=int, default=train_ivf.DEFAULT_KMEANS_SEED)
    p.add_argument("--use-gpu", action="store_true")
    return p


def regenerate_reports(work_dir: Path) -> dict:
    """
    Rebuild the markdown and text reports from a saved cluster_size_sweep.json.

    Lets the reporting be revised without paying for another sweep. Recomputes the
    derived cost columns and the winner so a change to how configurations are
    ranked applies to results already measured.

    Args:
        work_dir: Directory containing cluster_size_sweep.json.

    Returns:
        The updated sweep record.
    """
    work_dir = Path(work_dir)
    record = json.loads((work_dir / "cluster_size_sweep.json").read_text())
    b = record["baseline"]
    b.setdefault("centroid_dists", b["nlist"])
    b.setdefault("total_dists", b["nlist"] + b["candidates_mean"])

    for r in record["rows"]:
        r["centroid_dists"] = r["nlist"]
        r["total_dists"] = (
            r["nlist"] + r["candidates"] if r.get("candidates") else None
        )
        r["total_dists_vs_baseline"] = (
            r["total_dists"] / b["total_dists"] if r["total_dists"] else None
        )
        r["candidates_vs_baseline"] = (
            r["candidates"] / b["candidates_mean"] if r.get("candidates") else None
        )

    feasible = [r for r in record["rows"] if r["total_dists"] is not None]
    record["winner"] = min(feasible, key=lambda r: r["total_dists"]) if feasible else None

    (work_dir / "cluster_size_sweep.json").write_text(json.dumps(record, indent=2))
    write_markdown(work_dir / "cluster_size_sweep.md", record)
    write_recall_analysis(work_dir / "recall_analysis_constant_size.txt", record)
    print(f"Regenerated reports in {work_dir}")
    return record


def main():
    args = build_parser().parse_args()
    if args.report_only:
        regenerate_reports(Path(args.work_dir))
        return
    run_sweep(args)


if __name__ == "__main__":
    main()
