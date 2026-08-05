#!/usr/bin/env python3
"""
Measurement for IVF clustering: plaintext search, honest recall, balance diagnostics.

Two things here exist because the original harness could not answer the question
this project now needs answered.

**Plaintext search.** ``ivf_search_from_slots`` re-implements "pick the top-p
centroids, gather their slots, rerank exactly" in numpy rather than calling
``index.search``. That is not a workaround -- it is what the deployed FHE+PIR
pipeline actually does, and in constant-size mode the assignment deliberately
does not live inside the FAISS index at all (``index.add()`` would overwrite it
with nearest-centroid). Measuring the artifact under test beats measuring a
FAISS index that was built differently.

**Honest recall.** The original harness sampled its test queries *from the
database* and brute-forced ground truth over those same vectors, so every query
was its own guaranteed top-1 hit. That inflates the number, and it inflates it
in a way that interacts with the change under test: constant-size clustering
forces some vectors off their nearest centroid, which is precisely what
self-retrieval punishes hardest. Comparing cluster sizes on a self-retrieval
metric would systematically overstate the cost of balancing. Hence
``--recall-mode``:

    holdout    Hold queries out of the database and the clustering. The honest
               number, and the default for the sweep.
    perturbed  Keep the database whole (so the artifact measured IS the artifact
               shipped) but jitter the query off the stored vector.
    self       The original behaviour, retained for continuity and renamed to
               self_retrieval_recall@k wherever it is reported.

Latency here is numpy, not FAISS SIMD, so it is NOT comparable with the numbers
in recall_analysis.txt. Columns are labelled accordingly.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ivf_io import PADDING_SENTINEL

RECALL_MODES = ("holdout", "perturbed", "self")

# Reported nprobe grid for the retrievability curve.
DEFAULT_NPROBE_GRID = (1, 2, 4, 8, 16, 25, 32, 50, 64, 100, 200, 500)


# ---------------------------------------------------------------------------
# Query set construction
# ---------------------------------------------------------------------------

def build_query_set(
    vectors: np.ndarray,
    mode: str = "holdout",
    n_test: int = 1000,
    k: int = 10,
    seed: int = 20260804,
    query_noise: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Split vectors into a database and a query set, and compute exact ground truth.

    Must be called *before* clustering: in holdout mode the database is a strict
    subset, so the clustering has to be built on the returned ``db_vectors``.

    The split depends only on (len(vectors), mode, n_test, seed) -- never on the
    cluster size -- so every cluster size in a sweep is scored on an identical
    query set and the comparison is valid.

    Args:
        vectors: (N, dim) float32 full corpus.
        mode: One of RECALL_MODES.
        n_test: Number of queries.
        k: Ground-truth depth.
        seed: RNG seed; pin it or the sweep is not reproducible.
        query_noise: Gaussian sigma for `perturbed` mode, applied before
            renormalization.
        verbose: Print a summary.

    Returns:
        Dict with db_vectors, db_ids (positions in the original array), queries,
        ground_truth (indices into db_vectors), and a metric_name describing what
        the resulting recall actually measures.

    Raises:
        ValueError: On an unknown mode or an impossible n_test.
    """
    if mode not in RECALL_MODES:
        raise ValueError(f"unknown recall mode {mode!r}; expected one of {RECALL_MODES}")

    n_total = len(vectors)
    rng = np.random.default_rng(seed)
    n_test = int(min(n_test, n_total // 2 if mode == "holdout" else n_total))
    if n_test < 1:
        raise ValueError(f"n_test resolved to {n_test}; need at least 1 query")

    if mode == "holdout":
        perm = rng.permutation(n_total)
        query_pos, db_ids = perm[:n_test], np.sort(perm[n_test:])
        db_vectors = np.ascontiguousarray(vectors[db_ids])
        queries = np.ascontiguousarray(vectors[query_pos])
        metric_name = f"holdout_recall@{k}"
    else:
        db_ids = np.arange(n_total)
        db_vectors = vectors
        query_pos = rng.choice(n_total, size=n_test, replace=False)
        queries = np.ascontiguousarray(vectors[query_pos])
        if mode == "perturbed":
            queries = queries + rng.normal(
                0.0, query_noise, size=queries.shape
            ).astype(np.float32)
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = (queries / np.maximum(norms, 1e-12)).astype(np.float32)
            metric_name = f"perturbed_recall@{k}"
        else:
            metric_name = f"self_retrieval_recall@{k}"

    if verbose:
        print(
            f"\n  Query set: mode={mode} n_test={n_test} "
            f"db={len(db_vectors):,} metric={metric_name}"
        )

    ground_truth = brute_force_ground_truth(db_vectors, queries, k, verbose=verbose)

    return {
        "db_vectors": db_vectors,
        "db_ids": db_ids,
        "queries": queries,
        "ground_truth": ground_truth,
        "mode": mode,
        "metric_name": metric_name,
        "n_test": n_test,
        "k": k,
        "seed": seed,
        "query_noise": query_noise if mode == "perturbed" else None,
    }


def brute_force_ground_truth(
    db_vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    chunk: int = 256,
    verbose: bool = True,
) -> np.ndarray:
    """
    Exact top-k by brute force, in squared L2.

    Args:
        db_vectors: (N, dim) float32 database.
        queries: (Q, dim) float32 queries.
        k: Depth.
        chunk: Queries per batch.
        verbose: Print elapsed time.

    Returns:
        (Q, k) int64 indices into db_vectors, ascending by distance.
    """
    t0 = time.time()
    db_sq = np.einsum("ij,ij->i", db_vectors, db_vectors)
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        block = queries[s : s + chunk]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            d2 = db_sq[None, :] - 2.0 * (block @ db_vectors.T)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        rows = np.arange(len(block))[:, None]
        out[s : s + chunk] = part[rows, np.argsort(d2[rows, part], axis=1, kind="stable")]
    if verbose:
        print(f"    exact ground truth for {len(queries)} queries in {time.time() - t0:.2f}s")
    return out


# ---------------------------------------------------------------------------
# Plaintext IVF search
# ---------------------------------------------------------------------------

def ivf_search_from_slots(
    queries: np.ndarray,
    db_vectors: np.ndarray,
    centroids: np.ndarray,
    slots: np.ndarray,
    nprobe: int,
    k: int,
    transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Search by probing the top-``nprobe`` clusters and reranking their members exactly.

    Mirrors the deployed pipeline stage for stage, so it measures the exported
    artifact rather than a FAISS index that may have been built differently.

    Args:
        queries: (Q, dim) float32, in the *original* embedding space.
        db_vectors: (N, dim) float32 database, original space.
        centroids: (nlist, dim) float32. For ivf-opq these live in the rotated
            space, so pass ``transform``.
        slots: (nlist, S) int32 with PADDING_SENTINEL fill, or a ragged
            {cid: [ids]} mapping.
        nprobe: Clusters to probe, p.
        k: Neighbours to return.
        transform: Optional (dim, dim) rotation applied to queries before the
            centroid comparison, for ivf-opq.

    Returns:
        (indices, distances, latencies_ms). indices is (Q, k) int64 into
        db_vectors, -1 padded when a query found fewer than k candidates.
    """
    ragged = not isinstance(slots, np.ndarray)
    nlist = len(centroids)
    nprobe = int(min(nprobe, nlist))

    q_for_centroids = queries if transform is None else queries @ transform.T
    c_sq = np.einsum("ij,ij->i", centroids, centroids)

    indices = np.full((len(queries), k), -1, dtype=np.int64)
    distances = np.full((len(queries), k), np.inf, dtype=np.float32)
    latencies = np.empty(len(queries), dtype=np.float64)

    for qi in range(len(queries)):
        t0 = time.perf_counter()

        # Stage 1: rank centroids (the FHE stage in the deployed system).
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cd = c_sq - 2.0 * (centroids @ q_for_centroids[qi])
        probes = (
            np.argpartition(cd, nprobe - 1)[:nprobe] if nprobe < nlist else np.arange(nlist)
        )

        # Stage 2: gather the probed clusters' members (the PIR stage).
        if ragged:
            cand = np.concatenate([np.asarray(slots[int(c)], dtype=np.int64) for c in probes]) \
                if len(probes) else np.empty(0, dtype=np.int64)
        else:
            cand = slots[probes].reshape(-1).astype(np.int64)
            cand = cand[cand != PADDING_SENTINEL]

        if len(cand) == 0:
            latencies[qi] = (time.perf_counter() - t0) * 1000.0
            continue

        # Stage 3: exact rerank (client-side in the deployed system).
        sub = db_vectors[cand]
        diff = sub - queries[qi]
        d2 = np.einsum("ij,ij->i", diff, diff)
        kk = min(k, len(cand))
        top = np.argpartition(d2, kk - 1)[:kk]
        top = top[np.argsort(d2[top], kind="stable")]

        indices[qi, :kk] = cand[top]
        distances[qi, :kk] = d2[top]
        latencies[qi] = (time.perf_counter() - t0) * 1000.0

    return indices, distances, latencies


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Fraction of true neighbours found, matching the original harness's definition.

    Args:
        retrieved: (Q, k) indices, -1 for missing.
        ground_truth: (Q, k) true neighbour indices.

    Returns:
        recall@k in [0, 1].
    """
    correct = 0
    total = 0
    for r, g in zip(retrieved, ground_truth):
        gt = set(int(v) for v in g)
        correct += len(set(int(v) for v in r if v >= 0) & gt)
        total += len(gt)
    return correct / total if total else 0.0


def evaluate_clustering(
    query_set: dict,
    centroids: np.ndarray,
    slots,
    nprobe_values: Sequence[int],
    transform: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[int, dict]:
    """
    Measure recall and latency across several nprobe values.

    Args:
        query_set: Output of build_query_set.
        centroids: (nlist, dim) float32.
        slots: Slot array or ragged mapping.
        nprobe_values: Values of p to evaluate.
        transform: Optional OPQ rotation.
        verbose: Print a row per value.

    Returns:
        {nprobe: {recall, latency_ms_mean, latency_ms_p95, candidates_mean}}
    """
    out: Dict[int, dict] = {}
    k = query_set["k"]
    for p in sorted(set(int(v) for v in nprobe_values)):
        idx, _, lat = ivf_search_from_slots(
            query_set["queries"], query_set["db_vectors"], centroids, slots,
            nprobe=p, k=k, transform=transform,
        )
        rec = recall_at_k(idx, query_set["ground_truth"])
        found = int((idx >= 0).sum(axis=1).mean())
        out[p] = {
            "recall": rec,
            "latency_ms_mean": float(np.mean(lat)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "returned_mean": found,
        }
        if verbose:
            print(
                f"    p={p:<5d} {query_set['metric_name']}={rec:.4f}  "
                f"latency={np.mean(lat):.2f} ms (numpy)"
            )
    return out


# ---------------------------------------------------------------------------
# Balance diagnostics
# ---------------------------------------------------------------------------

def compute_balance_diagnostics(
    result,
    cluster_size: Optional[int],
    n_vectors: int,
    nprobe_grid: Sequence[int] = DEFAULT_NPROBE_GRID,
    dual_bound: Optional[float] = None,
) -> dict:
    """
    Quantify what constant-size assignment cost, relative to nearest-centroid.

    The repo previously reported no cluster-size statistics at all, so this is
    the first place the balance/quality trade-off becomes visible.

    Args:
        result: An AssignmentResult.
        cluster_size: Constant size n, or None for a variable-size run.
        n_vectors: Number of real vector ids.
        nprobe_grid: Values of p for the retrievability curve.
        dual_bound: Optional certified lower bound on total assignment cost.

    Returns:
        Diagnostics dict, safe to embed in ivf_metadata.json.

    Raises:
        AssertionError: If the constant-size invariant is violated. This is the
            invariant the whole FHE story rests on, so it is asserted rather
            than printed and hoped over.
    """
    slots = result.slots
    per_row = (slots != PADDING_SENTINEL).sum(axis=1)

    if cluster_size is not None:
        n_padded = int(slots.shape[0] * cluster_size - n_vectors)
        # Every cluster full except for the remainder tail.
        assert per_row.max(initial=0) <= cluster_size, "a cluster exceeds capacity"
        assert int(per_row.sum()) == n_vectors, (
            f"slots hold {int(per_row.sum())} real ids, expected {n_vectors}"
        )
        assert int((cluster_size - per_row).sum()) == n_padded, "padding accounting is wrong"

    d_assigned = result.dist_assigned.astype(np.float64)
    d_nearest = result.dist_nearest.astype(np.float64)
    delta = d_assigned - d_nearest

    total_assigned = float(d_assigned.sum())
    diag = {
        "distance_units": "squared_l2",
        "size_histogram": {
            int(s): int(c) for s, c in zip(*np.unique(per_row, return_counts=True))
        },
        "n_padded_slots": int(slots.shape[0] * slots.shape[1] - per_row.sum()),
        "padded_cluster_ids": [int(c) for c in np.flatnonzero(per_row < slots.shape[1])],
        "n_forced": int((result.rank > 0).sum()),
        "frac_forced": float((result.rank > 0).mean()),
        "n_outside_topm": int(result.diagnostics.get("n_outside_topm", 0)),
        "mean_sq_dist_assigned": float(d_assigned.mean()),
        "mean_sq_dist_nearest": float(d_nearest.mean()),
        "balance_penalty_ratio": float(d_assigned.mean() / d_nearest.mean())
        if d_nearest.mean() > 0 else 1.0,
        "delta_sq_dist": {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
            "p95": float(np.percentile(delta, 95)),
            "max": float(delta.max()),
        },
        "assigned_rank_histogram": {
            int(r): int(c) for r, c in zip(*np.unique(result.rank, return_counts=True))
        },
        "total_assignment_cost": total_assigned,
    }

    # Retrievability is only meaningful up to top_m: ranks of vectors assigned
    # outside the candidate set are recorded as exactly top_m, so any p > top_m
    # would count them as retrievable and saturate the curve at 1.0 artificially.
    top_m = int(result.diagnostics.get("top_m", result.rank.max() + 1))
    diag["retrievability_valid_up_to"] = top_m
    diag["retrievability_at_nprobe"] = {
        int(p): float((result.rank < p).mean())
        for p in sorted(set(nprobe_grid))
        if p <= top_m
    }

    if dual_bound is not None:
        diag["dual_lower_bound"] = float(dual_bound)
        diag["duality_gap_pct"] = (
            float(100.0 * (total_assigned - dual_bound) / abs(dual_bound))
            if dual_bound else None
        )

    # Per-cluster spread: is the penalty diffuse or concentrated in a few clusters?
    per_cluster_mean = np.full(slots.shape[0], np.nan)
    order = np.argsort(result.assign, kind="stable")
    sorted_assign = result.assign[order]
    bounds = np.searchsorted(sorted_assign, np.arange(slots.shape[0] + 1))
    for c in range(slots.shape[0]):
        lo, hi = bounds[c], bounds[c + 1]
        if hi > lo:
            per_cluster_mean[c] = d_assigned[order[lo:hi]].mean()
    finite = per_cluster_mean[np.isfinite(per_cluster_mean)]
    if len(finite):
        diag["per_cluster_mean_sq_dist"] = {
            "min": float(finite.min()),
            "p95": float(np.percentile(finite, 95)),
            "max": float(finite.max()),
        }

    return diag


def write_balance_report(output_dir: Path, diag: dict, header: str = "") -> Path:
    """
    Write cluster_balance.txt, a human-readable view of the diagnostics.

    Args:
        output_dir: Directory to write into.
        diag: Output of compute_balance_diagnostics.
        header: Optional lines describing the run.

    Returns:
        Path to the written report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cluster_balance.txt"

    lines = ["Cluster balance report", "=" * 60]
    if header:
        lines += [header, ""]

    hist = diag["size_histogram"]
    lines += [
        "Cluster occupancy (real ids per cluster)",
        f"  distinct sizes : {len(hist)}",
        "  histogram      : " + ", ".join(f"{s}x{c}" for s, c in sorted(hist.items())),
        f"  padded slots   : {diag['n_padded_slots']} "
        f"in cluster(s) {diag['padded_cluster_ids'] or 'none'}",
        "",
        f"Balance penalty (squared L2, vs unconstrained nearest centroid)",
        f"  mean dist assigned : {diag['mean_sq_dist_assigned']:.6f}",
        f"  mean dist nearest  : {diag['mean_sq_dist_nearest']:.6f}",
        f"  penalty ratio      : {diag['balance_penalty_ratio']:.6f}",
        f"  delta mean/med/p95/max : "
        f"{diag['delta_sq_dist']['mean']:.6f} / {diag['delta_sq_dist']['median']:.6f} / "
        f"{diag['delta_sq_dist']['p95']:.6f} / {diag['delta_sq_dist']['max']:.6f}",
        "",
        f"Displacement",
        f"  forced off nearest : {diag['n_forced']:,} ({diag['frac_forced'] * 100:.2f}%)",
        f"  outside top-m      : {diag['n_outside_topm']:,}",
        "",
    ]

    if "duality_gap_pct" in diag and diag["duality_gap_pct"] is not None:
        lines += [
            "Certified optimality",
            f"  total cost   : {diag['total_assignment_cost']:.4f}",
            f"  dual bound   : {diag['dual_lower_bound']:.4f}",
            f"  duality gap  : {diag['duality_gap_pct']:.3f}%",
            "",
        ]

    lines += [
        "Retrievability@p (analytic ceiling: fraction of vectors whose",
        "assigned cluster is within the query's top-p centroids)",
        f"  valid only for p <= top_m = {diag.get('retrievability_valid_up_to', '?')}; "
        "beyond that the metric saturates artificially",
    ]
    for p, v in sorted(diag["retrievability_at_nprobe"].items()):
        lines.append(f"  p={p:<5d} {v:.4f}")

    if "per_cluster_mean_sq_dist" in diag:
        s = diag["per_cluster_mean_sq_dist"]
        lines += [
            "",
            "Per-cluster mean distance spread",
            f"  min={s['min']:.6f}  p95={s['p95']:.6f}  max={s['max']:.6f}",
        ]

    path.write_text("\n".join(lines) + "\n")
    return path
