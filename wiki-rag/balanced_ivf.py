#!/usr/bin/env python3
"""
Capacity-constrained assignment of vectors to clusters of constant size.

FAISS's ``index.add()`` assigns every vector to its nearest centroid, so list
lengths are whatever Lloyd's algorithm happens to produce (measured on the
shipped artifact: min 1, max 103, mean 15.87 over 4096 clusters). An FHE
per-cluster kNN circuit wants a static shape instead, so this module assigns
*exactly* ``cluster_size`` vectors to each of ``nlist = ceil(N / cluster_size)``
clusters, keeping FAISS k-means for the centroid positions.

Strategies (``--assignment``):

    balanced       Price-balanced assignment (default). Dual ascent on the
                   transportation LP: each vector picks argmin_c (d + lambda_c)
                   at market-clearing prices, so there is no first-come bias.
    greedy         Global greedy over the top-m candidate pairs. Simpler, and
                   used as the repair pass inside `balanced` regardless.
    faiss-nearest  No balancing, variable sizes -- the zero-penalty reference
                   point that every balance diagnostic is measured against.
                   Not a constant-size mode.
    split-pad      Fallback: keep the nearest-centroid assignment, split lists
                   longer than n, pad shorter ones. See _assign_split_pad for
                   why this is a last resort.

Distances are **squared L2** throughout, matching FAISS's return convention.

faiss is optional here: pass a trained quantizer to reuse it, or let the module
fall back to a chunked numpy search. That keeps the assignment logic testable
without a faiss build.
"""

import time
from dataclasses import dataclass, field
from math import ceil
from typing import Callable, Dict, Optional

import numpy as np

from ivf_io import PADDING_SENTINEL, pack_slots

# Number of nearest centroids considered per vector. Bounding the candidate set
# is what makes this tractable: the dense 65,000 x 4,063 distance matrix would be
# 1.06 GB, while (N, 32) float32 is 8.3 MB.
DEFAULT_TOP_M = 32
DEFAULT_PRICE_ITERS = 200
DEFAULT_CHUNK = 8192


@dataclass
class AssignmentResult:
    """
    A capacity-feasible assignment.

    Attributes:
        slots: (nlist, cluster_size) int32, PADDING_SENTINEL in unused slots.
            The canonical artifact.
        assign: (N,) int32 cluster id per vector.
        slot_of: (N,) int32 position within the assigned cluster, so the flat
            slot address is ``assign * cluster_size + slot_of``.
        dist_assigned: (N,) float32 squared L2 to the assigned centroid.
        dist_nearest: (N,) float32 squared L2 to the *nearest* centroid, whether
            or not it was available. The gap is the balance penalty.
        rank: (N,) int32 rank of the assigned centroid within the vector's top-m
            (top_m means "outside the candidate set"). Drives retrievability@p.
        prices: (nlist,) final Lagrange multipliers for the `balanced` strategy,
            or None. Required to compute the certified dual bound.
        diagnostics: Free-form measurements; see compute_balance_diagnostics.
    """

    slots: np.ndarray
    assign: np.ndarray
    slot_of: np.ndarray
    dist_assigned: np.ndarray
    dist_nearest: np.ndarray
    rank: np.ndarray
    prices: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Candidate set
# ---------------------------------------------------------------------------

def _topm_candidates(
    vectors: np.ndarray,
    centroids: np.ndarray,
    top_m: int,
    quantizer=None,
    chunk: int = DEFAULT_CHUNK,
):
    """
    Squared-L2 distances to each vector's top-m nearest centroids.

    Args:
        vectors: (N, dim) float32.
        centroids: (nlist, dim) float32.
        top_m: Candidates per vector; clamped to nlist.
        quantizer: Optional trained faiss flat index over the centroids. Reused
            when given (it is exactly the IVF quantizer), avoiding a rebuild.
        chunk: Rows per batch, bounding the internal distance block.

    Returns:
        (D, I) with D (N, m) float32 ascending squared L2, I (N, m) int32
        centroid ids.
    """
    n_vectors = len(vectors)
    nlist = len(centroids)
    m = int(min(top_m, nlist))

    D = np.empty((n_vectors, m), dtype=np.float32)
    I = np.empty((n_vectors, m), dtype=np.int32)

    if quantizer is not None:
        for s in range(0, n_vectors, chunk):
            d, i = quantizer.search(vectors[s : s + chunk], m)
            D[s : s + chunk] = d
            I[s : s + chunk] = i
        return D, I

    # numpy fallback: ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2
    # errstate: Apple's Accelerate BLAS raises spurious FP warnings from matmul
    # on macOS; the results are verified equal to faiss to ~1e-7.
    c_sq = np.einsum("ij,ij->i", centroids, centroids)
    for s in range(0, n_vectors, chunk):
        block = vectors[s : s + chunk]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            d2 = c_sq[None, :] - 2.0 * (block @ centroids.T)
        d2 += np.einsum("ij,ij->i", block, block)[:, None]
        if m < nlist:
            part = np.argpartition(d2, m - 1, axis=1)[:, :m]
        else:
            part = np.broadcast_to(np.arange(nlist), d2.shape).copy()
        rows = np.arange(len(block))[:, None]
        part_d = d2[rows, part]
        order = np.argsort(part_d, axis=1, kind="stable")
        I[s : s + chunk] = part[rows, order]
        D[s : s + chunk] = np.maximum(part_d[rows, order], 0.0)
    return D, I


def _pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense squared-L2 distances, (len(a), len(b)) float32."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        d2 = np.einsum("ij,ij->i", b, b)[None, :] - 2.0 * (a @ b.T)
    d2 += np.einsum("ij,ij->i", a, a)[:, None]
    return np.maximum(d2, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Fill primitives
# ---------------------------------------------------------------------------

def _greedy_capacity_fill(
    D: np.ndarray,
    I: np.ndarray,
    assign: np.ndarray,
    capacity_left: np.ndarray,
) -> np.ndarray:
    """
    Assign unassigned vectors by global ascending distance, respecting capacity.

    Works both as a from-scratch strategy and as the repair pass after price
    balancing: it only touches vectors with ``assign == -1``.

    Args:
        D: (N, m) ascending squared distances.
        I: (N, m) candidate centroid ids.
        assign: (N,) int32, -1 for unassigned. Mutated in place.
        capacity_left: (nlist,) remaining capacity. Mutated in place.

    Returns:
        Indices of vectors still unassigned (their whole top-m filled up).
    """
    m = D.shape[1]
    pending = int((assign < 0).sum())
    if pending == 0:
        return np.empty(0, dtype=np.int64)

    # Restrict the sort to unassigned rows; at repair time this is a small set.
    rows = np.flatnonzero(assign < 0)
    sub_d = D[rows].ravel()
    order = np.argsort(sub_d, kind="stable")

    for p in order:
        r, c_rank = divmod(int(p), m)
        i = int(rows[r])
        if assign[i] >= 0:
            continue
        c = int(I[i, c_rank])
        if capacity_left[c] <= 0:
            continue
        assign[i] = c
        capacity_left[c] -= 1
        pending -= 1
        if pending == 0:
            break

    return np.flatnonzero(assign < 0)


def _exact_tail_fill(
    vectors: np.ndarray,
    centroids: np.ndarray,
    pool: np.ndarray,
    assign: np.ndarray,
    capacity_left: np.ndarray,
) -> None:
    """
    Place leftover vectors exactly, scanning only the still-open clusters.

    Guaranteed to terminate: ``sum(capacity_left) >= len(pool)`` by construction,
    and the dense block is small because both sets are small after balancing.

    Args:
        vectors: (N, dim) float32.
        centroids: (nlist, dim) float32.
        pool: Indices of unassigned vectors.
        assign: (N,) int32, mutated in place.
        capacity_left: (nlist,) mutated in place.
    """
    if len(pool) == 0:
        return
    open_c = np.flatnonzero(capacity_left > 0)
    assert capacity_left[open_c].sum() >= len(pool), (
        f"only {int(capacity_left[open_c].sum())} slots left for {len(pool)} vectors"
    )

    Dt = _pairwise_sq(vectors[pool], centroids[open_c])
    n_open = len(open_c)
    remaining = len(pool)
    for p in np.argsort(Dt.ravel(), kind="stable"):
        r, j = divmod(int(p), n_open)
        i = int(pool[r])
        if assign[i] >= 0:
            continue
        c = int(open_c[j])
        if capacity_left[c] <= 0:
            continue
        assign[i] = c
        capacity_left[c] -= 1
        remaining -= 1
        if remaining == 0:
            break


# ---------------------------------------------------------------------------
# Price balancing (dual ascent on the transportation LP)
# ---------------------------------------------------------------------------

def _price_balance(
    D: np.ndarray,
    I: np.ndarray,
    nlist: int,
    cluster_size: int,
    iters: int = DEFAULT_PRICE_ITERS,
    verbose: bool = True,
):
    """
    Find per-cluster prices that (approximately) clear the capacity market.

    Maintains lambda_c >= 0 and lets each vector pick argmin_c (d + lambda_c).
    Raising the price of an over-full cluster pushes its marginal members
    elsewhere. This is subgradient ascent on the dual of the transportation LP,
    so unlike greedy every vector re-chooses at the current prices each round --
    there is no first-come-first-served bias.

    Args:
        D: (N, m) ascending squared distances.
        I: (N, m) candidate centroid ids.
        nlist: Number of clusters.
        cluster_size: Capacity n per cluster.
        iters: Maximum subgradient iterations.
        verbose: Print convergence progress.

    Returns:
        (lam, best_over, history) -- prices at the least-overfull iterate, the
        total overflow there, and the per-iteration overflow trace.
    """
    n_vectors, m = D.shape
    rows = np.arange(n_vectors)
    lam = np.zeros(nlist, dtype=np.float32)

    # The step scale must be data-driven. Squared distances between normalized
    # BGE embeddings sit around 0.4 and the 1st-vs-2nd-nearest gap is ~0.01-0.05,
    # so any absolute constant would either do nothing or blow up.
    gap = float(np.mean(D[:, 1] - D[:, 0])) if m > 1 else 1.0
    step0 = max(gap, 1e-6) / 2.0

    best_lam = lam.copy()
    best_over = None
    history = []

    for t in range(1, iters + 1):
        choice = I[rows, np.argmin(D + lam[I], axis=1)]
        counts = np.bincount(choice, minlength=nlist)
        excess = counts.astype(np.float32) - cluster_size
        over = int(np.maximum(excess, 0).sum())
        history.append(over)

        if best_over is None or over < best_over:
            best_over, best_lam = over, lam.copy()
        if over == 0:
            if verbose:
                print(f"    prices cleared the market at iteration {t}")
            break

        lam = np.maximum(0.0, lam + (step0 / np.sqrt(t)) * (excess / cluster_size))

    if verbose:
        print(
            f"    price balancing: overflow {history[0]:,} -> {best_over:,} "
            f"over {len(history)} iterations"
        )
    return best_lam, best_over, history


def _hard_assign_with_eviction(
    D: np.ndarray,
    I: np.ndarray,
    lam: np.ndarray,
    nlist: int,
    cluster_size: int,
):
    """
    Commit each vector's priced choice, keeping the best members of each cluster.

    Where a cluster is still over-subscribed at the final prices, keep the
    ``cluster_size`` members with the smallest **raw** distance (not the priced
    distance -- prices are a means of allocation, not a measure of quality) and
    evict the rest for the repair pass.

    Returns:
        (assign, capacity_left) with -1 for evicted vectors.
    """
    n_vectors = D.shape[0]
    rows = np.arange(n_vectors)
    best = np.argmin(D + lam[I], axis=1)
    choice = I[rows, best].astype(np.int64)
    raw_d = D[rows, best]

    # Sort by (cluster, raw distance); position within the run gives the keep mask.
    order = np.lexsort((raw_d, choice))
    sorted_choice = choice[order]
    starts = np.searchsorted(sorted_choice, np.arange(nlist), side="left")
    within = np.arange(n_vectors) - starts[sorted_choice]
    keep = within < cluster_size

    assign = np.full(n_vectors, -1, dtype=np.int64)
    assign[order[keep]] = sorted_choice[keep]
    capacity_left = cluster_size - np.bincount(
        assign[assign >= 0], minlength=nlist
    ).astype(np.int64)
    return assign, capacity_left


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _finalize(
    vectors: np.ndarray,
    centroids: np.ndarray,
    assign: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    nlist: int,
    cluster_size: int,
    extra: dict,
    prices: Optional[np.ndarray] = None,
) -> AssignmentResult:
    """Compute distances/ranks and pack the slot array."""
    n_vectors = len(vectors)
    assert (assign >= 0).all(), f"{int((assign < 0).sum())} vectors left unassigned"

    # Distance to the assigned centroid: read from D where the assigned cluster
    # is inside the candidate set, else compute it directly.
    hit = I == assign[:, None]
    in_topm = hit.any(axis=1)
    rank = np.where(in_topm, hit.argmax(axis=1), D.shape[1]).astype(np.int32)

    dist_assigned = np.empty(n_vectors, dtype=np.float32)
    rows = np.arange(n_vectors)
    dist_assigned[in_topm] = D[rows[in_topm], rank[in_topm]]
    outside = np.flatnonzero(~in_topm)
    if len(outside):
        diff = vectors[outside] - centroids[assign[outside]]
        dist_assigned[outside] = np.einsum("ij,ij->i", diff, diff)

    dist_nearest = D[:, 0].astype(np.float32)

    slots = pack_slots(assign, dist_assigned, nlist, cluster_size)
    slot_of = np.empty(n_vectors, dtype=np.int32)
    flat = slots.reshape(-1)
    real = flat != PADDING_SENTINEL
    slot_of[flat[real]] = (np.flatnonzero(real) % cluster_size).astype(np.int32)

    return AssignmentResult(
        slots=slots,
        assign=assign.astype(np.int32),
        slot_of=slot_of,
        dist_assigned=dist_assigned,
        dist_nearest=dist_nearest,
        rank=rank,
        prices=prices,
        diagnostics=extra,
    )


def _assign_greedy(vectors, centroids, cluster_size, D, I, nlist, verbose, **kw):
    """Global greedy over top-m pairs, then exact tail fill."""
    assign = np.full(len(vectors), -1, dtype=np.int64)
    capacity_left = np.full(nlist, cluster_size, dtype=np.int64)
    pool = _greedy_capacity_fill(D, I, assign, capacity_left)
    n_outside = int(len(pool))
    if n_outside:
        if verbose:
            print(f"    {n_outside:,} vectors had their whole top-m full; exact tail fill")
        _exact_tail_fill(vectors, centroids, pool, assign, capacity_left)
    return assign, {"strategy": "greedy", "n_outside_topm": n_outside}, None


def _assign_balanced(
    vectors, centroids, cluster_size, D, I, nlist, verbose,
    price_iters=DEFAULT_PRICE_ITERS, **kw
):
    """Price-balanced assignment, then greedy repair, then exact tail fill."""
    lam, over, history = _price_balance(
        D, I, nlist, cluster_size, iters=price_iters, verbose=verbose
    )
    assign, capacity_left = _hard_assign_with_eviction(D, I, lam, nlist, cluster_size)
    n_evicted = int((assign < 0).sum())
    pool = _greedy_capacity_fill(D, I, assign, capacity_left)
    n_outside = int(len(pool))
    if n_outside:
        if verbose:
            print(f"    {n_outside:,} vectors had their whole top-m full; exact tail fill")
        _exact_tail_fill(vectors, centroids, pool, assign, capacity_left)
    return assign, {
        "strategy": "balanced",
        "price_iters_run": len(history),
        "overflow_after_prices": over,
        "n_evicted": n_evicted,
        "n_outside_topm": n_outside,
        "lambda_max": float(lam.max()),
        "lambda_nonzero_frac": float((lam > 0).mean()),
    }, lam


def dual_lower_bound(
    vectors: np.ndarray,
    centroids: np.ndarray,
    lam: np.ndarray,
    cluster_size: int,
    chunk: int = DEFAULT_CHUNK,
) -> float:
    """
    Lagrangian lower bound on the optimal balanced-assignment cost.

    ``L(lam) = sum_i min_c (d_ic + lam_c) - n * sum_c lam_c`` is a valid lower
    bound on the optimal total assignment cost, which turns "our clustering is
    good" into "within X% of the best possible balanced clustering".

    The inner minimum must run over **all** centroids: restricting it to the
    top-m candidate set can only raise it, which would invalidate the bound.

    Args:
        vectors: (N, dim) float32.
        centroids: (nlist, dim) float32.
        lam: (nlist,) final prices.
        cluster_size: Capacity n.
        chunk: Rows per batch.

    Returns:
        The bound, in the same squared-L2 units as dist_assigned.
    """
    total = 0.0
    c_sq = np.einsum("ij,ij->i", centroids, centroids) + lam
    for s in range(0, len(vectors), chunk):
        block = vectors[s : s + chunk]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            priced = c_sq[None, :] - 2.0 * (block @ centroids.T)
        priced += np.einsum("ij,ij->i", block, block)[:, None]
        total += float(priced.min(axis=1).sum())
    return total - cluster_size * float(lam.sum())


def _assign_faiss_nearest(vectors, centroids, cluster_size, D, I, nlist, verbose, **kw):
    """
    Nearest-centroid assignment with no capacity constraint.

    Not a constant-size mode -- it exists as the zero-balance-penalty reference
    that every diagnostic is measured against, and to verify that the new export
    path reproduces FAISS's own assignment exactly.
    """
    return I[:, 0].astype(np.int64), {"strategy": "faiss-nearest", "n_outside_topm": 0}, None


def _assign_split_pad(vectors, centroids, cluster_size, D, I, nlist, verbose, **kw):
    """
    Fallback: keep the nearest-centroid assignment, split long lists, pad short.

    This is a last resort, and degrades correctness as well as size:

      * The database inflates badly -- measured on the shipped distribution,
        1.52x at n=16 and 8.06x at n=128, versus 8-24 wasted slots for the
        balanced strategies.
      * Splitting a cluster emits **near-duplicate centroids**, so the top-p
        cluster selection burns several probes on chunks of one original
        cluster. That corrupts the selection stage itself, not just its cost.
      * nlist becomes an *output* (sum of ceil(L_c / n)) rather than
        ceil(N / n), so the sizing contract no longer holds.

    Returns a nearest-centroid assignment plus the chunk plan; the caller must
    rebuild centroids for the split chunks.
    """
    nearest = I[:, 0].astype(np.int64)
    counts = np.bincount(nearest, minlength=nlist)
    chunks_per_cluster = np.maximum(1, np.ceil(counts / cluster_size).astype(np.int64))
    return nearest, {
        "strategy": "split-pad",
        "n_outside_topm": 0,
        "derived_nlist": int(chunks_per_cluster.sum()),
        "duplicate_centroid_groups": {
            int(c): int(k) for c, k in enumerate(chunks_per_cluster) if k > 1
        },
        "total_slots": int(chunks_per_cluster.sum() * cluster_size),
    }, None


_STRATEGIES: Dict[str, Callable] = {
    "balanced": _assign_balanced,
    "greedy": _assign_greedy,
    "faiss-nearest": _assign_faiss_nearest,
    "split-pad": _assign_split_pad,
}

CONSTANT_SIZE_STRATEGIES = ("balanced", "greedy")


def assign_constant_size(
    vectors: np.ndarray,
    centroids: np.ndarray,
    cluster_size: int,
    *,
    strategy: str = "balanced",
    quantizer=None,
    top_m: int = DEFAULT_TOP_M,
    price_iters: int = DEFAULT_PRICE_ITERS,
    verbose: bool = True,
) -> AssignmentResult:
    """
    Assign exactly ``cluster_size`` vectors to each of ceil(N / cluster_size) clusters.

    Args:
        vectors: (N, dim) float32 database vectors. For ivf-opq these must
            already be in the OPQ-rotated space, matching the exported centroids.
        centroids: (nlist, dim) float32 trained centroids.
        cluster_size: Constant size n.
        strategy: One of _STRATEGIES.
        quantizer: Optional trained faiss flat index over the centroids, reused
            for the candidate search.
        top_m: Candidate centroids per vector.
        price_iters: Subgradient iterations for the `balanced` strategy.
        verbose: Print progress.

    Returns:
        An AssignmentResult whose slot array satisfies the constant-size invariant.

    Raises:
        ValueError: On an unknown strategy, or a non-constant-size strategy.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}"
        )
    if strategy not in CONSTANT_SIZE_STRATEGIES:
        raise ValueError(
            f"strategy {strategy!r} does not produce constant-size clusters; "
            f"it is a diagnostic/fallback path and must be driven explicitly"
        )

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    n_vectors = len(vectors)
    nlist = len(centroids)

    if nlist * cluster_size < n_vectors:
        raise ValueError(
            f"{nlist} clusters x {cluster_size} slots = {nlist * cluster_size} "
            f"cannot hold {n_vectors} vectors"
        )

    if verbose:
        print(f"\n  Assigning {n_vectors:,} vectors to {nlist:,} clusters of size {cluster_size}")
        print(f"    strategy={strategy} top_m={min(top_m, nlist)}")

    t0 = time.time()
    D, I = _topm_candidates(vectors, centroids, top_m, quantizer=quantizer)
    t_search = time.time() - t0
    if verbose:
        print(f"    candidate search: {t_search:.2f}s")

    t0 = time.time()
    assign, extra, prices = _STRATEGIES[strategy](
        vectors, centroids, cluster_size, D, I, nlist, verbose,
        price_iters=price_iters,
    )
    t_assign = time.time() - t0

    counts = np.bincount(assign, minlength=nlist)
    assert counts.max(initial=0) <= cluster_size, (
        f"capacity violated: cluster {int(np.argmax(counts))} holds {counts.max()} "
        f"> {cluster_size}"
    )

    # A vector whose entire top-m filled up had to be placed by the exact tail
    # fill, potentially far from any of its preferred centroids. A few are normal;
    # a lot means the candidate set is too small to express a good assignment, and
    # both the balance penalty and retrievability@p degrade. Warn rather than
    # fail -- the diagnostics are the tripwire, not an assertion.
    n_outside = int(extra.get("n_outside_topm", 0))
    if n_outside > 0.01 * n_vectors:
        print(
            f"    WARNING: {n_outside:,} vectors ({100.0 * n_outside / n_vectors:.1f}%) "
            f"were placed outside their top-{min(top_m, nlist)} candidates. "
            f"Consider raising --assign-topm; retrievability@p is only valid up to top_m."
        )

    result = _finalize(
        vectors, centroids, assign, D, I, nlist, cluster_size,
        {**extra, "top_m": int(min(top_m, nlist)),
         "candidate_search_seconds": round(t_search, 3),
         "assignment_seconds": round(t_assign, 3)},
        prices=prices,
    )
    if verbose:
        print(f"    assignment: {t_assign:.2f}s")
    return result


def nearest_centroid_lists(
    vectors: np.ndarray,
    centroids: np.ndarray,
    quantizer=None,
    chunk: int = DEFAULT_CHUNK,
) -> Dict[int, list]:
    """
    Plain nearest-centroid inverted lists, for the `faiss-nearest` baseline.

    Args:
        vectors: (N, dim) float32.
        centroids: (nlist, dim) float32.
        quantizer: Optional trained faiss flat index over the centroids.
        chunk: Rows per batch.

    Returns:
        {cluster_id: [vector ids...]} with every cluster id present.
    """
    _, I = _topm_candidates(vectors, centroids, 1, quantizer=quantizer, chunk=chunk)
    nearest = I[:, 0]
    lists = {c: [] for c in range(len(centroids))}
    for i, c in enumerate(nearest):
        lists[int(c)].append(i)
    return lists
