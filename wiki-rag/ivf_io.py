#!/usr/bin/env python3
"""
Shared I/O for IVF clustering artifacts.

Two clustering parameterizations exist in this project:

1. ``fixed-nlist`` (legacy) -- the number of clusters is fixed (K=4096) and
   cluster sizes are whatever k-means produces. Measured on the shipped
   ``prototype/data/lists.json``: 65,000 vectors, min 1, max 103, mean 15.87.

2. ``constant-size`` (new) -- the cluster size ``n`` is fixed and the number of
   clusters is derived as ``nlist = ceil(N / n)``. Every inverted list holds
   exactly ``n`` slots, short-filled with PADDING_SENTINEL. This gives the FHE
   per-cluster kNN circuit a static shape.

Consumers should never infer which parameterization they are looking at by
inspecting list lengths -- a variable-size run can coincidentally produce
equal-length lists on small data. Branch on ``metadata["sizing_mode"]`` instead,
which is why every run writes ``ivf_metadata.json``, including legacy runs.

Artifacts written next to each other in an output directory:
    ivf_metadata.json   always -- schema below
    centroids.npy       (nlist, dim) float32
    cluster_slots.npy   (nlist, n) int32, constant-size mode only, canonical
    lists.json          {"<cid>": [ids...]} -- compatibility view
    opq_transform.npy   (dim, dim) float32, ivf-opq only
"""

import json
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Schema version for ivf_metadata.json. Bump on any breaking key change.
METADATA_VERSION = 1

METADATA_FILENAME = "ivf_metadata.json"
CENTROIDS_FILENAME = "centroids.npy"
SLOTS_FILENAME = "cluster_slots.npy"
LISTS_FILENAME = "lists.json"
OPQ_FILENAME = "opq_transform.npy"

# Fills unused slots in constant-size mode. Negative so that any consumer that
# forgets to strip it produces an obviously wrong id rather than a subtly wrong
# result (a plausible-looking in-range id would be far more dangerous).
PADDING_SENTINEL = -1

SIZING_FIXED_NLIST = "fixed-nlist"
SIZING_CONSTANT_SIZE = "constant-size"
SIZING_CONSTANT_SIZE_SPLIT_PAD = "constant-size-split-pad"

# Default when neither --k nor --cluster-size is given: reproduces the historical
# behaviour of this script byte-for-byte.
DEFAULT_NLIST = 4096


# ---------------------------------------------------------------------------
# Cluster sizing configuration
# ---------------------------------------------------------------------------

def resolve_cluster_config(
    n_vectors: int,
    k: Optional[int] = None,
    cluster_size: Optional[int] = None,
) -> dict:
    """
    Resolve the cluster-sizing parameterization.

    Exactly one of ``k`` (fixed cluster count) or ``cluster_size`` (constant
    cluster size) may be given. If neither is given, falls back to the legacy
    default of K=DEFAULT_NLIST clusters with variable sizes.

    Args:
        n_vectors: Number of database vectors N. Only known after loading, which
            is why this cannot be resolved at argparse time.
        k: Legacy fixed cluster count.
        cluster_size: Constant cluster size n; nlist is derived as ceil(N / n).

    Returns:
        Dict with keys sizing_mode, nlist, cluster_size, total_slots,
        n_padded_slots.

    Raises:
        ValueError: If both k and cluster_size are given, or either is invalid.
    """
    if k is not None and cluster_size is not None:
        raise ValueError(
            "--k and --cluster-size are mutually exclusive: --k fixes the number "
            "of clusters (variable sizes), --cluster-size fixes the size "
            "(derived cluster count). Pick one."
        )

    if n_vectors <= 0:
        raise ValueError(f"n_vectors must be positive, got {n_vectors}")

    if cluster_size is not None:
        if cluster_size < 1:
            raise ValueError(f"--cluster-size must be >= 1, got {cluster_size}")
        if cluster_size > n_vectors:
            raise ValueError(
                f"--cluster-size {cluster_size} exceeds the database size "
                f"{n_vectors}; that would produce a single partly-empty cluster."
            )
        nlist = int(ceil(n_vectors / cluster_size))
        total_slots = nlist * cluster_size
        return {
            "sizing_mode": SIZING_CONSTANT_SIZE,
            "nlist": nlist,
            "cluster_size": int(cluster_size),
            "total_slots": total_slots,
            "n_padded_slots": total_slots - n_vectors,
        }

    nlist = DEFAULT_NLIST if k is None else int(k)
    if nlist < 1:
        raise ValueError(f"--k must be >= 1, got {nlist}")
    if nlist > n_vectors:
        raise ValueError(
            f"--k {nlist} exceeds the database size {n_vectors}; k-means cannot "
            f"produce more non-empty clusters than there are vectors."
        )
    return {
        "sizing_mode": SIZING_FIXED_NLIST,
        "nlist": nlist,
        "cluster_size": None,
        "total_slots": n_vectors,
        "n_padded_slots": 0,
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_ivf_metadata(output_dir: Path, metadata: dict) -> Path:
    """
    Write ivf_metadata.json, stamping the schema version.

    Args:
        output_dir: Directory to write into (created if absent).
        metadata: Metadata dict; see the module docstring for expected keys.

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"version": METADATA_VERSION, **metadata}
    path = output_dir / METADATA_FILENAME
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=_json_default)
    return path


def _json_default(obj):
    """Coerce numpy scalars/arrays that sneak into metadata dicts."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def read_ivf_metadata(path: Path) -> Optional[dict]:
    """
    Read ivf_metadata.json if present.

    Args:
        path: Either the metadata file itself or a directory containing it.

    Returns:
        The metadata dict, or None if there is no metadata file (a pre-metadata
        artifact such as the shipped prototype/data/lists.json).
    """
    path = Path(path)
    if path.is_dir():
        path = path / METADATA_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Slot array <-> lists.json
# ---------------------------------------------------------------------------

def pack_slots(
    assign: np.ndarray,
    order_key: np.ndarray,
    nlist: int,
    cluster_size: int,
) -> np.ndarray:
    """
    Build the canonical (nlist, cluster_size) int32 slot array.

    Members of each cluster are placed in ascending ``order_key`` (normally the
    distance to the assigned centroid), so slots[c, 0] is the tightest member.
    That ordering is deterministic and lets a consumer fetch a prefix of a
    cluster and still get its best candidates.

    Args:
        assign: (N,) cluster id per vector. Must satisfy the capacity constraint.
        order_key: (N,) sort key within a cluster, ascending.
        nlist: Number of clusters.
        cluster_size: Constant cluster size n.

    Returns:
        (nlist, cluster_size) int32 array, unused slots = PADDING_SENTINEL.

    Raises:
        ValueError: If any cluster holds more than cluster_size members.
    """
    assign = np.asarray(assign, dtype=np.int64)
    order_key = np.asarray(order_key, dtype=np.float64)

    counts = np.bincount(assign, minlength=nlist)
    if counts.max(initial=0) > cluster_size:
        over = int(np.argmax(counts))
        raise ValueError(
            f"cluster {over} holds {counts[over]} members, exceeding the "
            f"capacity {cluster_size}; the assignment is not capacity-feasible"
        )

    slots = np.full((nlist, cluster_size), PADDING_SENTINEL, dtype=np.int32)
    # Sort by (cluster, order_key) so a single pass fills each row in order.
    perm = np.lexsort((order_key, assign))
    sorted_assign = assign[perm]
    # Position of each element within its cluster run.
    starts = np.searchsorted(sorted_assign, np.arange(nlist), side="left")
    within = np.arange(len(perm)) - starts[sorted_assign]
    slots[sorted_assign, within] = perm.astype(np.int32)
    return slots


def slots_to_lists_json(slots: np.ndarray, style: str = "padded") -> Dict[str, List[int]]:
    """
    Render a slot array as the historical lists.json mapping.

    Args:
        slots: (nlist, n) int32 slot array.
        style: "padded" keeps PADDING_SENTINEL so every value has length n and
            the schema change is loudly visible; "trimmed" strips it for a
            consumer that genuinely cannot tolerate the sentinel.

    Returns:
        {"<cluster_id>": [vector ids...]}
    """
    if style not in ("padded", "trimmed"):
        raise ValueError(f"unknown lists.json style: {style!r}")
    out: Dict[str, List[int]] = {}
    for cid, row in enumerate(slots):
        ids = row.tolist()
        if style == "trimmed":
            ids = [i for i in ids if i != PADDING_SENTINEL]
        out[str(cid)] = ids
    return out


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

@dataclass
class Clustering:
    """
    A loaded clustering, uniform across both parameterizations.

    Attributes:
        nlist: Number of clusters.
        cluster_size: Constant size n, or None for a legacy variable-size run.
        n_vectors: Number of real (non-padding) vector ids.
        sizing_mode: One of the SIZING_* constants.
        slots: (nlist, n) int32 array in constant-size mode, else None.
        lists: Ragged {cid: [ids]} view, sentinel already stripped.
        centroids: (nlist, dim) float32, or None if not loaded.
        metadata: The raw ivf_metadata.json dict, or {} for pre-metadata data.
    """

    nlist: int
    cluster_size: Optional[int]
    n_vectors: int
    sizing_mode: str
    lists: Dict[int, List[int]]
    slots: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_constant_size(self) -> bool:
        """Whether every cluster holds exactly ``cluster_size`` slots."""
        return self.cluster_size is not None

    def members(self, cluster_id) -> List[int]:
        """
        Vector ids in a cluster, with padding sentinels removed.

        Args:
            cluster_id: Cluster id; accepts int or str for the convenience of
                call sites that previously indexed lists.json with a string.

        Returns:
            List of global vector ids, possibly empty.
        """
        return self.lists.get(int(cluster_id), [])

    def slot_address(self, cluster_id: int, index_in_cluster: int) -> int:
        """
        Flat slot address of a (cluster, offset) pair.

        Only meaningful in constant-size mode, where the address arithmetic is
        positional. Raises otherwise rather than returning a wrong answer.
        """
        if not self.is_constant_size:
            raise ValueError(
                "slot_address requires constant-size mode; a variable-size "
                "clustering has no positional address arithmetic"
            )
        return int(cluster_id) * self.cluster_size + int(index_in_cluster)

    def size_stats(self) -> dict:
        """Summary statistics of real (unpadded) cluster occupancy."""
        sizes = np.array([len(self.lists.get(c, [])) for c in range(self.nlist)])
        return {
            "min": int(sizes.min()) if len(sizes) else 0,
            "max": int(sizes.max()) if len(sizes) else 0,
            "mean": float(sizes.mean()) if len(sizes) else 0.0,
            "median": float(np.median(sizes)) if len(sizes) else 0.0,
            "n_empty": int((sizes == 0).sum()),
            "total": int(sizes.sum()),
        }


def load_clustering(path, load_centroids: bool = False) -> Clustering:
    """
    Load a clustering from either parameterization.

    Accepts an output directory (preferred -- picks up ivf_metadata.json and the
    canonical cluster_slots.npy) or a direct path to a lists.json for
    pre-metadata artifacts such as the shipped prototype/data/lists.json.

    Args:
        path: Output directory, or a lists.json file.
        load_centroids: Also load centroids.npy if present.

    Returns:
        A Clustering.

    Raises:
        FileNotFoundError: If no cluster assignment can be found.
    """
    path = Path(path)

    if path.is_dir():
        directory = path
        lists_path = directory / LISTS_FILENAME
    else:
        directory = path.parent
        lists_path = path

    metadata = read_ivf_metadata(directory) or {}
    slots_path = directory / SLOTS_FILENAME

    slots = None
    if slots_path.exists():
        slots = np.load(slots_path)
        if slots.ndim != 2:
            raise ValueError(
                f"{slots_path} must be 2-D (nlist, cluster_size), got shape {slots.shape}"
            )
        lists = {
            cid: [int(v) for v in row if v != PADDING_SENTINEL]
            for cid, row in enumerate(slots)
        }
        nlist, cluster_size = int(slots.shape[0]), int(slots.shape[1])
    elif lists_path.exists():
        with open(lists_path) as f:
            raw = json.load(f)
        lists = {
            int(cid): [int(v) for v in ids if v != PADDING_SENTINEL]
            for cid, ids in raw.items()
        }
        nlist = max(lists) + 1 if lists else 0
        # Trust metadata over inference. A variable-size run can coincidentally
        # produce equal-length lists, so length uniformity is NOT evidence of
        # constant-size mode.
        cluster_size = metadata.get("cluster_size")
        if cluster_size is not None:
            cluster_size = int(cluster_size)
            slots = np.full((nlist, cluster_size), PADDING_SENTINEL, dtype=np.int32)
            for cid, ids in lists.items():
                slots[cid, : len(ids)] = ids
    else:
        raise FileNotFoundError(
            f"no clustering found at {path}: expected {SLOTS_FILENAME} or "
            f"{LISTS_FILENAME}"
        )

    centroids = None
    if load_centroids:
        centroids_path = directory / CENTROIDS_FILENAME
        if centroids_path.exists():
            centroids = np.load(centroids_path)

    n_vectors = sum(len(v) for v in lists.values())
    sizing_mode = metadata.get(
        "sizing_mode",
        SIZING_CONSTANT_SIZE if cluster_size is not None else SIZING_FIXED_NLIST,
    )

    return Clustering(
        nlist=int(metadata.get("nlist", nlist)),
        cluster_size=cluster_size,
        n_vectors=n_vectors,
        sizing_mode=sizing_mode,
        lists=lists,
        slots=slots,
        centroids=centroids,
        metadata=metadata,
    )


def validate_slots(slots: np.ndarray, n_vectors: int) -> dict:
    """
    Check the invariants the whole constant-size story rests on.

    Every real vector id must appear exactly once, unused slots must all be the
    sentinel, and each row must be exactly ``cluster_size`` wide. These are
    asserted rather than reported because a violation makes every downstream
    number meaningless.

    Args:
        slots: (nlist, n) int32 slot array.
        n_vectors: Expected number of real vector ids.

    Returns:
        Dict of shape/occupancy facts.

    Raises:
        AssertionError: If any invariant is violated.
    """
    nlist, cluster_size = slots.shape
    flat = slots.reshape(-1)
    real = flat[flat != PADDING_SENTINEL]

    assert len(real) == n_vectors, (
        f"slot array holds {len(real)} real ids, expected {n_vectors}"
    )
    assert real.min() >= 0, f"negative non-sentinel id {real.min()} in slot array"
    unique = np.unique(real)
    assert len(unique) == n_vectors, (
        f"slot array holds {n_vectors - len(unique)} duplicate vector ids"
    )
    assert unique[0] == 0 and unique[-1] == n_vectors - 1, (
        f"vector ids are not exactly 0..{n_vectors - 1} "
        f"(got {unique[0]}..{unique[-1]})"
    )

    per_row = (slots != PADDING_SENTINEL).sum(axis=1)
    return {
        "nlist": int(nlist),
        "cluster_size": int(cluster_size),
        "total_slots": int(nlist * cluster_size),
        "n_real": int(len(real)),
        "n_padded_slots": int(nlist * cluster_size - len(real)),
        "n_full_clusters": int((per_row == cluster_size).sum()),
        "padded_cluster_ids": [int(c) for c in np.flatnonzero(per_row < cluster_size)],
    }
