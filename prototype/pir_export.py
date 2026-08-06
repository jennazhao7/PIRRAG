#!/usr/bin/env python3
"""
Lay out the PIR database in cluster-major order so a PIR address is arithmetic.

## The problem this solves

The retrieval pipeline addresses records as ``(cluster_id, index_in_cluster)``:
the FHE stage tells the client its top-p clusters, and PIR then fetches records
from inside them. But the PIR database is not built that way. Both exporters
write one record per vector in **global id order** --
``faiss_processor.py:76`` iterates ``range(num_vectors)``, and
``pickle_processor.py`` walks the docstore -- so nothing about a record's position
reflects which cluster it belongs to.

The gap has been bridged so far by carrying ``lists.json`` and translating
``(cluster_id, index) -> global id`` before the fetch. That costs a 768 KB side
table on the client, and it means the client holds the full cluster map of the
corpus.

Reordering the database into cluster-major order removes the translation
entirely:

    constant-size clustering:   address = cluster_id * cluster_size + index
    variable-size clustering:   address = offsets[cluster_id] + index

In the constant-size case the address is pure arithmetic and the client needs
nothing beyond the single integer ``cluster_size``. In the variable-size case it
needs an ``offsets`` array of ``nlist + 1`` integers -- 32 KB at nlist=4096
against 768 KB for ``lists.json``, a 24x reduction, and unlike ``lists.json`` it
reveals only cluster *sizes*, not which vector sits where.

Both layouts are supported on purpose: switching to constant-size clusters should
not be a precondition for cleaning up the PIR handoff.

## What it writes

    <output>                 the reordered database, same format as the input
    <output>.manifest.json   the addressing rule, row count, and offsets

## Usage

    # Reorder the document-text database (chr(31)-separated fixed-width records)
    python pir_export.py --clustering ../wiki-rag/ivf_output_n128 \\
        --records ../uniform_index.txt --output pir_db_n128.txt --verify

    # Reorder the vector database
    python pir_export.py --clustering ../wiki-rag/ivf_output_n128 \\
        --vectors-npy vectors.npy --output pir_vectors_n128.csv --verify
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_WIKI_RAG_DIR = Path(__file__).resolve().parents[1] / "wiki-rag"
if str(_WIKI_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_WIKI_RAG_DIR))

import ivf_io

# pickle_processor.py joins its fixed-width records with the ASCII unit separator.
RECORD_SEPARATOR = chr(31)


def build_row_order(clustering) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Compute the cluster-major row order and the addressing table.

    Args:
        clustering: An ivf_io.Clustering.

    Returns:
        (row_to_vector, offsets, width) where

        * ``row_to_vector[r]`` is the global vector id at PIR row ``r``, or -1 for
          a padding row;
        * ``offsets`` is None for constant-size (the address is arithmetic) or an
          ``(nlist + 1,)`` int64 array of row starts for variable-size;
        * ``width`` is the cluster width for constant-size, else -1.
    """
    nlist = clustering.nlist

    if clustering.is_constant_size:
        # Row-major traversal of the (nlist, n) slot array *is* the layout, so
        # address = cid * n + idx holds by construction.
        width = int(clustering.cluster_size)
        row_to_vector = clustering.slots.reshape(-1).astype(np.int64).copy()
        return row_to_vector, None, width

    # Variable-size: pack clusters back to back and record where each begins.
    sizes = np.array([len(clustering.members(c)) for c in range(nlist)], dtype=np.int64)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    row_to_vector = np.full(int(offsets[-1]), -1, dtype=np.int64)
    for c in range(nlist):
        ids = clustering.members(c)
        if ids:
            row_to_vector[offsets[c]: offsets[c] + len(ids)] = ids
    return row_to_vector, offsets, -1


def read_records(path: Path) -> Tuple[List[str], int]:
    """
    Read a chr(31)-separated fixed-width record file.

    Args:
        path: Path to the database (e.g. uniform_index.txt).

    Returns:
        (records, width). Every record is expected to be the same width, since
        the whole point of that file is uniformity; a mismatch is reported rather
        than silently reordered.

    Raises:
        ValueError: If the records are not all the same width.
    """
    text = Path(path).read_text()
    records = text.split(RECORD_SEPARATOR)
    # The writer appends a trailing separator, leaving an empty final element.
    if records and records[-1] == "":
        records.pop()
    widths = {len(r) for r in records}
    if len(widths) != 1:
        raise ValueError(
            f"{path} holds records of {len(widths)} different widths "
            f"(min {min(widths)}, max {max(widths)}). PIR needs fixed-width "
            f"records; regenerate it with pickle_processor.py."
        )
    return records, widths.pop()


def write_records(records: List[str], path: Path) -> None:
    """Write records in the same chr(31)-separated form pickle_processor.py uses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(r)
            f.write(RECORD_SEPARATOR)


def reorder_records(
    records: List[str], row_to_vector: np.ndarray, width: int
) -> List[str]:
    """
    Emit records in cluster-major order, blank-filling padding rows.

    Args:
        records: Source records indexed by global vector id.
        row_to_vector: Output of build_row_order.
        width: Record width, used to build the padding record.

    Returns:
        Reordered records, one per row.

    Raises:
        IndexError: If the clustering references a vector id the database lacks.
    """
    n_src = len(records)
    bad = row_to_vector[(row_to_vector >= n_src)]
    if len(bad):
        raise IndexError(
            f"the clustering references vector id {int(bad.max())} but the "
            f"database holds only {n_src} records. The clustering and the "
            f"database were built from different corpora -- check whether the "
            f"clustering was produced in holdout recall mode (see id_space in "
            f"ivf_metadata.json)."
        )
    # A padding row must still occupy a full record so every row is the same
    # width; the client discards it on the sentinel in the manifest.
    padding = " " * width
    return [padding if v < 0 else records[int(v)] for v in row_to_vector]


def reorder_vectors(vectors: np.ndarray, row_to_vector: np.ndarray) -> np.ndarray:
    """
    Emit vectors in cluster-major order, zero-filling padding rows.

    Args:
        vectors: (N, dim) source vectors indexed by global vector id.
        row_to_vector: Output of build_row_order.

    Returns:
        (n_rows, dim) reordered array.
    """
    if (row_to_vector >= len(vectors)).any():
        raise IndexError(
            f"the clustering references vector id "
            f"{int(row_to_vector.max())} but only {len(vectors)} vectors were "
            f"given; see id_space in ivf_metadata.json"
        )
    out = np.zeros((len(row_to_vector), vectors.shape[1]), dtype=vectors.dtype)
    real = row_to_vector >= 0
    out[real] = vectors[row_to_vector[real]]
    return out


def verify_addressing(clustering, row_to_vector: np.ndarray, offsets, width: int,
                      n_samples: int = 2000, seed: int = 0) -> dict:
    """
    Check that the documented address formula lands on the right vector.

    This is the property the whole handoff depends on, so it is checked directly
    rather than argued: for randomly chosen (cluster, index) pairs, the address
    computed by the formula the client will use must resolve to the vector id the
    clustering assigns to that position.

    Args:
        clustering: The ivf_io.Clustering.
        row_to_vector: Output of build_row_order.
        offsets: Offsets table, or None for constant-size.
        width: Cluster width for constant-size, else -1.
        n_samples: Number of (cluster, index) pairs to test.
        seed: RNG seed.

    Returns:
        Dict with n_checked and the formula used.

    Raises:
        AssertionError: If any address resolves to the wrong vector.
    """
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(n_samples):
        c = int(rng.integers(0, clustering.nlist))
        ids = clustering.members(c)
        if not ids:
            continue
        i = int(rng.integers(0, len(ids)))
        addr = c * width + i if offsets is None else int(offsets[c]) + i
        got = int(row_to_vector[addr])
        assert got == ids[i], (
            f"address {addr} for (cluster {c}, index {i}) resolved to vector "
            f"{got}, expected {ids[i]}"
        )
        checked += 1

    # Every real vector must appear exactly once, and only padding rows are -1.
    real = row_to_vector[row_to_vector >= 0]
    assert len(np.unique(real)) == len(real), "a vector id appears at two rows"
    assert len(real) == clustering.n_vectors, (
        f"layout holds {len(real)} real rows, clustering has {clustering.n_vectors}"
    )
    return {
        "n_checked": checked,
        "formula": ("cluster_id * cluster_size + index" if offsets is None
                    else "offsets[cluster_id] + index"),
    }


def export(args) -> dict:
    """Reorder a PIR database into cluster-major order and write the manifest."""
    clustering = ivf_io.load_clustering(args.clustering)
    row_to_vector, offsets, width = build_row_order(clustering)

    print(f"Clustering : {clustering.sizing_mode}, {clustering.nlist:,} clusters, "
          f"{clustering.n_vectors:,} vectors")
    if clustering.is_constant_size:
        print(f"Addressing : row = cluster_id * {width} + index   (arithmetic, "
              f"no side table)")
    else:
        print(f"Addressing : row = offsets[cluster_id] + index   "
              f"({clustering.nlist + 1} offsets, "
              f"{(clustering.nlist + 1) * 8 / 1024:.1f} KB)")
    print(f"Rows       : {len(row_to_vector):,} "
          f"({int((row_to_vector < 0).sum()):,} padding)")

    # A holdout clustering's ids index the held-out subset, not the corpus, so
    # reordering a full-corpus database by them silently pairs each row with the
    # wrong payload. Refuse rather than warn: the output would look perfectly
    # well-formed and be wrong in a way no downstream check would catch.
    id_space = clustering.metadata.get("id_space", "database")
    if id_space != "database" and not args.allow_holdout_ids:
        raise SystemExit(
            f"error: this clustering has id_space={id_space!r}, meaning it was built "
            f"in holdout recall mode and its vector ids index a held-out subset "
            f"rather than the full corpus.\n"
            f"  Reordering a full-corpus database by those ids would pair rows with "
            f"the wrong payloads, and the result would look valid.\n"
            f"  Rebuild the clustering for deployment:\n"
            f"      python train_ivf.py ... --recall-mode perturbed\n"
            f"  Or pass --allow-holdout-ids if you are deliberately exporting a "
            f"measurement artifact and the payload is the same subset."
        )
    if id_space != "database":
        print(f"  NOTE: exporting a holdout clustering (id_space={id_space!r}) "
              f"because --allow-holdout-ids was given. Not deployable.")

    out_path = Path(args.output)
    if args.records:
        records, rec_width = read_records(Path(args.records))
        print(f"Source     : {len(records):,} records of width {rec_width} "
              f"from {args.records}")
        write_records(reorder_records(records, row_to_vector, rec_width), out_path)
        payload_width = rec_width
    else:
        vectors = np.load(args.vectors_npy) if args.vectors_npy else None
        if vectors is None:
            import faiss
            idx = faiss.read_index(str(args.faiss_index_file))
            vectors = idx.reconstruct_n(0, idx.ntotal)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        print(f"Source     : {vectors.shape} vectors")
        reordered = reorder_vectors(vectors, row_to_vector)
        if out_path.suffix == ".npy":
            np.save(out_path, reordered)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                for row in reordered:
                    f.write(",".join(f"{v:.9g}" for v in row))
                    f.write("\n")
        payload_width = int(vectors.shape[1])

    manifest = {
        "layout": "cluster-major",
        "sizing_mode": clustering.sizing_mode,
        "nlist": clustering.nlist,
        "cluster_size": clustering.cluster_size,
        "n_rows": int(len(row_to_vector)),
        "n_real_rows": int((row_to_vector >= 0).sum()),
        "n_padding_rows": int((row_to_vector < 0).sum()),
        "padding_sentinel": ivf_io.PADDING_SENTINEL,
        "address_formula": ("cluster_id * cluster_size + index" if offsets is None
                            else "offsets[cluster_id] + index"),
        "offsets": offsets.tolist() if offsets is not None else None,
        "record_width": payload_width,
        "id_space": id_space,
        "source_clustering": str(args.clustering),
        "note": (
            "Rows are in cluster-major order, so the row index IS the PIR "
            "address. The client does not need lists.json. Padding rows are "
            "blank/zero and must be discarded by the client."
        ),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote      : {out_path}")
    print(f"             {manifest_path}")

    if args.verify:
        res = verify_addressing(clustering, row_to_vector, offsets, width)
        print(f"Verified   : {res['n_checked']:,} random (cluster, index) pairs "
              f"resolve correctly via `{res['formula']}`")
        print(f"             every vector appears exactly once; "
              f"only padding rows are sentinel")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reorder a PIR database into cluster-major order"
    )
    p.add_argument("--clustering", required=True,
                   help="IVF output directory (or a lists.json) from train_ivf.py")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--records", help="chr(31)-separated fixed-width record file, "
                                      "e.g. uniform_index.txt")
    src.add_argument("--vectors-npy", help="(N, dim) float32 .npy of vectors")
    src.add_argument("--faiss-index-file", help="Raw index.faiss to read vectors from")
    p.add_argument("--output", required=True, help="Destination database path")
    p.add_argument("--verify", action="store_true",
                   help="Check the address formula against the clustering")
    p.add_argument("--allow-holdout-ids", action="store_true",
                   help="Permit exporting a clustering built in holdout recall "
                        "mode, whose ids index a held-out subset. Only correct if "
                        "the payload is that same subset.")
    return p


def main():
    export(build_parser().parse_args())


if __name__ == "__main__":
    main()
