#!/usr/bin/env python3
"""
Prepare and validate FHE inputs for a clustering, whatever its cluster count.

## Does the existing FHE code work with equal-size clusters?

Yes, unchanged. Verified by reading the sources, not by running them (OpenFHE is
not installed in every dev environment; see `verify_kernel_oracle` for the part
that *is* checkable without it):

  * `openfhe_compute_distances_centroid_batched.cpp:87` takes the centroid count
    from the file (`centroids.size()`), never from a constant.
  * `:102` computes the batch count with a ceiling division, and `:121` clamps the
    last batch with `std::min(...)`, so a count that is not a multiple of
    `centroids_per_ciphertext` is handled -- relevant because 508 centroids is
    63.5 ciphertexts.
  * `n_centroids` is carried through metadata everywhere it is consumed
    (`openfhe_batched_workload.cpp:658`, `openfhe_decrypt_topk_centroid_batched.cpp:62`).
  * `prototype/fhe_backend.py:404-406` derives `n_centroids` from `len(centroids)`,
    and `_convert_centroids_to_text` (`:210`) exports whatever shape it is given.
  * The CKKS slot constraint is `centroids_per_ciphertext * padded_dim <=
    poly_modulus_degree / 2`, which involves the *packing* parameters only. The
    cluster count does not enter it.

So going from 4096 to 508 centroids requires **no code change** -- only
regenerating `centroids.txt`, which is what this script does. The centroid stage
gets proportionally cheaper: ceil(508/8) = 64 ciphertexts instead of 512.

## What equal-size clustering does NOT change

It is worth being precise, because it is easy to over-claim. The FHE layer in this
project computes exactly one thing: distances from the encrypted query to the
plaintext centroids, which the client decrypts to learn its top-p clusters. The
per-cluster kNN that follows is done **client-side in plaintext**, on records the
client just fetched over PIR (`plaintext_rag_pipeline.py: per_cluster_knn`). There
is no homomorphic per-cluster kNN in the codebase, so uniform cluster size does
not enable a new FHE computation here. What it buys is:

  1. a cheaper centroid stage, because nlist shrinks (existing code);
  2. uniform PIR records, so per-query cost stops revealing cluster occupancy;
  3. a fixed candidate count p*n, so the client-side rerank has a static shape.

If a homomorphic per-cluster kNN is ever added, the kernel in
`openfhe_compute_distances_centroid_batched.cpp` is already the right shape for it
-- it is generic "one encrypted query against M plaintext vectors" and does not
care that M happens to be a centroid count. That is a design note, not something
implemented here.

Usage:
    python fhe_prepare.py --ivf-dir ivf_output_n128 --out-dir openfhe_inputs
    python fhe_prepare.py --ivf-dir ivf_output_n128 --self-test
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

import ivf_io

# Verified layouts from openfhe_core/README.md:183-196.
DEFAULT_PADDED_DIM = 1024
DEFAULT_CENTROIDS_PER_CIPHERTEXT = 8
DEFAULT_POLY_MODULUS_DEGREE = 16384


def validate_ckks_layout(
    n_centroids: int,
    dim: int,
    padded_dim: int = DEFAULT_PADDED_DIM,
    centroids_per_ciphertext: int = DEFAULT_CENTROIDS_PER_CIPHERTEXT,
    poly_modulus_degree: int = DEFAULT_POLY_MODULUS_DEGREE,
) -> dict:
    """
    Check the CKKS packing parameters and report the resulting ciphertext cost.

    Args:
        n_centroids: Number of centroids to compare against, i.e. nlist.
        dim: Embedding dimension.
        padded_dim: Slot stride per vector; must be >= dim and a power of two.
        centroids_per_ciphertext: Vectors packed per ciphertext.
        poly_modulus_degree: Ring dimension; slots = degree / 2.

    Returns:
        Dict of the validated layout plus n_ciphertexts and last_batch_size.

    Raises:
        ValueError: If the layout does not fit the slot budget or dim > padded_dim.
    """
    slots = poly_modulus_degree // 2
    if dim > padded_dim:
        raise ValueError(f"dim {dim} exceeds padded_dim {padded_dim}")
    if padded_dim & (padded_dim - 1):
        raise ValueError(
            f"padded_dim {padded_dim} must be a power of two: the kernel's "
            f"rotation-sum halves the stride each step"
        )
    if centroids_per_ciphertext * padded_dim > slots:
        raise ValueError(
            f"centroids_per_ciphertext * padded_dim = "
            f"{centroids_per_ciphertext * padded_dim} exceeds the "
            f"{slots} available slots (poly_modulus_degree/2). Reduce "
            f"centroids_per_ciphertext to {slots // padded_dim} or raise the degree."
        )

    n_ciphertexts = -(-n_centroids // centroids_per_ciphertext)
    last = n_centroids - (n_ciphertexts - 1) * centroids_per_ciphertext
    return {
        "n_centroids": int(n_centroids),
        "dim": int(dim),
        "padded_dim": int(padded_dim),
        "centroids_per_ciphertext": int(centroids_per_ciphertext),
        "poly_modulus_degree": int(poly_modulus_degree),
        "slots_per_ciphertext": int(slots),
        "n_ciphertexts": int(n_ciphertexts),
        "last_batch_size": int(last),
        "last_batch_is_partial": bool(last != centroids_per_ciphertext),
        # The rotation-sum needs these indices to exist in the eval-key set.
        "required_rotation_indices": [
            int(step * centroids_per_ciphertext)
            for step in _powers_of_two_below(padded_dim)
        ],
    }


def _powers_of_two_below(limit: int):
    """Yield 1, 2, 4, ... < limit, matching the kernel's rotation-sum ladder."""
    step = 1
    while step < limit:
        yield step
        step <<= 1


def export_centroids_txt(centroids: np.ndarray, out_path: Path) -> Path:
    """
    Write centroids in the whitespace-delimited format the OpenFHE binaries read.

    Matches prototype/fhe_backend.py::_convert_centroids_to_text so the two paths
    produce byte-identical files.

    Args:
        centroids: (nlist, dim) float array.
        out_path: Destination .txt path.

    Returns:
        out_path.

    Raises:
        ValueError: If the array is not 2-D or holds non-finite values (the C++
            reader rejects those, and it is much cheaper to catch it here).
    """
    if centroids.ndim != 2:
        raise ValueError(f"expected 2-D centroids, got shape {centroids.shape}")
    if not np.isfinite(centroids).all():
        bad = int((~np.isfinite(centroids)).sum())
        raise ValueError(
            f"{bad} non-finite centroid values; the OpenFHE reader requires finite "
            f"floats. A zero-vector fallback in the exporter usually means a "
            f"centroid failed to reconstruct."
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in centroids.astype(np.float64):
            f.write(" ".join(f"{float(v):.12g}" for v in row))
            f.write("\n")
    return out_path


def kernel_oracle(
    query: np.ndarray,
    centroids: np.ndarray,
    padded_dim: int = DEFAULT_PADDED_DIM,
    centroids_per_ciphertext: int = DEFAULT_CENTROIDS_PER_CIPHERTEXT,
) -> np.ndarray:
    """
    Plaintext reproduction of the batched centroid-distance kernel.

    Mirrors the exact slot arithmetic of
    ``openfhe_compute_distances_centroid_batched.cpp``:

      * query packed at ``slot[d * cpc + b] = q[d]`` for every lane b (:70-74 of
        openfhe_encrypt_query_centroid_batched.cpp);
      * centroids packed at ``slot[d * cpc + b] = c_b[d]`` (:130-133);
      * elementwise multiply, then a rotation-sum with stride ``step * cpc`` for
        step = 1, 2, 4, ... < padded_dim (:141-145), which lands the dot product
        of lane b in slot b;
      * ``distance = (||q||^2 + ||c_b||^2) - 2 * dot`` (:147-151).

    This is what makes the FHE path checkable without OpenFHE: if this agrees with
    a direct squared-L2 computation, the packing and rotation schedule are right,
    and the only remaining risk is CKKS approximation error.

    Args:
        query: (dim,) float64.
        centroids: (M, dim) float64.
        padded_dim: Slot stride per vector.
        centroids_per_ciphertext: Lanes per ciphertext.

    Returns:
        (M,) squared L2 distances, assembled from the per-batch slot outputs.
    """
    dim = len(query)
    cpc = centroids_per_ciphertext
    slots = padded_dim * cpc
    n_batches = -(-len(centroids) // cpc)

    packed_query = np.zeros(slots)
    for d in range(dim):
        packed_query[d * cpc: d * cpc + cpc] = query[d]
    q_norm = float(query @ query)

    out = np.empty(len(centroids))
    for batch in range(n_batches):
        start = batch * cpc
        in_batch = min(cpc, len(centroids) - start)

        packed_c = np.zeros(slots)
        packed_norm = np.zeros(slots)
        for b in range(in_batch):
            c = centroids[start + b]
            packed_norm[b] = float(c @ c)
            for d in range(dim):
                packed_c[d * cpc + b] = c[d]

        dot = packed_query * packed_c
        step = 1
        while step < padded_dim:
            # EvalAtIndex(x, k) is a left rotation: slot i takes from slot i+k.
            dot = dot + np.roll(dot, -step * cpc)
            step <<= 1

        distance = (q_norm + packed_norm) - 2.0 * dot
        out[start: start + in_batch] = distance[:in_batch]
    return out


def verify_kernel_oracle(dim: int = 768, n_centroids: int = 508, seed: int = 0) -> dict:
    """
    Check the oracle against a direct squared-L2 computation.

    Exercises a centroid count that is deliberately *not* a multiple of the lane
    count (508 = 63 full batches + 1 lane of 4), which is exactly the case the
    partial-batch handling has to get right.

    Args:
        dim: Embedding dimension.
        n_centroids: Number of centroids.
        seed: RNG seed.

    Returns:
        Dict with max_abs_error and the layout used.

    Raises:
        AssertionError: If the oracle disagrees with the direct computation.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=dim)
    q /= np.linalg.norm(q)
    c = rng.normal(size=(n_centroids, dim))
    c /= np.linalg.norm(c, axis=1, keepdims=True)

    got = kernel_oracle(q, c)
    want = np.einsum("ij,ij->i", c - q, c - q)
    err = float(np.abs(got - want).max())
    assert err < 1e-9, f"kernel oracle disagrees with direct squared L2 by {err:.3e}"

    layout = validate_ckks_layout(n_centroids, dim)
    return {"max_abs_error": err, "layout": layout}


def prepare(ivf_dir: Path, out_dir: Path, **layout_kwargs) -> dict:
    """
    Export centroids.txt and an FHE layout manifest for a clustering directory.

    Args:
        ivf_dir: Output directory from train_ivf.py.
        out_dir: Where to write centroids.txt and fhe_layout.json.
        **layout_kwargs: Overrides for validate_ckks_layout.

    Returns:
        The layout manifest.
    """
    ivf_dir, out_dir = Path(ivf_dir), Path(out_dir)
    clustering = ivf_io.load_clustering(ivf_dir, load_centroids=True)
    if clustering.centroids is None:
        raise FileNotFoundError(f"no {ivf_io.CENTROIDS_FILENAME} in {ivf_dir}")

    centroids = clustering.centroids
    layout = validate_ckks_layout(len(centroids), centroids.shape[1], **layout_kwargs)

    txt = export_centroids_txt(centroids, out_dir / "centroids.txt")
    manifest = {
        **layout,
        "source_ivf_dir": str(ivf_dir),
        "centroids_txt": str(txt),
        "sizing_mode": clustering.sizing_mode,
        "cluster_size": clustering.cluster_size,
        "centroid_space": clustering.metadata.get("centroid_space", "raw"),
    }
    (out_dir / "fhe_layout.json").write_text(json.dumps(manifest, indent=2))

    print(f"Exported {len(centroids):,} centroids (dim {centroids.shape[1]}) -> {txt}")
    print(f"  sizing mode          : {clustering.sizing_mode}"
          + (f" (n={clustering.cluster_size})" if clustering.is_constant_size else ""))
    print(f"  centroid stage cost  : {layout['n_ciphertexts']:,} ciphertexts "
          f"({layout['centroids_per_ciphertext']} centroids each)")
    if layout["last_batch_is_partial"]:
        print(f"  last batch is partial: {layout['last_batch_size']} of "
              f"{layout['centroids_per_ciphertext']} lanes -- handled by the "
              f"std::min clamp in the kernel")
    if manifest["centroid_space"] != "raw":
        print(f"  WARNING: centroids are in the {manifest['centroid_space']} space. "
              f"The query must be rotated by opq_transform.npy before encryption, "
              f"or every distance will be wrong.")
    print(f"  wrote {out_dir / 'fhe_layout.json'}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export and validate FHE centroid inputs for any cluster count"
    )
    p.add_argument("--ivf-dir", type=str, help="Output directory from train_ivf.py")
    p.add_argument("--out-dir", type=str, default="./openfhe_inputs")
    p.add_argument("--padded-dim", type=int, default=DEFAULT_PADDED_DIM)
    p.add_argument("--centroids-per-ciphertext", type=int,
                   default=DEFAULT_CENTROIDS_PER_CIPHERTEXT)
    p.add_argument("--poly-modulus-degree", type=int, default=DEFAULT_POLY_MODULUS_DEGREE)
    p.add_argument("--self-test", action="store_true",
                   help="Verify the plaintext kernel oracle and exit")
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        for n in (4096, 508, 1000, 2032):
            r = verify_kernel_oracle(n_centroids=n)
            L = r["layout"]
            print(f"  n_centroids={n:<5d} ciphertexts={L['n_ciphertexts']:<5d} "
                  f"last_batch={L['last_batch_size']:<2d} "
                  f"partial={str(L['last_batch_is_partial']):<5s} "
                  f"max_err={r['max_abs_error']:.2e}")
        print("kernel oracle matches direct squared L2 at every centroid count")
        return
    if not args.ivf_dir:
        raise SystemExit("error: --ivf-dir is required (or use --self-test)")
    prepare(
        Path(args.ivf_dir), Path(args.out_dir),
        padded_dim=args.padded_dim,
        centroids_per_ciphertext=args.centroids_per_ciphertext,
        poly_modulus_degree=args.poly_modulus_degree,
    )


if __name__ == "__main__":
    main()
