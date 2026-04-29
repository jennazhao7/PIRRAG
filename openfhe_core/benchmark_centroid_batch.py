#!/usr/bin/env python3
"""Benchmark current OpenFHE centroid compute against centroid-batched compute."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def coeff_sizes_for_degree(degree: int) -> str:
    if degree == 16384:
        return "60,40,40,40,40,60"
    if degree == 32768:
        return "60,40,40,40,40,40,40,40,60"
    if degree == 65536:
        return "60,40,40,40,40,40,40,40,40,40,60"
    return "60,40,40,40,40,60"


def rotation_indices(centroids_per_ciphertext: int, padded_dim: int) -> str:
    values: List[str] = []
    step = 1
    while step < padded_dim:
        values.append(str(step * centroids_per_ciphertext))
        step <<= 1
    return ",".join(values)


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def first_rows(src: Path, dst: Path, n_rows: int, query_dim: int) -> int:
    kept = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line.split()) < query_dim:
                continue
            out.write(line + "\n")
            kept += 1
            if kept >= n_rows:
                break
    if kept == 0:
        raise RuntimeError(f"No usable centroid rows found in {src}")
    return kept


def write_first_query(centroids_file: Path, query_file: Path, query_dim: int) -> None:
    for line in centroids_file.read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) >= query_dim:
            query_file.write_text(" ".join(vals[:query_dim]) + "\n", encoding="utf-8")
            return
    raise RuntimeError(f"No query vector could be derived from {centroids_file}")


def read_topk(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark centroid-batched OpenFHE compute")
    parser.add_argument("--bin-dir", type=Path, default=Path("openfhe_core/build/bin"))
    parser.add_argument("--centroids-file", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("openfhe_core/centroid_batch_bench"))
    parser.add_argument("--poly-modulus-degree", type=int, default=16384)
    parser.add_argument("--query-dim", type=int, default=768)
    parser.add_argument("--padded-dim", type=int, default=1024)
    parser.add_argument("--centroids-per-ciphertext", type=int, default=8)
    parser.add_argument("--max-centroids", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    binaries = {
        "keygen": bin_dir / "openfhe_keygen",
        "encrypt": bin_dir / "openfhe_encrypt_query",
        "compute": bin_dir / "openfhe_compute_distances",
        "decrypt": bin_dir / "openfhe_decrypt_topk",
        "encrypt_cb": bin_dir / "openfhe_encrypt_query_centroid_batched",
        "compute_cb": bin_dir / "openfhe_compute_distances_centroid_batched",
        "decrypt_cb": bin_dir / "openfhe_decrypt_topk_centroid_batched",
    }
    for name, path in binaries.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} binary: {path}")

    centroids_subset = work_dir / "centroids_subset.txt"
    centroid_count = first_rows(
        args.centroids_file.resolve(),
        centroids_subset,
        args.max_centroids,
        args.query_dim,
    )
    query_file = work_dir / "query.txt"
    write_first_query(centroids_subset, query_file, args.query_dim)

    coeffs = coeff_sizes_for_degree(args.poly_modulus_degree)
    top_k = min(args.top_k, centroid_count)

    baseline = work_dir / "baseline"
    cb = work_dir / "centroid_batched"
    baseline_ctx = baseline / "context"
    baseline_enc = baseline / "encrypted_query"
    baseline_dist = baseline / "encrypted_distances"
    cb_ctx = cb / "context"
    cb_enc = cb / "encrypted_query"
    cb_dist = cb / "encrypted_distances"

    run_cmd([
        str(binaries["keygen"]),
        "--context-dir", str(baseline_ctx),
        "--poly-modulus-degree", str(args.poly_modulus_degree),
        "--coeff-mod-bit-sizes", coeffs,
    ])
    run_cmd([
        str(binaries["encrypt"]),
        "--context-dir", str(baseline_ctx),
        "--input-vector", str(query_file),
        "--output-dir", str(baseline_enc),
    ])
    t0 = time.perf_counter()
    run_cmd([
        str(binaries["compute"]),
        "--context-dir", str(baseline_ctx),
        "--centroids-file", str(centroids_subset),
        "--encrypted-query", str(baseline_enc / "encrypted_query.bin"),
        "--encrypted-norm", str(baseline_enc / "encrypted_norm_squared.bin"),
        "--output-dir", str(baseline_dist),
        "--num-threads", str(args.num_threads),
        "--batch-size", str(args.batch_size),
    ])
    baseline_compute = time.perf_counter() - t0
    run_cmd([
        str(binaries["decrypt"]),
        "--context-dir", str(baseline_ctx),
        "--encrypted-distances-dir", str(baseline_dist),
        "--top-k", str(top_k),
        "--output-json", str(baseline_dist / "top_k_results.json"),
    ])

    run_cmd([
        str(binaries["keygen"]),
        "--context-dir", str(cb_ctx),
        "--poly-modulus-degree", str(args.poly_modulus_degree),
        "--coeff-mod-bit-sizes", coeffs,
        "--rotation-indices", rotation_indices(args.centroids_per_ciphertext, args.padded_dim),
    ])
    run_cmd([
        str(binaries["encrypt_cb"]),
        "--context-dir", str(cb_ctx),
        "--input-vector", str(query_file),
        "--output-dir", str(cb_enc),
        "--centroids-per-ciphertext", str(args.centroids_per_ciphertext),
        "--padded-dim", str(args.padded_dim),
    ])
    t0 = time.perf_counter()
    run_cmd([
        str(binaries["compute_cb"]),
        "--context-dir", str(cb_ctx),
        "--centroids-file", str(centroids_subset),
        "--encrypted-query", str(cb_enc / "encrypted_query_centroid_batched.bin"),
        "--encrypted-norm", str(cb_enc / "encrypted_norm_centroid_batched.bin"),
        "--output-dir", str(cb_dist),
        "--centroids-per-ciphertext", str(args.centroids_per_ciphertext),
        "--padded-dim", str(args.padded_dim),
        "--num-threads", str(args.num_threads),
        "--batch-size", str(args.batch_size),
    ])
    cb_compute = time.perf_counter() - t0
    run_cmd([
        str(binaries["decrypt_cb"]),
        "--context-dir", str(cb_ctx),
        "--encrypted-distances-dir", str(cb_dist),
        "--top-k", str(top_k),
        "--output-json", str(cb_dist / "top_k_results.json"),
    ])

    baseline_topk = read_topk(baseline_dist / "top_k_results.json")
    cb_topk = read_topk(cb_dist / "top_k_results.json")
    baseline_indices = baseline_topk["centroid_indices"]
    cb_indices = cb_topk["centroid_indices"]
    topk_overlap = len(set(baseline_indices).intersection(cb_indices))

    result = {
        "poly_modulus_degree": args.poly_modulus_degree,
        "query_dim": args.query_dim,
        "padded_dim": args.padded_dim,
        "centroid_count": centroid_count,
        "centroids_per_ciphertext": args.centroids_per_ciphertext,
        "num_threads": args.num_threads,
        "batch_size": args.batch_size,
        "top_k": top_k,
        "baseline_compute_seconds": baseline_compute,
        "centroid_batched_compute_seconds": cb_compute,
        "speedup": baseline_compute / cb_compute if cb_compute > 0 else 0.0,
        "baseline_output_ciphertexts": centroid_count,
        "centroid_batched_output_ciphertexts": (
            centroid_count + args.centroids_per_ciphertext - 1
        ) // args.centroids_per_ciphertext,
        "topk_overlap": topk_overlap,
        "topk_overlap_ratio": topk_overlap / top_k if top_k > 0 else 0.0,
    }

    (work_dir / "centroid_batch_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (work_dir / "centroid_batch_benchmark.md").write_text(
        "\n".join(
            [
                "# Centroid-Batched OpenFHE Benchmark",
                "",
                f"- poly_modulus_degree: {result['poly_modulus_degree']}",
                f"- centroid_count: {result['centroid_count']}",
                f"- centroids_per_ciphertext: {result['centroids_per_ciphertext']}",
                f"- baseline_compute_seconds: {result['baseline_compute_seconds']:.3f}",
                f"- centroid_batched_compute_seconds: {result['centroid_batched_compute_seconds']:.3f}",
                f"- speedup: {result['speedup']:.3f}x",
                f"- baseline_output_ciphertexts: {result['baseline_output_ciphertexts']}",
                f"- centroid_batched_output_ciphertexts: {result['centroid_batched_output_ciphertexts']}",
                f"- topk_overlap_ratio: {result['topk_overlap_ratio']:.3f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
