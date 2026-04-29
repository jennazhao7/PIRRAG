#!/usr/bin/env python3
"""Benchmark query+centroid batching for many encrypted queries."""

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


def rotation_indices(lanes: int, padded_dim: int) -> str:
    values: List[str] = []
    step = 1
    while step < padded_dim:
        values.append(str(step * lanes))
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
            vals = line.split()
            if len(vals) < query_dim:
                continue
            out.write(" ".join(vals[:query_dim]) + "\n")
            kept += 1
            if kept >= n_rows:
                break
    if kept == 0:
        raise RuntimeError(f"No usable rows found in {src}")
    return kept


def write_query_file(queries_file: Path, query_index: int, out_file: Path) -> None:
    lines = [line for line in queries_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_file.write_text(lines[query_index] + "\n", encoding="utf-8")


def read_multi_topk(path: Path) -> List[List[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row["centroid_indices"] for row in data["results"]]


def read_single_topk(path: Path) -> List[int]:
    return json.loads(path.read_text(encoding="utf-8"))["centroid_indices"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenFHE query+centroid batching")
    parser.add_argument("--bin-dir", type=Path, default=Path("openfhe_core/build/bin"))
    parser.add_argument("--centroids-file", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("openfhe_core/query_centroid_batch_bench"))
    parser.add_argument("--poly-modulus-degree", type=int, default=16384)
    parser.add_argument("--query-dim", type=int, default=768)
    parser.add_argument("--padded-dim", type=int, default=1024)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--max-centroids", type=int, default=512)
    parser.add_argument("--queries-per-batch", type=int, default=2)
    parser.add_argument("--centroids-per-batch", type=int, default=4)
    parser.add_argument("--single-centroids-per-batch", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    bins = {
        "keygen": bin_dir / "openfhe_keygen",
        "encrypt_single": bin_dir / "openfhe_encrypt_query_centroid_batched",
        "compute_single": bin_dir / "openfhe_compute_distances_centroid_batched",
        "decrypt_single": bin_dir / "openfhe_decrypt_topk_centroid_batched",
        "encrypt_2d": bin_dir / "openfhe_encrypt_queries_centroid_batched",
        "compute_2d": bin_dir / "openfhe_compute_distances_query_centroid_batched",
        "decrypt_2d": bin_dir / "openfhe_decrypt_topk_query_centroid_batched",
    }
    for name, path in bins.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} binary: {path}")

    centroids_subset = work_dir / "centroids_subset.txt"
    centroid_count = first_rows(
        args.centroids_file.resolve(),
        centroids_subset,
        args.max_centroids,
        args.query_dim,
    )
    queries_file = work_dir / "queries.txt"
    query_count = first_rows(
        args.centroids_file.resolve(),
        queries_file,
        args.num_queries,
        args.query_dim,
    )
    top_k = min(args.top_k, centroid_count)
    coeffs = coeff_sizes_for_degree(args.poly_modulus_degree)

    lanes_single = args.single_centroids_per_batch
    lanes_2d = args.queries_per_batch * args.centroids_per_batch

    single_dir = work_dir / "single_query_centroid_batched"
    single_ctx = single_dir / "context"
    run_cmd([
        str(bins["keygen"]),
        "--context-dir", str(single_ctx),
        "--poly-modulus-degree", str(args.poly_modulus_degree),
        "--coeff-mod-bit-sizes", coeffs,
        "--rotation-indices", rotation_indices(lanes_single, args.padded_dim),
    ])

    single_compute_seconds = 0.0
    single_topks: List[List[int]] = []
    for q in range(query_count):
        qdir = single_dir / f"query_{q:04d}"
        enc = qdir / "encrypted_query"
        dist = qdir / "encrypted_distances"
        query_file = qdir / "query.txt"
        qdir.mkdir(parents=True, exist_ok=True)
        write_query_file(queries_file, q, query_file)
        run_cmd([
            str(bins["encrypt_single"]),
            "--context-dir", str(single_ctx),
            "--input-vector", str(query_file),
            "--output-dir", str(enc),
            "--centroids-per-ciphertext", str(args.single_centroids_per_batch),
            "--padded-dim", str(args.padded_dim),
        ])
        t0 = time.perf_counter()
        run_cmd([
            str(bins["compute_single"]),
            "--context-dir", str(single_ctx),
            "--centroids-file", str(centroids_subset),
            "--encrypted-query", str(enc / "encrypted_query_centroid_batched.bin"),
            "--encrypted-norm", str(enc / "encrypted_norm_centroid_batched.bin"),
            "--output-dir", str(dist),
            "--centroids-per-ciphertext", str(args.single_centroids_per_batch),
            "--padded-dim", str(args.padded_dim),
            "--num-threads", str(args.num_threads),
            "--batch-size", str(args.batch_size),
        ])
        single_compute_seconds += time.perf_counter() - t0
        run_cmd([
            str(bins["decrypt_single"]),
            "--context-dir", str(single_ctx),
            "--encrypted-distances-dir", str(dist),
            "--top-k", str(top_k),
            "--output-json", str(dist / "top_k_results.json"),
        ])
        single_topks.append(read_single_topk(dist / "top_k_results.json"))

    combo_dir = work_dir / "query_centroid_batched"
    combo_ctx = combo_dir / "context"
    combo_enc = combo_dir / "encrypted_queries"
    combo_dist = combo_dir / "encrypted_distances"
    run_cmd([
        str(bins["keygen"]),
        "--context-dir", str(combo_ctx),
        "--poly-modulus-degree", str(args.poly_modulus_degree),
        "--coeff-mod-bit-sizes", coeffs,
        "--rotation-indices", rotation_indices(lanes_2d, args.padded_dim),
    ])
    run_cmd([
        str(bins["encrypt_2d"]),
        "--context-dir", str(combo_ctx),
        "--input-matrix", str(queries_file),
        "--output-dir", str(combo_enc),
        "--queries-per-batch", str(args.queries_per_batch),
        "--centroids-per-batch", str(args.centroids_per_batch),
        "--padded-dim", str(args.padded_dim),
    ])
    t0 = time.perf_counter()
    run_cmd([
        str(bins["compute_2d"]),
        "--context-dir", str(combo_ctx),
        "--centroids-file", str(centroids_subset),
        "--encrypted-queries-dir", str(combo_enc),
        "--output-dir", str(combo_dist),
        "--num-threads", str(args.num_threads),
        "--batch-size", str(args.batch_size),
    ])
    combo_compute_seconds = time.perf_counter() - t0
    run_cmd([
        str(bins["decrypt_2d"]),
        "--context-dir", str(combo_ctx),
        "--encrypted-distances-dir", str(combo_dist),
        "--top-k", str(top_k),
        "--output-json", str(combo_dist / "top_k_results.json"),
    ])
    combo_topks = read_multi_topk(combo_dist / "top_k_results.json")

    overlaps = []
    for q in range(query_count):
        overlaps.append(len(set(single_topks[q]).intersection(combo_topks[q])))

    result: Dict = {
        "poly_modulus_degree": args.poly_modulus_degree,
        "query_dim": args.query_dim,
        "padded_dim": args.padded_dim,
        "query_count": query_count,
        "centroid_count": centroid_count,
        "queries_per_batch": args.queries_per_batch,
        "centroids_per_batch": args.centroids_per_batch,
        "single_centroids_per_batch": args.single_centroids_per_batch,
        "num_threads": args.num_threads,
        "batch_size": args.batch_size,
        "top_k": top_k,
        "single_query_centroid_batched_compute_seconds": single_compute_seconds,
        "query_centroid_batched_compute_seconds": combo_compute_seconds,
        "throughput_speedup": single_compute_seconds / combo_compute_seconds
        if combo_compute_seconds > 0 else 0.0,
        "single_query_seconds_per_query": single_compute_seconds / query_count,
        "combo_seconds_per_query": combo_compute_seconds / query_count,
        "topk_overlaps": overlaps,
        "min_topk_overlap_ratio": min(overlaps) / top_k if top_k > 0 else 0.0,
    }

    (work_dir / "query_centroid_batch_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
