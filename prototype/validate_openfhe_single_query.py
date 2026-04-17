#!/usr/bin/env python3
"""
Single-query OpenFHE validation helper.

Runs encryption -> server distance computation -> decryption and compares top-k with
plaintext baseline using the same centroids.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fhe_query_client import FHEQueryClient
from fhe_query_server import FHEQueryServer


def plaintext_top_k(query_vector: np.ndarray, centroids: np.ndarray, top_k: int):
    query_norm = np.sum(query_vector**2)
    centroid_norms = np.sum(centroids**2, axis=1)
    dots = centroids @ query_vector
    distances = query_norm + centroid_norms - (2.0 * dots)
    idx = np.argsort(distances)[:top_k]
    return distances[idx].astype(np.float32), idx.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Validate OpenFHE single-query parity")
    parser.add_argument("--query-vector", type=str, required=True)
    parser.add_argument("--centroids-path", type=str, required=True)
    parser.add_argument("--work-dir", type=str, default="./openfhe_validation_work")
    parser.add_argument("--context-path", type=str, default="./fhe_context")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--backend", type=str, default="openfhe_cpp")
    parser.add_argument("--max-distance-delta", type=float, default=0.15)
    parser.add_argument("--min-overlap", type=float, default=0.8)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    encrypted_queries_dir = work_dir / "encrypted_queries"
    encrypted_distances_dir = work_dir / "encrypted_distances"
    decrypted_results_dir = work_dir / "decrypted_results"
    work_dir.mkdir(parents=True, exist_ok=True)

    query_vec = np.load(args.query_vector).astype(np.float32).flatten()
    centroids = np.load(args.centroids_path).astype(np.float32)
    if centroids.ndim != 2:
        raise SystemExit("centroids-path must point to a 2D .npy array")
    if centroids.shape[1] != query_vec.shape[0]:
        raise SystemExit(
            f"Query dim ({query_vec.shape[0]}) != centroid dim ({centroids.shape[1]})"
        )

    client = FHEQueryClient(
        context_path=args.context_path,
        backend=args.backend,
    )
    server = FHEQueryServer(
        context_path=args.context_path,
        centroids_path=args.centroids_path,
        backend=args.backend,
    )

    client.process_query(query_vec, encrypted_queries_dir)
    server.compute_distances(
        encrypted_query_path=encrypted_queries_dir / "encrypted_query.bin",
        encrypted_norm_path=encrypted_queries_dir / "encrypted_norm_squared.bin",
        output_path=encrypted_distances_dir,
    )
    fhe_distances, fhe_indices = client.decrypt_distances(encrypted_distances_dir, args.top_k)
    client.save_decrypted_results(fhe_distances, fhe_indices, decrypted_results_dir)

    plain_distances, plain_indices = plaintext_top_k(query_vec, centroids, args.top_k)

    overlap = (
        len(set(map(int, fhe_indices.tolist())) & set(map(int, plain_indices.tolist())))
        / max(1, args.top_k)
    )
    paired = min(len(fhe_distances), len(plain_distances))
    max_delta = float(np.max(np.abs(fhe_distances[:paired] - plain_distances[:paired])))

    report = {
        "backend": args.backend,
        "top_k": args.top_k,
        "overlap_ratio": overlap,
        "max_abs_distance_delta": max_delta,
        "thresholds": {
            "max_distance_delta": args.max_distance_delta,
            "min_overlap": args.min_overlap,
        },
        "passed": bool(overlap >= args.min_overlap and max_delta <= args.max_distance_delta),
        "fhe_indices": fhe_indices.tolist(),
        "plaintext_indices": plain_indices.tolist(),
    }
    report_file = work_dir / "parity_report.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Report written to: {report_file}")
    if not report["passed"]:
        raise SystemExit("OpenFHE single-query parity check did not meet thresholds")


if __name__ == "__main__":
    main()

