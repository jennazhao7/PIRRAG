#!/usr/bin/env python3
"""Sweep OpenFHE batched parameters to optimize throughput.

This keeps query dimension fixed by default (768), so query embedding fidelity is unchanged.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def parse_csv_ints(value: str) -> List[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError(f"Expected at least one integer in: {value}")
    return out


def coeff_sizes_for_degree(degree: int) -> str:
    if degree == 8192:
        return "60,40,40,60"
    if degree == 16384:
        return "60,40,40,40,40,60"
    if degree == 32768:
        return "60,40,40,40,40,40,40,40,60"
    # Reasonable fallback for other degrees.
    return "60,40,40,60"


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def build_query_matrix_from_centroids(
    centroids_file: Path,
    output_queries_file: Path,
    num_queries: int,
    query_dim: int,
) -> int:
    kept = 0
    output_queries_file.parent.mkdir(parents=True, exist_ok=True)
    with centroids_file.open("r", encoding="utf-8") as src, output_queries_file.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) < query_dim:
                continue
            dst.write(" ".join(vals[:query_dim]) + "\n")
            kept += 1
            if kept >= num_queries:
                break
    if kept == 0:
        raise RuntimeError(
            f"No usable rows found in {centroids_file} for query_dim={query_dim}"
        )
    return kept


def build_centroid_subset(
    centroids_file: Path,
    output_centroids_file: Path,
    max_centroids: int,
    query_dim: int,
) -> int:
    if max_centroids <= 0:
        return 0

    kept = 0
    output_centroids_file.parent.mkdir(parents=True, exist_ok=True)
    with centroids_file.open("r", encoding="utf-8") as src, output_centroids_file.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) < query_dim:
                continue
            dst.write(line + "\n")
            kept += 1
            if kept >= max_centroids:
                break
    if kept == 0:
        raise RuntimeError(
            f"No usable centroid rows found in {centroids_file} for query_dim={query_dim}"
        )
    return kept


def derive_speedups(results: List[Dict]) -> Dict:
    if not results:
        return {}

    baseline_rows = [row for row in results if row["poly_modulus_degree"] == 8192]
    if not baseline_rows:
        baseline_rows = [min(results, key=lambda row: row["queries_per_ciphertext"])]
    baseline = max(baseline_rows, key=lambda row: row["queries_per_second"])
    baseline_qps = baseline["queries_per_second"]

    for row in results:
        row["speedup_vs_baseline"] = (
            row["queries_per_second"] / baseline_qps if baseline_qps > 0 else 0.0
        )
    return baseline


def write_markdown_report(summary: Dict, path: Path) -> None:
    results = summary["results"]
    best = summary.get("best")
    best_over_five = summary.get("best_over_five")
    lines = [
        "# OpenFHE Batched Parameter Sweep",
        "",
        f"- total_runs: {summary['total_runs']}",
        f"- query_dim: {summary['query_dim']}",
        f"- query_count: {summary['query_count']}",
        f"- centroid_count: {summary['centroid_count']}",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Best Config",
                "",
                f"- poly_modulus_degree: {best['poly_modulus_degree']}",
                f"- queries_per_ciphertext: {best['queries_per_ciphertext']}",
                f"- num_threads: {best['num_threads']}",
                f"- batch_size: {best['batch_size']}",
                f"- compute_seconds: {best['compute_seconds']:.3f}",
                f"- queries_per_second: {best['queries_per_second']:.6f}",
                "",
            ]
        )
    if best_over_five:
        lines.extend(
            [
                "## Best Config With >5 Queries/Ciphertext",
                "",
                f"- poly_modulus_degree: {best_over_five['poly_modulus_degree']}",
                f"- queries_per_ciphertext: {best_over_five['queries_per_ciphertext']}",
                f"- num_threads: {best_over_five['num_threads']}",
                f"- batch_size: {best_over_five['batch_size']}",
                f"- compute_seconds: {best_over_five['compute_seconds']:.3f}",
                f"- queries_per_second: {best_over_five['queries_per_second']:.6f}",
                f"- speedup_vs_baseline: {best_over_five['speedup_vs_baseline']:.3f}x",
                "",
            ]
        )
    lines.extend(
        [
            "## All Runs (sorted by qps)",
            "",
            "| degree | q/ciphertext | threads | batch | compute_s | qps | speedup_vs_8192 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        lines.append(
            f"| {row['poly_modulus_degree']} | {row['queries_per_ciphertext']} | "
            f"{row['num_threads']} | {row['batch_size']} | "
            f"{row['compute_seconds']:.3f} | {row['queries_per_second']:.6f} | "
            f"{row['speedup_vs_baseline']:.3f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep OpenFHE batched compute parameters")
    parser.add_argument("--bin-dir", type=Path, default=Path("openfhe_core/build/bin"))
    parser.add_argument("--centroids-file", type=Path, required=True)
    parser.add_argument("--queries-file", type=Path, default=None)
    parser.add_argument("--query-dim", type=int, default=768)
    parser.add_argument("--num-queries", type=int, default=30)
    parser.add_argument(
        "--max-centroids",
        type=int,
        default=0,
        help="Use the first N centroids for a faster sweep; 0 uses the full file",
    )
    parser.add_argument("--poly-degrees", type=str, default="8192,16384,32768")
    parser.add_argument("--thread-options", type=str, default="0,8,16,20")
    parser.add_argument("--batch-size-options", type=str, default="32,64,128")
    parser.add_argument("--work-dir", type=Path, default=Path("openfhe_core/sweep_runs"))
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    centroids_file = args.centroids_file.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    keygen_bin = bin_dir / "openfhe_keygen"
    encrypt_bin = bin_dir / "openfhe_encrypt_queries_batched"
    compute_bin = bin_dir / "openfhe_compute_distances_batched"
    for path in (keygen_bin, encrypt_bin, compute_bin):
        if not path.exists():
            raise FileNotFoundError(f"Missing binary: {path}")
    if not centroids_file.exists():
        raise FileNotFoundError(f"Missing centroids file: {centroids_file}")

    if args.queries_file:
        queries_file = args.queries_file.resolve()
        if not queries_file.exists():
            raise FileNotFoundError(f"Missing queries file: {queries_file}")
        query_count = sum(1 for _ in queries_file.open("r", encoding="utf-8") if _.strip())
    else:
        queries_file = work_dir / f"queries_from_centroids_q{args.num_queries}_d{args.query_dim}.txt"
        query_count = build_query_matrix_from_centroids(
            centroids_file=centroids_file,
            output_queries_file=queries_file,
            num_queries=args.num_queries,
            query_dim=args.query_dim,
        )

    if args.max_centroids > 0:
        sweep_centroids_file = work_dir / f"centroids_first_{args.max_centroids}_d{args.query_dim}.txt"
        centroid_count = build_centroid_subset(
            centroids_file=centroids_file,
            output_centroids_file=sweep_centroids_file,
            max_centroids=args.max_centroids,
            query_dim=args.query_dim,
        )
    else:
        sweep_centroids_file = centroids_file
        centroid_count = sum(
            1 for line in centroids_file.open("r", encoding="utf-8") if line.strip()
        )

    poly_degrees = parse_csv_ints(args.poly_degrees)
    thread_options = parse_csv_ints(args.thread_options)
    batch_size_options = parse_csv_ints(args.batch_size_options)

    results: List[Dict] = []
    for degree in poly_degrees:
        degree_dir = work_dir / f"degree_{degree}"
        context_dir = degree_dir / "context"
        encrypted_queries_dir = degree_dir / "encrypted_queries"
        context_dir.mkdir(parents=True, exist_ok=True)
        encrypted_queries_dir.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [
                str(keygen_bin),
                "--context-dir",
                str(context_dir),
                "--poly-modulus-degree",
                str(degree),
                "--coeff-mod-bit-sizes",
                coeff_sizes_for_degree(degree),
            ]
        )
        run_cmd(
            [
                str(encrypt_bin),
                "--context-dir",
                str(context_dir),
                "--input-matrix",
                str(queries_file),
                "--output-dir",
                str(encrypted_queries_dir),
                "--poly-modulus-degree",
                str(degree),
                "--query-dim",
                str(args.query_dim),
            ]
        )

        metadata = json.loads(
            (encrypted_queries_dir / "queries_metadata.json").read_text(encoding="utf-8")
        )
        queries_per_ciphertext = int(metadata["queries_per_ciphertext"])
        n_queries = int(metadata["n_queries"])

        for num_threads, batch_size in itertools.product(thread_options, batch_size_options):
            run_name = f"deg{degree}_t{num_threads}_b{batch_size}"
            out_dir = degree_dir / f"dist_{run_name}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            run_cmd(
                [
                    str(compute_bin),
                    "--context-dir",
                    str(context_dir),
                    "--centroids-file",
                    str(sweep_centroids_file),
                    "--encrypted-queries-dir",
                    str(encrypted_queries_dir),
                    "--output-dir",
                    str(out_dir),
                    "--num-threads",
                    str(num_threads),
                    "--batch-size",
                    str(batch_size),
                ]
            )
            elapsed = time.perf_counter() - t0

            result = {
                "poly_modulus_degree": degree,
                "queries_per_ciphertext": queries_per_ciphertext,
                "n_queries": n_queries,
                "num_threads": num_threads,
                "batch_size": batch_size,
                "compute_seconds": elapsed,
                "queries_per_second": (n_queries / elapsed) if elapsed > 0 else 0.0,
                "output_dir": str(out_dir),
            }
            results.append(result)
            print(
                f"[{run_name}] q/c={queries_per_ciphertext}, "
                f"compute={elapsed:.3f}s, qps={result['queries_per_second']:.6f}"
            )

            if not args.keep_artifacts:
                shutil.rmtree(out_dir, ignore_errors=True)

    results.sort(key=lambda x: x["queries_per_second"], reverse=True)
    baseline = derive_speedups(results)
    best = results[0] if results else None
    over_five = [row for row in results if row["queries_per_ciphertext"] > 5]
    best_over_five = over_five[0] if over_five else None
    summary = {
        "query_dim": args.query_dim,
        "query_count": query_count,
        "queries_file": str(queries_file),
        "centroids_file": str(sweep_centroids_file),
        "source_centroids_file": str(centroids_file),
        "centroid_count": centroid_count,
        "total_runs": len(results),
        "baseline": baseline,
        "best": best,
        "best_over_five": best_over_five,
        "results": results,
    }
    summary_json = work_dir / "batched_param_sweep_summary.json"
    summary_md = work_dir / "batched_param_sweep_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary, summary_md)

    print(f"\nWrote sweep summary: {summary_json}")
    print(f"Wrote sweep report:  {summary_md}")
    if best:
        print(
            "Best config: "
            f"degree={best['poly_modulus_degree']}, q/c={best['queries_per_ciphertext']}, "
            f"threads={best['num_threads']}, batch={best['batch_size']}, "
            f"qps={best['queries_per_second']:.6f}"
        )
    if best_over_five:
        print(
            "Best config with >5 queries/ciphertext: "
            f"degree={best_over_five['poly_modulus_degree']}, "
            f"q/c={best_over_five['queries_per_ciphertext']}, "
            f"threads={best_over_five['num_threads']}, batch={best_over_five['batch_size']}, "
            f"qps={best_over_five['queries_per_second']:.6f}, "
            f"speedup_vs_baseline={best_over_five['speedup_vs_baseline']:.3f}x"
        )


if __name__ == "__main__":
    main()

