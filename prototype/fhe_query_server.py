#!/usr/bin/env python3
"""
Server-side single-query FHE distance computation wrapper.
"""

import argparse
from pathlib import Path
from typing import Optional

from fhe_backend import BackendError, create_backend

class FHEQueryServer:
    """
    Server wrapper delegating crypto operations to configured backend.
    """
    
    def __init__(
        self,
        context_path: Path,
        centroids_path: Path,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: Optional[list] = None,
        backend: str = "openfhe_cpp",
    ):
        self.context_path = Path(context_path)
        self.centroids_path = Path(centroids_path)
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.backend_name = backend
        if self.backend_name == "openfhe_cpp" and self.poly_modulus_degree < 16384:
            self.poly_modulus_degree = 16384
        self.backend = create_backend(backend)
        self.backend.ensure_context(
            self.context_path,
            self.poly_modulus_degree,
            self.coeff_mod_bit_sizes,
        )

    def compute_distances(
        self,
        encrypted_query_path: Path,
        encrypted_norm_path: Path,
        output_path: Path,
        batch_size: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        self.backend.compute_distances(
            context_path=self.context_path,
            centroids_path=self.centroids_path,
            encrypted_query_path=Path(encrypted_query_path),
            encrypted_norm_path=Path(encrypted_norm_path),
            output_path=Path(output_path),
            batch_size=batch_size,
            num_threads=num_threads,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Server-side FHE distance computation"
    )
    parser.add_argument(
        "--context-path",
        type=str,
        default=None,
        help="Path to encryption context directory (will look for context_public.json). "
             "If not provided, will try to load from encrypted query directory."
    )
    parser.add_argument(
        "--centroids-path",
        type=str,
        required=True,
        help="Path to centroids.npy file"
    )
    parser.add_argument(
        "--encrypted-query",
        type=str,
        required=True,
        help="Path to encrypted_query.bin"
    )
    parser.add_argument(
        "--encrypted-norm",
        type=str,
        required=True,
        help="Path to encrypted_norm_squared.bin"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./encrypted_distances",
        help="Output directory for encrypted distances"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Centroid chunk size for OpenFHE parallel scheduling"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads for OpenFHE server compute"
    )
    parser.add_argument(
        "--poly-modulus-degree",
        type=int,
        default=8192,
        help="Polynomial modulus degree (must match client)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openfhe_cpp",
        choices=["openfhe_cpp", "tenseal"],
        help="Crypto backend implementation",
    )
    
    args = parser.parse_args()
    
    # Determine context path: use provided path, or try to find in encrypted query directory
    if args.context_path is None:
        encrypted_query_dir = Path(args.encrypted_query).parent
        if (encrypted_query_dir / "context.bin").exists() or (
            encrypted_query_dir / "context_public.json"
        ).exists():
            args.context_path = str(encrypted_query_dir)
            print(f"Found context in encrypted query directory: {args.context_path}")
        else:
            args.context_path = "./fhe_context"
            print(f"Using default context path: {args.context_path}")
    
    print("=" * 70)
    print("FHE Query Server - Distance Computation")
    print("=" * 70)
    print()
    
    try:
        server = FHEQueryServer(
            context_path=args.context_path,
            centroids_path=args.centroids_path,
            poly_modulus_degree=args.poly_modulus_degree,
            backend=args.backend,
        )
        server.compute_distances(
            encrypted_query_path=Path(args.encrypted_query),
            encrypted_norm_path=Path(args.encrypted_norm),
            output_path=Path(args.output_dir),
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )
    except BackendError as exc:
        raise SystemExit(f"Failed with backend '{args.backend}': {exc}") from exc

    print(f"Computed encrypted distances in: {args.output_dir}")


if __name__ == "__main__":
    main()

