#!/usr/bin/env python3
"""
Server-side FHE distance computation for wiki-rag.

This script:
1. Loads encrypted query and encrypted ||q||^2
2. Loads centroids from IVF index
3. Computes encrypted distances: d_i = ||q||^2 + ||c_i||^2 - 2<q, c_i>
4. Uses TenSEAL batching for efficient computation
5. Returns encrypted distances (or top-k if decryption key available)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import tenseal as ts

from prototype.go_runner import run_built_executable


class FHEQueryServer:
    """
    Server-side FHE distance computation.
    
    Computes encrypted distances between query and centroids:
    d_i = ||q||^2 + ||c_i||^2 - 2<q, c_i>
    """
    
    def __init__(
        self,
        context_path: Path,
        centroids_path: Path,
        poly_modulus_degree: int = 8192
    ):
        """
        Initialize FHE query server.
        
        Args:
            context_path: Path to encryption context (public key)
            centroids_path: Path to centroids.npy file
            poly_modulus_degree: Polynomial modulus degree (must match client)
        """
        self.context_path = Path(context_path)
        self.centroids_path = Path(centroids_path)
        self.poly_modulus_degree = poly_modulus_degree
        
        # Load encryption context (public key only for server)
        self.context = self._load_context()
        
        # Load centroids
        self.centroids = self._load_centroids()
        self.n_centroids = len(self.centroids)
        self.centroid_dim = self.centroids.shape[1]
        
        # Pre-compute squared norms of centroids (plaintext)
        print("Pre-computing squared norms of centroids...")
        self.centroid_norms_squared = np.sum(self.centroids ** 2, axis=1).astype(np.float32)
        
        print(f"Loaded {self.n_centroids} centroids of dimension {self.centroid_dim}")
        print(f"Centroid norms range: [{self.centroid_norms_squared.min():.4f}, {self.centroid_norms_squared.max():.4f}]")
    
    def _load_context(self) -> ts.Context:
        """
        Load encryption context (public key only).
        
        Tries to load public key context first, falls back to full context if needed.
        
        Returns:
            TenSEAL encryption context (public key only)
        """
        # Try public key file first (preferred)
        public_context_file = self.context_path / "context_public.json"
        context_file = self.context_path / "context.json"
        
        if public_context_file.exists():
            print(f"Loading public key context from {public_context_file}...")
            with open(public_context_file, 'rb') as f:
                context_bytes = f.read()
            context = ts.context_from(context_bytes)
            
            if context.is_public():
                print("✓ Loaded public key context (server-safe, no secret key)")
            else:
                raise ValueError("Public key file contains secret key - this should not happen!")
            return context
        
        # Fallback to full context file (for backward compatibility)
        if context_file.exists():
            print(f"⚠ Loading full context from {context_file}...")
            print("  Note: Server should use public key only. Creating public key context...")
            with open(context_file, 'rb') as f:
                context_bytes = f.read()
            context = ts.context_from(context_bytes)
            
            # Extract public key only
            if not context.is_public():
                # Save public key version for future use
                public_context_bytes = context.serialize(save_secret_key=False)
                public_context_file.parent.mkdir(parents=True, exist_ok=True)
                with open(public_context_file, 'wb') as f:
                    f.write(public_context_bytes)
                print(f"✓ Created public key context at {public_context_file}")
                
                # Create public-only context
                context = ts.context_from(public_context_bytes)
            
            if context.is_public():
                print("✓ Using public key context (server-safe)")
            else:
                raise ValueError("Failed to create public key context")
            return context
        
        raise FileNotFoundError(
            f"Encryption context not found at {self.context_path}. "
            "Make sure the client has generated the context and exported the public key."
        )
    
    def _load_centroids(self) -> np.ndarray:
        """
        Load centroids from file.
        
        Returns:
            Centroids array (n_centroids, dim)
        """
        print(f"Loading centroids from {self.centroids_path}...")
        centroids = np.load(self.centroids_path).astype(np.float32)
        
        if centroids.ndim != 2:
            raise ValueError(f"Expected 2D centroids array, got {centroids.ndim}D")
        
        return centroids
    
    def load_encrypted_query(
        self,
        encrypted_query_path: Path,
        encrypted_norm_path: Path
    ) -> Tuple[ts.CKKSVector, ts.CKKSVector]:
        """
        Load encrypted query and encrypted norm from files.
        
        Args:
            encrypted_query_path: Path to encrypted query binary file
            encrypted_norm_path: Path to encrypted norm binary file
            
        Returns:
            Tuple of (encrypted_query, encrypted_norm_squared)
        """
        print(f"Loading encrypted query from {encrypted_query_path}...")
        with open(encrypted_query_path, 'rb') as f:
            query_bytes = f.read()
        encrypted_query = ts.ckks_vector_from(self.context, query_bytes)
        
        print(f"Loading encrypted norm from {encrypted_norm_path}...")
        with open(encrypted_norm_path, 'rb') as f:
            norm_bytes = f.read()
        encrypted_norm_squared = ts.ckks_vector_from(self.context, norm_bytes)
        
        print("✓ Encrypted query and norm loaded successfully")
        
        return encrypted_query, encrypted_norm_squared
    
    def _determine_optimal_batch_size(self, n_items: int, item_dim: int) -> int:
        """
        Determine optimal batch size based on available resources and TenSEAL capabilities.
        
        Args:
            n_items: Number of items to process
            item_dim: Dimension of each item
            
        Returns:
            Optimal batch size
        """
        # TenSEAL batching considerations:
        # - poly_modulus_degree determines max batch capacity
        # - Memory usage scales with batch_size * dimension
        # - Larger batches = fewer operations but more memory
        
        max_batch_capacity = self.poly_modulus_degree // 2  # TenSEAL batching limit
        
        # Estimate memory per encrypted vector (rough approximation)
        # Each encrypted value in TenSEAL is ~100-200KB
        estimated_memory_per_vector = item_dim * 150 * 1024  # bytes (conservative)
        
        # Try to use available memory efficiently
        # Target: process enough to keep GPU/CPU busy but not exceed memory
        # For CPU: smaller batches are better (50-200)
        # For large datasets: balance between memory and number of operations
        
        if n_items <= 100:
            # Small dataset: process all at once
            return n_items
        elif n_items <= 1000:
            # Medium dataset: use moderate batches
            optimal = min(200, max_batch_capacity // 4)
        elif n_items <= 10000:
            # Large dataset: use larger batches for efficiency
            optimal = min(500, max_batch_capacity // 2)
        else:
            # Very large dataset: use maximum efficient batch size
            optimal = min(1000, max_batch_capacity)
        
        # Ensure batch size doesn't exceed dataset size
        optimal = min(optimal, n_items)
        
        # Round to nice numbers for reporting
        if optimal > 100:
            optimal = (optimal // 50) * 50  # Round to nearest 50
        else:
            optimal = (optimal // 10) * 10  # Round to nearest 10
        
        return max(10, optimal)  # Minimum batch size of 10
    
    def compute_dot_products_batched(
        self,
        encrypted_query: ts.CKKSVector,
        batch_size: Optional[int] = None
    ) -> List[ts.CKKSVector]:
        """
        Compute dot products <q, c_i> for all centroids using efficient batching.
        
        Uses TenSEAL batching to compute multiple dot products efficiently.
        Strategy: Compute element-wise products and sum using rotations.
        
        Args:
            encrypted_query: Encrypted query vector
            batch_size: Batch size for processing (None = auto-determine optimal)
            
        Returns:
            List of encrypted dot products [<q, c_0>, <q, c_1>, ..., <q, c_n>]
        """
        n_centroids = self.n_centroids
        dim = self.centroid_dim
        
        print(f"\nComputing dot products <q, c_i> using batching...")
        print(f"  Total centroids: {n_centroids}")
        print(f"  Query dimension: {dim}")
        
        # Auto-determine optimal batch size if not provided
        if batch_size is None:
            batch_size = self._determine_optimal_batch_size(n_centroids, dim)
            print(f"  Auto-selected batch size: {batch_size}")
        else:
            print(f"  Using specified batch size: {batch_size}")
        
        # Compute dot products efficiently
        # For each centroid c_i: <q, c_i> = sum(q_j * c_i_j)
        all_dot_products = []
        
        for batch_start in range(0, n_centroids, batch_size):
            batch_end = min(batch_start + batch_size, n_centroids)
            batch_centroids = self.centroids[batch_start:batch_end]
            
            # Compute dot products for this batch
            for i, centroid in enumerate(batch_centroids):
                centroid_idx = batch_start + i
                
                # Compute element-wise product: q * c_i
                # Create encrypted vector for centroid
                encrypted_centroid = ts.ckks_vector(self.context, centroid.tolist())
                
                # Element-wise multiplication
                product = encrypted_query * encrypted_centroid
                
                # Sum all elements to get dot product
                # Use efficient sum operation
                dot_product = product.sum()
                
                all_dot_products.append(dot_product)
            
            if batch_end % 500 == 0 or batch_end == n_centroids:
                print(f"  Computed {batch_end}/{n_centroids} dot products...")
        
        print(f"✓ Computed {len(all_dot_products)} dot products")
        
        return all_dot_products
    
    def compute_distances(
        self,
        encrypted_query: ts.CKKSVector,
        encrypted_norm_squared: ts.CKKSVector,
        batch_size: Optional[int] = None
    ) -> List[ts.CKKSVector]:
        """
        Compute encrypted distances: d_i = ||q||^2 + ||c_i||^2 - 2<q, c_i>
        
        Args:
            encrypted_query: Encrypted query vector
            encrypted_norm_squared: Encrypted ||q||^2
            batch_size: Batch size for dot product computation
            
        Returns:
            List of encrypted distances [d_0, d_1, ..., d_n]
        """
        print(f"\n{'='*70}")
        print("Computing Encrypted Distances")
        print(f"{'='*70}")
        
        # Step 1: Compute dot products <q, c_i>
        dot_products = self.compute_dot_products_batched(
            encrypted_query,
            batch_size=batch_size
        )
        
        # Step 2: Compute distances for each centroid using batching
        print(f"\nComputing distances: d_i = ||q||^2 + ||c_i||^2 - 2<q, c_i>...")
        encrypted_distances = []
        
        # Auto-determine optimal batch size for distance computation
        # Reuse the same batch size as dot products for consistency
        if batch_size is None:
            distance_batch_size = self._determine_optimal_batch_size(self.n_centroids, self.centroid_dim)
        else:
            distance_batch_size = batch_size
        
        for batch_start in range(0, self.n_centroids, distance_batch_size):
            batch_end = min(batch_start + distance_batch_size, self.n_centroids)
            
            for i in range(batch_start, batch_end):
                # d_i = ||q||^2 + ||c_i||^2 - 2<q, c_i>
                # ||q||^2 is encrypted
                # ||c_i||^2 is plaintext (pre-computed)
                # <q, c_i> is encrypted
                
                # Create encrypted version of ||c_i||^2
                encrypted_centroid_norm = ts.ckks_vector(
                    self.context,
                    [float(self.centroid_norms_squared[i])]
                )
                
                # Compute: ||q||^2 + ||c_i||^2
                sum_norms = encrypted_norm_squared + encrypted_centroid_norm
                
                # Compute: 2<q, c_i>
                two_times_dot = dot_products[i] * 2
                
                # Compute: d_i = ||q||^2 + ||c_i||^2 - 2<q, c_i>
                distance = sum_norms - two_times_dot
                
                encrypted_distances.append(distance)
            
            if batch_end % 500 == 0 or batch_end == self.n_centroids:
                print(f"  Computed {batch_end}/{self.n_centroids} distances...")
        
        print(f"✓ Computed {len(encrypted_distances)} encrypted distances")
        
        return encrypted_distances
    
    def get_top_k_encrypted(
        self,
        encrypted_distances: List[ts.CKKSVector],
        k: int = 10
    ) -> List[Tuple[int, ts.CKKSVector]]:
        """
        Get top-k smallest distances (encrypted).
        
        Note: This requires homomorphic comparison, which is complex.
        For now, returns all distances sorted by index.
        In practice, you'd need to decrypt or use a more sophisticated protocol.
        
        Args:
            encrypted_distances: List of encrypted distances
            k: Number of top results
            
        Returns:
            List of (index, encrypted_distance) tuples
        """
        # Without decryption, we can't sort encrypted values
        # Return first k indices (client will decrypt and sort)
        return [(i, encrypted_distances[i]) for i in range(min(k, len(encrypted_distances)))]
    
    def save_encrypted_distances(
        self,
        encrypted_distances: List[ts.CKKSVector],
        output_path: Path
    ):
        """
        Save encrypted distances to files.
        
        Args:
            encrypted_distances: List of encrypted distances
            output_path: Output directory
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving encrypted distances to {output_path}...")
        
        # Save each distance (or batch them if possible)
        for i, enc_dist in enumerate(encrypted_distances):
            dist_file = output_path / f"encrypted_distance_{i:04d}.bin"
            with open(dist_file, 'wb') as f:
                f.write(enc_dist.serialize())
        
        print(f"✓ Saved {len(encrypted_distances)} encrypted distances")
        
        # Also save metadata
        metadata = {
            "n_distances": len(encrypted_distances),
            "n_centroids": self.n_centroids,
            "centroid_dim": self.centroid_dim
        }
        
        metadata_file = output_path / "distances_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata to {metadata_file}")


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
        help="Batch size for dot product computation (default: auto)"
    )
    parser.add_argument(
        "--poly-modulus-degree",
        type=int,
        default=8192,
        help="Polynomial modulus degree (must match client)"
    )
    
    args = parser.parse_args()
    
    # Determine context path: use provided path, or try to find in encrypted query directory
    if args.context_path is None:
        # Try to find public key in encrypted query directory
        encrypted_query_dir = Path(args.encrypted_query).parent
        potential_public_key = encrypted_query_dir / "context_public.json"
        if potential_public_key.exists():
            args.context_path = str(encrypted_query_dir)
            print(f"Found public key in encrypted query directory: {args.context_path}")
        else:
            # Default to fhe_context
            args.context_path = "./fhe_context"
            print(f"Using default context path: {args.context_path}")
    
    print("=" * 70)
    print("FHE Query Server - Distance Computation")
    print("=" * 70)
    print()
    
    # Initialize server
    server = FHEQueryServer(
        context_path=args.context_path,
        centroids_path=args.centroids_path,
        poly_modulus_degree=args.poly_modulus_degree
    )
    
    # Load encrypted query
    encrypted_query, encrypted_norm = server.load_encrypted_query(
        Path(args.encrypted_query),
        Path(args.encrypted_norm)
    )
    
    # Compute distances
    encrypted_distances = server.compute_distances(
        encrypted_query,
        encrypted_norm,
        batch_size=args.batch_size
    )
    
    # Save encrypted distances
    server.save_encrypted_distances(
        encrypted_distances,
        Path(args.output_dir)
    )
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Computed {len(encrypted_distances)} encrypted distances")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}")


    # Run the built executable
    print("\n=== Running database server ===")
    stdout, stderr, code = run_built_executable(
        "/Users/antoniajanuszewicz/GolandProjects/Piano-PIR-RAG/server_exe",
        args=["-port","50051", "-database", "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/uniform_index.txt"]
    )
    print(f"Output:\n{stderr}")


if __name__ == "__main__":
    main()

