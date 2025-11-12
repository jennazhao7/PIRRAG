#!/usr/bin/env python3
"""
Train IVF index on wiki-rag embeddings using FAISS Python.

This script:
1. Loads embeddings from FAISS vectorstore
2. Trains IVF index with K=4096 clusters (IVF-flat first, optional PQ/OPQ)
3. Measures recall/latency
4. Exports centroids.npy + lists.json

Usage examples:
    # Basic IVF-flat training with K=4096
    python train_ivf.py --faiss-path wiki_rag_data/wiki_index__top_100000__2025-04-11 --k 4096

    # With recall/latency measurement
    python train_ivf.py --faiss-path wiki_rag_data/wiki_index__top_100000__2025-04-11 --k 4096 --test-recall

    # IVF-PQ with custom parameters
    python train_ivf.py --faiss-path wiki_rag_data/wiki_index__top_100000__2025-04-11 --k 4096 --index-type ivf-pq --m 64 --n-bits 8

    # IVF-OPQ
    python train_ivf.py --faiss-path wiki_rag_data/wiki_index__top_100000__2025-04-11 --k 4096 --index-type ivf-opq --m 64 --n-bits 8

    # With GPU (much faster for large datasets)
    python train_ivf.py --faiss-path wiki_rag_data/wiki_index__top_100000__2025-04-11 --k 4096 --use-gpu
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from wiki_rag.rag import PromptedBGE

# Check if GPU is available
def get_gpu_resources():
    """Get GPU resources if available."""
    try:
        # Check if FAISS GPU support is available
        res = faiss.StandardGpuResources()
        return res
    except AttributeError:
        # FAISS not compiled with GPU support
        return None
    except Exception as e:
        print(f"Warning: Could not initialize GPU resources: {e}")
        return None


def load_embeddings_from_faiss(faiss_path: Path) -> Tuple[np.ndarray, int]:
    """
    Load embeddings from FAISS vectorstore.
    
    Args:
        faiss_path: Path to FAISS index directory
        
    Returns:
        Tuple of (embeddings array, embedding dimension)
    """
    print(f"Loading FAISS index from {faiss_path}...")
    
    # Load embeddings model (same as used to create the index)
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    
    # Load FAISS vectorstore
    vectorstore = FAISS.load_local(
        str(faiss_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Get FAISS index
    index = vectorstore.index
    n_vectors = index.ntotal
    embedding_dim = index.d
    
    print(f"Found {n_vectors} vectors with dimension {embedding_dim}")
    
    # Extract all vectors
    print("Extracting vectors from FAISS index...")
    try:
        # Try to reconstruct all vectors at once (most efficient)
        vectors = index.reconstruct_n(0, n_vectors)
        print(f"Successfully reconstructed {len(vectors)} vectors")
    except Exception as e:
        print(f"Could not reconstruct all vectors at once: {e}")
        print("Reconstructing vectors one by one (this may take a while)...")
        vectors = []
        for i in range(n_vectors):
            if (i + 1) % 10000 == 0:
                print(f"  Reconstructed {i + 1}/{n_vectors} vectors...")
            vec = index.reconstruct(i)
            vectors.append(vec)
        vectors = np.array(vectors)
    
    vectors = np.array(vectors, dtype=np.float32)
    print(f"Extracted embeddings shape: {vectors.shape}")
    
    return vectors, embedding_dim


def train_ivf_flat(
    vectors: np.ndarray,
    n_clusters: int = 4096,
    n_probe: int = 10,
    niter: int = 20,
    verbose: bool = True,
    use_gpu: bool = False
) -> faiss.IndexIVFFlat:
    """
    Train IVF-flat index.
    
    Args:
        vectors: Training vectors (n_vectors, dim)
        n_clusters: Number of clusters (K)
        n_probe: Number of clusters to probe during search
        niter: Number of k-means iterations
        verbose: Whether to print progress
        
    Returns:
        Trained IVF-flat index
    """
    n_vectors, dim = vectors.shape
    
    print(f"\nTraining IVF-flat index...")
    print(f"  Vectors: {n_vectors:,}")
    print(f"  Dimension: {dim}")
    print(f"  Clusters (K): {n_clusters}")
    print(f"  n_probe: {n_probe}")
    print(f"  k-means iterations: {niter}")
    print(f"  Using GPU: {use_gpu}")
    
    # Create quantizer (flat index for cluster centroids)
    quantizer = faiss.IndexFlatL2(dim)
    
    # Create IVF index
    index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
    index.nprobe = n_probe
    
    # Move to GPU if requested
    gpu_res = None
    if use_gpu:
        gpu_res = get_gpu_resources()
        if gpu_res is not None:
            print("  Moving index to GPU...")
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        else:
            print("  Warning: GPU requested but not available, using CPU")
            use_gpu = False
    
    # Train the index
    print("\nTraining k-means clustering...")
    start_time = time.time()
    index.train(vectors)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")
    
    # Add vectors to index
    print("\nAdding vectors to index...")
    start_time = time.time()
    index.add(vectors)
    add_time = time.time() - start_time
    print(f"Added {n_vectors:,} vectors in {add_time:.2f}s")
    
    # Move back to CPU for export (if on GPU)
    if use_gpu and gpu_res is not None:
        print("  Moving index back to CPU for export...")
        index = faiss.index_gpu_to_cpu(index)
    
    return index


def train_ivf_pq(
    vectors: np.ndarray,
    n_clusters: int = 4096,
    m: int = 64,
    n_bits: int = 8,
    n_probe: int = 10,
    niter: int = 20,
    verbose: bool = True,
    use_gpu: bool = False
) -> faiss.IndexIVFPQ:
    """
    Train IVF-PQ index (Product Quantization).
    
    Args:
        vectors: Training vectors (n_vectors, dim)
        n_clusters: Number of clusters (K)
        m: Number of subquantizers (dim must be divisible by m)
        n_bits: Number of bits per subquantizer (2^n_bits centroids per subquantizer)
        n_probe: Number of clusters to probe during search
        niter: Number of k-means iterations
        verbose: Whether to print progress
        
    Returns:
        Trained IVF-PQ index
    """
    n_vectors, dim = vectors.shape
    
    if dim % m != 0:
        raise ValueError(f"Dimension {dim} must be divisible by m={m}")
    
    print(f"\nTraining IVF-PQ index...")
    print(f"  Vectors: {n_vectors:,}")
    print(f"  Dimension: {dim}")
    print(f"  Clusters (K): {n_clusters}")
    print(f"  Subquantizers (m): {m}")
    print(f"  Bits per subquantizer: {n_bits}")
    print(f"  n_probe: {n_probe}")
    print(f"  k-means iterations: {niter}")
    print(f"  Using GPU: {use_gpu}")
    
    # Create quantizer (flat index for cluster centroids)
    quantizer = faiss.IndexFlatL2(dim)
    
    # Create IVF-PQ index
    index = faiss.IndexIVFPQ(quantizer, dim, n_clusters, m, n_bits)
    index.nprobe = n_probe
    
    # Move to GPU if requested
    gpu_res = None
    if use_gpu:
        gpu_res = get_gpu_resources()
        if gpu_res is not None:
            print("  Moving index to GPU...")
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        else:
            print("  Warning: GPU requested but not available, using CPU")
            use_gpu = False
    
    # Train the index
    print("\nTraining k-means clustering and PQ...")
    start_time = time.time()
    index.train(vectors)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")
    
    # Add vectors to index
    print("\nAdding vectors to index...")
    start_time = time.time()
    index.add(vectors)
    add_time = time.time() - start_time
    print(f"Added {n_vectors:,} vectors in {add_time:.2f}s")
    
    # Move back to CPU for export (if on GPU)
    if use_gpu and gpu_res is not None:
        print("  Moving index back to CPU for export...")
        index = faiss.index_gpu_to_cpu(index)
    
    return index


def train_ivf_opq(
    vectors: np.ndarray,
    n_clusters: int = 4096,
    m: int = 64,
    n_bits: int = 8,
    n_probe: int = 10,
    niter: int = 20,
    verbose: bool = True,
    use_gpu: bool = False
) -> faiss.IndexIVF:
    """
    Train IVF-OPQ index (Optimized Product Quantization with rotation).
    
    Args:
        vectors: Training vectors (n_vectors, dim)
        n_clusters: Number of clusters (K)
        m: Number of subquantizers (dim must be divisible by m)
        n_bits: Number of bits per subquantizer
        n_probe: Number of clusters to probe during search
        niter: Number of k-means iterations
        verbose: Whether to print progress
        
    Returns:
        Trained IVF-OPQ index
    """
    n_vectors, dim = vectors.shape
    
    if dim % m != 0:
        raise ValueError(f"Dimension {dim} must be divisible by m={m}")
    
    print(f"\nTraining IVF-OPQ index...")
    print(f"  Vectors: {n_vectors:,}")
    print(f"  Dimension: {dim}")
    print(f"  Clusters (K): {n_clusters}")
    print(f"  Subquantizers (m): {m}")
    print(f"  Bits per subquantizer: {n_bits}")
    print(f"  n_probe: {n_probe}")
    print(f"  k-means iterations: {niter}")
    print(f"  Using GPU: {use_gpu}")
    
    # Create quantizer (flat index for cluster centroids)
    quantizer = faiss.IndexFlatL2(dim)
    
    # Create OPQ preprocessor
    opq = faiss.OPQMatrix(dim, m)
    
    # Create IVF-PQ index
    pq_index = faiss.IndexIVFPQ(quantizer, dim, n_clusters, m, n_bits)
    pq_index.nprobe = n_probe
    
    # Wrap with OPQ
    index = faiss.IndexPreTransform(opq, pq_index)
    
    # Move to GPU if requested
    gpu_res = None
    if use_gpu:
        gpu_res = get_gpu_resources()
        if gpu_res is not None:
            print("  Moving index to GPU...")
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        else:
            print("  Warning: GPU requested but not available, using CPU")
            use_gpu = False
    
    # Train the index
    print("\nTraining OPQ rotation, k-means clustering, and PQ...")
    start_time = time.time()
    index.train(vectors)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")
    
    # Add vectors to index
    print("\nAdding vectors to index...")
    start_time = time.time()
    index.add(vectors)
    add_time = time.time() - start_time
    print(f"Added {n_vectors:,} vectors in {add_time:.2f}s")
    
    # Move back to CPU for export (if on GPU)
    if use_gpu and gpu_res is not None:
        print("  Moving index back to CPU for export...")
        index = faiss.index_gpu_to_cpu(index)
    
    return index


def measure_recall_latency(
    index: faiss.Index,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    n_queries: Optional[int] = None
) -> Tuple[float, float]:
    """
    Measure recall@k and average query latency.
    
    Args:
        index: FAISS index
        query_vectors: Query vectors (n_queries, dim)
        ground_truth: Ground truth nearest neighbors (n_queries, k)
        k: Number of neighbors to retrieve
        n_queries: Number of queries to test (None = all)
        
    Returns:
        Tuple of (recall@k, average_latency_ms)
    """
    if n_queries is None:
        n_queries = len(query_vectors)
    else:
        n_queries = min(n_queries, len(query_vectors))
    
    query_vectors = query_vectors[:n_queries]
    ground_truth = ground_truth[:n_queries]
    
    print(f"\nMeasuring recall@k and latency on {n_queries} queries...")
    
    # Ensure index is trained and has vectors
    if not index.is_trained:
        raise ValueError("Index must be trained before measuring recall")
    if index.ntotal == 0:
        raise ValueError("Index must have vectors before measuring recall")
    
    # Search
    latencies = []
    correct = 0
    total = 0
    
    for i in range(n_queries):
        query = query_vectors[i:i+1]
        
        # Measure latency
        start_time = time.time()
        distances, indices = index.search(query, k)
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        # Compute recall
        retrieved = set(indices[0])
        gt_set = set(ground_truth[i])
        correct += len(retrieved & gt_set)
        total += len(gt_set)
    
    recall = correct / total if total > 0 else 0.0
    avg_latency_ms = np.mean(latencies)
    
    print(f"  Recall@{k}: {recall:.4f} ({correct}/{total})")
    print(f"  Average latency: {avg_latency_ms:.2f} ms")
    print(f"  Min latency: {np.min(latencies):.2f} ms")
    print(f"  Max latency: {np.max(latencies):.2f} ms")
    print(f"  P50 latency: {np.median(latencies):.2f} ms")
    print(f"  P95 latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 latency: {np.percentile(latencies, 99):.2f} ms")
    
    return recall, avg_latency_ms


def compute_ground_truth(
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """
    Compute ground truth nearest neighbors using brute force.
    
    Args:
        vectors: Database vectors (n_vectors, dim)
        query_vectors: Query vectors (n_queries, dim)
        k: Number of neighbors
        
    Returns:
        Ground truth indices (n_queries, k)
    """
    print(f"\nComputing ground truth nearest neighbors (brute force)...")
    start_time = time.time()
    
    # Use FAISS brute force index for exact search
    dim = vectors.shape[1]
    brute_index = faiss.IndexFlatL2(dim)
    brute_index.add(vectors)
    
    distances, indices = brute_index.search(query_vectors, k)
    
    elapsed = time.time() - start_time
    print(f"Computed ground truth in {elapsed:.2f}s")
    
    return indices


def export_centroids_and_lists(
    index: faiss.Index,
    output_dir: Path
):
    """
    Export centroids and inverted lists from IVF index.
    
    Args:
        index: FAISS IVF index
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting centroids and lists to {output_dir}...")
    
    # Get the actual IVF index (unwrap if wrapped in IndexPreTransform)
    ivf_index = index
    transform_matrix = None
    if isinstance(index, faiss.IndexPreTransform):
        # Extract transformation matrix if OPQ
        if isinstance(index.chain[0], faiss.OPQMatrix):
            transform_matrix = faiss.vector_to_array(index.chain[0].A).reshape(
                index.chain[0].d, index.chain[0].d
            )
        ivf_index = index.index
    
    if not isinstance(ivf_index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
        raise ValueError(f"Index type {type(ivf_index)} is not an IVF index")
    
    # Extract centroids
    quantizer = ivf_index.quantizer
    # Check if quantizer is a flat index (handle both IndexFlat and optimized versions)
    is_flat = (hasattr(quantizer, 'reconstruct') and 
               hasattr(quantizer, 'ntotal') and
               quantizer.ntotal == ivf_index.nlist)
    
    if is_flat:
        # Get centroids from quantizer
        n_clusters = ivf_index.nlist
        dim = ivf_index.d
        
        print(f"  Extracting {n_clusters} centroids...")
        centroids = []
        for i in range(n_clusters):
            try:
                centroid = quantizer.reconstruct(i)
                centroids.append(centroid)
            except Exception as e:
                print(f"  Warning: Could not reconstruct centroid {i}: {e}")
                # Use zero vector as fallback
                centroids.append(np.zeros(dim, dtype=np.float32))
        
        centroids = np.array(centroids, dtype=np.float32)
        
        print(f"  Extracted {len(centroids)} centroids of dimension {centroids.shape[1]}")
        
        # Save centroids
        centroids_path = output_dir / "centroids.npy"
        np.save(centroids_path, centroids)
        print(f"  Saved centroids to {centroids_path}")
        
        # Save transformation matrix if OPQ
        if transform_matrix is not None:
            transform_path = output_dir / "opq_transform.npy"
            np.save(transform_path, transform_matrix)
            print(f"  Saved OPQ transformation matrix to {transform_path}")
    else:
        raise ValueError(f"Unsupported quantizer type: {type(quantizer)}. Expected flat quantizer.")
    
    # Extract inverted lists using FAISS's direct_map or by reconstruction
    print(f"  Extracting inverted lists for {n_clusters} clusters...")
    
    # Try to use direct_map if available (faster)
    lists_data = {}
    for cluster_id in range(n_clusters):
        lists_data[str(cluster_id)] = []
    
    # Method 1: Try to access invlists directly (if available)
    try:
        invlists = ivf_index.invlists
        n_clusters = invlists.nlist
        
        for cluster_id in range(n_clusters):
            list_size = invlists.list_size(cluster_id)
            if list_size > 0:
                # Get IDs from inverted list
                ids_ptr = invlists.get_ids(cluster_id)
                # Convert C array to numpy array
                vector_ids = faiss.rev_swig_ptr(ids_ptr, list_size)
                # Convert to list (vector_ids is already a numpy array view)
                lists_data[str(cluster_id)] = [int(vector_ids[i]) for i in range(list_size)]
        
        print(f"  Extracted lists using direct invlists access")
    except Exception as e:
        # Method 2: Reconstruct by finding nearest centroids (slower but works)
        print(f"  Direct access failed ({e}), using reconstruction method...")
        print("  This may take a while for large datasets...")
        
        n_vectors = ivf_index.ntotal
        batch_size = 10000
        
        # Reconstruct vectors in batches and find nearest centroids
        for i in range(0, n_vectors, batch_size):
            end_idx = min(i + batch_size, n_vectors)
            batch_ids = np.arange(i, end_idx, dtype=np.int64)
            
            # Reconstruct vectors
            batch_vectors = []
            for vec_id in batch_ids:
                try:
                    vec = ivf_index.reconstruct(int(vec_id))
                    batch_vectors.append(vec)
                except:
                    # Skip if reconstruction fails
                    continue
            
            if len(batch_vectors) == 0:
                continue
                
            batch_vectors = np.array(batch_vectors, dtype=np.float32)
            
            # Find nearest centroid for each vector
            distances, nearest_clusters = quantizer.search(batch_vectors, 1)
            
            for j, cluster_id in enumerate(nearest_clusters[:, 0]):
                vec_id = int(batch_ids[j])
                lists_data[str(int(cluster_id))].append(vec_id)
            
            if (i + batch_size) % 50000 == 0:
                print(f"    Processed {min(i + batch_size, n_vectors)}/{n_vectors} vectors...")
    
    # Save lists
    lists_path = output_dir / "lists.json"
    with open(lists_path, 'w') as f:
        json.dump(lists_data, f, indent=2)
    
    total_vectors = sum(len(v) for v in lists_data.values())
    print(f"  Saved inverted lists to {lists_path}")
    print(f"  Total vectors in lists: {total_vectors:,}")
    print(f"  Expected vectors: {ivf_index.ntotal:,}")
    
    if total_vectors != ivf_index.ntotal:
        print(f"  Warning: Vector count mismatch (may be due to reconstruction method)")
    
    return centroids_path, lists_path


def main():
    parser = argparse.ArgumentParser(
        description="Train IVF index on wiki-rag embeddings"
    )
    parser.add_argument(
        "--faiss-path",
        type=str,
        required=True,
        help="Path to FAISS index directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ivf_output",
        help="Output directory for centroids.npy and lists.json"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4096,
        help="Number of clusters (K) for IVF"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["ivf-flat", "ivf-pq", "ivf-opq"],
        default="ivf-flat",
        help="Type of IVF index to train"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=64,
        help="Number of subquantizers for PQ/OPQ (dim must be divisible by m)"
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=8,
        help="Number of bits per subquantizer for PQ/OPQ"
    )
    parser.add_argument(
        "--n-probe",
        type=int,
        default=10,
        help="Number of clusters to probe during search"
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=20,
        help="Number of k-means iterations"
    )
    parser.add_argument(
        "--test-recall",
        action="store_true",
        help="Test recall and latency (requires ground truth computation)"
    )
    parser.add_argument(
        "--n-test-queries",
        type=int,
        default=1000,
        help="Number of test queries for recall measurement"
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=10,
        help="k for recall@k measurement"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for training (much faster for large datasets)"
    )
    
    args = parser.parse_args()
    
    # Check GPU availability
    if args.use_gpu:
        gpu_res = get_gpu_resources()
        if gpu_res is None:
            print("Warning: --use-gpu specified but GPU not available. Falling back to CPU.")
            print("  Make sure FAISS is compiled with GPU support and CUDA is available.")
            args.use_gpu = False
        else:
            print("GPU support detected and enabled!")
    else:
        print("Using CPU (use --use-gpu for faster training on large datasets)")
    
    # Load embeddings
    faiss_path = Path(args.faiss_path)
    vectors, dim = load_embeddings_from_faiss(faiss_path)
    
    # Train index
    print(f"\n{'='*70}")
    print(f"Training {args.index_type.upper()} index")
    print(f"{'='*70}")
    
    if args.index_type == "ivf-flat":
        index = train_ivf_flat(
            vectors,
            n_clusters=args.k,
            n_probe=args.n_probe,
            niter=args.niter,
            use_gpu=args.use_gpu
        )
    elif args.index_type == "ivf-pq":
        index = train_ivf_pq(
            vectors,
            n_clusters=args.k,
            m=args.m,
            n_bits=args.n_bits,
            n_probe=args.n_probe,
            niter=args.niter,
            use_gpu=args.use_gpu
        )
    elif args.index_type == "ivf-opq":
        index = train_ivf_opq(
            vectors,
            n_clusters=args.k,
            m=args.m,
            n_bits=args.n_bits,
            n_probe=args.n_probe,
            niter=args.niter,
            use_gpu=args.use_gpu
        )
    
    # Measure recall/latency if requested
    if args.test_recall:
        print(f"\n{'='*70}")
        print("Measuring Recall and Latency")
        print(f"{'='*70}")
        
        # Create test queries (sample from vectors)
        np.random.seed(42)  # For reproducibility
        n_test = min(args.n_test_queries, len(vectors))
        test_indices = np.random.choice(len(vectors), n_test, replace=False)
        query_vectors = vectors[test_indices]
        
        # Compute ground truth
        ground_truth = compute_ground_truth(vectors, query_vectors, k=args.recall_k)
        
        # Measure recall and latency
        recall, avg_latency = measure_recall_latency(
            index,
            query_vectors,
            ground_truth,
            k=args.recall_k,
            n_queries=n_test
        )
    
    # Export centroids and lists
    print(f"\n{'='*70}")
    print("Exporting Centroids and Lists")
    print(f"{'='*70}")
    
    output_dir = Path(args.output_dir)
    centroids_path, lists_path = export_centroids_and_lists(index, output_dir)
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Index type: {args.index_type}")
    print(f"Clusters (K): {args.k}")
    print(f"Vectors: {len(vectors):,}")
    print(f"Dimension: {dim}")
    print(f"Centroids saved to: {centroids_path}")
    print(f"Lists saved to: {lists_path}")
    if args.test_recall:
        print(f"Recall@{args.recall_k}: {recall:.4f}")
        print(f"Average latency: {avg_latency:.2f} ms")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

