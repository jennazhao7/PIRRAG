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

import balanced_ivf
import ivf_eval
import ivf_io

# Default k-means seed. Pinned rather than left to FAISS's default so that a
# cluster-size sweep is reproducible and comparable across runs.
DEFAULT_KMEANS_SEED = 1234


def _configure_clustering(
    ivf_index: faiss.Index,
    niter: int,
    seed: int = DEFAULT_KMEANS_SEED,
    min_points_per_centroid: int = 1,
    verbose: bool = False,
) -> dict:
    """
    Apply k-means clustering parameters to an IVF index.

    This exists because ``--niter`` was previously accepted, printed, and then
    silently discarded -- nothing ever set ``index.cp.niter``, so FAISS's default
    of 10 governed every run regardless of what the log claimed.

    ``min_points_per_centroid`` is lowered from FAISS's default of 39 because
    points-per-centroid is approximately the target cluster size: at n=16 or
    n=32 the default would trip FAISS's "please provide at least N training
    points" path, while n=64 and n=128 would not. Left alone, a sweep would
    silently straddle two different clustering regimes.

    Must be called on the **CPU** index, before any ``index_cpu_to_gpu`` move --
    the GPU index does not expose ``cp`` the same way, so setting it afterwards
    is a silent no-op.

    Args:
        ivf_index: The IVF index (for ivf-opq, the inner IndexIVFPQ, before it is
            wrapped in IndexPreTransform).
        niter: k-means iterations.
        seed: k-means RNG seed.
        min_points_per_centroid: Lower bound before FAISS subsamples/warns.
        verbose: Whether FAISS should print clustering progress.

    Returns:
        The parameters actually applied, for the metadata record.
    """
    cp = ivf_index.cp
    cp.niter = int(niter)
    cp.seed = int(seed)
    cp.min_points_per_centroid = int(min_points_per_centroid)
    cp.verbose = bool(verbose)
    return {
        "niter": int(cp.niter),
        "seed": int(cp.seed),
        "min_points_per_centroid": int(cp.min_points_per_centroid),
        "max_points_per_centroid": int(cp.max_points_per_centroid),
    }


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

    # Imported lazily: this pulls in torch/sentence-transformers, which takes
    # seconds and is only needed to satisfy FAISS.load_local's constructor. Keeping
    # it out of module scope lets the sweep and the assignment code be imported
    # (and tested) without a model environment. See also --vectors-npy.
    from langchain_community.vectorstores import FAISS

    from wiki_rag.rag import PromptedBGE

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
    use_gpu: bool = False,
    skip_add: bool = False,
    seed: int = DEFAULT_KMEANS_SEED
) -> faiss.IndexIVFFlat:
    """
    Train IVF-flat index.

    Args:
        vectors: Training vectors (n_vectors, dim)
        n_clusters: Number of clusters (K)
        n_probe: Number of clusters to probe during search
        niter: Number of k-means iterations
        verbose: Whether to print progress
        skip_add: Train centroids but do not add vectors. Used by constant-size
            mode, where the assignment is computed externally and index.add()
            would overwrite it with plain nearest-centroid.
        seed: k-means RNG seed

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

    # Must precede any GPU move: the GPU index does not expose cp the same way.
    _configure_clustering(index, niter, seed=seed, verbose=False)

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
    if skip_add:
        print("\nSkipping index.add(): assignment is computed externally "
              "(constant-size mode)")
    else:
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
    use_gpu: bool = False,
    skip_add: bool = False,
    seed: int = DEFAULT_KMEANS_SEED
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
        skip_add: Train but do not add vectors (constant-size mode). Note this
            leaves PQ codes unwritten, so coded recall is not measurable.
        seed: k-means RNG seed

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

    # Coarse-quantizer k-means only. The PQ codebook has its own iteration count
    # (index.pq.cp.niter), deliberately left at the FAISS default.
    _configure_clustering(index, niter, seed=seed, verbose=False)

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
    if skip_add:
        print("\nSkipping index.add(): assignment is computed externally "
              "(constant-size mode)")
    else:
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
    use_gpu: bool = False,
    skip_add: bool = False,
    seed: int = DEFAULT_KMEANS_SEED
) -> faiss.IndexIVF:
    """
    Train IVF-OPQ index (Optimized Product Quantization with rotation).

    Note that the coarse centroids of an OPQ index live in the **rotated** space,
    so the exported centroids.npy is rotated too. Anything comparing a query
    against those centroids must apply the same rotation first; the exported
    metadata records this as centroid_space="opq-rotated".

    Args:
        vectors: Training vectors (n_vectors, dim)
        n_clusters: Number of clusters (K)
        m: Number of subquantizers (dim must be divisible by m)
        n_bits: Number of bits per subquantizer
        n_probe: Number of clusters to probe during search
        niter: Number of k-means iterations
        verbose: Whether to print progress
        skip_add: Train but do not add vectors (constant-size mode)
        seed: k-means RNG seed

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

    # Configure the inner IVF before it is wrapped -- IndexPreTransform does not
    # forward .cp. OPQMatrix.niter is a separate knob, left at the FAISS default.
    _configure_clustering(pq_index, niter, seed=seed, verbose=False)

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
    if skip_add:
        print("\nSkipping index.add(): assignment is computed externally "
              "(constant-size mode)")
    else:
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
    output_dir: Path,
    slots: Optional[np.ndarray] = None,
    lists_json_style: str = "padded",
    n_vectors: Optional[int] = None
):
    """
    Export centroids and inverted lists from IVF index.

    Args:
        index: FAISS IVF index
        output_dir: Output directory
        slots: (nlist, cluster_size) int32 slot array from a constant-size
            assignment. When given, the inverted lists come from here and the
            index's own invlists are ignored -- in constant-size mode the index
            was never populated, because index.add() would have overwritten the
            assignment with plain nearest-centroid.
        lists_json_style: "padded" (keep the sentinel, so every value has length
            cluster_size), "trimmed", or "none" to skip lists.json entirely.
        n_vectors: Expected vector count. Required when slots is given, since the
            index's ntotal is 0 in that mode.

    Returns:
        Dict of written paths plus the centroids array.
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
    
    # Constant-size mode: the assignment is authoritative and already computed.
    if slots is not None:
        if n_vectors is None:
            raise ValueError("n_vectors is required when exporting a slot array")
        if len(slots) != n_clusters:
            raise ValueError(
                f"slot array has {len(slots)} rows but there are {n_clusters} centroids"
            )

        stats = ivf_io.validate_slots(slots, n_vectors)
        slots_path = output_dir / ivf_io.SLOTS_FILENAME
        np.save(slots_path, slots)
        print(f"  Saved slot array {slots.shape} to {slots_path}")
        print(
            f"  Every cluster holds exactly {stats['cluster_size']} slots "
            f"({stats['n_real']:,} real + {stats['n_padded_slots']} padded)"
        )

        lists_path = None
        if lists_json_style != "none":
            lists_data = ivf_io.slots_to_lists_json(slots, style=lists_json_style)
            lists_path = output_dir / ivf_io.LISTS_FILENAME
            with open(lists_path, "w") as f:
                json.dump(lists_data, f, indent=2)
            print(f"  Saved {lists_json_style} lists.json to {lists_path}")

        return {
            "centroids_path": centroids_path,
            "lists_path": lists_path,
            "slots_path": slots_path,
            "centroids": centroids,
            "transform_matrix": transform_matrix,
            "slot_stats": stats,
        }

    # Legacy mode: read the assignment back out of the index's inverted lists.
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
    expected = ivf_index.ntotal if n_vectors is None else n_vectors
    print(f"  Saved inverted lists to {lists_path}")
    print(f"  Total vectors in lists: {total_vectors:,}")
    print(f"  Expected vectors: {expected:,}")

    if total_vectors != expected:
        print(f"  Warning: Vector count mismatch (may be due to reconstruction method)")

    sizes = np.array([len(v) for v in lists_data.values()])
    print(
        f"  Cluster size distribution: min={sizes.min()} max={sizes.max()} "
        f"mean={sizes.mean():.2f} median={np.median(sizes):.1f} "
        f"empty={(sizes == 0).sum()}"
    )

    return {
        "centroids_path": centroids_path,
        "lists_path": lists_path,
        "slots_path": None,
        "centroids": centroids,
        "transform_matrix": transform_matrix,
        "slot_stats": None,
    }


def unwrap_ivf(index: faiss.Index):
    """
    Unwrap an index to its IVF core and OPQ rotation, if any.

    Args:
        index: A trained IVF index, possibly wrapped in IndexPreTransform.

    Returns:
        (ivf_index, transform_matrix) where transform_matrix is the (dim, dim)
        OPQ rotation or None.
    """
    transform_matrix = None
    ivf_index = index
    if isinstance(index, faiss.IndexPreTransform):
        if isinstance(index.chain[0], faiss.OPQMatrix):
            transform_matrix = faiss.vector_to_array(index.chain[0].A).reshape(
                index.chain[0].d, index.chain[0].d
            )
        ivf_index = index.index
    return ivf_index, transform_matrix


def extract_centroids(ivf_index: faiss.Index) -> np.ndarray:
    """
    Read the coarse centroids out of an IVF index's quantizer.

    Args:
        ivf_index: A trained IVF index (not the IndexPreTransform wrapper).

    Returns:
        (nlist, dim) float32 centroids.
    """
    quantizer = ivf_index.quantizer
    return np.array(
        [quantizer.reconstruct(i) for i in range(ivf_index.nlist)], dtype=np.float32
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train IVF index on wiki-rag embeddings"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--faiss-path",
        type=str,
        help="Path to FAISS index directory"
    )
    source.add_argument(
        "--vectors-npy",
        type=str,
        help="Path to a (N, dim) float32 .npy of embeddings, bypassing the "
             "FAISS/embedding-model load entirely. Much faster for sweeps."
    )
    source.add_argument(
        "--faiss-index-file",
        type=str,
        help="Path to a raw index.faiss. Read with faiss.read_index, so no "
             "embedding model or langchain is needed -- the model is only "
             "required to satisfy FAISS.load_local's constructor, never to read "
             "stored vectors."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ivf_output",
        help="Output directory for centroids.npy, cluster_slots.npy, lists.json"
    )

    # Cluster sizing: fix the count (legacy) or fix the size (new).
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument(
        "--k",
        type=int,
        default=None,
        help="LEGACY: fixed number of clusters, variable cluster sizes. "
             f"Defaults to {ivf_io.DEFAULT_NLIST} if neither sizing flag is given."
    )
    sizing.add_argument(
        "--cluster-size",
        type=int,
        default=None,
        help="Constant cluster size n. The number of clusters is derived as "
             "ceil(N / n) and every inverted list holds exactly n slots, "
             "short-filled with -1. Mutually exclusive with --k."
    )
    parser.add_argument(
        "--assignment",
        type=str,
        choices=sorted(balanced_ivf._STRATEGIES),
        default="balanced",
        help="Assignment strategy for constant-size mode (default: balanced)"
    )
    parser.add_argument(
        "--assign-topm",
        type=int,
        default=balanced_ivf.DEFAULT_TOP_M,
        help="Candidate centroids considered per vector during assignment"
    )
    parser.add_argument(
        "--price-iters",
        type=int,
        default=balanced_ivf.DEFAULT_PRICE_ITERS,
        help="Subgradient iterations for the balanced assignment strategy"
    )
    parser.add_argument(
        "--lists-json",
        type=str,
        choices=["padded", "trimmed", "none"],
        default="padded",
        help="How to render lists.json in constant-size mode (default: padded, "
             "so every value has length n and the schema change is visible)"
    )
    parser.add_argument(
        "--dual-bound",
        action="store_true",
        help="Compute the certified lower bound on assignment cost (one extra "
             "full pass over all centroids)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_KMEANS_SEED,
        help="k-means RNG seed"
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
        "--recall-mode",
        type=str,
        choices=list(ivf_eval.RECALL_MODES),
        default="holdout",
        help="How test queries are drawn. 'holdout' removes them from the "
             "database and the clustering (the honest number). 'perturbed' keeps "
             "the database whole and jitters the query. 'self' is the original "
             "behaviour, where every query is its own guaranteed top-1 hit -- it "
             "is reported as self_retrieval_recall@k, not recall@k."
    )
    parser.add_argument(
        "--query-noise",
        type=float,
        default=0.05,
        help="Gaussian sigma applied to queries in --recall-mode perturbed"
    )
    parser.add_argument(
        "--nprobe-grid",
        type=int,
        nargs="*",
        default=None,
        help="nprobe values to evaluate (default: --n-probe only)"
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
    return parser


def run_pipeline(
    args,
    vectors: Optional[np.ndarray] = None,
    dim: Optional[int] = None,
    query_set: Optional[dict] = None,
    verbose: bool = True
) -> dict:
    """
    Train, assign, evaluate and export one clustering configuration.

    Returns the metadata dict, which doubles as the per-row record for a sweep.
    Accepts pre-loaded vectors and a pre-built query set so a sweep can pay the
    load and ground-truth costs once across many cluster sizes.

    Args:
        args: Parsed arguments from build_parser().
        vectors: Optional pre-loaded (N, dim) float32 corpus.
        dim: Optional embedding dimension (inferred from vectors if omitted).
        query_set: Optional pre-built query set from ivf_eval.build_query_set.
            Sharing one across cluster sizes is what makes a sweep comparable.
        verbose: Print progress.

    Returns:
        The metadata dict as written to ivf_metadata.json.
    """
    # Check GPU availability
    if args.use_gpu:
        if get_gpu_resources() is None:
            print("Warning: --use-gpu specified but GPU not available. Falling back to CPU.")
            print("  Make sure FAISS is compiled with GPU support and CUDA is available.")
            args.use_gpu = False
        else:
            print("GPU support detected and enabled!")
    elif verbose:
        print("Using CPU (use --use-gpu for faster training on large datasets)")

    # ---- Load the corpus -------------------------------------------------
    if vectors is None:
        if args.vectors_npy:
            print(f"Loading vectors from {args.vectors_npy}...")
            vectors = np.ascontiguousarray(np.load(args.vectors_npy), dtype=np.float32)
            print(f"Loaded embeddings shape: {vectors.shape}")
        elif getattr(args, "faiss_index_file", None):
            print(f"Reading FAISS index from {args.faiss_index_file}...")
            raw = faiss.read_index(str(args.faiss_index_file))
            print(f"  ntotal={raw.ntotal:,} d={raw.d}")
            vectors = np.ascontiguousarray(
                raw.reconstruct_n(0, raw.ntotal), dtype=np.float32
            )
        else:
            vectors, dim = load_embeddings_from_faiss(Path(args.faiss_path))
    if dim is None:
        dim = int(vectors.shape[1])

    # ---- Query set, built BEFORE clustering -----------------------------
    # In holdout mode the database is a strict subset, so the clustering must be
    # trained on the reduced set or the held-out queries would not be held out.
    if args.test_recall and query_set is None:
        query_set = ivf_eval.build_query_set(
            vectors,
            mode=args.recall_mode,
            n_test=args.n_test_queries,
            k=args.recall_k,
            seed=args.seed,
            query_noise=args.query_noise,
            verbose=verbose,
        )
    db_vectors = query_set["db_vectors"] if query_set is not None else vectors
    n_vectors = len(db_vectors)

    # ---- Resolve the sizing parameterization ----------------------------
    cfg = ivf_io.resolve_cluster_config(
        n_vectors, k=args.k, cluster_size=args.cluster_size
    )
    constant_size = cfg["sizing_mode"] == ivf_io.SIZING_CONSTANT_SIZE
    if constant_size and args.assignment not in balanced_ivf.CONSTANT_SIZE_STRATEGIES:
        raise ValueError(
            f"--assignment {args.assignment} does not produce constant-size "
            f"clusters; use one of {balanced_ivf.CONSTANT_SIZE_STRATEGIES}"
        )

    print(f"\n{'='*70}")
    print(f"Training {args.index_type.upper()} index")
    print(f"  sizing: {cfg['sizing_mode']}  nlist={cfg['nlist']:,}"
          + (f"  cluster_size={cfg['cluster_size']}" if constant_size else ""))
    print(f"{'='*70}")

    trainers = {
        "ivf-flat": lambda: train_ivf_flat(
            db_vectors, n_clusters=cfg["nlist"], n_probe=args.n_probe,
            niter=args.niter, use_gpu=args.use_gpu, skip_add=constant_size,
            seed=args.seed,
        ),
        "ivf-pq": lambda: train_ivf_pq(
            db_vectors, n_clusters=cfg["nlist"], m=args.m, n_bits=args.n_bits,
            n_probe=args.n_probe, niter=args.niter, use_gpu=args.use_gpu,
            skip_add=constant_size, seed=args.seed,
        ),
        "ivf-opq": lambda: train_ivf_opq(
            db_vectors, n_clusters=cfg["nlist"], m=args.m, n_bits=args.n_bits,
            n_probe=args.n_probe, niter=args.niter, use_gpu=args.use_gpu,
            skip_add=constant_size, seed=args.seed,
        ),
    }
    index = trainers[args.index_type]()
    ivf_index, transform_matrix = unwrap_ivf(index)
    kmeans_params = _configure_clustering(ivf_index, args.niter, seed=args.seed)

    if constant_size and args.index_type != "ivf-flat":
        print("  Note: PQ codes are not written in constant-size mode, so "
              "PQ-coded recall is not measured.")

    # ---- Assignment ------------------------------------------------------
    slots = None
    assignment = None
    diagnostics = None
    if constant_size:
        print(f"\n{'='*70}")
        print("Capacity-Constrained Assignment")
        print(f"{'='*70}")

        centroids = extract_centroids(ivf_index)
        # For ivf-opq the centroids live in the rotated space, so the assignment
        # must be computed there too.
        assign_vectors = db_vectors
        if transform_matrix is not None:
            assign_vectors = np.ascontiguousarray(db_vectors @ transform_matrix.T)

        assignment = balanced_ivf.assign_constant_size(
            assign_vectors, centroids, cfg["cluster_size"],
            strategy=args.assignment, quantizer=ivf_index.quantizer,
            top_m=args.assign_topm, price_iters=args.price_iters, verbose=verbose,
        )
        slots = assignment.slots

        dual = None
        if args.dual_bound:
            if assignment.prices is None:
                print("  --dual-bound needs the balanced strategy's prices; "
                      f"strategy {args.assignment!r} does not produce them. Skipping.")
            else:
                print("  Computing certified dual lower bound "
                      "(one full pass over all centroids)...")
                t0 = time.time()
                dual = balanced_ivf.dual_lower_bound(
                    assign_vectors, centroids, assignment.prices, cfg["cluster_size"]
                )
                print(f"    dual bound {dual:.4f} in {time.time() - t0:.2f}s")

        diagnostics = ivf_eval.compute_balance_diagnostics(
            assignment, cfg["cluster_size"], n_vectors, dual_bound=dual
        )

    # ---- Export ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("Exporting Centroids and Lists")
    print(f"{'='*70}")

    output_dir = Path(args.output_dir)
    exported = export_centroids_and_lists(
        index, output_dir, slots=slots,
        lists_json_style=args.lists_json, n_vectors=n_vectors,
    )
    centroids = exported["centroids"]

    # In holdout mode the exported ids index the held-out DB subset, not the
    # original corpus. Record the mapping so the artifact is not mistaken for a
    # production one.
    id_space = "database"
    if query_set is not None and query_set["mode"] == "holdout":
        id_space = "holdout_subset"
        np.save(output_dir / "db_ids.npy", query_set["db_ids"])
        print(f"  Saved holdout DB id mapping to {output_dir / 'db_ids.npy'}")

    # ---- Evaluate --------------------------------------------------------
    recall_results = {}
    if args.test_recall:
        print(f"\n{'='*70}")
        print(f"Measuring {query_set['metric_name']} and latency (numpy)")
        print(f"{'='*70}")
        grid = args.nprobe_grid if args.nprobe_grid else [args.n_probe]
        lists_for_search = (
            slots if slots is not None
            else balanced_ivf.nearest_centroid_lists(
                db_vectors, centroids, quantizer=ivf_index.quantizer
            )
        )
        recall_results = ivf_eval.evaluate_clustering(
            query_set, centroids, lists_for_search, grid,
            transform=transform_matrix, verbose=verbose,
        )

    # ---- Metadata --------------------------------------------------------
    metadata = {
        "sizing_mode": cfg["sizing_mode"],
        "n_vectors": int(n_vectors),
        "dim": int(dim),
        "nlist": int(cfg["nlist"]),
        "cluster_size": cfg["cluster_size"],
        "total_slots": int(cfg["total_slots"]),
        "n_padded_slots": int(cfg["n_padded_slots"]),
        "padding_sentinel": ivf_io.PADDING_SENTINEL,
        "id_mapping": (
            "global_vector_id = slots[cid][idx]; "
            "slot_address = cid * cluster_size + idx"
            if constant_size else "global_vector_id = lists[cid][idx]"
        ),
        "id_space": id_space,
        "assignment": args.assignment if constant_size else "faiss-nearest",
        "assignment_params": (
            {"top_m": args.assign_topm, "price_iters": args.price_iters}
            if constant_size else {}
        ),
        "index_type": args.index_type,
        "centroid_space": "opq-rotated" if transform_matrix is not None else "raw",
        "nprobe": int(args.n_probe),
        "recall_k": int(args.recall_k),
        "pq": ({"m": args.m, "n_bits": args.n_bits}
               if args.index_type in ("ivf-pq", "ivf-opq") else None),
        "kmeans": kmeans_params,
        "recall_mode": query_set["mode"] if query_set else None,
        "recall_metric_name": query_set["metric_name"] if query_set else None,
        "recall_by_nprobe": {str(p): v for p, v in recall_results.items()},
        "artifacts": {
            "centroids": ivf_io.CENTROIDS_FILENAME,
            "slots": ivf_io.SLOTS_FILENAME if slots is not None else None,
            "lists": ivf_io.LISTS_FILENAME if args.lists_json != "none" else None,
        },
        "diagnostics": diagnostics,
    }
    if assignment is not None:
        metadata["assignment_params"].update(assignment.diagnostics)

    metadata_path = ivf_io.write_ivf_metadata(output_dir, metadata)
    print(f"  Saved metadata to {metadata_path}")

    if diagnostics is not None:
        report_path = ivf_eval.write_balance_report(
            output_dir, diagnostics,
            header=(f"cluster_size={cfg['cluster_size']} nlist={cfg['nlist']} "
                    f"n_vectors={n_vectors} strategy={args.assignment}"),
        )
        print(f"  Saved balance report to {report_path}")

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Index type: {args.index_type}")
    print(f"Sizing mode: {cfg['sizing_mode']}")
    print(f"Clusters (nlist): {cfg['nlist']:,}")
    if constant_size:
        print(f"Cluster size (n): {cfg['cluster_size']}")
        print(f"Padded slots: {cfg['n_padded_slots']}")
        print(f"Balance penalty ratio: {diagnostics['balance_penalty_ratio']:.6f}")
    print(f"Vectors: {n_vectors:,}")
    print(f"Dimension: {dim}")
    print(f"Output: {output_dir}")
    for p, v in sorted(recall_results.items()):
        print(f"{query_set['metric_name']} @ nprobe={p}: {v['recall']:.4f} "
              f"({v['latency_ms_mean']:.2f} ms, numpy)")
    print(f"{'='*70}")

    return metadata


def main():
    args = build_parser().parse_args()
    try:
        run_pipeline(args)
    except ValueError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()

