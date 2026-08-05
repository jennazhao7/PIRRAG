#!/usr/bin/env python3
"""
Compute ground truth: Top-5 vectors within each of the top-100 clusters.

For each cluster in the top-100, finds the 5 closest vectors to the query.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_utils import PromptedBGE

# The clustering I/O helpers live with the training code. These prototype scripts
# are run standalone rather than installed, so point at that directory directly.
_WIKI_RAG_DIR = Path(__file__).resolve().parents[1] / "wiki-rag"
if str(_WIKI_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_WIKI_RAG_DIR))

import ivf_io


def load_query_embedding(query_text: str) -> np.ndarray:
    """Load or compute query embedding."""
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    query_vec = embeddings.embed_query(query_text)
    return np.array(query_vec, dtype=np.float32)


def load_vectors_from_faiss(faiss_path: Path) -> Tuple[np.ndarray, int]:
    """Load all vectors from FAISS index."""
    faiss_path = Path(faiss_path)
    
    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: {faiss_path}\n"
            "Please provide the path to the FAISS index directory containing index.faiss and index.pkl"
        )
    
    print(f"Loading vectors from FAISS index: {faiss_path}...")
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    
    try:
        vectorstore = FAISS.load_local(
            str(faiss_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load FAISS index from {faiss_path}: {e}\n"
            "Make sure the directory contains index.faiss and index.pkl files"
        )
    
    index = vectorstore.index
    n_vectors = index.ntotal
    dim = index.d
    
    print(f"Found {n_vectors} vectors with dimension {dim}")
    
    # Extract all vectors
    try:
        vectors = index.reconstruct_n(0, n_vectors)
        print(f"Successfully reconstructed {len(vectors)} vectors")
    except Exception as e:
        print(f"Could not reconstruct all at once: {e}")
        print("Reconstructing one by one (this may take a while)...")
        vectors = []
        for i in range(n_vectors):
            if (i + 1) % 10000 == 0:
                print(f"  Reconstructed {i + 1}/{n_vectors}...")
            vec = index.reconstruct(i)
            vectors.append(vec)
        vectors = np.array(vectors)
    
    return np.array(vectors, dtype=np.float32), dim


def compute_ground_truth(
    query_vector: np.ndarray,
    top_cluster_indices: List[int],
    lists: Dict[str, List[int]],
    all_vectors: np.ndarray,
    top_k_per_cluster: int = 5
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Compute ground truth: top-k vectors within each top cluster.
    
    Args:
        query_vector: Query embedding vector
        top_cluster_indices: List of top cluster indices
        lists: Cluster-to-vectors mapping
        all_vectors: All document vectors
        top_k_per_cluster: Number of top vectors per cluster
        
    Returns:
        Dict mapping cluster_id -> [(vector_idx, distance), ...]
    """
    print(f"\nComputing ground truth for {len(top_cluster_indices)} clusters...")
    print(f"  Top-k per cluster: {top_k_per_cluster}")
    
    ground_truth = {}
    
    for cluster_idx in top_cluster_indices:
        cluster_id = str(cluster_idx)
        
        if cluster_id not in lists:
            print(f"  Warning: Cluster {cluster_idx} not found in lists")
            ground_truth[cluster_idx] = []
            continue
        
        # Get vectors in this cluster
        vector_indices = lists[cluster_id]
        
        if len(vector_indices) == 0:
            ground_truth[cluster_idx] = []
            continue
        
        # Filter vector indices to only those that exist in all_vectors
        valid_indices = [idx for idx in vector_indices if idx < len(all_vectors)]
        
        if len(valid_indices) == 0:
            print(f"  Warning: Cluster {cluster_idx} has no valid vectors (all indices out of range)")
            ground_truth[cluster_idx] = []
            continue
        
        # Extract vectors for this cluster
        cluster_vectors = all_vectors[valid_indices]
        
        # Compute distances: ||q - v||^2 = ||q||^2 + ||v||^2 - 2<q, v>
        query_norm_sq = np.sum(query_vector ** 2)
        vector_norms_sq = np.sum(cluster_vectors ** 2, axis=1)
        dot_products = np.dot(cluster_vectors, query_vector)
        
        distances = query_norm_sq + vector_norms_sq - 2 * dot_products
        
        # Get top-k smallest distances
        top_k_indices_local = np.argsort(distances)[:top_k_per_cluster]
        
        # Map back to global vector indices (using valid_indices, not original vector_indices)
        top_results = [
            (valid_indices[idx], float(distances[idx]))
            for idx in top_k_indices_local
        ]
        
        ground_truth[cluster_idx] = top_results
    
    print(f"✓ Computed ground truth for {len(ground_truth)} clusters")
    
    return ground_truth


def save_ground_truth(
    ground_truth: Dict[int, List[Tuple[int, float]]],
    output_path: Path,
    query_text: str = None
):
    """Save ground truth to files."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    gt_json = {
        "query": query_text,
        "n_clusters": len(ground_truth),
        "top_k_per_cluster": len(list(ground_truth.values())[0]) if ground_truth else 0,
        "ground_truth": {
            str(cluster_id): [
                {"vector_index": vec_idx, "distance": float(dist)}
                for vec_idx, dist in results
            ]
            for cluster_id, results in ground_truth.items()
        }
    }
    
    # Save JSON
    json_file = output_path / "ground_truth.json"
    with open(json_file, 'w') as f:
        json.dump(gt_json, f, indent=2)
    print(f"✓ Saved ground truth to {json_file}")
    
    # Save summary
    summary_file = output_path / "ground_truth_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Ground Truth Summary\n")
        f.write("=" * 70 + "\n\n")
        if query_text:
            f.write(f"Query: {query_text}\n\n")
        f.write(f"Number of clusters: {len(ground_truth)}\n")
        f.write(f"Top-k per cluster: {gt_json['top_k_per_cluster']}\n\n")
        f.write("Top-10 Clusters:\n")
        for i, (cluster_id, results) in enumerate(list(ground_truth.items())[:10]):
            f.write(f"\nCluster {cluster_id}:\n")
            for j, (vec_idx, dist) in enumerate(results):
                f.write(f"  {j+1}. Vector {vec_idx}: distance = {dist:.4f}\n")
    
    print(f"✓ Saved summary to {summary_file}")
    
    return json_file, summary_file


def main():
    parser = argparse.ArgumentParser(
        description="Compute ground truth: top-5 vectors within top-100 clusters"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text"
    )
    parser.add_argument(
        "--top-results",
        type=str,
        default="./workflow_decrypted_results/top_k_results.json",
        help="Path to top-k results JSON file"
    )
    parser.add_argument(
        "--lists",
        type=str,
        default="./data/lists.json",
        help="Path to lists.json file"
    )
    parser.add_argument(
        "--faiss-path",
        type=str,
        required=True,
        help="Path to FAISS index directory (to load vectors). Example: ../wiki-rag/wiki_rag_data/wiki_index__top_100000__2025-04-11"
    )
    parser.add_argument(
        "--top-clusters",
        type=int,
        default=100,
        help="Number of top clusters to process"
    )
    parser.add_argument(
        "--top-k-per-cluster",
        type=int,
        default=5,
        help="Number of top vectors per cluster"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ground_truth",
        help="Output directory for ground truth"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Computing Ground Truth")
    print("=" * 70)
    print()
    
    # Load top cluster indices
    print(f"Loading top-{args.top_clusters} cluster indices...")
    with open(args.top_results, 'r') as f:
        results = json.load(f)
    top_cluster_indices = results['centroid_indices'][:args.top_clusters]
    print(f"✓ Loaded {len(top_cluster_indices)} cluster indices")
    
    # Load cluster lists. Works with both the legacy variable-size layout and the
    # constant-cluster-size layout, whose lists are padded to a fixed width with a
    # sentinel; load_clustering strips those so the code below never sees them.
    print(f"\nLoading cluster lists from {args.lists}...")
    clustering = ivf_io.load_clustering(args.lists)
    lists = {str(cid): ids for cid, ids in clustering.lists.items()}
    print(f"✓ Loaded {len(lists)} clusters ({clustering.sizing_mode})")
    if clustering.is_constant_size:
        print(f"  {clustering.cluster_size} slots per cluster, "
              f"{clustering.n_vectors:,} real vectors")
    
    # Load query embedding
    print(f"\nComputing query embedding...")
    query_vector = load_query_embedding(args.query)
    print(f"✓ Query vector shape: {query_vector.shape}")
    
    # Load all vectors from FAISS
    if not args.faiss_path:
        raise ValueError(
            "FAISS path is required for ground truth computation. "
            "Please provide --faiss-path pointing to the FAISS index directory."
        )
    
    print(f"\nLoading vectors from FAISS index...")
    all_vectors, dim = load_vectors_from_faiss(Path(args.faiss_path))
    print(f"✓ Loaded {len(all_vectors)} vectors")
    
    # Compute ground truth
    ground_truth = compute_ground_truth(
        query_vector,
        top_cluster_indices,
        lists,
        all_vectors,
        top_k_per_cluster=args.top_k_per_cluster
    )
    
    # Save ground truth
    print(f"\n{'='*70}")
    print("Saving Ground Truth")
    print(f"{'='*70}")
    json_file, summary_file = save_ground_truth(
        ground_truth,
        Path(args.output_dir),
        query_text=args.query
    )
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Query: {args.query}")
    print(f"Top clusters processed: {len(top_cluster_indices)}")
    print(f"Top-k per cluster: {args.top_k_per_cluster}")
    print(f"Total vectors found: {sum(len(v) for v in ground_truth.values())}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

