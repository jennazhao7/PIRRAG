#!/usr/bin/env python3
"""
Plaintext RAG Pipeline after FHE Query.

Pipeline:
1. Load top-P cluster IDs from FHE query results
2. Embed query → vector q
3. Within each of those clusters, get top-k vectors (per-cluster KNN)
4. Merge all candidates and do a global top-10 (local KNN / sort)
5. Map those 10 vector IDs → RAG chunks using the pickle mapping
6. Build RAG prompt and call LLM

Works with both clustering parameterizations: the legacy fixed-cluster-count
layout (variable sizes) and the constant-cluster-size layout, whose inverted
lists are padded to a fixed width with a sentinel that must be skipped. See
wiki-rag/ivf_io.py for the schema.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import faiss
from langchain_community.vectorstores import FAISS

# The clustering I/O helpers live with the training code. These prototype scripts
# are run standalone rather than installed, so point at that directory directly.
_WIKI_RAG_DIR = Path(__file__).resolve().parents[2] / "wiki-rag"
if str(_WIKI_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_WIKI_RAG_DIR))

import ivf_io
from langchain_core.documents import Document
import sys
from pathlib import Path
from prototype.go_runner import run_built_executable
import re
# Add parent directory to path to import rag_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_utils import PromptedBGE


def load_top_clusters(top_results_path: str, n_clusters: int = 100) -> List[int]:
    """Load top-N cluster IDs from FHE query results."""
    with open(top_results_path, 'r') as f:
        results = json.load(f)
    return results['centroid_indices'][:n_clusters]


def load_cluster_lists(lists_path: str) -> Dict[str, List[int]]:
    """
    Load the cluster-to-vectors mapping.

    Handles both layouts. Under constant cluster size every list is padded to a
    fixed width with ivf_io.PADDING_SENTINEL; those entries are stripped here so
    that callers never see them. Keys stay strings for backward compatibility
    with the existing call sites.
    """
    clustering = ivf_io.load_clustering(lists_path)
    if clustering.is_constant_size:
        print(f"  Constant-size clustering: {clustering.nlist} clusters "
              f"x {clustering.cluster_size} slots "
              f"({clustering.n_vectors:,} real vectors)")
    return {str(cid): ids for cid, ids in clustering.lists.items()}


def embed_query(query_text: str) -> np.ndarray:
    """Embed query → vector q."""
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    query_vec = embeddings.embed_query(query_text)
    return np.array(query_vec, dtype=np.float32)


def per_cluster_knn(
    query_vector: np.ndarray,
    cluster_id: int,
    vector_indices: List[int],
    all_vectors: np.ndarray,
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Get top-k vectors within a cluster (per-cluster KNN).
    
    Returns:
        List of (vector_index, distance) tuples
    """
    if len(vector_indices) == 0:
        return []
    
    # Filter to valid indices
    valid_indices = [idx for idx in vector_indices if idx < len(all_vectors)]
    if len(valid_indices) == 0:
        return []
    
    # Extract vectors for this cluster
    cluster_vectors = all_vectors[valid_indices]
    
    # Compute distances: ||q - v||^2 = ||q||^2 + ||v||^2 - 2<q, v>
    query_norm_sq = np.sum(query_vector ** 2)
    vector_norms_sq = np.sum(cluster_vectors ** 2, axis=1)
    dot_products = np.dot(cluster_vectors, query_vector)
    
    distances = query_norm_sq + vector_norms_sq - 2 * dot_products
    
    # Get top-k smallest distances
    top_k_indices_local = np.argsort(distances)[:top_k]
    
    # Map back to global vector indices
    results = [
        (valid_indices[idx], float(distances[idx]))
        for idx in top_k_indices_local
    ]
    
    return results


def global_top_k(
    candidates: List[Tuple[int, float]],
    query_vector: np.ndarray,
    all_vectors: np.ndarray,
    top_k: int = 10,
    use_faiss: bool = False
) -> List[Tuple[int, float]]:
    """
    Merge all candidates and do a global top-k sort.
    
    Can use pure NumPy (default) or FAISS IndexFlatL2 for KNN.
    
    Args:
        candidates: List of (vector_index, distance) tuples
        query_vector: Query embedding vector
        all_vectors: All document vectors
        top_k: Number of top results to return
        use_faiss: Whether to use FAISS IndexFlatL2 (vs pure NumPy)
        
    Returns:
        Top-k (vector_index, distance) tuples sorted by distance
    """
    if use_faiss and len(candidates) > 0:
        # ====================================================================
        # FAISS INTERACTION: Building temporary FAISS IndexFlatL2 for KNN
        # ====================================================================
        # - Creates a small FAISS index on candidate vectors
        # - Uses FAISS search() for efficient distance computation
        # - Alternative to pure NumPy sorting (faster for large candidate sets)
        candidate_indices = [idx for idx, _ in candidates]
        candidate_vectors = all_vectors[candidate_indices]
        
        # Create FAISS index
        dim = candidate_vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(candidate_vectors.astype('float32'))
        
        # Query
        query_reshaped = query_vector.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_reshaped, min(top_k, len(candidates)))
        
        # Map back to original indices
        results = [
            (candidate_indices[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]
        return results
    else:
        # Pure NumPy: sort by distance (ascending)
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        return candidates_sorted[:top_k]


def pir_global_top_k(
        indexes: List[str],
        query_vector: np.ndarray,
        all_vectors: np.ndarray,
        top_k: int = 10,
        use_faiss: bool = True
) -> List[Tuple[int, float]]:
    """
    Merge all candidates and do a global top-k sort.

    Can use pure NumPy (default) or FAISS IndexFlatL2 for KNN.

    Args:
        candidates: List of (vector_index, distance) tuples
        query_vector: Query embedding vector
        all_vectors: All document vectors
        top_k: Number of top results to return
        use_faiss: Whether to use FAISS IndexFlatL2 (vs pure NumPy)

    Returns:
        Top-k (vector_index, distance) tuples sorted by distance
    """
    if use_faiss and len(indexes) > 0:
        # ====================================================================
        # FAISS INTERACTION: Building temporary FAISS IndexFlatL2 for KNN
        # ====================================================================
        # - Creates a small FAISS index on candidate vectors
        # - Uses FAISS search() for efficient distance computation
        # - Alternative to pure NumPy sorting (faster for large candidate sets)
        candidate_indices = indexes
        candidate_vectors = all_vectors

        # Create FAISS index
        dim = candidate_vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(candidate_vectors.astype('float32'))

        # Query
        query_reshaped = query_vector.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_reshaped, k=top_k)

        # Map back to original indices
        results = [
            (candidate_indices[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]
        return results
    return [(0,0)]


def extract_query_results(log_string):
    """
    Extract cluster IDs, indices, and vector arrays from query result log strings.

    Parameters:
    -----------
    log_string : str
        The log string containing query results with cluster information

    Returns:
    --------
    tuple: (cluster_ids, indices, vectors)
        cluster_ids: list of int - the cluster IDs for each query result
        indices: list of int - the extracted indices
        vectors: np.ndarray - array of shape (n, vector_dim) containing the vectors
    """
    cluster_ids = []
    indices = []
    vectors = []

    # Split into lines
    lines = log_string.strip().split('\n')

    for line in lines:
        # Look for lines that contain "Final query result from cluster"
        if "Final query result from cluster" in line:
            # Extract the cluster ID and index
            # Pattern: "from cluster <cluster_id> at index <index>:"
            match = re.search(r'from cluster (\d+) at index (\d+):', line)
            if match:
                cluster_id = int(match.group(1))
                index = int(match.group(2))

                # Extract the vector (everything inside the square brackets)
                vector_match = re.search(r'\[(.*?)\]', line)
                if vector_match:
                    vector_str = vector_match.group(1)
                    # Split by whitespace and convert to floats
                    vector = np.array([float(x) for x in vector_str.split()])

                    cluster_ids.append(cluster_id)
                    indices.append(index)
                    vectors.append(vector)

    # Convert list of vectors to numpy array
    if vectors:
        vectors = np.array(vectors)
    else:
        vectors = np.array([])

    return cluster_ids, indices, vectors

def load_faiss_vectorstore(faiss_path: Path) -> FAISS:
    """
    Load FAISS vectorstore.
    
    ====================================================================
    FAISS INTERACTION: Loading FAISS index from disk
    ====================================================================
    - Loads index.faiss (vector index) and index.pkl (documents/metadata)
    - Returns FAISS vectorstore object containing both vectors and documents
    """
    print(f"Loading FAISS vectorstore from {faiss_path}...")
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    vectorstore = FAISS.load_local(
        str(faiss_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Loaded vectorstore with {vectorstore.index.ntotal} vectors")
    return vectorstore


def get_documents_by_indices(
    vectorstore: FAISS,
    vector_indices: List[int]
) -> List[Document]:
    """
    Map vector IDs → RAG chunks using the pickle mapping.
    
    ====================================================================
    DOCUMENT/PICKLE INTERACTION: Retrieving actual document content
    ====================================================================
    - Accesses docstore (loaded from index.pkl) to get document objects
    - Uses index_to_docstore_id mapping to translate vector index → doc ID
    - Returns Document objects with page_content and metadata (title, etc.)
    
    Args:
        vectorstore: FAISS vectorstore
        vector_indices: List of vector indices
        
    Returns:
        List of Document objects
    """
    documents = []
    
    # ====================================================================
    # DOCUMENT/PICKLE INTERACTION: Accessing docstore
    # ====================================================================
    # - docstore contains the actual document text and metadata
    # - Loaded from index.pkl file (pickle format)
    # - docstore._dict maps doc_id → Document object
    docstore = vectorstore.docstore
    
    for idx in vector_indices:
        try:
            # ====================================================================
            # DOCUMENT/PICKLE INTERACTION: Mapping vector index → document
            # ====================================================================
            # - index_to_docstore_id: maps FAISS vector index → document ID
            # - docstore._dict[doc_id]: retrieves Document object with content
            # - Document contains: page_content (text) and metadata (title, etc.)
            if hasattr(vectorstore, 'index_to_docstore_id'):
                doc_id = vectorstore.index_to_docstore_id.get(idx)
                if doc_id and doc_id in docstore._dict:
                    doc = docstore._dict[doc_id]
                    documents.append(doc)
                else:
                    # Fallback: try direct index access
                    if idx in docstore._dict:
                        doc = docstore._dict[idx]
                        documents.append(doc)
                    else:
                        print(f"  Warning: Could not find document for vector index {idx}")
            else:
                # Fallback: try direct access
                if idx in docstore._dict:
                    doc = docstore._dict[idx]
                    documents.append(doc)
        except Exception as e:
            print(f"  Warning: Error getting document for index {idx}: {e}")
    
    return documents


def save_decrypted_results(
            top_k_distances: np.ndarray,
            top_k_indices: np.ndarray,
            output_path: Path,
            centroids_path: Path = None
    ):
        """
        Save decrypted top-k results to files.

        Args:
            top_k_distances: Top-k distances (sorted)
            top_k_indices: Top-k centroid indices (sorted)
            output_path: Output directory
            centroids_path: Optional path to centroids.npy for additional info
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save distances and indices
        results = {
            "top_k": len(top_k_distances),
            "distances": top_k_distances.tolist(),
            "centroid_indices": top_k_indices.tolist(),
            "min_distance": float(top_k_distances.min()),
            "max_distance": float(top_k_distances.max()),
            "mean_distance": float(top_k_distances.mean())
        }

        results_file = output_path / "ground_truth.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved results to {results_file}")

        # Save as numpy arrays for easy loading
        # np.save(output_path / "top_k_distances.npy", top_k_distances)
        # np.save(output_path / "top_k_indices.npy", top_k_indices)
        # print(f"✓ Saved numpy arrays to {output_path}")

        # Print summary
        print(f"\nTop-{len(top_k_distances)} Results:")
        print(f"  Best match: Centroid {top_k_indices[0]} (distance: {top_k_distances[0]:.4f})")
        print(f"  Worst match: Centroid {top_k_indices[-1]} (distance: {top_k_distances[-1]:.4f})")

        return results_file

def extract_text_query_results(log_string):
    """
    Extract indices and text results from query result log strings.

    Parameters:
    -----------
    log_string : str
        The log string containing text query results

    Returns:
    --------
    tuple: (indices, texts)
        indices: list of int - the extracted indices
        texts: list of str - the corresponding text results
    """
    indices = []
    texts = []

    # Split into lines
    lines = log_string.strip().split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for lines that contain "Final query result at index"
        if "Final query result at index" in line:
            # Extract the index number
            match = re.search(r'at index (\d+):', line)
            if match:
                index = int(match.group(1))

                # Extract the text after the colon
                # Split at ': ' and take everything after
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    text = parts[1].strip()

                    indices.append(index)
                    texts.append(text)

        i += 1

    return indices, texts

def pir_get_documents_by_indices(
        vector_indices: List[int]
) -> List[Document]:
    stdout, stderr, code = run_built_executable(
        "/Users/antoniajanuszewicz/GolandProjects/Piano-PIR-RAG/client_exe",
        args=["-ip", "localhost:50051", "-thread", "1", "-input",
              "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/rag_operations/ground_truth.json"],
        timeout=60
    )
    #print(stderr)
    indices, text = extract_text_query_results(stderr)
    documents = text

    # ====================================================================
    # DOCUMENT/PICKLE INTERACTION: Accessing docstore
    # ====================================================================
    # - docstore contains the actual document text and metadata
    # - Loaded from index.pkl file (pickle format)
    # - docstore._dict maps doc_id → Document object


    return documents

def build_rag_prompt(query: str, documents: List[Document]) -> str:
    """
    Build RAG prompt from query and retrieved documents.
    
    Args:
        query: User query
        documents: Retrieved documents
        
    Returns:
        Formatted prompt string
    """
    context_parts = []
    for i, doc in enumerate(documents, 1):
        if (type(doc) == str):
            context_parts.append(f"[{i}] Content: {doc}")
        else:
            title = doc.metadata.get('title', 'Unknown')
            content = doc.page_content[:500]  # Truncate long content
            context_parts.append(f"[{i}] Title: {title}\nContent: {content}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following context documents, answer the question.

Context Documents:
{context}

Question: {query}

Answer:"""
    
    return prompt


def call_llm(prompt: str, model_name: str = None) -> str:
    """
    Call LLM with the RAG prompt.
    
    For now, returns a placeholder. In production, this would call
    OpenAI API, Anthropic API, or a local LLM.
    """
    # Placeholder - in production, replace with actual LLM call
    return f"[LLM Response Placeholder]\n\nPrompt length: {len(prompt)} characters\n\nThis would call an LLM API (OpenAI, Anthropic, etc.) with the prompt above."


def run_plaintext_rag_pipeline(
    query: str,
    top_clusters: List[int],
    lists: Dict[str, List[int]],
    faiss_path: Path,
    faiss_vectorstore: FAISS = None,
    per_cluster_k: int = 5,
    global_k: int = 10,
    use_faiss_for_knn: bool = False
) -> Dict:
    """
    Run the complete plaintext RAG pipeline.
    
    Args:
        query: User query text
        top_clusters: List of top cluster IDs
        lists: Cluster-to-vectors mapping
        faiss_path: Path to FAISS index
        faiss_vectorstore: Optional pre-loaded vectorstore
        per_cluster_k: Top-k vectors per cluster
        global_k: Global top-k to return
        use_faiss_for_knn: Whether to use FAISS for KNN (vs pure NumPy)
        
    Returns:
        Dictionary with pipeline results
    """
    print("=" * 70)
    print("Plaintext RAG Pipeline")
    print("=" * 70)
    print(f"\nQuery: {query}")
    print(f"Top clusters: {len(top_clusters)}")
    print(f"Per-cluster K: {per_cluster_k}")
    print(f"Global K: {global_k}")
    
    # Step 1: Embed query
    print(f"\n{'='*70}")
    print("Step 1: Embedding query...")
    query_vector = embed_query(query)
    print(f"✓ Query vector shape: {query_vector.shape}")
    
    # Step 2: Load FAISS vectorstore if not provided
    # ====================================================================
    # FAISS INTERACTION: Loading FAISS index (index.faiss + index.pkl)
    # ====================================================================
    if faiss_vectorstore is None:
        faiss_vectorstore = load_faiss_vectorstore(faiss_path)
    
    # Extract all vectors
    print(f"\n{'='*70}")
    print("Step 2: Extracting vectors from FAISS...")
    # ====================================================================
    # FAISS INTERACTION: Reconstructing vectors from FAISS index
    # ====================================================================
    # - Uses index.reconstruct_n() or index.reconstruct() to get raw vectors
    # - These are the numerical embeddings stored in the FAISS index
    # - No document content here, just vector data
    index = faiss_vectorstore.index
    n_vectors = index.ntotal
    
    try:
        all_vectors = index.reconstruct_n(0, n_vectors)
        print(f"✓ Extracted {len(all_vectors)} vectors")
    except Exception as e:
        print(f"Reconstructing vectors one by one...")
        all_vectors = []
        for i in range(n_vectors):
            if (i + 1) % 10000 == 0:
                print(f"  Reconstructed {i + 1}/{n_vectors}...")
            vec = index.reconstruct(i)
            all_vectors.append(vec)
        all_vectors = np.array(all_vectors, dtype=np.float32)
    
    # Step 3: Per-cluster KNN
    print(f"\n{'='*70}")
    print(f"Step 3: Per-cluster KNN (top-{per_cluster_k} per cluster)...")
    all_candidates = []
    
    for cluster_id in top_clusters:
        cluster_id_str = str(cluster_id)
        if cluster_id_str not in lists:
            continue
        
        vector_indices = lists[cluster_id_str]
        cluster_results = per_cluster_knn(
            query_vector,
            cluster_id,
            vector_indices,
            all_vectors,
            top_k=per_cluster_k
        )
        all_candidates.extend(cluster_results)
    
    print(f"✓ Found {len(all_candidates)} candidates across {len(top_clusters)} clusters")
#replace this step



    # Step 4: Global top-k
    print(f"\n{'='*70}")
    print(f"Step 4: Global top-{global_k} selection...")
    if use_faiss_for_knn:
        print("  Using FAISS IndexFlatL2 for KNN...")
    else:
        print("  Using pure NumPy for sorting...")

    #run executable
    #retrieve info from it

    stdout, stderr, code = run_built_executable(
        "/Users/antoniajanuszewicz/GolandProjects/Piano-PIR-RAG/client_exe",
        args=["-ip", "localhost:50052", "-thread", "1", "-input",
              "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/decrypted_results/top_k_results.json"],
        timeout=60
    )
    _, indices, vectors = extract_query_results(stderr)

    top_k_results = pir_global_top_k(
        indices,
        query_vector,
        vectors,
        top_k=global_k,
        use_faiss=True
    )
    """
    top_k_results = global_top_k(
        all_candidates,
        query_vector,
        all_vectors,
        top_k=global_k,
        use_faiss=use_faiss_for_knn
    )"""
    top_k_indices = [idx for idx, _ in top_k_results]
    top_k_distances = [dist for _, dist in top_k_results]

    save_decrypted_results(np.array(top_k_distances),np.array(top_k_indices),Path("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/rag_operations"))
    
    print(f"✓ Selected top-{len(top_k_results)} vectors:")
    for i, (idx, dist) in enumerate(top_k_results, 1):
        print(f"  {i}. Vector {idx}: distance = {dist:.4f}")
   #replace this step



    # Step 5: Map vector IDs → documents
    print(f"\n{'='*70}")
    print("Step 5: Mapping vector IDs to documents...")
    # ====================================================================
    # DOCUMENT/PICKLE INTERACTION: Getting actual document content
    # ====================================================================
    # - Translates vector indices → Document objects via docstore
    # - Documents contain page_content (text) and metadata (title)
    #documents = get_documents_by_indices(faiss_vectorstore, top_k_indices)
    documents = pir_get_documents_by_indices(top_k_indices)
    #print(documents)
    print(top_k_indices)
    print(f"✓ Retrieved {len(documents)} documents")
    
    # Step 6: Build RAG prompt
    print(f"\n{'='*70}")
    print("Step 6: Building RAG prompt...")
    rag_prompt = build_rag_prompt(query, documents)
    print(f"✓ Prompt length: {len(rag_prompt)} characters")
    
    # Step 7: Call LLM
    print(f"\n{'='*70}")
    print("Step 7: Calling LLM...")
    llm_response = call_llm(rag_prompt)
    print(f"✓ LLM response generated")
    
    print(f"\n{'='*70}")
    print("Pipeline Complete!")
    print(f"{'='*70}")
    
    return {
        "query": query,
        "top_clusters": top_clusters,
        "candidates_count": len(all_candidates),
        "top_k_indices": top_k_indices,
        "top_k_distances": top_k_distances,
        "documents": [
            {
                "index": idx,
                #"title": doc.metadata.get('title', 'Unknown'),
                "content_preview": doc[:200],
                "distance": dist
            }
            for idx, dist, doc in zip(top_k_indices, top_k_distances, documents)
        ],
        "rag_prompt": rag_prompt,
        "llm_response": llm_response
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plaintext RAG pipeline after FHE query"
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
        default="../workflow_decrypted_results/top_k_results.json",
        help="Path to top-k results JSON file from FHE query"
    )
    parser.add_argument(
        "--lists",
        type=str,
        default="../data/lists.json",
        help="Path to lists.json file"
    )
    parser.add_argument(
        "--faiss-path",
        type=str,
        required=True,
        help="Path to FAISS index directory"
    )
    parser.add_argument(
        "--top-clusters",
        type=int,
        default=100,
        help="Number of top clusters to use"
    )
    parser.add_argument(
        "--per-cluster-k",
        type=int,
        default=5,
        help="Top-k vectors per cluster"
    )
    parser.add_argument(
        "--global-k",
        type=int,
        default=10,
        help="Global top-k to return"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./rag_output.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Load top clusters
    top_clusters = load_top_clusters(args.top_results, n_clusters=args.top_clusters)
    
    # Load cluster lists
    lists = load_cluster_lists(args.lists)
    
    # Run pipeline
    results = run_plaintext_rag_pipeline(
        query=args.query,
        top_clusters=top_clusters,
        lists=lists,
        faiss_path=Path(args.faiss_path),
        per_cluster_k=args.per_cluster_k,
        global_k=args.global_k
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert documents to JSON-serializable format
    results_json = {
        "query": results["query"],
        "top_clusters": results["top_clusters"],
        "candidates_count": results["candidates_count"],
        "top_k_results": [
            {
                "vector_index": idx,
                "distance": dist,
                #"title": doc["title"],
                "content_preview": doc
            }
            for idx, dist, doc in zip(
                results["top_k_indices"],
                results["top_k_distances"],
                results["documents"]
            )
        ],
        "rag_prompt": results["rag_prompt"],
        "llm_response": results["llm_response"]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()

