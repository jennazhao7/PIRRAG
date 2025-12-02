#!/usr/bin/env python3
"""
Client-side FHE query encryption for wiki-rag.

This script:
1. Takes a client query (text or embedding vector)
2. Encrypts the query vector using FHE (TenSEAL/CKKS)
3. Encrypts ||q||^2 (squared norm) for distance computations
4. Exports encrypted data for server-side processing
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import tenseal as ts
import re
from typing import List, Tuple, Dict
import faiss

from langchain_community.embeddings import HuggingFaceEmbeddings

from prototype.go_runner import run_built_executable
from rag_utils import PromptedBGE


class FHEQueryClient:
    """
    Client-side FHE query encryption.
    
    Handles:
    - Query text → embedding → encryption
    - Query vector → encryption
    - Squared norm encryption
    """
    
    def __init__(
        self,
        context_path: Path = None,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list = [60, 40, 40, 60],
        embedding_model_name: str = "BAAI/bge-base-en"
    ):
        """
        Initialize FHE query client.
        
        Args:
            context_path: Path to save/load encryption context (keys)
            poly_modulus_degree: Polynomial modulus degree (8192 or 16384)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
            embedding_model_name: Name of embedding model
        """
        self.context_path = Path(context_path) if context_path else Path("./fhe_context")
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embeddings = PromptedBGE(model_name=embedding_model_name)
        self.embedding_dim = 768  # BGE-base-en dimension
        
        # Initialize or load encryption context
        self.context = self._setup_context()
    
    def _setup_context(self) -> ts.Context:
        """
        Setup or load encryption context.
        
        Returns:
            TenSEAL encryption context
        """
        context_file = self.context_path / "context.json"
        
        if context_file.exists():
            print(f"Loading encryption context from {context_file}...")
            with open(context_file, 'rb') as f:
                context_bytes = f.read()
            context = ts.context_from(context_bytes)
            print("✓ Context loaded successfully")
        else:
            print("Creating new encryption context...")
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
            )
            
            # Generate Galois keys (needed for rotations)
            print("Generating Galois keys...")
            context.generate_galois_keys()
            
            # Set global scale for CKKS
            context.global_scale = 2**40
            
            # Save context (with secret key for client use)
            self.context_path.mkdir(parents=True, exist_ok=True)
            with open(context_file, 'wb') as f:
                f.write(context.serialize(save_secret_key=True))
            print(f"✓ Context saved to {context_file}")
            
            # Also save public key context (for server)
            public_context_file = self.context_path / "context_public.json"
            with open(public_context_file, 'wb') as f:
                f.write(context.serialize(save_secret_key=False))
            print(f"✓ Public key context saved to {public_context_file} (for server)")
        
        return context
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Convert query text to embedding vector.
        
        Args:
            query_text: Query text string
            
        Returns:
            Query embedding vector (768D)
        """
        embedding = self.embeddings.embed_query(query_text)
        return np.array(embedding, dtype=np.float32)
    
    def encrypt_query(self, query_vector: np.ndarray) -> ts.CKKSVector:
        """
        Encrypt query vector using FHE.
        
        Args:
            query_vector: Query embedding vector
            
        Returns:
            Encrypted query vector
        """
        # Ensure vector is 1D and float32
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()
        query_vector = query_vector.astype(np.float32)
        
        # Encrypt using CKKS
        encrypted_query = ts.ckks_vector(self.context, query_vector.tolist())
        
        return encrypted_query
    
    def encrypt_squared_norm(self, query_vector: np.ndarray) -> ts.CKKSVector:
        """
        Encrypt squared norm ||q||^2 of query vector.
        
        Args:
            query_vector: Query embedding vector
            
        Returns:
            Encrypted squared norm (scalar)
        """
        # Compute squared norm
        squared_norm = np.sum(query_vector ** 2)
        
        # Encrypt as a single-element vector
        encrypted_norm = ts.ckks_vector(self.context, [float(squared_norm)])
        
        return encrypted_norm
    
    def process_query(
        self,
        query: Union[str, np.ndarray],
        return_plaintext_norm: bool = False
    ) -> Tuple[ts.CKKSVector, ts.CKKSVector, dict]:
        """
        Process query: embed (if text), encrypt vector and squared norm.
        
        Args:
            query: Query text string or embedding vector
            return_plaintext_norm: Whether to return plaintext norm for verification
            
        Returns:
            Tuple of (encrypted_query, encrypted_norm_squared, metadata)
        """
        # Get query vector
        if isinstance(query, str):
            print(f"Embedding query: '{query[:50]}...'")
            query_vector = self.embed_query(query)
        else:
            query_vector = np.array(query, dtype=np.float32)
            if query_vector.ndim > 1:
                query_vector = query_vector.flatten()
        
        print(f"Query vector shape: {query_vector.shape}")
        print(f"Query vector range: [{query_vector.min():.4f}, {query_vector.max():.4f}]")
        
        # Encrypt query vector
        print("Encrypting query vector...")
        encrypted_query = self.encrypt_query(query_vector)
        
        # Encrypt squared norm
        print("Encrypting squared norm ||q||^2...")
        encrypted_norm_squared = self.encrypt_squared_norm(query_vector)
        
        # Compute plaintext norm for verification
        plaintext_norm_squared = float(np.sum(query_vector ** 2))
        
        metadata = {
            "query_dim": len(query_vector),
            "plaintext_norm_squared": plaintext_norm_squared,
            "plaintext_norm": float(np.sqrt(plaintext_norm_squared)),
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_mod_bit_sizes": self.coeff_mod_bit_sizes
        }
        
        print(f"✓ Query encrypted successfully")
        print(f"  Plaintext ||q||^2: {plaintext_norm_squared:.4f}")
        print(f"  Plaintext ||q||: {np.sqrt(plaintext_norm_squared):.4f}")
        
        return encrypted_query, encrypted_norm_squared, metadata
    
    def export_public_key(self, output_path: Path = None) -> Path:
        """
        Export public key context (without secret key) for server use.
        
        Args:
            output_path: Optional output path (default: context_path/context_public.json)
            
        Returns:
            Path to public key file
        """
        if output_path is None:
            public_key_file = self.context_path / "context_public.json"
        else:
            public_key_file = Path(output_path)
        
        # Serialize context without secret key
        public_context_bytes = self.context.serialize(save_secret_key=False)
        
        public_key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(public_key_file, 'wb') as f:
            f.write(public_context_bytes)
        
        print(f"✓ Exported public key to {public_key_file}")
        return public_key_file
    
    def save_encrypted_query(
        self,
        encrypted_query: ts.CKKSVector,
        encrypted_norm_squared: ts.CKKSVector,
        metadata: dict,
        output_path: Path,
        include_public_key: bool = True
    ):
        """
        Save encrypted query and metadata to files.
        
        Args:
            encrypted_query: Encrypted query vector
            encrypted_norm_squared: Encrypted squared norm
            metadata: Query metadata
            output_path: Output directory path
            include_public_key: Whether to copy public key to output directory
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Serialize encrypted vectors
        query_bytes = encrypted_query.serialize()
        norm_bytes = encrypted_norm_squared.serialize()
        
        # Save encrypted query
        query_file = output_path / "encrypted_query.bin"
        with open(query_file, 'wb') as f:
            f.write(query_bytes)
        print(f"✓ Saved encrypted query to {query_file} ({len(query_bytes)} bytes)")
        
        # Save encrypted norm
        norm_file = output_path / "encrypted_norm_squared.bin"
        with open(norm_file, 'wb') as f:
            f.write(norm_bytes)
        print(f"✓ Saved encrypted norm to {norm_file} ({len(norm_bytes)} bytes)")
        
        # Save metadata
        metadata_file = output_path / "query_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_file}")
        
        # Optionally copy public key to output directory for server
        if include_public_key:
            public_key_file = output_path / "context_public.json"
            public_context_bytes = self.context.serialize(save_secret_key=False)
            with open(public_key_file, 'wb') as f:
                f.write(public_context_bytes)
            print(f"✓ Saved public key context to {public_key_file} (for server)")
        
        return query_file, norm_file, metadata_file
    
    def decrypt_distances(
        self,
        encrypted_distances_dir: Path,
        top_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decrypt encrypted distances and return top-k smallest distances.
        
        Args:
            encrypted_distances_dir: Directory containing encrypted distance files
            top_k: Number of top distances to return
            
        Returns:
            Tuple of (top_k_distances, top_k_indices) sorted by distance (smallest first)
        """
        encrypted_distances_dir = Path(encrypted_distances_dir)
        
        # Load metadata
        metadata_file = encrypted_distances_dir / "distances_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        n_distances = metadata.get("n_distances", 0)
        print(f"Decrypting {n_distances} encrypted distances...")
        
        # Load and decrypt all distances
        all_distances = []
        
        for i in range(n_distances):
            dist_file = encrypted_distances_dir / f"encrypted_distance_{i:04d}.bin"
            
            if not dist_file.exists():
                print(f"Warning: Distance file {dist_file} not found, skipping...")
                continue
            
            # Load encrypted distance
            with open(dist_file, 'rb') as f:
                encrypted_bytes = f.read()
            
            # Decrypt
            encrypted_distance = ts.ckks_vector_from(self.context, encrypted_bytes)
            distance_value = encrypted_distance.decrypt()[0]  # Decrypt and get scalar
            all_distances.append(distance_value)
            
            if (i + 1) % 500 == 0 or (i + 1) == n_distances:
                print(f"  Decrypted {i + 1}/{n_distances} distances...")
        
        all_distances = np.array(all_distances, dtype=np.float32)
        print(f"✓ Decrypted {len(all_distances)} distances")
        
        # Get top-k smallest distances
        top_k_indices = np.argsort(all_distances)[:top_k]
        top_k_distances = all_distances[top_k_indices]
        
        print(f"✓ Selected top-{top_k} smallest distances")
        print(f"  Distance range: [{top_k_distances.min():.4f}, {top_k_distances.max():.4f}]")
        
        return top_k_distances, top_k_indices
    
    def save_decrypted_results(
        self,
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
        
        results_file = output_path / "top_k_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved results to {results_file}")
        
        # Save as numpy arrays for easy loading
        np.save(output_path / "top_k_distances.npy", top_k_distances)
        np.save(output_path / "top_k_indices.npy", top_k_indices)
        print(f"✓ Saved numpy arrays to {output_path}")
        
        # Print summary
        print(f"\nTop-{len(top_k_distances)} Results:")
        print(f"  Best match: Centroid {top_k_indices[0]} (distance: {top_k_distances[0]:.4f})")
        print(f"  Worst match: Centroid {top_k_indices[-1]} (distance: {top_k_distances[-1]:.4f})")
        
        return results_file


def main():
    parser = argparse.ArgumentParser(
        description="Client-side FHE query encryption for wiki-rag"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        help="Query text to encrypt (not required if --decrypt-distances is used)"
    )
    parser.add_argument(
        "--context-path",
        type=str,
        default="./fhe_context",
        help="Path to encryption context (keys)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./encrypted_queries",
        help="Output directory for encrypted query files"
    )
    parser.add_argument(
        "--poly-modulus-degree",
        type=int,
        default=8192,
        choices=[8192, 16384],
        help="Polynomial modulus degree (8192 or 16384)"
    )
    parser.add_argument(
        "--query-vector",
        type=str,
        default=None,
        help="Path to .npy file with query vector (alternative to --query text)"
    )
    parser.add_argument(
        "--decrypt-distances",
        type=str,
        default=None,
        help="Decrypt encrypted distances from server (path to encrypted_distances directory)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top distances to return (default: 100)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./decrypted_results",
        help="Output directory for decrypted results"
    )
    
    args = parser.parse_args()
    
    # Handle decryption mode
    if args.decrypt_distances:
        print("=" * 70)
        print("FHE Query Client - Distance Decryption")
        print("=" * 70)
        print()
        
        # Initialize client
        client = FHEQueryClient(context_path=args.context_path)
        
        # Decrypt distances
        top_k_distances, top_k_indices = client.decrypt_distances(
            Path(args.decrypt_distances),
            top_k=args.top_k
        )
        
        # Save results
        client.save_decrypted_results(
            top_k_distances,
            top_k_indices,
            Path(args.results_dir)
        )

        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"Decrypted {len(top_k_distances)} top distances")
        print(f"Results saved to: {args.results_dir}")
        print(f"{'='*70}")

        # Step 4: Global top-k
        global_k = 10
        print(f"\n{'=' * 70}")
        print(f"Step 4: Global top-{global_k} selection...")

        query_vector = embed_query(args.query)

        # run executable
        # retrieve info from it

        stdout, stderr, code = run_built_executable(
            "/Users/antoniajanuszewicz/GolandProjects/Piano-PIR-RAG/client_exe",
            args=["-ip", "localhost:50052", "-thread", "1", "-input",
                  "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/decrypted_results/top_k_results.json",
                  "-extra_input", "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/data/lists.json"],
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

        save_top_k_results(np.array(top_k_distances), np.array(top_k_indices),
                               Path("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/ground_truth"))

        print(f"✓ Selected top-{len(top_k_results)} vectors:")
        for i, (idx, dist) in enumerate(top_k_results, 1):
            print(f"  {i}. Vector {idx}: distance = {dist:.4f}")
        # replace this step

        # Step 5: Map vector IDs → documents
        print(f"\n{'=' * 70}")
        print("Step 5: Mapping vector IDs to documents...")
        # ====================================================================
        # DOCUMENT/PICKLE INTERACTION: Getting actual document content
        # ====================================================================
        # - Translates vector indices → Document objects via docstore
        # - Documents contain page_content (text) and metadata (title)
        # documents = get_documents_by_indices(faiss_vectorstore, top_k_indices)
        documents = pir_get_documents_by_indices(top_k_indices)
        print(documents)
        print(top_k_indices)
        print(f"✓ Retrieved {len(documents)} documents")

        return
    
    # Encryption mode - query is required
    if not args.query and not args.query_vector:
        parser.error("--query or --query-vector is required for encryption mode")
    
    print("=" * 70)
    print("FHE Query Encryption Client")
    print("=" * 70)
    print()
    
    # Initialize client
    client = FHEQueryClient(
        context_path=args.context_path,
        poly_modulus_degree=args.poly_modulus_degree
    )
    
    # Process query
    if args.query_vector:
        print(f"Loading query vector from {args.query_vector}...")
        query_vector = np.load(args.query_vector)
        encrypted_query, encrypted_norm, metadata = client.process_query(query_vector)
    else:
        encrypted_query, encrypted_norm, metadata = client.process_query(args.query)
    
    # Save encrypted query
    print(f"\n{'='*70}")
    print("Saving Encrypted Query")
    print(f"{'='*70}")
    output_path = Path(args.output_dir)
    query_file, norm_file, metadata_file = client.save_encrypted_query(
        encrypted_query,
        encrypted_norm,
        metadata,
        output_path
    )
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Query: {args.query[:100] if len(args.query) > 100 else args.query}")
    print(f"Query dimension: {metadata['query_dim']}")
    print(f"Plaintext ||q||^2: {metadata['plaintext_norm_squared']:.4f}")
    print(f"Plaintext ||q||: {metadata['plaintext_norm']:.4f}")
    print(f"\nOutput files:")
    print(f"  - Encrypted query: {query_file}")
    print(f"  - Encrypted norm: {norm_file}")
    print(f"  - Metadata: {metadata_file}")
    print(f"{'='*70}")


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

def pir_get_documents_by_indices(
        vector_indices: List[int]
) -> List[str]:
    stdout, stderr, code = run_built_executable(
        "/Users/antoniajanuszewicz/GolandProjects/Piano-PIR-RAG/client_exe",
        args=["-ip", "localhost:50051", "-thread", "1", "-input",
              "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/ground_truth/ground_truth.json"],
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

def save_top_k_results(
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

def embed_query(query_text: str) -> np.ndarray:
    """Embed query → vector q."""
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    query_vec = embeddings.embed_query(query_text)
    return np.array(query_vec, dtype=np.float32)

if __name__ == "__main__":
    main()

