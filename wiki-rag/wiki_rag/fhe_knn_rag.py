"""
FHE KNN wrapper for wiki-rag
Integrates Zama's Concrete ML KNeighborsClassifier for encrypted document retrieval
"""
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import time

try:
    from concrete.ml.sklearn import KNeighborsClassifier
    CONCRETE_ML_AVAILABLE = True
except ImportError:
    CONCRETE_ML_AVAILABLE = False
    KNeighborsClassifier = None

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FHEKNNRAG:
    """
    Wrapper that uses FHE KNN for encrypted document retrieval.
    
    This class:
    1. Extracts embeddings from FAISS vectorstore
    2. Applies PCA for dimensionality reduction (required for FHE)
    3. Trains a Concrete ML KNeighborsClassifier
    4. Provides encrypted search functionality
    """
    
    def __init__(
        self,
        vectorstore: FAISS,
        embeddings: Embeddings,
        n_neighbors: int = 5,
        n_bits: int = 3,
        pca_components: int = 50,
        compile_fhe: bool = True,
        cache_dir: Optional[Path] = None
    ):
        if not CONCRETE_ML_AVAILABLE:
            raise ImportError(
                "Concrete ML is not installed. Please install it with: pip install concrete-ml"
            )
        """
        Initialize FHE KNN RAG wrapper.
        
        Args:
            vectorstore: FAISS vectorstore containing documents
            embeddings: Embedding model for query encoding
            n_neighbors: Number of nearest neighbors (top-k)
            n_bits: Quantization bits for FHE (lower = faster, less accurate)
            pca_components: Number of PCA components (must be < embedding dim)
            compile_fhe: Whether to compile FHE circuit immediately
            cache_dir: Directory to cache FHE model
        """
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.n_neighbors = n_neighbors
        self.n_bits = n_bits
        self.pca_components = pca_components
        self.cache_dir = cache_dir or Path("fhe_knn_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Extract embeddings and documents from FAISS
        print("Extracting embeddings from FAISS vectorstore...")
        self._extract_faiss_data()
        
        # Setup dimensionality reduction
        print(f"Applying PCA: {self.embedding_dim} -> {pca_components} dimensions")
        self._setup_pca()
        
        # Train FHE KNN
        print("Training FHE KNN classifier...")
        self._train_fhe_knn()
        
        # Compile FHE circuit if requested
        if compile_fhe:
            print("Compiling FHE circuit (this may take a while)...")
            self._compile_fhe()
    
    def _extract_faiss_data(self):
        """Extract all embeddings and documents from FAISS vectorstore."""
        # Get FAISS index
        index = self.vectorstore.index
        
        # Get number of vectors
        n_vectors = index.ntotal
        embedding_dim = index.d
        
        print(f"Found {n_vectors} documents with {embedding_dim}D embeddings")
        
        # Extract all vectors and documents
        all_vectors = []
        all_documents = []
        
        # Try to reconstruct vectors from FAISS index
        try:
            # Reconstruct all vectors at once (if supported)
            reconstructed_vectors = index.reconstruct_n(0, n_vectors)
            print("Successfully reconstructed vectors from FAISS index")
        except Exception as e:
            print(f"Could not reconstruct all vectors at once: {e}")
            print("Reconstructing vectors one by one...")
            reconstructed_vectors = []
            for i in range(min(n_vectors, 10000)):  # Limit for testing
                try:
                    vec = index.reconstruct(i)
                    reconstructed_vectors.append(vec)
                except Exception as e2:
                    print(f"Warning: Could not reconstruct vector {i}: {e2}")
                    break
            reconstructed_vectors = np.array(reconstructed_vectors)
            n_vectors = len(reconstructed_vectors)
        
        # Get documents from docstore
        # FAISS vectorstore stores documents in docstore
        docstore = self.vectorstore.docstore
        
        # Try different methods to get documents
        if hasattr(docstore, '_dict'):
            # In-memory docstore
            doc_dict = docstore._dict
            doc_ids = list(doc_dict.keys())
            print(f"Found {len(doc_ids)} documents in docstore")
            
            # Match documents with vectors
            # Note: FAISS index order may not match docstore order
            # We'll re-embed documents to ensure consistency
            print("Re-embedding documents to ensure consistency...")
            for i, doc_id in enumerate(doc_ids[:n_vectors]):
                try:
                    doc = doc_dict[doc_id]
                    all_documents.append(doc)
                    # Re-embed document to get embedding
                    doc_text = doc.page_content
                    embedding = self.embeddings.embed_documents([doc_text])[0]
                    all_vectors.append(embedding)
                except Exception as e:
                    print(f"Warning: Could not process document {doc_id}: {e}")
                    continue
        else:
            # Fallback: use reconstructed vectors and try to get documents
            print("Using reconstructed vectors (document matching may be approximate)")
            for i in range(n_vectors):
                try:
                    # Try to get document by various methods
                    if hasattr(self.vectorstore, 'index_to_docstore_id'):
                        doc_id = self.vectorstore.index_to_docstore_id.get(i)
                        if doc_id:
                            doc = docstore.search(doc_id)
                            all_documents.append(doc)
                    else:
                        # Create placeholder document
                        all_documents.append(None)
                    
                    all_vectors.append(reconstructed_vectors[i])
                except Exception as e:
                    print(f"Warning: Could not get document {i}: {e}")
                    continue
        
        # Filter out None documents
        valid_indices = [i for i, doc in enumerate(all_documents) if doc is not None]
        self.all_vectors = np.array([all_vectors[i] for i in valid_indices], dtype=np.float32)
        self.all_documents = [all_documents[i] for i in valid_indices]
        self.embedding_dim = self.all_vectors.shape[1] if len(self.all_vectors) > 0 else embedding_dim
        
        print(f"Extracted {len(self.all_vectors)} vectors and {len(self.all_documents)} documents")
    
    def _setup_pca(self):
        """Setup PCA for dimensionality reduction."""
        # Standardize embeddings
        self.scaler = StandardScaler()
        vectors_scaled = self.scaler.fit_transform(self.all_vectors)
        
        # Apply PCA
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        self.reduced_vectors = self.pca.fit_transform(vectors_scaled).astype(np.float32)
        
        print(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
    
    def _train_fhe_knn(self):
        """Train Concrete ML KNeighborsClassifier."""
        # Create labels (document indices)
        # These will be used to map back to documents
        labels = np.arange(len(self.reduced_vectors), dtype=np.int64)
        
        # Train FHE KNN
        self.fhe_knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            n_bits=self.n_bits
        )
        
        print(f"Fitting KNN with {len(self.reduced_vectors)} samples, "
              f"k={self.n_neighbors}, n_bits={self.n_bits}")
        
        start_time = time.time()
        self.fhe_knn.fit(self.reduced_vectors, labels)
        fit_time = time.time() - start_time
        print(f"KNN fit completed in {fit_time:.2f}s")
    
    def _compile_fhe(self):
        """Compile FHE circuit."""
        print("Compiling FHE circuit (this may take several minutes)...")
        start_time = time.time()
        
        # Use a calibration set (small batch) for compilation
        # Concrete ML uses this to determine input/output ranges
        calibration_size = min(8, len(self.reduced_vectors))
        calibration_samples = self.reduced_vectors[:calibration_size].astype(np.float32)
        
        # Try multiple p_error values (probability of error in FHE operations)
        # This is crucial for successful compilation
        p_errors = [0.01, 0.02, 0.05]
        compile_success = False
        used_p_error = None
        
        for p_err in p_errors:
            try:
                print(f"  Trying p_error={p_err}...")
                self.fhe_knn.compile(calibration_samples, p_error=p_err)
                used_p_error = p_err
                compile_success = True
                break
            except Exception as e:
                print(f"  ✗ Failed with p_error={p_err}: {str(e)[:100]}")
                if p_err == p_errors[-1]:  # Last attempt
                    raise RuntimeError(f"FHE compilation failed with all p_error values. Last error: {e}")
        
        compile_time = time.time() - start_time
        print(f"FHE compilation completed in {compile_time:.2f}s (p_error={used_p_error})")
        
        # Save compiled model
        model_path = self.cache_dir / "fhe_knn_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'fhe_knn': self.fhe_knn,
                'pca': self.pca,
                'scaler': self.scaler,
                'n_neighbors': self.n_neighbors,
                'n_bits': self.n_bits,
                'pca_components': self.pca_components,
                'p_error': used_p_error
            }, f)
        print(f"Saved compiled model to {model_path}")
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        fhe_mode: str = "simulate"
    ) -> List[Document]:
        """
        Search for similar documents using FHE KNN.
        
        Args:
            query: Query string
            k: Number of results (defaults to n_neighbors)
            fhe_mode: "simulate", "execute", or "clear"
        
        Returns:
            List of Document objects
        
        Note:
            FHE KNN predict() returns the most common class among k neighbors.
            For true top-k retrieval, we query multiple times with slight variations
            or fall back to clear-text distance computation for the top-k.
        """
        if k is None:
            k = self.n_neighbors
        
        # Embed query
        query_embedding = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Apply same preprocessing as training
        query_scaled = self.scaler.transform(query_embedding)
        query_reduced = self.pca.transform(query_scaled).astype(np.float32)
        
        # For top-k retrieval, we use a hybrid approach:
        # 1. Use FHE KNN to get the predicted document (most common among k neighbors)
        # 2. Use clear-text distance to get additional top-k documents
        
        # Step 1: Get FHE prediction
        if fhe_mode == "clear":
            # Clear text prediction (fastest, for testing)
            fhe_pred = self.fhe_knn.predict(query_reduced)
        elif fhe_mode == "simulate":
            # FHE simulation (faster than real FHE)
            fhe_pred = self.fhe_knn.predict(query_reduced, fhe="simulate")
        elif fhe_mode == "execute":
            # Real FHE execution (slowest, most secure)
            fhe_pred = self.fhe_knn.predict(query_reduced, fhe="execute")
        else:
            raise ValueError(f"Unknown fhe_mode: {fhe_mode}")
        
        pred_idx = int(fhe_pred[0])
        
        # Step 2: Compute distances in clear-text to get top-k
        # This is a hybrid approach: FHE for the main prediction, clear for top-k
        distances = np.linalg.norm(
            self.reduced_vectors - query_reduced,
            axis=1
        )
        
        # Get top-k indices
        top_k_indices = np.argsort(distances)[:k]
        
        # Ensure predicted document is in results (if not already)
        if pred_idx not in top_k_indices and pred_idx < len(self.all_documents):
            # Replace the last one with the FHE prediction
            top_k_indices = list(top_k_indices[:-1]) + [pred_idx]
        
        # Get documents
        results = []
        for idx in top_k_indices:
            if idx < len(self.all_documents):
                results.append(self.all_documents[idx])
        
        return results[:k]
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        fhe_mode: str = "simulate"
    ):
        """
        Search with scores (returns (Document, score) tuples).
        
        Note: FHE KNN doesn't directly provide scores, so we return
        a placeholder score of 1.0 for retrieved documents.
        """
        documents = self.similarity_search(query, k, fhe_mode)
        return [(doc, 1.0) for doc in documents]
    
    @classmethod
    def from_vectorstore(
        cls,
        vectorstore: FAISS,
        embeddings: Embeddings,
        **kwargs
    ) -> 'FHEKNNRAG':
        """Create FHE KNN RAG from existing FAISS vectorstore."""
        return cls(vectorstore, embeddings, **kwargs)

