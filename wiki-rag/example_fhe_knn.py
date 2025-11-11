#!/usr/bin/env python3
"""
Example: Using FHE KNN with wiki-rag

This demonstrates how to use Zama's Fully Homomorphic Encryption (FHE)
KNN for encrypted document retrieval in wiki-rag.
"""
from pathlib import Path
from wiki_rag import rag
from wiki_rag.fhe_knn_rag import FHEKNNRAG

def main():
    print("=" * 60)
    print("Wiki-RAG with FHE KNN Example")
    print("=" * 60)
    
    # Step 1: Load the standard FAISS vectorstore
    print("\n1. Loading FAISS vectorstore from HuggingFace...")
    BAAI_embedding = rag.PromptedBGE(model_name="BAAI/bge-base-en")
    
    faiss_name = "wiki_index__top_100000__2025-04-11"
    vectorstore = rag.download_and_build_rag_from_huggingface(
        embeddings=BAAI_embedding,
        rag_name=faiss_name,
        save_dir=Path("wiki_rag_data")
    )
    print(f"   ✓ Loaded {vectorstore.index.ntotal} documents")
    
    # Step 2: Create FHE KNN wrapper
    print("\n2. Creating FHE KNN wrapper...")
    print("   Note: This will extract embeddings, apply PCA, and train FHE KNN")
    print("   This may take several minutes on first run...")
    
    fhe_rag = FHEKNNRAG(
        vectorstore=vectorstore,
        embeddings=BAAI_embedding,
        n_neighbors=5,          # Top-5 documents
        n_bits=3,               # Quantization bits (lower = faster, less accurate)
        pca_components=50,       # Reduce to 50D (from 768D for bge-base-en)
        compile_fhe=True,        # Compile FHE circuit immediately
        cache_dir=Path("fhe_knn_cache")
    )
    print("   ✓ FHE KNN wrapper created and compiled")
    
    # Step 3: Test queries with different FHE modes
    queries = [
        "Biochemistry",
        "Machine learning",
        "Quantum physics"
    ]
    
    print("\n3. Testing queries with FHE KNN...")
    for query in queries:
        print(f"\n   Query: '{query}'")
        print("   " + "-" * 50)
        
        # Test with FHE simulation (faster, for development)
        print("   FHE Mode: simulate")
        results = fhe_rag.similarity_search(query, k=3, fhe_mode="simulate")
        for i, doc in enumerate(results[:3], 1):
            title = doc.metadata.get("title", "Unknown")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"   {i}. {title}")
            print(f"      {preview}...")
        
        # Optional: Test with real FHE execution (slow, but fully encrypted)
        # Uncomment to test:
        # print("\n   FHE Mode: execute (real FHE - may be slow)")
        # results_fhe = fhe_rag.similarity_search(query, k=3, fhe_mode="execute")
        # for i, doc in enumerate(results_fhe[:3], 1):
        #     title = doc.metadata.get("title", "Unknown")
        #     print(f"   {i}. {title}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("\nNext steps:")
    print("  - Adjust n_bits (lower = faster, less accurate)")
    print("  - Adjust pca_components (lower = faster, less accurate)")
    print("  - Use fhe_mode='execute' for real encrypted queries")
    print("=" * 60)

if __name__ == "__main__":
    main()

