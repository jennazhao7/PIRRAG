#!/usr/bin/env python3
"""
Direct RAG Query without IVF.

Performs a direct FAISS similarity search without using IVF clusters.
This serves as a baseline comparison to the IVF-based approach.
"""

import argparse
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
import sys
from pathlib import Path
# Add parent directory to path to import rag_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_utils import PromptedBGE


def direct_rag_query(
    query: str,
    faiss_path: Path,
    k: int = 10
):
    """
    Perform direct FAISS similarity search without IVF.
    
    Args:
        query: Query text
        faiss_path: Path to FAISS index directory
        k: Number of results to return
        
    Returns:
        Dictionary with query results
    """
    print("=" * 70)
    print("Direct RAG Query (No IVF)")
    print("=" * 70)
    print(f"\nQuery: {query}")
    print(f"Top-k: {k}")
    
    # Load FAISS vectorstore
    print(f"\n{'='*70}")
    print("Loading FAISS vectorstore...")
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    vectorstore = FAISS.load_local(
        str(faiss_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Loaded vectorstore with {vectorstore.index.ntotal} vectors")
    
    # Perform direct similarity search
    print(f"\n{'='*70}")
    print(f"Performing direct similarity search (top-{k})...")
    results = vectorstore.similarity_search_with_score(query, k=k)
    print(f"✓ Retrieved {len(results)} results")
    
    # Extract results
    top_results = []
    for i, (doc, score) in enumerate(results, 1):
        result = {
            "rank": i,
            "score": float(score),
            "distance": float(score),  # FAISS returns L2 distance
            "title": doc.metadata.get('title', 'Unknown'),
            "content": doc.page_content[:500],  # First 500 chars
            "full_content": doc.page_content
        }
        top_results.append(result)
        
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"   Content preview: {doc.page_content[:150]}...")
    
    print(f"\n{'='*70}")
    print("Query Complete!")
    print(f"{'='*70}")
    
    return {
        "query": query,
        "method": "direct_faiss_search",
        "k": k,
        "total_vectors": vectorstore.index.ntotal,
        "results": top_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Direct RAG query without IVF"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text"
    )
    parser.add_argument(
        "--faiss-path",
        type=str,
        required=True,
        help="Path to FAISS index directory"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to return"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./direct_rag_output.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Run query
    results = direct_rag_query(
        query=args.query,
        faiss_path=Path(args.faiss_path),
        k=args.k
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()

