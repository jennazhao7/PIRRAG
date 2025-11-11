#!/usr/bin/env python3
"""Test script to verify wiki-rag setup"""
from pathlib import Path
from wiki_rag import rag

print("Testing wiki-rag setup...")

# Test 1: Check if we can import and create the embedding model
print("\n1. Testing embedding model initialization...")
try:
    BAAI_embedding = rag.PromptedBGE(model_name="BAAI/bge-base-en")
    print("   ✓ Embedding model created successfully")
except Exception as e:
    print(f"   ✗ Error creating embedding model: {e}")
    exit(1)

# Test 2: Try to download and build RAG from HuggingFace
print("\n2. Testing RAG download from HuggingFace...")
try:
    faiss_name = "wiki_index__top_100000__2025-04-11"
    save_dir = Path("wiki_rag_data")
    print(f"   Downloading RAG: {faiss_name}")
    print("   (This may take a few minutes on first run...)")
    
    vectorstore = rag.download_and_build_rag_from_huggingface(
        embeddings=BAAI_embedding,
        rag_name=faiss_name,
        save_dir=save_dir
    )
    print("   ✓ RAG downloaded and loaded successfully")
except Exception as e:
    print(f"   ✗ Error downloading/loading RAG: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Test a query
print("\n3. Testing RAG query...")
try:
    query = "Biochemistry"
    print(f"   Query: '{query}'")
    responses = vectorstore.similarity_search(query, k=3)
    print(f"   ✓ Query successful, found {len(responses)} results")
    
    # Print results
    print("\n   Results:")
    for i, result in enumerate(responses[:3], 1):
        title = result.metadata.get("title", "Unknown")
        content_preview = result.page_content[:100].replace("\n", " ")
        print(f"   {i}. {title}")
        print(f"      {content_preview}...")
except Exception as e:
    print(f"   ✗ Error querying RAG: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ All tests passed! Wiki-rag is working correctly.")
print(f"\nRAG data saved to: {save_dir.absolute()}")

