#!/usr/bin/env python3
"""
Quick start example for wiki-rag
Based on the README example
"""
from pathlib import Path
from wiki_rag import rag

# Get RAG from HuggingFace
print("Initializing embedding model...")
BAAI_embedding = rag.PromptedBGE(model_name="BAAI/bge-base-en")

print("Downloading RAG from HuggingFace...")
faiss_name = "wiki_index__top_100000__2025-04-11"
vectorstore = rag.download_and_build_rag_from_huggingface(
    embeddings=BAAI_embedding,
    rag_name=faiss_name,
    save_dir=Path("wiki_rag_data")
)

# Query RAG
print("\nQuerying RAG...")
query = "Biochemistry"
responses = vectorstore.similarity_search(query, k=3)

# Print Results
print(f"\nTop {len(responses)} results for '{query}':")
for i, result in enumerate(responses, 1):
    title = result.metadata.get("title", "Unknown")
    content_preview = result.page_content[:100].replace("\n", " ")
    print(f"{i}. Wiki Page: '{title}'")
    print(f"   {content_preview}...\n")

