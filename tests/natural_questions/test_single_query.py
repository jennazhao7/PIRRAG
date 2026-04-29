#!/usr/bin/env python3
"""
Quick test script for Natural Questions retrieval evaluation.
Tests a single query to verify the pipeline works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "prototype"))

from rag_utils import PromptedBGE
from langchain_community.vectorstores import FAISS
from datasets import load_dataset

def test_single_query():
    """Test retrieval on a single Natural Questions example."""
    
    print("=" * 70)
    print("Natural Questions Single Query Test")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading Natural Questions dataset...")
    dataset = load_dataset("sentence-transformers/natural-questions", "pair", split="train")
    example = dataset[0]
    
    question = example['query']
    answer = example['answer']
    
    print(f"\nQuestion: {question}")
    print(f"Answer (first 200 chars): {answer[:200]}...")
    
    # Load embeddings and index
    print("\nLoading embedding model and FAISS index...")
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    vectorstore = FAISS.load_local(
        "wiki-rag/wiki_rag_data/wiki_index__top_100000__2025-04-11",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✓ Loaded index with {vectorstore.index.ntotal} vectors")
    
    # Retrieve documents
    print(f"\nRetrieving documents for question...")
    results = vectorstore.similarity_search_with_score(question, k=20)
    
    print(f"✓ Retrieved {len(results)} documents")
    print("\nTop 5 retrieved documents:")
    for i, (doc, score) in enumerate(results[:5], 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Title: {doc.metadata.get('title', 'N/A')}")
        print(f"   Text: {doc.page_content[:150]}...")
    
    # Test answer recall
    print("\n" + "=" * 70)
    print("Testing AnswerRecall@K")
    print("=" * 70)
    
    import re
    def normalize_string(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Extract years from answer
    normalized_answer = normalize_string(answer)
    years = re.findall(r'\b((19|20)\d{2})\b', normalized_answer)
    gold_answers = [normalize_string(y[0]) for y in years]
    
    # Also add first sentence
    first_sentence = normalized_answer.split('.')[0]
    if len(first_sentence) > 10:
        gold_answers.append(first_sentence)
    
    print(f"\nGold answers extracted: {gold_answers[:5]}")
    
    # Check recall
    retrieved_text = " ".join([normalize_string(doc.page_content) for doc, _ in results])
    
    for k in [1, 5, 20]:
        top_k_text = " ".join([normalize_string(doc.page_content) for doc, _ in results[:k]])
        recall = any(gold in top_k_text for gold in gold_answers if gold)
        print(f"AnswerRecall@{k}: {recall}")
    
    print("\n" + "=" * 70)
    print("✓ Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_single_query()


