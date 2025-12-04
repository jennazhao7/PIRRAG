#!/usr/bin/env python3
"""
Natural Questions Retrieval Evaluation Script

Evaluates retrieval performance on Natural Questions dataset with:
- AnswerRecall@K: Check if any gold answer is substring of any retrieved doc
- Normalizes strings for comparison
- Evaluates for K ∈ {1, 5, 20, 50}

Uses wiki-rag FAISS index for retrieval.
Optional reader: datarpit/distilbert-base-uncased-finetuned-natural-questions
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Set, Optional
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "prototype"))

from rag_utils import PromptedBGE
from langchain_community.vectorstores import FAISS

# Datasets import
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  Warning: datasets not available. Install with: pip install datasets")

# Transformers import (for optional reader)
try:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: transformers not available. Install with: pip install transformers torch")


class NaturalQuestionsRetrievalEvaluator:
    """
    Evaluates retrieval performance on Natural Questions dataset.
    Focuses on AnswerRecall@K metric.
    """
    
    def __init__(
        self,
        wiki_rag_index_path: Path,
        embedding_model_name: str = "BAAI/bge-base-en",
        reader_model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Natural Questions retrieval evaluator.
        
        Args:
            wiki_rag_index_path: Path to wiki-rag FAISS index directory
            embedding_model_name: Name of embedding model for RAG retrieval
            reader_model_name: Optional reader model (e.g., datarpit/distilbert-base-uncased-finetuned-natural-questions)
            device: Device to run models on
        """
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embeddings = PromptedBGE(model_name=embedding_model_name)
        
        # Load wiki-rag FAISS index
        print(f"Loading wiki-rag FAISS index from {wiki_rag_index_path}...")
        self.load_wiki_rag_index(wiki_rag_index_path)
        
        self.device = device
        
        # Load optional reader model
        if reader_model_name:
            print(f"Loading reader model: {reader_model_name}...")
            try:
                self.reader_tokenizer = AutoTokenizer.from_pretrained(reader_model_name)
                self.reader_model = AutoModelForQuestionAnswering.from_pretrained(reader_model_name)
                self.reader_model.to(device)
                self.reader_model.eval()
                self.has_reader = True
                print("✓ Reader model loaded")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load reader model: {e}")
                print("  Continuing with retrieval-only evaluation...")
                self.has_reader = False
        else:
            self.has_reader = False
    
    def load_wiki_rag_index(self, index_path: Path):
        """Load wiki-rag FAISS index."""
        try:
            self.vectorstore = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✓ Loaded wiki-rag index with {self.vectorstore.index.ntotal} vectors")
            
            # Check metadata structure
            sample_docs = self.vectorstore.similarity_search("test", k=1)
            if sample_docs:
                sample_metadata = sample_docs[0].metadata
                print(f"✓ Sample metadata keys: {list(sample_metadata.keys())}")
        except Exception as e:
            print(f"✗ Error loading wiki-rag index: {e}")
            raise
    
    def normalize_string(self, text: str) -> str:
        """
        Normalize string for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text (lowercase, whitespace normalized, punctuation removed)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove punctuation (optional - can be made configurable)
        # text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def extract_gold_answers(self, example: Dict) -> List[str]:
        """
        Extract gold answers from Natural Questions example.
        
        Natural Questions has 'answer' field which contains long answer text.
        We extract short answer strings (multiple acceptable aliases) for retrieval evaluation.
        
        Strategy:
        1. Extract first sentence (often contains the answer)
        2. Extract key phrases (dates, names, numbers, etc.)
        3. Extract noun phrases and important entities
        
        Args:
            example: Natural Questions example dictionary
            
        Returns:
            List of normalized gold answer strings (short phrases)
        """
        # Natural Questions dataset has 'answer' field
        answer = example.get('answer', '')
        
        if not answer:
            return []
        
        # Handle both string and list formats
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, list):
            answers = answer
        else:
            answers = []
        
        normalized_answers = []
        
        for ans in answers:
            if not ans:
                continue
            
            ans_str = str(ans)
            normalized = self.normalize_string(ans_str)
            
            # Extract multiple answer candidates
            # Priority: specific answers (years, numbers, names) > phrases > sentences
            
            candidates = []
            
            # 1. Extract years (4-digit numbers) - HIGH PRIORITY (often direct answers)
            # Extract full 4-digit years
            full_years = re.findall(r'\b((19|20)\d{2})\b', normalized)
            years = [y[0] for y in full_years]  # Extract the full year, not just prefix
            candidates.extend(years)
            
            # 2. Extract standalone numbers (1-4 digits) - often answers
            # Look for numbers that appear standalone or with context
            numbers = re.findall(r'\b(\d{1,4})\b', normalized[:800])  # First 800 chars
            # Remove duplicates and very common numbers
            seen_nums = set()
            for num in numbers:
                if num not in seen_nums and num not in ['1', '2', '3']:  # Skip very common
                    seen_nums.add(num)
                    candidates.append(num)
            
            # 3. First sentence (often contains the direct answer)
            first_sentence = normalized.split('.')[0].strip()
            if len(first_sentence) > 10 and len(first_sentence) < 200:
                candidates.append(first_sentence)
            
            # 4. Extract key phrases containing numbers/years (e.g., "in 2017", "since 1995")
            # Look for patterns like "in YEAR", "since YEAR", "YEAR season", etc.
            year_phrases = re.findall(r'\b(in|since|during|from|until|by)\s+((19|20)\d{2})\b', normalized[:500])
            for match in year_phrases[:5]:
                candidates.append(f"{match[0]} {match[1]}")
                candidates.append(match[1])  # Also add just the year
            
            # 5. Extract first 2 sentences (for context)
            sentences = re.split(r'[.!?]\s+', normalized)
            for sent in sentences[:2]:
                sent = sent.strip()
                if 20 <= len(sent) <= 200:
                    candidates.append(sent)
            
            # 6. Extract noun phrases from first sentence (2-5 words)
            first_sent_words = first_sentence.split()
            if len(first_sent_words) >= 2:
                for length in [2, 3, 4, 5]:
                    for i in range(len(first_sent_words) - length + 1):
                        phrase = ' '.join(first_sent_words[i:i+length])
                        if 10 <= len(phrase) <= 120:
                            candidates.append(phrase)
                        if len(candidates) > 50:
                            break
                    if len(candidates) > 50:
                        break
            
            # Normalize all candidates
            normalized_candidates = [self.normalize_string(c) for c in candidates if c]
            
            # Remove duplicates and very short/long candidates
            seen = set()
            for cand in normalized_candidates:
                if cand and 5 <= len(cand) <= 500 and cand not in seen:
                    seen.add(cand)
                    normalized_answers.append(cand)
        
        # Limit to top 10 most relevant answers (prioritize shorter, more specific ones)
        # Sort by length (shorter = more specific) and take first 10
        normalized_answers.sort(key=len)
        return normalized_answers[:10]
    
    def retrieve_documents(self, query: str, top_k: int = 100) -> List[Dict]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries with 'text', 'metadata', 'score', 'rank'
        """
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        documents = []
        for rank, (doc, score) in enumerate(results, 1):
            documents.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'rank': rank
            })
        
        return documents
    
    def calculate_answer_recall(
        self,
        retrieved_docs: List[Dict],
        gold_answers: List[str],
        k: int
    ) -> bool:
        """
        Calculate AnswerRecall@K: Check if any gold answer is substring of any retrieved doc.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            gold_answers: List of normalized gold answer strings
            k: Top-K to consider
            
        Returns:
            True if any gold answer is found as substring in top-K retrieved docs
        """
        if not gold_answers:
            return False  # No gold answers means no recall
        
        top_k_docs = retrieved_docs[:k]
        
        # Combine all retrieved text
        retrieved_text = " ".join([self.normalize_string(doc['text']) for doc in top_k_docs])
        
        # Check if any gold answer is a substring of retrieved text
        for gold_answer in gold_answers:
            if gold_answer and gold_answer in retrieved_text:
                return True
        
        return False
    
    def evaluate_on_natural_questions(
        self,
        nq_dataset,
        output_path: Optional[Path] = None,
        top_k_values: List[int] = [1, 5, 20, 50],
        max_examples: Optional[int] = None,
        debug: bool = False
    ) -> Dict:
        """
        Evaluate retrieval performance on Natural Questions dataset.
        
        Args:
            nq_dataset: Natural Questions dataset (from Hugging Face)
            output_path: Path to save results JSON file
            top_k_values: List of K values to evaluate
            max_examples: Maximum number of examples to process
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Processing {len(nq_dataset)} examples...")
        if max_examples:
            nq_dataset = nq_dataset.select(range(min(max_examples, len(nq_dataset))))
            print(f"Limited to {len(nq_dataset)} examples")
        
        start_time = time.time()
        retrieval_times = []
        
        # Metrics storage
        metrics = {f"answer_recall@{k}": [] for k in top_k_values}
        
        results = []
        total_with_answers = 0
        
        for example in tqdm(nq_dataset, desc="Evaluating retrieval"):
            # Extract question and answer
            question = example.get('query', '') or example.get('question', '')
            example_id = example.get('id', 'unknown')
            
            # Extract gold answers
            gold_answers = self.extract_gold_answers(example)
            
            # Skip if no gold answers
            if not gold_answers:
                continue
            
            total_with_answers += 1
            
            # Debug output for first example
            if debug and len(results) == 0:
                print(f"\n{'='*70}")
                print("DEBUG: First Example")
                print(f"{'='*70}")
                print(f"Question: {question}")
                print(f"Gold answers extracted ({len(gold_answers)}):")
                for i, ans in enumerate(gold_answers[:5], 1):
                    print(f"  {i}. {ans[:100]}...")
                print()
            
            # Retrieve documents
            retrieval_start = time.time()
            retrieved_docs = self.retrieve_documents(question, top_k=max(top_k_values))
            retrieval_time = time.time() - retrieval_start
            retrieval_times.append(retrieval_time)
            
            # Calculate AnswerRecall@K for each K
            example_metrics = {}
            for k in top_k_values:
                answer_recall = self.calculate_answer_recall(retrieved_docs, gold_answers, k)
                metrics[f"answer_recall@{k}"].append(1.0 if answer_recall else 0.0)
                example_metrics[f"answer_recall@{k}"] = answer_recall
            
            # Debug output for first example
            if debug and len(results) == 0:
                print(f"Retrieved documents (top 3):")
                for i, doc in enumerate(retrieved_docs[:3], 1):
                    print(f"  {i}. Score: {doc['score']:.4f}")
                    print(f"     Text: {doc['text'][:150]}...")
                    print(f"     Title: {doc['metadata'].get('title', 'N/A')}")
                print(f"\nAnswerRecall results:")
                for k in top_k_values:
                    print(f"  @{k}: {example_metrics[f'answer_recall@{k}']}")
                print(f"{'='*70}\n")
            
            results.append({
                'id': example_id,
                'question': question,
                'gold_answers': gold_answers,
                'metrics': example_metrics,
                'retrieval_time': retrieval_time,
                'num_retrieved': len(retrieved_docs)
            })
        
        # Calculate aggregate metrics
        total_time = time.time() - start_time
        avg_retrieval_time = np.mean(retrieval_times) if retrieval_times else 0.0
        
        aggregate_metrics = {}
        for metric_name, values in metrics.items():
            aggregate_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'total': len(values),
                'correct': sum(values)
            }
        
        evaluation_results = {
            'total_examples': len(results),
            'total_with_answers': total_with_answers,
            'top_k_values': top_k_values,
            'metrics': aggregate_metrics,
            'timing': {
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60.0,
                'avg_retrieval_time_seconds': avg_retrieval_time,
                'avg_retrieval_time_ms': avg_retrieval_time * 1000,
                'throughput_examples_per_second': len(results) / total_time if total_time > 0 else 0.0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Print results
        print(f"\n{'='*70}")
        print("Natural Questions Retrieval Evaluation Results")
        print(f"{'='*70}")
        print(f"Total examples processed: {len(results)}")
        print(f"Examples with gold answers: {total_with_answers}")
        print(f"\nRetrieval Performance:")
        print(f"  Avg retrieval time: {avg_retrieval_time*1000:.2f} ms per query")
        print(f"  Throughput: {evaluation_results['timing']['throughput_examples_per_second']:.2f} queries/second")
        
        print(f"\n{'='*70}")
        print("AnswerRecall@K Metrics")
        print(f"{'='*70}")
        print(f"{'Metric':<25} {'Recall':<10} {'Correct':<10} {'Total':<10} {'Std':<10}")
        print("-" * 70)
        
        for k in top_k_values:
            metric_name = f"answer_recall@{k}"
            if metric_name in aggregate_metrics:
                recall = aggregate_metrics[metric_name]['mean']
                correct = aggregate_metrics[metric_name]['correct']
                total = aggregate_metrics[metric_name]['total']
                std = aggregate_metrics[metric_name]['std']
                print(f"{metric_name:<25} {recall:<10.4f} {correct:<10} {total:<10} {std:<10.4f}")
        
        # Save results
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_data = {
                'evaluation_results': evaluation_results,
                'detailed_results': results[:1000]  # Save first 1000 for inspection
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to {output_path}")
        
        return evaluation_results


def load_natural_questions_dataset():
    """
    Load Natural Questions dataset from Hugging Face.
    
    Returns:
        Natural Questions dataset
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    print("Loading Natural Questions dataset from Hugging Face...")
    # Load from sentence-transformers/natural-questions
    dataset = load_dataset("sentence-transformers/natural-questions", "pair", split="train")
    
    print(f"✓ Loaded Natural Questions dataset: {len(dataset)} examples")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Natural Questions Retrieval Evaluation"
    )
    parser.add_argument(
        "--wiki-rag-index",
        type=str,
        default="wiki-rag/wiki_rag_data/wiki_index__top_100000__2025-04-11",
        help="Path to wiki-rag FAISS index directory"
    )
    parser.add_argument(
        "--reader-model",
        type=str,
        default="datarpit/distilbert-base-uncased-finetuned-natural-questions",
        help="Reader model for answer extraction (optional, for retrieval-only set to None)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs='+',
        default=[1, 5, 20, 50],
        help="Top-K values to evaluate (can specify multiple)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON file"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (for testing)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en",
        help="Embedding model name for RAG retrieval"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run models on (cuda/cpu, auto-detected if not specified)"
    )
    parser.add_argument(
        "--no-reader",
        action="store_true",
        help="Skip loading reader model (retrieval-only evaluation)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for first example"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Load Natural Questions dataset
    nq_dataset = load_natural_questions_dataset()
    
    # Initialize evaluator
    reader_model = None if args.no_reader else args.reader_model
    evaluator = NaturalQuestionsRetrievalEvaluator(
        wiki_rag_index_path=Path(args.wiki_rag_index),
        embedding_model_name=args.embedding_model,
        reader_model_name=reader_model,
        device=device
    )
    
    # Run evaluation
    output_path = Path(args.output) if args.output else None
    results = evaluator.evaluate_on_natural_questions(
        nq_dataset=nq_dataset,
        output_path=output_path,
        top_k_values=args.top_k,
        max_examples=args.max_examples,
        debug=args.debug
    )
    
    return results


if __name__ == "__main__":
    main()

