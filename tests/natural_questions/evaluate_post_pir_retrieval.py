#!/usr/bin/env python3
"""
Natural Questions Post-PIR Retrieval Evaluation

Evaluates retrieval performance AFTER PIR (Private Information Retrieval) has retrieved actual documents.

This script evaluates the actual documents returned by PIR, not just centroid selection.

Input Format:
- PIR output files containing actual retrieved documents per query
- Format: JSON files with structure:
  {
    "query_idx": int,
    "top_k_indices": List[int],
    "documents": List[str],  # Actual document texts retrieved by PIR
    ...
  }

Metrics:
- AnswerRecall@K: Percentage of queries where at least one answer string is found in top-K retrieved documents
- AnswerCoverage@K: Average fraction of answer strings found in top-K retrieved documents
- MRR (Mean Reciprocal Rank): Average reciprocal rank of first document containing answer
- Precision@K: Fraction of retrieved documents that contain answer strings
- ExactMatch@K: Percentage of queries where exact answer string is found
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class NaturalQuestionsPostPIRRetrievalEvaluator:
    """
    Evaluates post-PIR retrieval performance on Natural Questions dataset.
    Evaluates actual documents retrieved by PIR.
    """
    
    def __init__(
        self,
        pir_output_path: Path,
        pir_input_path: Optional[Path] = None
    ):
        """
        Initialize Natural Questions post-PIR retrieval evaluator.
        
        Args:
            pir_output_path: Path to PIR output directory or JSON file containing retrieved documents
            pir_input_path: Optional path to original pir_input.json (for query metadata)
        """
        # Load PIR output (retrieved documents)
        print(f"Loading PIR output from {pir_output_path}...")
        self.pir_results = self._load_pir_output(pir_output_path)
        print(f"✓ Loaded {len(self.pir_results)} queries with PIR-retrieved documents")
        
        # Load original pir_input for query metadata
        if pir_input_path:
            print(f"Loading original queries from {pir_input_path}...")
            with open(pir_input_path, 'r') as f:
                self.original_queries = {q['query_id']: q for q in json.load(f)}
            print(f"✓ Loaded {len(self.original_queries)} original queries")
        else:
            self.original_queries = {}
    
    def _load_pir_output(self, path: Path) -> List[Dict]:
        """Load PIR output files."""
        path = Path(path)
        
        if path.is_file():
            with open(path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if 'documents' in data or 'query_idx' in data:
                    return [data]
                elif 'results' in data:
                    return data['results']
                else:
                    return [data]
            else:
                raise ValueError(f"Unexpected format in {path}")
        
        elif path.is_dir():
            results = []
            query_files = sorted(path.glob("query_*_documents.json"))
            
            for query_file in query_files:
                with open(query_file, 'r') as f:
                    results.append(json.load(f))
            
            if not results:
                query_files = sorted(path.glob("*.json"))
                for query_file in query_files:
                    with open(query_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            results.extend(data)
                        elif isinstance(data, dict) and ('documents' in data or 'query_idx' in data):
                            results.append(data)
            
            return results
        else:
            raise ValueError(f"Path {path} does not exist")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def extract_answer_strings(self, query_result: Optional[Dict]) -> Set[str]:
        """
        Extract answer strings from Natural Questions query result.
        
        Returns:
            Set of normalized answer strings
        """
        if not query_result:
            return set()
        
        answers = set()
        
        # Try short_answer_strings first
        short_answers = query_result.get("short_answer_strings", [])
        if short_answers:
            for ans in short_answers:
                if ans and ans.strip():
                    answers.add(self.normalize_text(ans))
        
        # Also try answer field
        answer = query_result.get("answer", "")
        if answer and answer.strip():
            # Split long answers into sentences/segments
            sentences = re.split(r'[.!?]\s+', answer)
            for sent in sentences:
                sent = sent.strip()
                if sent and len(sent) > 3:  # Ignore very short segments
                    answers.add(self.normalize_text(sent))
        
        return answers
    
    def check_answer_in_document(self, answer: str, document: str) -> bool:
        """Check if answer string appears in document content."""
        doc_normalized = self.normalize_text(document)
        return answer in doc_normalized
    
    def evaluate_query(
        self,
        pir_result: Dict,
        original_query: Optional[Dict] = None,
        k_values: List[int] = [1, 5, 10, 20, 50, 100]
    ) -> Dict:
        """Evaluate retrieval for a single query using PIR-retrieved documents."""
        documents = pir_result.get("documents", [])
        query_idx = pir_result.get("query_idx", None)
        
        # Get answer strings from original query if available
        answer_strings = self.extract_answer_strings(original_query)
        
        if not answer_strings:
            return {f"metrics_k{k}": {} for k in k_values}
        
        metrics = {}
        
        for k in k_values:
            k_docs = documents[:k] if k else documents
            
            # Check which answers are found in retrieved documents
            found_answers = set()
            answer_found_at_rank = {}
            docs_with_answers = []
            first_relevant_rank = None
            
            for rank, doc in enumerate(k_docs, 1):
                doc_has_answer = False
                for answer in answer_strings:
                    if answer not in found_answers and self.check_answer_in_document(answer, doc):
                        found_answers.add(answer)
                        if answer not in answer_found_at_rank:
                            answer_found_at_rank[answer] = rank
                        doc_has_answer = True
                
                if doc_has_answer:
                    docs_with_answers.append(rank)
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
            
            # Calculate metrics
            # AnswerRecall@K: At least one answer found
            answer_recall = 1.0 if found_answers else 0.0
            
            # AnswerCoverage@K: Fraction of answers found
            if answer_strings:
                answer_coverage = len(found_answers) / len(answer_strings)
            else:
                answer_coverage = 0.0
            
            # Precision@K: Fraction of documents containing answers
            if k_docs:
                precision = len(docs_with_answers) / len(k_docs)
            else:
                precision = 0.0
            
            # MRR: Reciprocal rank of first document containing any answer
            mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            
            # ExactMatch@K: Check if any answer exactly matches document content
            exact_match = 0.0
            for doc in k_docs:
                doc_content = self.normalize_text(doc)
                for answer in answer_strings:
                    # Check for exact match (answer is a complete word/phrase in document)
                    if answer in doc_content:
                        # Additional check: answer should be a complete phrase
                        pattern = r'\b' + re.escape(answer) + r'\b'
                        if re.search(pattern, doc_content):
                            exact_match = 1.0
                            break
                if exact_match:
                    break
            
            metrics[f"metrics_k{k}"] = {
                "answer_recall": answer_recall,
                "answer_coverage": answer_coverage,
                "precision": precision,
                "mrr": mrr,
                "exact_match": exact_match,
                "num_found_answers": len(found_answers),
                "num_total_answers": len(answer_strings),
                "num_retrieved_docs": len(k_docs),
            }
        
        return metrics
    
    def evaluate_all(self, k_values: List[int] = [1, 5, 10, 20, 50, 100]) -> Dict:
        """Evaluate all queries."""
        print(f"\nEvaluating {len(self.pir_results)} queries...")
        
        all_metrics = {f"metrics_k{k}": defaultdict(list) for k in k_values}
        
        for pir_result in tqdm(self.pir_results, desc="Evaluating queries"):
            query_idx = pir_result.get("query_idx")
            original_query = self.original_queries.get(query_idx) if query_idx is not None else None
            
            query_metrics = self.evaluate_query(pir_result, original_query, k_values)
            
            for k in k_values:
                key = f"metrics_k{k}"
                if key in query_metrics and query_metrics[key]:
                    for metric_name, metric_value in query_metrics[key].items():
                        all_metrics[key][metric_name].append(metric_value)
        
        # Aggregate metrics
        aggregated = {}
        for k in k_values:
            key = f"metrics_k{k}"
            if all_metrics[key]:
                aggregated[f"k{k}"] = {
                    "answer_recall": np.mean(all_metrics[key]["answer_recall"]) if all_metrics[key]["answer_recall"] else 0.0,
                    "answer_coverage": np.mean(all_metrics[key]["answer_coverage"]) if all_metrics[key]["answer_coverage"] else 0.0,
                    "precision": np.mean(all_metrics[key]["precision"]) if all_metrics[key]["precision"] else 0.0,
                    "mrr": np.mean(all_metrics[key]["mrr"]) if all_metrics[key]["mrr"] else 0.0,
                    "exact_match": np.mean(all_metrics[key]["exact_match"]) if all_metrics[key]["exact_match"] else 0.0,
                    "avg_found_answers": np.mean(all_metrics[key]["num_found_answers"]) if all_metrics[key]["num_found_answers"] else 0.0,
                }
        
        return {
            "total_queries": len(self.pir_results),
            "metrics": aggregated
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Natural Questions post-PIR retrieval performance")
    parser.add_argument(
        "--pir-output",
        type=str,
        required=True,
        help="Path to PIR output (directory with query_*_documents.json files or single JSON file)"
    )
    parser.add_argument(
        "--pir-input",
        type=str,
        default=None,
        help="Optional: Path to original nq_pir_input.json (for query metadata)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="nq_post_pir_evaluation_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50, 100],
        help="K values to evaluate"
    )
    
    args = parser.parse_args()
    
    evaluator = NaturalQuestionsPostPIRRetrievalEvaluator(
        pir_output_path=Path(args.pir_output),
        pir_input_path=Path(args.pir_input) if args.pir_input else None,
    )
    
    results = evaluator.evaluate_all(k_values=args.k_values)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("Natural Questions Post-PIR Retrieval Evaluation Results")
    print(f"{'='*70}")
    print(f"Total queries: {results['total_queries']}")
    print(f"\nMetrics:")
    print(f"{'K':<6} {'AnswerRecall':<15} {'Coverage':<12} {'Precision':<12} {'MRR':<10} {'ExactMatch':<12}")
    print("-" * 70)
    
    for k in args.k_values:
        key = f"k{k}"
        if key in results["metrics"]:
            m = results["metrics"][key]
            print(f"{k:<6} {m['answer_recall']*100:>6.2f}%{'':<7} {m['answer_coverage']*100:>6.2f}%{'':<8} {m['precision']*100:>6.2f}%{'':<8} {m['mrr']:>6.4f}{'':<6} {m['exact_match']*100:>6.2f}%")
    
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

