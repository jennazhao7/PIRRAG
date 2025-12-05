#!/usr/bin/env python3
"""
HotpotQA Post-PIR Retrieval Evaluation

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
- TitleRecall@K (Any): Percentage of queries where at least one supporting page title is found in top-K
- TitleRecall@K (All): Percentage of queries where all supporting page titles are found in top-K
- SentenceRecall@K: Percentage of queries where at least one supporting sentence is found
- SupportingFactRecall@K: Percentage of supporting facts found in top-K
- MRR (Mean Reciprocal Rank): Average reciprocal rank of first supporting page
- Precision@K: Fraction of retrieved documents that are supporting pages
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


class HotpotQAPostPIRRetrievalEvaluator:
    """
    Evaluates post-PIR retrieval performance on HotpotQA dataset.
    Evaluates actual documents retrieved by PIR.
    """
    
    def __init__(
        self,
        pir_output_path: Path,
        pir_input_path: Optional[Path] = None
    ):
        """
        Initialize HotpotQA post-PIR retrieval evaluator.
        
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
    
    def extract_supporting_facts(self, supporting_facts: Dict) -> Dict[str, Set]:
        """
        Extract supporting facts information from HotpotQA structure.
        
        Supporting facts format: {"title": [[sentence_id, ...], ...]}
        
        Returns:
            Dictionary with 'titles', 'sentences', 'title_sentence_pairs' sets
        """
        titles = set()
        sentences = set()
        title_sentence_pairs = set()
        
        if isinstance(supporting_facts, dict):
            for title, sentence_lists in supporting_facts.items():
                titles.add(title)
                for sentence_list in sentence_lists:
                    if isinstance(sentence_list, list):
                        for sentence_id in sentence_list:
                            sentences.add(sentence_id)
                            title_sentence_pairs.add((title, sentence_id))
        
        return {
            "titles": titles,
            "sentences": sentences,
            "title_sentence_pairs": title_sentence_pairs
        }
    
    def normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        return title.strip().lower().replace("_", " ")
    
    def extract_title_from_document(self, document: str) -> Optional[str]:
        """
        Try to extract title from document text.
        For HotpotQA, documents may contain title information.
        """
        # Try to find title patterns in document
        # Common patterns: "Title: ...", "## ...", etc.
        lines = document.split('\n')
        for line in lines[:5]:  # Check first few lines
            line = line.strip()
            if line:
                # Remove markdown headers
                line = re.sub(r'^#+\s*', '', line)
                # Remove "Title:" prefix
                line = re.sub(r'^Title:\s*', '', line, flags=re.IGNORECASE)
                if line:
                    return self.normalize_title(line)
        return None
    
    def check_supporting_facts_in_document(self, document: str, facts_info: Dict[str, Set]) -> Dict[str, bool]:
        """Check if document contains supporting facts."""
        doc_normalized = self.normalize_text(document)
        doc_title = self.extract_title_from_document(document)
        
        # Check title match
        title_match = False
        if doc_title:
            gt_titles = {self.normalize_title(t) for t in facts_info["titles"]}
            title_match = doc_title in gt_titles
        
        # Check if document mentions any supporting titles
        title_mentioned = False
        for title in facts_info["titles"]:
            title_normalized = self.normalize_title(title)
            if title_normalized in doc_normalized:
                title_mentioned = True
                break
        
        return {
            "title_match": title_match,
            "title_mentioned": title_mentioned,
            "has_content": len(doc_normalized) > 0
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()
    
    def evaluate_query(
        self,
        pir_result: Dict,
        original_query: Optional[Dict] = None,
        k_values: List[int] = [1, 5, 10, 20, 50, 100]
    ) -> Dict:
        """Evaluate retrieval for a single query using PIR-retrieved documents."""
        documents = pir_result.get("documents", [])
        query_idx = pir_result.get("query_idx", None)
        
        # Get supporting facts from original query if available
        if original_query:
            supporting_facts = original_query.get("supporting_facts", {})
        else:
            supporting_facts = {}
        
        if not supporting_facts:
            return {f"metrics_k{k}": {} for k in k_values}
        
        # Extract supporting facts info
        facts_info = self.extract_supporting_facts(supporting_facts)
        if not facts_info["titles"]:
            return {f"metrics_k{k}": {} for k in k_values}
        
        metrics = {}
        
        for k in k_values:
            k_docs = documents[:k] if k else documents
            
            # Check each document for supporting facts
            docs_with_titles = set()
            docs_with_any_title = set()
            first_relevant_rank = None
            
            for rank, doc in enumerate(k_docs, 1):
                checks = self.check_supporting_facts_in_document(doc, facts_info)
                
                if checks["title_match"] or checks["title_mentioned"]:
                    docs_with_any_title.add(rank)
                    # Try to identify which title
                    doc_title = self.extract_title_from_document(doc)
                    if doc_title:
                        gt_titles = {self.normalize_title(t) for t in facts_info["titles"]}
                        if doc_title in gt_titles:
                            docs_with_titles.add(doc_title)
                    
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
            
            # Normalize ground truth titles
            gt_titles = {self.normalize_title(t) for t in facts_info["titles"]}
            
            # Calculate metrics
            # TitleRecall@K (Any): At least one supporting title found
            title_recall_any = 1.0 if docs_with_any_title else 0.0
            
            # TitleRecall@K (All): All supporting titles found
            if gt_titles:
                title_recall_all = 1.0 if gt_titles.issubset(docs_with_titles) else 0.0
            else:
                title_recall_all = 0.0
            
            # Precision@K: Fraction of retrieved documents that are supporting pages
            if k_docs:
                precision = len(docs_with_any_title) / len(k_docs)
            else:
                precision = 0.0
            
            # MRR: Reciprocal rank of first supporting title
            mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            
            metrics[f"metrics_k{k}"] = {
                "title_recall_any": title_recall_any,
                "title_recall_all": title_recall_all,
                "precision": precision,
                "mrr": mrr,
                "num_docs_with_titles": len(docs_with_any_title),
                "num_retrieved_docs": len(k_docs),
                "num_supporting_titles": len(gt_titles),
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
                    "title_recall_any": np.mean(all_metrics[key]["title_recall_any"]) if all_metrics[key]["title_recall_any"] else 0.0,
                    "title_recall_all": np.mean(all_metrics[key]["title_recall_all"]) if all_metrics[key]["title_recall_all"] else 0.0,
                    "precision": np.mean(all_metrics[key]["precision"]) if all_metrics[key]["precision"] else 0.0,
                    "mrr": np.mean(all_metrics[key]["mrr"]) if all_metrics[key]["mrr"] else 0.0,
                    "avg_docs_with_titles": np.mean(all_metrics[key]["num_docs_with_titles"]) if all_metrics[key]["num_docs_with_titles"] else 0.0,
                }
        
        return {
            "total_queries": len(self.pir_results),
            "metrics": aggregated
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA post-PIR retrieval performance")
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
        help="Optional: Path to original hotpot_pir_input.json (for query metadata)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hotpot_post_pir_evaluation_results.json",
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
    
    evaluator = HotpotQAPostPIRRetrievalEvaluator(
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
    print("HotpotQA Post-PIR Retrieval Evaluation Results")
    print(f"{'='*70}")
    print(f"Total queries: {results['total_queries']}")
    print(f"\nMetrics:")
    print(f"{'K':<6} {'TitleRecall(Any)':<18} {'TitleRecall(All)':<18} {'Precision':<12} {'MRR':<10}")
    print("-" * 70)
    
    for k in args.k_values:
        key = f"k{k}"
        if key in results["metrics"]:
            m = results["metrics"][key]
            print(f"{k:<6} {m['title_recall_any']*100:>6.2f}%{'':<10} {m['title_recall_all']*100:>6.2f}%{'':<10} {m['precision']*100:>6.2f}%{'':<8} {m['mrr']:>6.4f}")
    
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

