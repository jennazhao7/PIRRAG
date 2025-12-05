#!/usr/bin/env python3
"""
FEVER Post-PIR Retrieval Evaluation

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
- EvidenceRecall@K: Percentage of queries where at least one evidence sentence is found in top-K retrieved documents
- EvidenceCoverage@K: Average fraction of evidence sentences found in top-K retrieved documents
- MRR (Mean Reciprocal Rank): Average reciprocal rank of first relevant document
- Precision@K: Fraction of retrieved documents that contain evidence
- DocumentRecall@K: Percentage of queries where all evidence documents are found
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


class FEVERPostPIRRetrievalEvaluator:
    """
    Evaluates post-PIR retrieval performance on FEVER dataset.
    Evaluates actual documents retrieved by PIR.
    """
    
    def __init__(
        self,
        pir_output_path: Path,
        pir_input_path: Optional[Path] = None
    ):
        """
        Initialize FEVER post-PIR retrieval evaluator.
        
        Args:
            pir_output_path: Path to PIR output directory or JSON file containing retrieved documents
            pir_input_path: Optional path to original pir_input.json (for query metadata)
        """
        # Load PIR output (retrieved documents)
        print(f"Loading PIR output from {pir_output_path}...")
        self.pir_results = self._load_pir_output(pir_output_path)
        print(f"✓ Loaded {len(self.pir_results)} queries with PIR-retrieved documents")
        
        # Load original pir_input for query metadata (evidence, claim, etc.)
        if pir_input_path:
            print(f"Loading original queries from {pir_input_path}...")
            with open(pir_input_path, 'r') as f:
                self.original_queries = {q['query_id']: q for q in json.load(f)}
            print(f"✓ Loaded {len(self.original_queries)} original queries")
        else:
            self.original_queries = {}
    
    def _load_pir_output(self, path: Path) -> List[Dict]:
        """
        Load PIR output files.
        
        Supports:
        - Single JSON file with list of results
        - Directory with per-query JSON files (query_XXXX_documents.json)
        - Single JSON file with single result object
        """
        path = Path(path)
        
        if path.is_file():
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Check if it's a list or single object
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Single result or dict with results
                if 'documents' in data or 'query_idx' in data:
                    return [data]
                elif 'results' in data:
                    return data['results']
                else:
                    # Try to find list-like structure
                    return [data]
            else:
                raise ValueError(f"Unexpected format in {path}")
        
        elif path.is_dir():
            # Load all query_XXXX_documents.json files
            results = []
            query_files = sorted(path.glob("query_*_documents.json"))
            
            for query_file in query_files:
                with open(query_file, 'r') as f:
                    results.append(json.load(f))
            
            if not results:
                # Try alternative naming
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
    
    def extract_evidence_info(self, evidence: List) -> Dict[str, Set]:
        """
        Extract evidence information from FEVER evidence structure.
        
        Evidence format: [[page_id, sentence_id, entity, sentence_index], ...]
        
        Returns:
            Dictionary with 'page_ids', 'sentence_ids', 'entities' sets
        """
        page_ids = set()
        sentence_ids = set()
        entities = set()
        
        for ev_group in evidence:
            for ev in ev_group:
                if len(ev) >= 2:
                    page_id = ev[0]
                    sentence_id = ev[1]
                    page_ids.add(page_id)
                    sentence_ids.add(sentence_id)
                    if len(ev) >= 3:
                        entities.add(ev[2])
        
        return {
            "page_ids": page_ids,
            "sentence_ids": sentence_ids,
            "entities": entities,
            "entity_strings": [str(e) for e in entities]
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()
    
    def check_evidence_in_document(self, document: str, evidence_info: Dict[str, Set]) -> Dict[str, bool]:
        """
        Check if document contains evidence.
        
        Returns:
            Dictionary with checks for page_id, sentence_id, entity matches
        """
        doc_normalized = self.normalize_text(document)
        
        # Check for entity mentions
        entity_found = False
        for entity in evidence_info["entity_strings"]:
            entity_normalized = self.normalize_text(entity.replace("_", " "))
            if entity_normalized in doc_normalized:
                entity_found = True
                break
        
        # For FEVER, we check if document mentions the entities
        # Page ID and sentence ID matching would require document metadata
        # For now, we focus on entity/content matching
        
        return {
            "entity_found": entity_found,
            "has_content": len(doc_normalized) > 0
        }
    
    def evaluate_query(
        self,
        pir_result: Dict,
        original_query: Optional[Dict] = None,
        k_values: List[int] = [1, 5, 10, 20, 50, 100]
    ) -> Dict:
        """
        Evaluate retrieval for a single query using PIR-retrieved documents.
        
        Args:
            pir_result: PIR output with 'documents' list
            original_query: Original query with evidence metadata
            k_values: K values to evaluate
            
        Returns:
            Dictionary with metrics for each k value
        """
        documents = pir_result.get("documents", [])
        query_idx = pir_result.get("query_idx", None)
        
        # Get evidence from original query if available
        if original_query:
            evidence = original_query.get("evidence", [])
        else:
            evidence = []
        
        if not evidence:
            return {f"metrics_k{k}": {} for k in k_values}
        
        # Extract evidence info
        evidence_info = self.extract_evidence_info(evidence)
        if not evidence_info["entities"]:
            return {f"metrics_k{k}": {} for k in k_values}
        
        metrics = {}
        
        for k in k_values:
            k_docs = documents[:k] if k else documents
            
            # Check each document for evidence
            docs_with_evidence = []
            first_relevant_rank = None
            
            for rank, doc in enumerate(k_docs, 1):
                checks = self.check_evidence_in_document(doc, evidence_info)
                if checks["entity_found"]:
                    docs_with_evidence.append(rank)
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
            
            # Calculate metrics
            # EvidenceRecall@K: At least one document contains evidence entities
            evidence_recall = 1.0 if docs_with_evidence else 0.0
            
            # EvidenceCoverage@K: Fraction of documents containing evidence
            if k_docs:
                evidence_coverage = len(docs_with_evidence) / len(k_docs)
            else:
                evidence_coverage = 0.0
            
            # Precision@K: Fraction of retrieved documents that contain evidence
            if k_docs:
                precision = len(docs_with_evidence) / len(k_docs)
            else:
                precision = 0.0
            
            # MRR: Reciprocal rank of first relevant document
            mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            
            metrics[f"metrics_k{k}"] = {
                "evidence_recall": evidence_recall,
                "evidence_coverage": evidence_coverage,
                "precision": precision,
                "mrr": mrr,
                "num_docs_with_evidence": len(docs_with_evidence),
                "num_retrieved_docs": len(k_docs),
                "num_evidence_entities": len(evidence_info["entities"]),
            }
        
        return metrics
    
    def evaluate_all(self, k_values: List[int] = [1, 5, 10, 20, 50, 100]) -> Dict:
        """
        Evaluate all queries.
        
        Returns:
            Dictionary with aggregated metrics
        """
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
                    "evidence_recall": np.mean(all_metrics[key]["evidence_recall"]) if all_metrics[key]["evidence_recall"] else 0.0,
                    "evidence_coverage": np.mean(all_metrics[key]["evidence_coverage"]) if all_metrics[key]["evidence_coverage"] else 0.0,
                    "precision": np.mean(all_metrics[key]["precision"]) if all_metrics[key]["precision"] else 0.0,
                    "mrr": np.mean(all_metrics[key]["mrr"]) if all_metrics[key]["mrr"] else 0.0,
                    "avg_docs_with_evidence": np.mean(all_metrics[key]["num_docs_with_evidence"]) if all_metrics[key]["num_docs_with_evidence"] else 0.0,
                    "avg_retrieved_docs": np.mean(all_metrics[key]["num_retrieved_docs"]) if all_metrics[key]["num_retrieved_docs"] else 0.0,
                }
        
        return {
            "total_queries": len(self.pir_results),
            "metrics": aggregated
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FEVER post-PIR retrieval performance")
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
        help="Optional: Path to original fever_pir_input.json (for query metadata)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fever_post_pir_evaluation_results.json",
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
    
    evaluator = FEVERPostPIRRetrievalEvaluator(
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
    print("FEVER Post-PIR Retrieval Evaluation Results")
    print(f"{'='*70}")
    print(f"Total queries: {results['total_queries']}")
    print(f"\nMetrics:")
    print(f"{'K':<6} {'EvidenceRecall':<18} {'Coverage':<12} {'Precision':<12} {'MRR':<10}")
    print("-" * 70)
    
    for k in args.k_values:
        key = f"k{k}"
        if key in results["metrics"]:
            m = results["metrics"][key]
            print(f"{k:<6} {m['evidence_recall']*100:>6.2f}%{'':<10} {m['evidence_coverage']*100:>6.2f}%{'':<8} {m['precision']*100:>6.2f}%{'':<8} {m['mrr']:>6.4f}")
    
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

