#!/usr/bin/env python3
"""
Run PIR using single-query client in a loop for batched queries.

This avoids the buffer overflow issue in the batched client by processing
each query individually using the single-query client that works.
"""

import json
import subprocess
import sys
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


def create_single_query_input(query_data: Dict, output_path: Path):
    """Create input file for single-query client from one query."""
    # The single-query client expects ground_truth.json format:
    # {
    #   "top_k": 100,
    #   "distances": [...],
    #   "centroid_indices": [...]
    # }
    # But we only have centroid_indices from the batched input
    # Create dummy distances (all zeros) - the client will compute real distances
    centroid_indices = query_data.get("top_k_centroid_indices", [])
    num_results = query_data.get("num_results", len(centroid_indices))
    
    single_query = {
        "top_k": num_results,
        "distances": [0.0] * len(centroid_indices),  # Dummy distances
        "centroid_indices": centroid_indices[:num_results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(single_query, f, indent=2)


def extract_vectors_from_stderr(stderr: str) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Extract vector indices and vectors from PIR client stderr output.
    
    Returns:
        (cluster_ids, indices, vectors) tuple
    """
    cluster_ids = []
    indices = []
    vectors = []
    
    lines = stderr.strip().split('\n')
    
    for line in lines:
        if "Final query result from cluster" in line:
            # Older/verbose client output that includes both cluster and index
            match = re.search(r'from cluster (\d+) at index (\d+):', line)
            if match:
                cluster_id = int(match.group(1))
                index = int(match.group(2))
                
                # Extract the vector (everything inside square brackets)
                vector_match = re.search(r'\[(.*?)\]', line)
                if vector_match:
                    vector_str = vector_match.group(1)
                    # Split by whitespace and convert to floats
                    vector = np.array([float(x) for x in vector_str.split()])
                    
                    cluster_ids.append(cluster_id)
                    indices.append(index)
                    vectors.append(vector)

        elif "Final query result at index" in line:
            # Fallback output format (no cluster id; index only, often with binary blob)
            match = re.search(r'Final query result at index (\d+):', line)
            if match:
                index = int(match.group(1))
                # We do not attempt to parse the vector payload when it is printed as binary.
                cluster_ids.append(index)  # best-effort: treat index as cluster id placeholder
                indices.append(index)
    
    if vectors:
        vectors = np.array(vectors)
    else:
        vectors = np.array([])
    
    return cluster_ids, indices, vectors


def run_single_query_client(
    query_data: Dict,
    query_id: int,
    server_ip: str,
    extra_input: Path,
    client_binary: Path,
    temp_dir: Path,
    output_dir: Path = None,
    cluster_to_vectors: Optional[Dict[str, List[int]]] = None
) -> tuple:
    """
    Run single-query client on one query.
    
    Returns:
        (success: bool, result_data: dict or None)
    """
    # Create temporary input file for this query
    temp_input = temp_dir / f"query_{query_id}_input.json"
    create_single_query_input(query_data, temp_input)
    
    cmd = [
        str(client_binary),
        "-ip", server_ip,
        "-thread", "1",
        "-input", str(temp_input),
        "-extra_input", str(extra_input)
    ]
    
    if query_id % 10 == 0 or query_id == 0:
        print(f"Processing query {query_id}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout per query
        )
        
        if result.returncode != 0:
            print(f"\n✗ Error processing query {query_id}:")
            print(result.stderr[:500])  # First 500 chars of error
            return False, None
        
        # Save raw stdout/stderr for debugging
        if output_dir:
            logs_dir = output_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            with open(logs_dir / f"query_{query_id}_stdout.log", "w") as f_out:
                f_out.write(result.stdout or "")
            with open(logs_dir / f"query_{query_id}_stderr.log", "w") as f_err:
                f_err.write(result.stderr or "")

        # Extract results from stderr
        cluster_ids, indices, vectors = extract_vectors_from_stderr(result.stderr)

        # Map (cluster_id, index_in_cluster) -> global vector id when possible
        vector_ids: List[int] = []
        for cid, idx in zip(cluster_ids, indices):
            if cluster_to_vectors and str(cid) in cluster_to_vectors:
                vec_list = cluster_to_vectors[str(cid)]
                if 0 <= idx < len(vec_list):
                    vector_ids.append(vec_list[idx])
                    continue
            # Fallback: if mapping unavailable, keep the index itself
            vector_ids.append(idx)

        # Create result data
        result_data = {
            "query_id": query_data.get("query_id", query_id),
            # For downstream doc retrieval, use global vector ids when available
            "top_k_indices": vector_ids,
            "cluster_ids": cluster_ids,
            "num_results": len(vector_ids)
        }
        
        # Save individual result if output_dir specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            result_file = output_dir / f"query_{query_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
        
        return True, result_data
        
    except subprocess.TimeoutExpired:
        print(f"\n✗ Timeout processing query {query_id}")
        return False, None
    except Exception as e:
        print(f"\n✗ Error processing query {query_id}: {e}")
        return False, None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run PIR using single-query client in a loop"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to batched PIR input JSON file"
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="localhost:50052",
        help="PIR server IP and port (default: localhost:50052)"
    )
    parser.add_argument(
        "--extra-input",
        type=str,
        default="/home/jzhao7/RAGPIR/prototype/data/lists.json",
        help="Path to lists.json file"
    )
    parser.add_argument(
        "--client-binary",
        type=str,
        default="/home/jzhao7/RAGPIR/___go_build_easypir_client_new_linux",
        help="Path to single-query PIR client binary"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to process (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save PIR results (default: <input_dir>/pir_results)"
    )
    
    args = parser.parse_args()
    
    # Load input
    input_path = Path(args.input)
    print(f"Loading PIR input from: {input_path}")
    queries = load_pir_input(input_path)
    print(f"Total queries: {len(queries)}")
    
    if args.max_queries:
        queries = queries[:args.max_queries]
        print(f"Processing first {len(queries)} queries (--max-queries={args.max_queries})")
    
    # Create temp directory for input files
    temp_dir = Path(tempfile.mkdtemp(prefix="pir_single_query_"))
    print(f"Temporary files in: {temp_dir}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "pir_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Load cluster->vector mapping (lists.json) if available
    cluster_to_vectors = None
    try:
        with open(args.extra_input, "r") as f_lists:
            cluster_to_vectors = json.load(f_lists)
        print(f"Loaded cluster->vector mapping from {args.extra_input}")
    except Exception as e:
        print(f"Warning: could not load cluster mapping from {args.extra_input}: {e}")

    # Process each query
    print(f"\n{'='*70}")
    print(f"Processing {len(queries)} queries using single-query client")
    print(f"{'='*70}\n")
    
    successful = 0
    failed = 0
    all_results = []
    
    try:
        for i, query_data in enumerate(queries):
            success, result_data = run_single_query_client(
                query_data,
                i,
                args.server_ip,
                Path(args.extra_input),
                Path(args.client_binary),
                temp_dir,
                output_dir,
                cluster_to_vectors
            )
            
            if success and result_data:
                successful += 1
                all_results.append(result_data)
            else:
                failed += 1
                if failed <= 5:  # Show first 5 errors
                    print(f"  Query {i} failed")
    
    finally:
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        except:
            pass
    
    # Save aggregated results
    aggregated_file = output_dir / "all_results.json"
    with open(aggregated_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved aggregated results to {aggregated_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total queries: {len(queries)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*70}")
    
    if failed > 0:
        print(f"\n⚠ {failed} queries failed")
        sys.exit(1)
    else:
        print("\n✓ All queries processed successfully!")


def load_pir_input(input_path: Path) -> List[Dict]:
    """Load PIR input JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    main()

