#!/usr/bin/env python3
"""
Convert PIR client output to format expected by post-PIR evaluation scripts.

The PIR client outputs vectors/indices, but the evaluation scripts need documents.
This script:
1. Takes PIR client output (vectors/indices)
2. Retrieves documents using PIR document server (port 50051)
3. Formats results for evaluation scripts
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional
import tempfile


def retrieve_documents_via_pir(
    vector_indices: List[int],
    server_ip: str,
    client_binary: Path,
    temp_dir: Path
) -> List[str]:
    """
    Retrieve documents via PIR using vector indices.
    
    Args:
        vector_indices: List of vector indices to retrieve
        server_ip: PIR server IP (port 50051 for documents)
        client_binary: Path to PIR client binary
        temp_dir: Temporary directory for input files
        
    Returns:
        List of document texts
    """
    # Create input file for document retrieval
    # Format: ground_truth.json with centroid_indices
    input_data = {
        "top_k": len(vector_indices),
        "distances": [0.0] * len(vector_indices),  # Dummy distances
        "centroid_indices": vector_indices
    }
    
    input_file = temp_dir / "doc_retrieval_input.json"
    with open(input_file, 'w') as f:
        json.dump(input_data, f, indent=2)
    
    # Run PIR client for document retrieval
    cmd = [
        str(client_binary),
        "-ip", server_ip,
        "-thread", "1",
        "-input", str(input_file),
        "-extra_input", "/home/jzhao7/RAGPIR/prototype/data/lists.json"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"Warning: PIR document retrieval failed: {result.stderr[:200]}")
            return []
        
        # Extract documents from stderr (PIR client outputs to stderr)
        documents = extract_documents_from_stderr(result.stderr)
        return documents
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


def extract_documents_from_stderr(stderr: str) -> List[str]:
    """
    Extract document texts from PIR client stderr output.
    
    The PIR client outputs documents in stderr with format:
    "Final query result at index X: <document text>"
    """
    documents = []
    lines = stderr.strip().split('\n')
    
    for line in lines:
        if "Final query result at index" in line:
            # Extract text after ": "
            parts = line.split(': ', 1)
            if len(parts) == 2:
                documents.append(parts[1].strip())
    
    return documents


def convert_pir_output_to_eval_format(
    pir_output_path: Path,
    pir_input_path: Path,
    output_path: Path,
    document_server_ip: str = "localhost:50051",
    client_binary: Path = Path("/home/jzhao7/RAGPIR/___go_build_easypir_client_new_linux")
):
    """
    Convert PIR output to format expected by evaluation scripts.
    
    Args:
        pir_output_path: Path to PIR output (vectors/indices)
        pir_input_path: Path to original pir_input.json
        output_path: Output path for converted format
        document_server_ip: PIR document server IP
        client_binary: PIR client binary path
    """
    # Load original queries
    print(f"Loading original queries from {pir_input_path}...")
    with open(pir_input_path, 'r') as f:
        original_queries = {q['query_id']: q for q in json.load(f)}
    print(f"✓ Loaded {len(original_queries)} queries")
    
    # Load PIR output (this might be in various formats)
    print(f"\nLoading PIR output from {pir_output_path}...")
    pir_results = load_pir_output(pir_output_path)
    print(f"✓ Loaded {len(pir_results)} PIR results")
    
    # Convert each result
    converted_results = []
    temp_dir = Path(tempfile.mkdtemp(prefix="pir_doc_retrieval_"))
    
    try:
        for i, result in enumerate(pir_results):
            query_id = result.get('query_id', i)
            
            # Get vector indices from PIR result
            # The format might vary - try different field names
            vector_indices = (
                result.get('top_k_indices') or
                result.get('vector_indices') or
                result.get('indices') or
                result.get('top_k_centroid_indices', [])
            )
            
            if not vector_indices:
                print(f"Warning: No vector indices found for query {query_id}")
                continue
            
            print(f"Query {query_id}: Retrieving {len(vector_indices)} documents...")
            
            # Retrieve documents via PIR
            documents = retrieve_documents_via_pir(
                vector_indices[:100],  # Limit to top 100
                document_server_ip,
                client_binary,
                temp_dir
            )
            
            # Format for evaluation script
            converted_result = {
                "query_id": query_id,
                "query_idx": query_id,
                "top_k_indices": vector_indices,
                "documents": documents,
                "num_documents": len(documents)
            }
            
            # Add original query metadata if available
            if query_id in original_queries:
                converted_result.update({
                    "claim": original_queries[query_id].get("claim"),
                    "label": original_queries[query_id].get("label"),
                    "evidence": original_queries[query_id].get("evidence", [])
                })
            
            converted_results.append(converted_result)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(pir_results)} queries...")
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    # Save converted results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    print(f"\n✓ Converted {len(converted_results)} results")
    print(f"✓ Saved to {output_path}")


def load_pir_output(path: Path) -> List[Dict]:
    """Load PIR output in various formats."""
    path = Path(path)
    
    if path.is_file():
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'results' in data:
                return data['results']
            else:
                return [data]
        else:
            raise ValueError(f"Unexpected format in {path}")
    
    elif path.is_dir():
        # Load all JSON files
        results = []
        for json_file in sorted(path.glob("*.json")):
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                elif isinstance(data, dict):
                    results.append(data)
        return results
    
    else:
        raise FileNotFoundError(f"PIR output not found: {path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert PIR output to evaluation format"
    )
    parser.add_argument(
        "--pir-output",
        type=str,
        required=True,
        help="Path to PIR output (file or directory)"
    )
    parser.add_argument(
        "--pir-input",
        type=str,
        required=True,
        help="Path to original pir_input.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for converted format"
    )
    parser.add_argument(
        "--document-server-ip",
        type=str,
        default="localhost:50051",
        help="PIR document server IP (default: localhost:50051)"
    )
    parser.add_argument(
        "--client-binary",
        type=str,
        default="/home/jzhao7/RAGPIR/___go_build_easypir_client_new_linux",
        help="Path to PIR client binary"
    )
    
    args = parser.parse_args()
    
    convert_pir_output_to_eval_format(
        Path(args.pir_output),
        Path(args.pir_input),
        Path(args.output),
        args.document_server_ip,
        Path(args.client_binary)
    )


if __name__ == "__main__":
    main()




