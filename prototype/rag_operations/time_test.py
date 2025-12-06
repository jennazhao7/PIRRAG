import time
import numpy as np
import pandas as pd
import json
import sys
import os
import torch
from datetime import datetime
from typing import Dict, Any, List, Tuple
import argparse
from pathlib import Path

def run_pir_rag_experiment(self, embeddings: np.ndarray, documents: List[str],
                           queries: List[np.ndarray], k_clusters: int = 5,
                           cluster_top_k: int = 3, top_k: int = 10,
                           dataset_name: str = None) -> Dict[str, Any]:
    """Run PIR-RAG experiment with detailed timing."""
    dataset_info = f" on {dataset_name}" if dataset_name else ""
    print(f"Running PIR-RAG experiment{dataset_info} (k_clusters={k_clusters}, cluster_top_k={cluster_top_k})")

    # Setup phase
    setup_start = time.perf_counter()
    client = PIRRAGClient()
    server = PIRRAGServer()

    # Setup timing breakdown
    cluster_start = time.perf_counter()
    # Server does clustering and setup first
    server_setup_result = server.setup(embeddings, documents, k_clusters)
    clustering_time = time.perf_counter() - cluster_start

    server_setup_start = time.perf_counter()
    # Client setup with centroids from server
    client_setup_result = client.setup(server.centroids)
    server_setup_time = time.perf_counter() - server_setup_start

    total_setup_time = time.perf_counter() - setup_start

    # Query phase - detailed timing for each step
    query_times = []
    step_times = []
    communication_costs = []

    for i, query_embedding in enumerate(queries):
        print(f"  Query {i + 1}/{len(queries)}")

        query_start = time.perf_counter()

        # Step 1: Find relevant clusters
        cluster_start = time.perf_counter()
        query_tensor = torch.tensor(query_embedding) if not isinstance(query_embedding,
                                                                       torch.Tensor) else query_embedding
        relevant_clusters = client.find_relevant_clusters(query_tensor, top_k=cluster_top_k)
        cluster_time = time.perf_counter() - cluster_start

        # Step 2: PIR retrieval (now returns URLs and embeddings together)
        pir_start = time.perf_counter()
        doc_tuples, pir_metrics = client.pir_retrieve(relevant_clusters, server)
        pir_time = time.perf_counter() - pir_start

        # Step 3: Reranking (using embeddings from PIR, no server request)
        rerank_start = time.perf_counter()
        final_results = client.rerank_documents(query_tensor, doc_tuples, top_k=top_k)
        rerank_time = time.perf_counter() - rerank_start

        total_query_time = time.perf_counter() - query_start

        query_times.append(total_query_time)
        step_times.append({
            'cluster_selection_time': cluster_time,
            'pir_retrieval_time': pir_time,
            'reranking_time': rerank_time,
            'query_gen_time': pir_metrics.get('total_query_gen_time', 0),
            'server_time': pir_metrics.get('total_server_time', 0),
            'decode_time': pir_metrics.get('total_decode_time', 0)
        })
        communication_costs.append({
            'upload_bytes': pir_metrics.get('total_upload_bytes', 0),
            'download_bytes': pir_metrics.get('total_download_bytes', 0)
        })

    return {
        'system': 'PIR-RAG',
        'dataset_name': dataset_name,
        'setup_time': total_setup_time,
        'clustering_time': clustering_time,
        'server_setup_time': server_setup_time,
        'avg_query_time': np.mean(query_times),
        'std_query_time': np.std(query_times),
        'query_times': query_times,
        'step_times': step_times,
        'avg_upload_bytes': np.mean([c['upload_bytes'] for c in communication_costs]),
        'avg_download_bytes': np.mean([c['download_bytes'] for c in communication_costs]),
        'communication_costs': communication_costs,
        'parameters': {'k_clusters': k_clusters, 'cluster_top_k': cluster_top_k, 'top_k': top_k},
        'n_documents': len(documents),
        'embedding_dim': embeddings.shape[1]
    }
