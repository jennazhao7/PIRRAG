# RAG Operations

This directory contains scripts for performing RAG (Retrieval-Augmented Generation) queries.

## Files

### Scripts

- **`direct_rag_query.py`**: Direct FAISS similarity search without IVF clustering
  - Performs a standard similarity search across all vectors
  - No cluster filtering - searches entire index
  - Baseline comparison for IVF-based approaches

- **`plaintext_rag_pipeline.py`**: IVF-based RAG pipeline after FHE query
  - Uses top-100 clusters from FHE query results
  - Per-cluster KNN (top-5 per cluster)
  - Global top-10 selection
  - Maps to documents and builds RAG prompt

### Outputs

- **`direct_rag_output.json`**: Results from direct FAISS search
- **`rag_output.json`**: Results from IVF-based RAG pipeline

## Usage

### Direct RAG Query

```bash
cd prototype/rag_operations
python direct_rag_query.py \
    --query "what is machine learning" \
    --faiss-path ../../wiki-rag/wiki_rag_data/wiki_index__top_100000__2025-04-11 \
    --k 10 \
    --output ./direct_rag_output.json
```

### Plaintext RAG Pipeline

```bash
cd prototype/rag_operations
python plaintext_rag_pipeline.py \
    --query "What is machine learning?" \
    --top-results ../workflow_decrypted_results/top_k_results.json \
    --lists ../data/lists.json \
    --faiss-path ../../wiki-rag/wiki_rag_data/wiki_index__top_100000__2025-04-11 \
    --per-cluster-k 5 \
    --global-k 10 \
    --output ./rag_output.json
```

**Key Parameters:**
- `--per-cluster-k 5`: Get top-5 vectors within each of the top-100 clusters (default: 5)
- `--global-k 10`: Select top-10 from all ~500 candidates (default: 10)
- `--top-clusters 100`: Number of top clusters to use (default: 100)

## Comparison

- **Direct RAG**: Searches all 65,000 vectors directly (slower, exact)
- **IVF-based RAG**: Uses top-100 clusters → ~500 candidates → top-10 (faster, approximate)

Both methods typically return similar top results, with the IVF approach filtering through clusters first.

