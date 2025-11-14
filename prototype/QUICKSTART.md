# Quick Start Guide

Get the FHE query prototype running in 3 steps!

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

## Step-by-Step

### 1. Encrypt a Query (Client)

```bash
python fhe_query_client.py \
    --query "What is machine learning?" \
    --output-dir ./encrypted_queries
```

**Output**: `encrypted_queries/` directory with encrypted query files

### 2. Compute Distances (Server)

```bash
python fhe_query_server.py \
    --centroids-path ./data/centroids.npy \
    --encrypted-query ./encrypted_queries/encrypted_query.bin \
    --encrypted-norm ./encrypted_queries/encrypted_norm_squared.bin \
    --output-dir ./encrypted_distances
```

**Output**: `encrypted_distances/` directory with encrypted distance files

### 3. Decrypt Results (Client)

```bash
python fhe_query_client.py \
    --decrypt-distances ./encrypted_distances \
    --top-k 100 \
    --results-dir ./decrypted_results
```

**Output**: `decrypted_results/` directory with top-100 results

## One-Command Workflow

```bash
./example_workflow.sh
```

This runs all three steps automatically!

## What You Get

After running all steps, you'll have:
- **Top-100 closest centroids** to your query
- **Distances** (smaller = closer match)
- **Centroid indices** (which clusters match best)

## Troubleshooting

**Missing centroids.npy?**
- Already included in `./data/centroids.npy`
- If missing, check `data/README.md`

**TenSEAL errors?**
- Ensure TenSEAL is installed: `pip install tenseal`
- Check Python version: 3.8+ required

## Next Steps

Use the top-k centroid indices to:
1. Retrieve corresponding document lists from `data/lists.json`
2. Perform further search within those clusters
3. Return relevant documents to users

