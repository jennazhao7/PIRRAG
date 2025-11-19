# Ground Truth Computation

## Overview

This script computes ground truth: for each of the top-100 clusters, it finds the top-5 closest vectors within that cluster.

## Requirements

**Important**: This script requires access to the **full** FAISS index containing all 65,000 document vectors.

The FAISS index should be in a directory containing:
- `index.faiss` - FAISS index file (must contain ~65,000 vectors to match lists.json)
- `index.pkl` - Pickle file with document metadata

**Note**: The `lists.json` file references vector indices 0-64999, so the FAISS index must contain at least 65,000 vectors. A small test index (e.g., 11 vectors) will not work.

## Usage

```bash
python compute_ground_truth.py \
    --query "What is machine learning?" \
    --top-results ./workflow_decrypted_results/top_k_results.json \
    --faiss-path /path/to/faiss/index/directory \
    --output-dir ./ground_truth
```

## Example

If your FAISS index is located at `/data/wiki_index__top_100000__2025-04-11/`:

```bash
python compute_ground_truth.py \
    --query "What is machine learning?" \
    --top-results ./workflow_decrypted_results/top_k_results.json \
    --faiss-path /data/wiki_index__top_100000__2025-04-11 \
    --output-dir ./ground_truth
```

## Output

The script creates:
- `ground_truth/ground_truth.json` - Complete ground truth data
- `ground_truth/ground_truth_summary.txt` - Human-readable summary

## Output Format

`ground_truth.json` contains:
```json
{
  "query": "What is machine learning?",
  "n_clusters": 100,
  "top_k_per_cluster": 5,
  "ground_truth": {
    "1537": [
      {"vector_index": 12345, "distance": 0.1234},
      {"vector_index": 12346, "distance": 0.1456},
      ...
    ],
    ...
  }
}
```

## Note

If you don't have access to the FAISS index, you cannot compute true ground truth. The script requires the actual document vectors to compute accurate distances within each cluster.

