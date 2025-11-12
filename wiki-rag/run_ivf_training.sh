#!/bin/bash
# Script to run IVF training on wiki-rag

cd /home/jzhao7/RAGPIR/wiki-rag

echo "=========================================="
echo "Training IVF index on wiki-rag embeddings"
echo "=========================================="
echo ""

# Basic IVF-flat training with K=4096 (CPU only)
python3 train_ivf.py \
    --faiss-path wiki_rag_data/wiki_index__top_100000__2025-04-11 \
    --k 4096 \
    --output-dir ./ivf_output \
    --test-recall \
    --n-test-queries 1000

echo ""
echo "=========================================="
echo "Training complete!"
echo "Output files:"
echo "  - centroids.npy: Cluster centroids"
echo "  - lists.json: Inverted lists"
echo "=========================================="

