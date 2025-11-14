#!/bin/bash
# Example workflow for FHE query prototype

set -e  # Exit on error

echo "=========================================="
echo "FHE Query Prototype - Example Workflow"
echo "=========================================="
echo ""

# Configuration
QUERY="What is artificial intelligence?"
CENTROIDS_PATH="./data/centroids.npy"
ENCRYPTED_QUERIES_DIR="./encrypted_queries"
ENCRYPTED_DISTANCES_DIR="./encrypted_distances"
DECRYPTED_RESULTS_DIR="./decrypted_results"
TOP_K=100

# Step 1: Client encrypts query
echo "Step 1: Encrypting query..."
echo "Query: $QUERY"
python fhe_query_client.py \
    --query "$QUERY" \
    --output-dir "$ENCRYPTED_QUERIES_DIR"

echo ""
echo "✓ Query encrypted successfully"
echo ""

# Step 2: Server computes distances
echo "Step 2: Computing encrypted distances..."
if [ ! -f "$CENTROIDS_PATH" ]; then
    echo "Error: Centroids file not found at $CENTROIDS_PATH"
    echo "Please provide a centroids.npy file"
    exit 1
fi

python fhe_query_server.py \
    --centroids-path "$CENTROIDS_PATH" \
    --encrypted-query "$ENCRYPTED_QUERIES_DIR/encrypted_query.bin" \
    --encrypted-norm "$ENCRYPTED_QUERIES_DIR/encrypted_norm_squared.bin" \
    --output-dir "$ENCRYPTED_DISTANCES_DIR"

echo ""
echo "✓ Distances computed successfully"
echo ""

# Step 3: Client decrypts and gets top-k
echo "Step 3: Decrypting distances and selecting top-$TOP_K..."
python fhe_query_client.py \
    --decrypt-distances "$ENCRYPTED_DISTANCES_DIR" \
    --top-k $TOP_K \
    --results-dir "$DECRYPTED_RESULTS_DIR"

echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $DECRYPTED_RESULTS_DIR"
echo "  - top_k_results.json: JSON results"
echo "  - top_k_distances.npy: Distance array"
echo "  - top_k_indices.npy: Centroid indices"
echo ""

