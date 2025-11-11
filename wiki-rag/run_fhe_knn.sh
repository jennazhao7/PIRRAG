#!/bin/bash
# Helper script to run FHE KNN examples with the concreteml environment

# Activate concreteml environment
source /home/jzhao7/miniconda3/etc/profile.d/conda.sh
conda activate concreteml

# Change to wiki-rag directory
cd "$(dirname "$0")"

# Run the example
echo "Running FHE KNN example with concreteml environment..."
echo ""

if [ "$1" = "example" ]; then
    python example_fhe_knn.py
elif [ "$1" = "server" ]; then
    export USE_FHE_KNN=true
    python wiki_rag/rag_server_api.py
else
    echo "Usage: $0 [example|server]"
    echo ""
    echo "  example - Run the FHE KNN example script"
    echo "  server  - Run the RAG server with FHE KNN enabled"
    exit 1
fi


