#!/bin/bash
# Install FAISS GPU support

echo "Installing faiss-gpu..."
echo "Note: This will replace faiss-cpu"

# Option 1: Using conda (recommended if using conda)
# conda install -c pytorch faiss-gpu -y

# Option 2: Using pip (if conda doesn't work)
pip uninstall faiss-cpu -y
pip install faiss-gpu

echo ""
echo "Verifying installation..."
python3 -c "import faiss; print('FAISS version:', faiss.__version__); res = faiss.StandardGpuResources(); print('✓ GPU support available!')"

