# FAISS GPU Installation Guide

## Current Situation
- **Python**: 3.13 (not supported by faiss-gpu)
- **CUDA**: 12.8 available
- **GPU**: Quadro RTX 5000 (16GB)
- **Current FAISS**: CPU-only version

## Options

### Option 1: Use CPU Version (Recommended for now)
The CPU version works fine, just slower. For 100k vectors:
- Training time: ~5-15 minutes (still reasonable)
- All functionality works correctly

**Just run:**
```bash
python3 train_ivf.py \
    --faiss-path wiki_rag_data/wiki_index__top_100000__2025-04-11 \
    --k 4096 \
    --output-dir ./ivf_output \
    --test-recall
```

### Option 2: Create Conda Environment with Python 3.12 (For GPU)
If you want GPU acceleration, create a separate environment:

```bash
# Create new environment with Python 3.12
conda create -n faiss-gpu python=3.12 -y
conda activate faiss-gpu

# Install faiss-gpu
conda install -c pytorch faiss-gpu -y

# Install other dependencies
pip install langchain-community numpy

# Run training
python train_ivf.py --faiss-path ... --use-gpu
```

### Option 3: Build FAISS from Source (Advanced)
Build FAISS with GPU support for Python 3.13:
- More complex, requires CUDA toolkit
- See: https://github.com/facebookresearch/faiss/wiki/Installing-Faiss

## Recommendation
**Start with CPU version** - it's ready to use now and will work fine. 
You can always set up GPU later if needed.

