# Data Directory

This directory contains the IVF index data needed for the FHE query server.

## Files

- **centroids.npy**: Cluster centroids from IVF training
  - Shape: (4096, 768) - 4096 clusters, 768-dimensional embeddings
  - Format: NumPy float32 array
  
- **lists.json**: Inverted lists mapping cluster IDs to vector indices
  - Format: JSON object mapping cluster_id (string) -> list of vector indices
  - Used to identify which vectors belong to each cluster

## Usage

These files are automatically used by `fhe_query_server.py` when you specify:
```bash
--centroids-path ./data/centroids.npy
```

The server will:
1. Load centroids for distance computation
2. Pre-compute squared norms of centroids
3. Use lists.json for cluster-to-vector mapping (if needed)

## Generating These Files

If you need to regenerate these files, use the IVF training script:
```bash
python train_ivf.py \
    --faiss-path <path_to_faiss_index> \
    --k 4096 \
    --output-dir <output_directory>
```

This will create `centroids.npy` and `lists.json` in the output directory.

