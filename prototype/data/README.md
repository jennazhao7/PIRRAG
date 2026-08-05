# Data Directory

This directory contains the IVF index data needed for the FHE query server.

## Two clustering layouts

The clustering can be parameterized two ways, and consumers must branch on
`ivf_metadata.json`'s `sizing_mode` rather than guessing from list lengths:

- **`fixed-nlist`** (legacy, what is checked in here): the cluster count is fixed
  and sizes vary. The shipped `lists.json` has 4096 clusters over 65,000 vectors
  with sizes from 1 to 103 (mean 15.87).
- **`constant-size`**: the cluster size `n` is fixed and the count is derived as
  `ceil(N / n)`. Every list holds exactly `n` slots, short-filled with `-1`. This
  gives the FHE per-cluster kNN circuit a static shape and makes the candidate
  count per query exactly `nprobe * n` instead of data-dependent.

## Files

- **centroids.npy**: Cluster centroids from IVF training
  - Shape: (nlist, 768). The checked-in artifact is (4096, 768); under
    constant-size mode nlist is `ceil(N / n)`, so it varies with `n`.
  - Format: NumPy float32 array
  - For `ivf-opq` these centroids are in the **OPQ-rotated** space; see
    `centroid_space` in the metadata.

- **lists.json**: Inverted lists mapping cluster IDs to vector indices
  - Format: JSON object mapping cluster_id (string) -> list of vector indices
  - Under constant-size mode every list has length exactly `n`, padded with `-1`.
    Strip the sentinel before use, or load via `wiki-rag/ivf_io.load_clustering`,
    which handles both layouts.

- **cluster_slots.npy** (constant-size only): the canonical `(nlist, n)` int32
  slot array. `lists.json` is a compatibility view derived from it.

- **ivf_metadata.json**: sizing mode, nlist, cluster size, padding sentinel,
  k-means parameters, recall mode, and balance diagnostics. Written by every run,
  including legacy ones.

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

If you need to regenerate these files, use the IVF training script.

Legacy layout (fixed cluster count, variable sizes) — reproduces what is checked
in here:
```bash
python train_ivf.py \
    --faiss-path <path_to_faiss_index> \
    --k 4096 \
    --output-dir <output_directory>
```

Constant cluster size (every list exactly `n` wide):
```bash
python train_ivf.py \
    --faiss-index-file <path_to_faiss_index>/index.faiss \
    --cluster-size 64 \
    --output-dir <output_directory>
```

`--faiss-index-file` reads the vectors straight off `index.faiss`, so no
embedding model or langchain is needed. Both forms write `centroids.npy`,
`lists.json` and `ivf_metadata.json`; constant-size additionally writes
`cluster_slots.npy` and `cluster_balance.txt`.

To find which cluster size matches the baseline's recall at the lowest candidate
budget, see `wiki-rag/sweep_cluster_size.py`.

