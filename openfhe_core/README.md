# OpenFHE Single-Query FHE Package

This directory contains the OpenFHE C++ implementation used by the Python
single-query wrappers in `prototype/`.

## What is implemented

- OpenFHE C++ executables:
  - `openfhe_keygen`
  - `openfhe_encrypt_query`
  - `openfhe_compute_distances`
  - `openfhe_decrypt_topk`
- Python backend integration:
  - `prototype/fhe_backend.py` (`openfhe_cpp` backend)
  - `prototype/fhe_query_client.py`
  - `prototype/fhe_query_server.py`
- Parallel tuning knobs for server distance compute:
  - `--num-threads`
  - `--batch-size`

## Minimal files collaborators need

Use the bundle manifest in `openfhe_core/BUNDLE_MANIFEST.md`.

Core required files:

- `openfhe_core/CMakeLists.txt`
- `openfhe_core/include/io_utils.h`
- `openfhe_core/src/io_utils.cpp`
- `openfhe_core/src/openfhe_keygen.cpp`
- `openfhe_core/src/openfhe_encrypt_query.cpp`
- `openfhe_core/src/openfhe_compute_distances.cpp`
- `openfhe_core/src/openfhe_decrypt_topk.cpp`
- `prototype/fhe_backend.py`
- `prototype/fhe_query_client.py`
- `prototype/fhe_query_server.py`

Optional (only needed for text-query mode):

- `prototype/rag_utils.py` and its model dependencies

## Create a collaborator bundle

From repo root:

```bash
bash openfhe_core/create_openfhe_bundle.sh
```

This creates:

- `openfhe_core/dist/openfhe_single_query_bundle.tar.gz`

## Build

If OpenFHE is installed at `/usr/local/lib/OpenFHE`:

```bash
cmake -S openfhe_core -B openfhe_core/build -DOpenFHE_DIR=/usr/local/lib/OpenFHE
cmake --build openfhe_core/build -j
```

Expected binaries:

- `openfhe_core/build/bin/openfhe_keygen`
- `openfhe_core/build/bin/openfhe_encrypt_query`
- `openfhe_core/build/bin/openfhe_compute_distances`
- `openfhe_core/build/bin/openfhe_decrypt_topk`

## Run (vector input, recommended)

1) Encrypt:

```bash
python prototype/fhe_query_client.py \
  --backend openfhe_cpp \
  --query-vector /path/to/query.npy \
  --context-path /path/to/context_dir \
  --output-dir /path/to/encrypted_queries
```

2) Server compute:

```bash
python prototype/fhe_query_server.py \
  --backend openfhe_cpp \
  --context-path /path/to/context_dir \
  --centroids-path /path/to/centroids.npy \
  --encrypted-query /path/to/encrypted_queries/encrypted_query.bin \
  --encrypted-norm /path/to/encrypted_queries/encrypted_norm_squared.bin \
  --output-dir /path/to/encrypted_distances \
  --num-threads 20 \
  --batch-size 128
```

3) Decrypt top-k:

```bash
python prototype/fhe_query_client.py \
  --backend openfhe_cpp \
  --decrypt-distances /path/to/encrypted_distances \
  --context-path /path/to/context_dir \
  --results-dir /path/to/decrypted_results \
  --top-k 10
```

## Output artifacts

Context files:

- `context.bin`
- `public_key.bin`
- `secret_key.bin`
- `eval_mult_keys.bin`
- `eval_sum_keys.bin`

Query files:

- `encrypted_query.bin`
- `encrypted_norm_squared.bin`
- `query_metadata.json`

Distance files:

- `encrypted_distance_XXXX.bin`
- `distances_metadata.json`

Result files:

- `top_k_results.json`
- `top_k_distances.npy`
- `top_k_indices.npy`

## Current benchmark result (real centroids)

Environment:

- 4096 centroids, dim 768
- OpenFHE backend, ring dimension 16384
- 20 CPU threads

Observed server distance compute time:

- tuned: `--num-threads 20 --batch-size 128` -> about **216.89s** (~3m 37s)

This is a major improvement from the earlier untuned run.

