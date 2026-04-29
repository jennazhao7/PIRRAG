# OpenFHE Single-Query FHE Package

This directory contains the OpenFHE C++ implementation used by the Python
single-query wrappers in `prototype/`.

## What is implemented

- OpenFHE C++ executables:
  - `openfhe_keygen`
  - `openfhe_encrypt_query`
  - `openfhe_encrypt_query_centroid_batched`
  - `openfhe_encrypt_queries_centroid_batched`
  - `openfhe_encrypt_queries_batched`
  - `openfhe_compute_distances`
  - `openfhe_compute_distances_centroid_batched`
  - `openfhe_compute_distances_query_centroid_batched`
  - `openfhe_compute_distances_batched`
  - `openfhe_decrypt_topk`
  - `openfhe_decrypt_topk_centroid_batched`
  - `openfhe_decrypt_topk_query_centroid_batched`
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
- `openfhe_core/build/bin/openfhe_encrypt_query_centroid_batched`
- `openfhe_core/build/bin/openfhe_encrypt_queries_centroid_batched`
- `openfhe_core/build/bin/openfhe_compute_distances`
- `openfhe_core/build/bin/openfhe_compute_distances_centroid_batched`
- `openfhe_core/build/bin/openfhe_compute_distances_query_centroid_batched`
- `openfhe_core/build/bin/openfhe_decrypt_topk`
- `openfhe_core/build/bin/openfhe_decrypt_topk_centroid_batched`
- `openfhe_core/build/bin/openfhe_decrypt_topk_query_centroid_batched`

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

## Batched OpenFHE parameters

The batched OpenFHE binaries use the same slot layout contract as the batching design:

- `slots_per_ciphertext = poly_modulus_degree / 2`
- `queries_per_ciphertext = floor(slots_per_ciphertext / query_dim)`

The centroid-batched and query+centroid-batched paths use `padded_dim=1024`
for 768-dimensional embeddings and pack lanes as:

- centroid-batched single-query: `1 query * centroids_per_ciphertext`
- query+centroid-batched: `queries_per_batch * centroids_per_batch`

The packed lane count must fit in the CKKS slot budget:

```text
packed_lanes <= slots_per_ciphertext / padded_dim
```

Example with `query_dim=768`:

- `poly_modulus_degree=8192` -> `4096` slots -> `5` queries per ciphertext
- `poly_modulus_degree=16384` -> `8192` slots -> `10` queries per ciphertext
- `poly_modulus_degree=32768` -> `16384` slots -> `21` queries per ciphertext

Example with `padded_dim=1024`:

- `poly_modulus_degree=16384` -> `8192` slots -> `8` packed lanes
- good single-query centroid layout: `1 query * 8 centroids`
- good many-query layout: `2 queries * 4 centroids`

## Which batched path to use

Use **centroid batching** when serving one encrypted query at a time against many
plaintext centroids. This is the main online retrieval bottleneck and the best
verified no-accuracy-drop optimization so far. It keeps the query count at `1`
and spends the available slots on more centroid lanes.

Use **query+centroid batching** when several encrypted queries are already
available together and throughput matters more than the latency of one query.
It shares the same slot budget between query lanes and centroid lanes, so it
reduces repeated per-query overhead but does not multiply capacity for free.
At `poly_modulus_degree=16384` and `padded_dim=1024`, the tested layout is
`queries_per_batch=2` and `centroids_per_batch=4`.

The older query-only batched binaries (`openfhe_encrypt_queries_batched` and
`openfhe_compute_distances_batched`) are useful for parameter experiments, but
they are not the recommended optimized retrieval path yet because they still
write per-query/per-centroid distance ciphertexts.

## Optimized parameter sweep (batching >5)

Use the sweep harness to benchmark degree + threading + OpenMP schedule batch size:

```bash
python openfhe_core/sweep_batched_params.py \
  --centroids-file prototype/real_openfhe_tuned/encrypted_distances_bs64b/centroids.txt \
  --query-dim 768 \
  --num-queries 30 \
  --poly-degrees 8192,16384,32768 \
  --thread-options 0,8,16,20 \
  --batch-size-options 32,64,128
```

Outputs:

- `openfhe_core/sweep_runs/batched_param_sweep_summary.json`
- `openfhe_core/sweep_runs/batched_param_sweep_summary.md`

## Centroid-batched single-query acceleration

For the common bottleneck case (one encrypted query against 4096 centroids), use centroid batching:

```bash
python openfhe_core/benchmark_centroid_batch.py \
  --bin-dir openfhe_core/build/bin \
  --centroids-file prototype/real_openfhe_tuned/encrypted_distances_bs64b/centroids.txt \
  --work-dir openfhe_core/centroid_batch_bench_4096 \
  --poly-modulus-degree 16384 \
  --query-dim 768 \
  --padded-dim 1024 \
  --centroids-per-ciphertext 8 \
  --max-centroids 4096 \
  --num-threads 20 \
  --batch-size 1 \
  --top-k 100
```

Measured on 4096 centroids:

- baseline OpenFHE compute: `419.709s`
- centroid-batched OpenFHE compute: `49.511s`
- speedup: `8.477x`
- top-100 overlap: `100/100`

See `openfhe_core/CENTROID_BATCH_SPEEDUP_REPORT.md`.

Recommended settings:

```text
poly_modulus_degree=16384
padded_dim=1024
centroids_per_ciphertext=8
num_threads=20
batch_size=1
```

Underlying binaries:

- `openfhe_encrypt_query_centroid_batched`
- `openfhe_compute_distances_centroid_batched`
- `openfhe_decrypt_topk_centroid_batched`

## Query+centroid-batched many-query acceleration

For workloads with multiple encrypted queries available at once, use
query+centroid batching:

```bash
python openfhe_core/benchmark_query_centroid_batch.py \
  --bin-dir openfhe_core/build/bin \
  --centroids-file prototype/real_openfhe_tuned/encrypted_distances_bs64b/centroids.txt \
  --work-dir openfhe_core/query_centroid_batch_bench_q4_c512_16384 \
  --poly-modulus-degree 16384 \
  --query-dim 768 \
  --padded-dim 1024 \
  --num-queries 4 \
  --max-centroids 512 \
  --queries-per-batch 2 \
  --centroids-per-batch 4 \
  --single-centroids-per-batch 8 \
  --num-threads 20 \
  --batch-size 1 \
  --top-k 100
```

Measured against running the centroid-batched single-query path once per query:

- 4 queries x 512 centroids: `30.590s` -> `25.293s`, speedup `1.209x`, top-100 overlap `100/100` for every query
- 4 queries x 128 centroids: `10.309s` -> `6.305s`, speedup `1.635x`, top-50 overlap `50/50` for every query

Recommended settings for the best tested exact layout:

```text
poly_modulus_degree=16384
padded_dim=1024
queries_per_batch=2
centroids_per_batch=4
num_threads=20
batch_size=1
```

Underlying binaries:

- `openfhe_encrypt_queries_centroid_batched`
- `openfhe_compute_distances_query_centroid_batched`
- `openfhe_decrypt_topk_query_centroid_batched`

See `openfhe_core/QUERY_CENTROID_BATCH_REPORT.md`.

