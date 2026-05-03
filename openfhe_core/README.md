# OpenFHE Workloads

This directory contains the OpenFHE C++ binaries used for encrypted query-to-centroid distance computation. Build once, then reuse the same binaries for every query. Regenerate keys only when OpenFHE parameters or batching layout change.

## What Is Included

Core files:

- `CMakeLists.txt`
- `include/io_utils.h`
- `src/io_utils.cpp`
- `src/openfhe_keygen.cpp`
- `src/openfhe_encrypt_query.cpp`
- `src/openfhe_compute_distances.cpp`
- `src/openfhe_decrypt_topk.cpp`

Batched workload files:

- `src/openfhe_batched_workload.cpp`
- `src/openfhe_encrypt_query_centroid_batched.cpp`
- `src/openfhe_compute_distances_centroid_batched.cpp`
- `src/openfhe_decrypt_topk_centroid_batched.cpp`
- `src/openfhe_encrypt_queries_centroid_batched.cpp`
- `src/openfhe_compute_distances_query_centroid_batched.cpp`
- `src/openfhe_decrypt_topk_query_centroid_batched.cpp`
- `src/openfhe_encrypt_queries_batched.cpp`
- `src/openfhe_compute_distances_batched.cpp`
- `src/openfhe_compute_distances_serial.cpp`

Benchmark helpers:

- `benchmark_centroid_batch.py`
- `benchmark_query_centroid_batch.py`
- `sweep_batched_params.py`

Python integration for the existing single-query workflow lives in:

- `prototype/fhe_backend.py`
- `prototype/fhe_query_client.py`
- `prototype/fhe_query_server.py`

## Build

If OpenFHE is installed at `/usr/local/lib/OpenFHE`:

```bash
cmake -S openfhe_core -B openfhe_core/build -DOpenFHE_DIR=/usr/local/lib/OpenFHE
cmake --build openfhe_core/build -j
```

Expected binaries include:

- `openfhe_core/build/bin/openfhe_keygen`
- `openfhe_core/build/bin/openfhe_encrypt_query`
- `openfhe_core/build/bin/openfhe_compute_distances`
- `openfhe_core/build/bin/openfhe_decrypt_topk`
- `openfhe_core/build/bin/openfhe_encrypt_query_centroid_batched`
- `openfhe_core/build/bin/openfhe_compute_distances_centroid_batched`
- `openfhe_core/build/bin/openfhe_decrypt_topk_centroid_batched`
- `openfhe_core/build/bin/openfhe_encrypt_queries_centroid_batched`
- `openfhe_core/build/bin/openfhe_compute_distances_query_centroid_batched`
- `openfhe_core/build/bin/openfhe_decrypt_topk_query_centroid_batched`

## One-File Batched Wrapper

Use `src/openfhe_batched_workload.cpp` when handing the batched FHE workflow to a collaborator. It is a self-contained C++ entry point for key generation, encryption, distance computation, and top-k decryption for the two recommended batched paths:

- Centroid batching: one encrypted query against many plaintext centroids.
- Query+centroid batching: multiple encrypted queries processed together.

This source file includes its own argument parsing, OpenFHE context/key serialization, ciphertext IO, metadata writing, and top-k JSON output. A collaborator can copy this one `.cpp` file into another project and link it against OpenFHE and OpenMP; it does not require `include/io_utils.h`, `src/io_utils.cpp`, or the split batched binary sources.

Compile this file into an executable named `openfhe_batched_workload` using the same OpenFHE include paths and libraries as the rest of the project. The source is intentionally standalone so it can be dropped into the collaborator's PIR build system without also copying the OpenFHE helper library.

The binary keeps the same file formats as the split binaries, so existing benchmark scripts can still read the outputs. Input vectors are whitespace-delimited text files. For matrix input, each row is one vector.

Sample input files are in `openfhe_core/sample_inputs/`:

- `centroids_4d.txt`: four 4-dimensional centroid rows.
- `query_4d.txt`: one 4-dimensional query vector for `run-centroid`.
- `queries_4d.txt`: two 4-dimensional query rows for `run-query-centroid`.

The text format is intentionally simple:

```text
1 0 0 0
0 1 0 0
0 0 1 0
1 1 0 0
```

To convert a NumPy centroid matrix to this format:

```bash
python - <<'PY'
import numpy as np

centroids = np.load("/path/to/centroids.npy")
np.savetxt("/path/to/centroids.txt", centroids, fmt="%.9g")
PY
```

The real centroid file for the current 4096-centroid workload is:

- `openfhe_core/centroids.txt`

It is already converted to the wrapper format: `4096` rows, `768` whitespace-delimited finite floats per row.

Centroid-batched end-to-end run:

```bash
openfhe_core/build/bin/openfhe_batched_workload run-centroid \
  --context-dir /path/to/context_dir \
  --input-vector /path/to/query.txt \
  --centroids-file /path/to/centroids.txt \
  --work-dir /path/to/work_dir \
  --output-json /path/to/top_k_results.json \
  --poly-modulus-degree 16384 \
  --padded-dim 1024 \
  --centroids-per-ciphertext 8 \
  --num-threads 20 \
  --batch-size 1 \
  --top-k 100
```

Query+centroid-batched end-to-end run:

```bash
openfhe_core/build/bin/openfhe_batched_workload run-query-centroid \
  --context-dir /path/to/context_dir \
  --input-matrix /path/to/queries.txt \
  --centroids-file /path/to/centroids.txt \
  --work-dir /path/to/work_dir \
  --output-json /path/to/top_k_results.json \
  --poly-modulus-degree 16384 \
  --padded-dim 1024 \
  --queries-per-batch 2 \
  --centroids-per-batch 4 \
  --num-threads 20 \
  --batch-size 1 \
  --top-k 100
```

The same binary also exposes step-by-step subcommands for wrappers that want to insert PIR logic between stages:

- `keygen-centroid`, `encrypt-centroid`, `compute-centroid`, `decrypt-centroid`
- `keygen-query-centroid`, `encrypt-query-centroid`, `compute-query-centroid`, `decrypt-query-centroid`

Run `openfhe_core/build/bin/openfhe_batched_workload help` for the complete command list.

Smoke test used for the one-file wrapper:

```bash
openfhe_core/build/bin/openfhe_batched_workload run-centroid \
  --context-dir openfhe_core/one_file_wrapper_smoke/context_centroid \
  --input-vector openfhe_core/one_file_wrapper_smoke/query.txt \
  --centroids-file openfhe_core/one_file_wrapper_smoke/centroids.txt \
  --work-dir openfhe_core/one_file_wrapper_smoke/centroid_run \
  --output-json openfhe_core/one_file_wrapper_smoke/centroid_topk.json \
  --poly-modulus-degree 16384 \
  --padded-dim 8 \
  --centroids-per-ciphertext 2 \
  --num-threads 2 \
  --batch-size 1 \
  --top-k 2 \
  --security-level none
```

The same test also passed for `run-query-centroid` with two query rows and four centroid rows. The expected nearest centroids were recovered for both queries.

## Which Path To Use

Use the standard single-query path for baseline compatibility through the Python wrappers. It produces one encrypted distance ciphertext per centroid.

Use centroid batching for the main online bottleneck: one encrypted query against many plaintext centroids. This is the best verified low-latency path. It packs multiple centroids into each ciphertext and keeps top-k results exact in the measured runs.

Use query+centroid batching when several encrypted queries are available together and throughput matters more than one-query latency. The slot budget is shared between query lanes and centroid lanes, so the gain is smaller than centroid batching but useful for grouped workloads.

The query-only batched binaries are available for parameter sweeps and compatibility experiments, but they are not the recommended optimized retrieval path because they still emit per-query/per-centroid distance ciphertexts.

## Parameters

For 768-dimensional embeddings, use `padded_dim=1024` in the centroid-batched and query+centroid-batched paths.

CKKS slot budget:

```text
slots_per_ciphertext = poly_modulus_degree / 2
packed_lanes <= slots_per_ciphertext / padded_dim
```

Recommended exact layouts:

- Single-query centroid batching: `poly_modulus_degree=16384`, `padded_dim=1024`, `centroids_per_ciphertext=8`
- Query+centroid batching: `poly_modulus_degree=16384`, `padded_dim=1024`, `queries_per_batch=2`, `centroids_per_batch=4`
- Threading: `num_threads=20`, `batch_size=1` for the batched C++ kernels

Generate keys with the rotation indices required by the chosen layout. Reuse the same context and keys for all queries that use the same parameters and layout.

## Standard Single-Query Workflow

Encrypt:

```bash
python prototype/fhe_query_client.py \
  --backend openfhe_cpp \
  --query-vector /path/to/query.npy \
  --context-path /path/to/context_dir \
  --output-dir /path/to/encrypted_queries
```

Compute:

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

Decrypt:

```bash
python prototype/fhe_query_client.py \
  --backend openfhe_cpp \
  --decrypt-distances /path/to/encrypted_distances \
  --context-path /path/to/context_dir \
  --results-dir /path/to/decrypted_results \
  --top-k 10
```

## Centroid-Batched Benchmark

Use this for one encrypted query against many centroids:

```bash
python openfhe_core/benchmark_centroid_batch.py \
  --bin-dir openfhe_core/build/bin \
  --centroids-file openfhe_core/centroids.txt \
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

Verified result on 4096 centroids:

- Baseline OpenFHE compute: `419.709s`
- Centroid-batched compute: `49.511s`
- Speedup: `8.477x`
- Top-100 overlap: `100/100`

## Query+Centroid-Batched Benchmark

Use this when multiple encrypted queries are processed together:

```bash
python openfhe_core/benchmark_query_centroid_batch.py \
  --bin-dir openfhe_core/build/bin \
  --centroids-file openfhe_core/centroids.txt \
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

Verified throughput results against running centroid batching once per query:

- 4 queries x 512 centroids: `30.590s` to `25.293s`, speedup `1.209x`, top-100 overlap `100/100` for every query
- 4 queries x 128 centroids: `10.309s` to `6.305s`, speedup `1.635x`, top-50 overlap `50/50` for every query

## Output Files

Context files:

- `context.bin`
- `public_key.bin`
- `secret_key.bin`
- `eval_mult_keys.bin`
- `eval_sum_keys.bin`
- `eval_automorphism_keys.bin` when rotation keys are generated

Standard query files:

- `encrypted_query.bin`
- `encrypted_norm_squared.bin`
- `query_metadata.json`

Batched query files use path-specific metadata such as:

- `centroid_batch_query_metadata.json`
- `query_centroid_batch_metadata.json`

Distance output directories contain encrypted distance ciphertexts plus metadata. Decryption writes top-k JSON and, through the Python wrapper, NumPy result files.

## Collaborator Bundle

Create a source bundle from the repo root:

```bash
bash openfhe_core/create_openfhe_bundle.sh
```

This creates:

- `openfhe_core/dist/openfhe_single_query_bundle.tar.gz`

Minimal source files for collaborator execution are listed in this README under `What Is Included`. Runtime data such as `centroids.npy`, query vectors, and embedding dependencies are not bundled.

