# OpenFHE Batch-Ready Design (Phase 2)

This design extends the single-query OpenFHE pipeline to packed multi-query CKKS while preserving the current file contract used by `prototype/fhe_query_client_batched.py` and `prototype/fhe_query_server_batched.py`.

## Goals

- Keep Python orchestration unchanged where possible.
- Move packed ciphertext math into C++ OpenFHE binaries.
- Keep output files deterministic and versioned.
- Preserve compatibility with existing downstream decryption/top-k scripts.

## Slot Layout Contract

For each ciphertext:

- `slots_per_ciphertext = ring_dim / 2`
- `query_dim = embedding_dim_after_optional_reduction`
- `queries_per_ciphertext = floor(slots_per_ciphertext / query_dim)`
- Query `j` occupies slot range `[j * query_dim, (j + 1) * query_dim)`

Metadata keys:

- `format_version`: `"openfhe_batch_v1"`
- `n_queries`
- `n_ciphertexts`
- `queries_per_ciphertext`
- `query_dim`
- `slots_per_ciphertext`
- `poly_modulus_degree`

## C++ Binaries To Add

- `openfhe_encrypt_queries_batched`
  - Input: packed plaintext query matrix file + context dir
  - Output: `encrypted_query_batch_*.bin`, `encrypted_norm_batch_*.bin`, `queries_metadata.json`
- `openfhe_compute_distances_batched`
  - Input: encrypted query batches + plaintext centroids file
  - Output: `distance_{centroid}_query_{query}.bin` + `distances_metadata.json`
- `openfhe_decrypt_batched_topk`
  - Input: encrypted distances + context dir
  - Output: `batched_top_k_results.json` and optional per-query `.npy` emitted by Python wrapper

## Distance Computation Strategy

For each centroid:

1. Construct a packed plaintext centroid by repeating centroid values into each query segment.
2. Compute ciphertext-plaintext multiply once per encrypted query batch.
3. Use masked `EvalSum` blocks per query segment to obtain `<q_j, c_i>`.
4. Compute `d_i^j = ||q_j||^2 + ||c_i||^2 - 2<q_j, c_i>`.

Optimization note:

- Replace per-query masking with rotate-and-accumulate tree once correctness is stable.

## Compatibility Plan

- Keep current filenames consumed by batched runners.
- Add `backend` and `format_version` fields to metadata.
- Python wrappers accept both legacy and new metadata while migration is active.

## Validation Plan

- Correctness: per-query top-k overlap against plaintext baseline.
- Numerical tolerance: max absolute distance delta and rank stability.
- Throughput: report centroids/sec and queries/sec against TenSEAL batched baseline.

