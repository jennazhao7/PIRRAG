# OpenFHE Batching Speed Report

## 1. Are we using OpenFHE for FHE computation?

Partially.

- The single-query Python path defaults to the OpenFHE C++ backend (`openfhe_cpp`). That backend shells out to `openfhe_keygen`, `openfhe_encrypt_query`, `openfhe_compute_distances`, and `openfhe_decrypt_topk`.
- The older Python batched path in `prototype/fhe_query_client_batched.py` and `prototype/fhe_query_server_batched.py` still imports and uses TenSEAL directly.
- The new `openfhe_core` batched binaries (`openfhe_encrypt_queries_batched`, `openfhe_compute_distances_batched`) use the OpenFHE C++ library directly, but the Python batched wrappers are not yet integrated with those binaries.

So: OpenFHE is used for the single-query workflow and for the new C++ batched binaries. If you launch the old `prototype/*_batched.py` scripts, batching is not OpenFHE end-to-end yet.

## 2. Supported batching parameters

This OpenFHE install rejects `poly_modulus_degree=8192` for CKKS:

```text
The specified ring dimension (8192) does not comply with HE standards recommendation (16384).
```

With 768-d embeddings, supported ring dimensions pack:

| poly_modulus_degree | slots | queries/ciphertext |
|---:|---:|---:|
| 16384 | 8192 | 10 |
| 32768 | 16384 | 21 |
| 65536 | 32768 | 42 |

## 3. Speed sweep

Command shape:

```bash
python openfhe_core/sweep_batched_params.py \
  --bin-dir openfhe_core/build/bin \
  --centroids-file prototype/real_openfhe_tuned/encrypted_distances_bs64b/centroids.txt \
  --query-dim 768 \
  --num-queries 21 \
  --max-centroids 2 \
  --poly-degrees 16384,32768,65536 \
  --thread-options 20 \
  --batch-size-options 32 \
  --work-dir openfhe_core/sweep_runs_tiny
```

Measured centroid-compute results:

| poly_modulus_degree | queries/ciphertext | queries | centroids | compute_s | qps | relative to 16384 |
|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 10 | 21 | 2 | 70.735 | 0.2969 | 1.000x |
| 65536 | 42 | 21 | 2 | 155.112 | 0.1354 | 0.456x |
| 32768 | 21 | 21 | 2 | 265.275 | 0.0792 | 0.267x |

Single-query OpenFHE baseline on the same 2-centroid input:

| poly_modulus_degree | queries | centroids | compute_s | qps |
|---:|---:|---:|---:|---:|
| 16384 | 1 | 2 | 1.962 | 0.5098 |

For this current batched kernel, `16384` is the best tested ring dimension. Increasing ring dimension does batch more queries per ciphertext, but it is slower overall for centroid computation in the measured kernel.

## 4. Interpretation

The current C++ batched implementation is functionally OpenFHE-backed, but it is not an optimized SIMD batching kernel yet. It performs per-query extraction/masking and writes per-query/per-centroid distance ciphertexts, which means the larger ring dimensions do not translate into speedup.

Speed-wise recommendation for the current implementation:

- Use `poly_modulus_degree=16384`.
- Treat `queries_per_ciphertext=10` as the best no-accuracy-drop batching point currently measured.
- Do not move to `32768` or `65536` for speed unless the batched kernel is optimized.

To get real batching speedup beyond 10 queries/ciphertext, the next implementation target should be changing the C++ batched distance output to keep distances packed per centroid/query-batch and avoid the per-query mask/sum/save loop.
