# Centroid-Batched OpenFHE Speedup Report

## Summary

For the actual bottleneck workload (one encrypted query against 4096 plaintext centroids), centroid batching is the effective OpenFHE batching strategy.

The implemented centroid-batched path keeps the similarity math unchanged and uses all 768 embedding dimensions. It changes only the CKKS slot layout:

- Baseline: one encrypted distance ciphertext per centroid.
- Centroid-batched: one encrypted distance ciphertext per group of 8 centroids.

## Full 4096-centroid benchmark

Command:

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

Results:

| mode | centroids | output ciphertexts | compute seconds | speedup | top-100 overlap |
|---|---:|---:|---:|---:|---:|
| baseline OpenFHE | 4096 | 4096 | 419.709 | 1.000x | reference |
| centroid-batched OpenFHE | 4096 | 512 | 49.511 | 8.477x | 100/100 |

## Smaller validation runs

| centroids | baseline seconds | centroid-batched seconds | speedup | top-k overlap |
|---:|---:|---:|---:|---:|
| 16 | 2.992 | 1.349 | 2.219x | 10/10 |
| 128 | 13.452 | 2.415 | 5.570x | 100/100 |
| 4096 | 419.709 | 49.511 | 8.477x | 100/100 |

## Implemented binaries

- `openfhe_keygen`
  - now accepts `--rotation-indices` for packed rotate-add reductions.
- `openfhe_encrypt_query_centroid_batched`
  - encrypts one query repeated across centroid lanes.
- `openfhe_compute_distances_centroid_batched`
  - computes packed distances for multiple centroids per ciphertext.
- `openfhe_decrypt_topk_centroid_batched`
  - decrypts packed distance blocks and returns exact top-k centroids.

## Recommendation

Use:

- `poly_modulus_degree=16384`
- `padded_dim=1024`
- `centroids_per_ciphertext=8`
- `num_threads=20`

This is the best verified no-accuracy-drop optimization so far for one encrypted query against 4096 centroids.
