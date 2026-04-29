# Query + Centroid Batching Experiment Report

## Goal

Test a clean separate pipeline for workloads with many encrypted queries available at once, while preserving the verified single-query centroid-batched path.

Centroids remain plaintext. The only encrypted values are query vectors and output distances.

## Implemented experimental binaries

- `openfhe_encrypt_queries_centroid_batched`
  - encrypts multiple queries in a 2D query-lane/centroid-lane layout.
- `openfhe_compute_distances_query_centroid_batched`
  - computes packed distances for query batches against plaintext centroid batches.
- `openfhe_decrypt_topk_query_centroid_batched`
  - decrypts packed outputs and returns top-k per query.

## Packing layout

For `Q` queries and `C` plaintext centroid lanes:

```text
slots used = padded_dim * Q * C
```

At `poly_modulus_degree=16384`, CKKS has 8192 slots. With `padded_dim=1024`:

```text
Q * C <= 8
```

The tested exact 2D layout was:

```text
Q = 2 queries
C = 4 plaintext centroids
Q*C = 8 lanes
```

The current single-query centroid-batched baseline uses:

```text
Q = 1 query
C = 8 plaintext centroids
Q*C = 8 lanes
```

## Benchmarks

### 4 queries × 128 centroids

| mode | ring | Q | C | compute seconds | seconds/query | top-k overlap |
|---|---:|---:|---:|---:|---:|---:|
| single-query centroid batching | 16384 | 1 | 8 | 10.309 | 2.577 | reference |
| query+centroid batching | 16384 | 2 | 4 | 6.305 | 1.576 | 50/50 each |

Throughput speedup: `1.635x`

### 4 queries × 512 centroids

| mode | ring | Q | C | compute seconds | seconds/query | top-k overlap |
|---|---:|---:|---:|---:|---:|---:|
| single-query centroid batching | 16384 | 1 | 8 | 30.590 | 7.647 | reference |
| query+centroid batching | 16384 | 2 | 4 | 25.293 | 6.323 | 100/100 each |

Throughput speedup: `1.209x`

### Larger ring test

For 4 queries × 128 centroids:

| mode | ring | Q | C | compute seconds | seconds/query | top-k overlap |
|---|---:|---:|---:|---:|---:|---:|
| single-query centroid batching | 32768 | 1 | 8 | 30.585 | 7.646 | reference |
| query+centroid batching | 32768 | 2 | 8 | 12.640 | 3.160 | 50/50 each |

This improves relative to the same-ring baseline, but in absolute time it is slower than the `16384` query+centroid run (`6.305s`) on the same 4-query/128-centroid problem.

## Interpretation

Query+centroid batching works and is exact, but it does not multiply capacity for free. The slot budget is shared:

```text
Q * C <= slots / padded_dim
```

At ring `16384`, `Q=2,C=4` and `Q=1,C=8` both use the same 8 lanes. Therefore, 2D batching mostly reduces repeated per-query overhead; it does not reduce the total number of FHE packed blocks for large centroid counts.

For many-query throughput, the best tested exact layout is:

```text
poly_modulus_degree=16384
padded_dim=1024
queries_per_batch=2
centroids_per_batch=4
num_threads=20
batch_size=1
```

For single-query latency, keep using:

```text
poly_modulus_degree=16384
padded_dim=1024
centroids_per_ciphertext=8
num_threads=20
batch_size=1
```
