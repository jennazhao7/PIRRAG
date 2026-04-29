# Plaintext Matrix / Prepacked Centroid Experiment Report

## Goal

Try a cleaner experimental path that preserves the verified centroid-batched kernel and tests whether extra plaintext-side preprocessing reduces query-time FHE centroid compute.

## Experiment 1: Prepacked plaintext centroid blocks

Added a separate binary:

- `openfhe_compute_distances_centroid_prepacked`

This binary keeps the existing exact `B=8` centroid-batched layout but moves plaintext packing into a prepack stage before the timed compute loop:

- precompute packed centroid plaintexts
- precompute packed centroid-norm plaintexts
- reuse those plaintext objects during the compute loop
- optionally skip ciphertext file writes with `--skip-writes`

This does not modify the existing verified binary:

- `openfhe_compute_distances_centroid_batched`

## 4096-centroid result

Config:

- `poly_modulus_degree=16384`
- `padded_dim=1024`
- `centroids_per_ciphertext=8`
- `num_threads=20`
- `batch_size=1`

| mode | prepack seconds | compute seconds | notes |
|---|---:|---:|---|
| prepacked with writes | 2.738 | 43.254 | 512 packed distance files |
| prepacked skip writes | 2.797 | 43.259 | no distance ciphertext writes |

Conclusion:

- Prepacking plaintext blocks does not materially improve the current kernel.
- Skipping output writes also does not materially improve compute time.
- The bottleneck is the OpenFHE ciphertext operations, especially the rotate-add reduction, not plaintext packing or filesystem writes.

## Experiment 2: Full plaintext matrix diagonal complexity

For an encrypted query of dimension 768 and 4096 plaintext centroids, embedding the centroid matrix into an 8192-slot CKKS vector space gives:

```text
slots = 8192
rows = 4096
cols = 768
nonzero matrix entries = 3,145,728
unique diagonal offsets = 4,863
```

Rough BSGS estimates:

| baby step | rough rotations | plaintext multiplications |
|---:|---:|---:|
| 8 | 616 | 4863 |
| 16 | 320 | 4863 |
| 32 | 184 | 4863 |
| 64 | 140 | 4863 |
| 128 | 166 | 4863 |

Conclusion:

- A direct diagonal matrix-vector multiplication is not automatically better for this rectangular sparse layout.
- It reduces the conceptual block loop, but introduces thousands of plaintext diagonals/multiplications.
- A production version would need a more specialized low-rank/block-sparse layout or OpenFHE-native linear-transform support tuned for this exact matrix shape.

## Current best verified path

The best exact no-accuracy-drop path remains:

- `openfhe_compute_distances_centroid_batched`
- `poly_modulus_degree=16384`
- `padded_dim=1024`
- `centroids_per_ciphertext=8`
- `num_threads=20`
- `batch_size=1`

Measured full 4096-centroid compute:

- original baseline: about `419.7s`
- centroid-batched: about `42-50s`
- speedup: about `8.5x-10x`
- top-100 overlap: `100/100`
