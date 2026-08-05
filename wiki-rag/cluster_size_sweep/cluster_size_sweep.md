# Constant cluster size sweep

- Database: **64,000** vectors, dim 768
- Metric: **holdout_recall@10** over 1000 queries (seed 1234, shared across every row)
- Index: ivf-flat, niter=20

## Baseline (fixed cluster count, variable sizes)

| nlist | nprobe | cluster sizes | centroid dists | candidates | total dists | % DB | holdout_recall@10 |
|---|---|---|---|---|---|---|---|
| 4096 | 100 | min 1, max 102, mean 15.62 | 4,096 | ~1,562 (varies) | 5,658 | 2.44% | 0.9519 |

## Constant size: smallest p reaching recall 0.9519

Ranked by **total distance evaluations** (centroid stage + candidate stage),
since both are homomorphic comparisons and constant-size clustering trades
one against the other.

| n | x = ceil(N/n) | pad | p | centroid | candidates (p*n) | total | vs base | % DB | holdout_recall@10 | balance penalty | forced | ms (numpy) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 4000 | 0 | 419 | 4,000 | 6,704 | 10,704 | 1.89x | 10.47% | 0.9519 | 1.1487 | 33.6% | 3.37 |
| 32 | 2000 | 0 | 163 | 2,000 | 5,216 | 7,216 | 1.28x | 8.15% | 0.9519 | 1.1029 | 32.2% | 2.53 |
| 64 | 1000 | 0 | 85 | 1,000 | 5,440 | 6,440 | 1.14x | 8.50% | 0.9521 | 1.0807 | 28.6% | 2.67 |
| 128 | 500 | 0 | 43 | 500 | 5,504 | 6,004 | 1.06x | 8.60% | 0.9529 | 1.0543 | 25.2% | 3.48 |

**Equivalent operating point: n = 128, p = 43.** 6,004 total distance evaluations versus 5,658 for the baseline (1.06x), at the same recall, with every cluster exactly 128 wide.

The candidate stage alone costs 3.52x the baseline -- equal sizes give up the baseline's implicit adaptivity, where a query landing in a dense region automatically pulls larger clusters. What pays for it is the centroid stage: 4,096 -> 500 centroids, 8.2x fewer encrypted comparisons.

## Notes

- `total dists` counts nlist centroid comparisons plus p*n candidate
  comparisons. Ranking on candidates alone would pick a different (worse) n.
- Latency is numpy, **not** comparable with `recall_analysis.txt`, which was
  measured through FAISS's SIMD search path.
- The baseline recall here is re-measured in `holdout` mode,
  not taken from `recall_analysis.txt` (whose queries came from the database
  itself, making every query its own guaranteed top-1 hit).
- With constant size, candidates scanned is exactly `p*n` for every query.
  The baseline's candidate count varies per query with the sizes of whichever
  clusters were probed.
