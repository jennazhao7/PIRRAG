# Padded Naive Clustering — the alternative to forced-equal-size

**Verdict: the idea works, but only in one specific form, and it wins on a
different axis than expected.**

- Padding naive clusters out to the **maximum** list width — the obvious reading of
  "make them look the same size" — is the **worst** option tested. 75–85% of every
  fetch is padding.
- **Splitting** naive clusters into fixed-width chunks is the version that works,
  and it **beats forced-equal-size on PIR bandwidth by 42%** (3,168 records vs
  5,504) at identical recall.
- On *total* homomorphic work the two approaches are a **dead heat** (5,985 vs
  6,004, a 0.3% difference).

So which approach is better depends entirely on whether PIR bandwidth or the FHE
centroid stage is your binding constraint. That is a real decision, not a
tie-break, and §4 lays it out.

---

## 1. The hypothesis being tested

Forced-equal-size clustering (see `../wiki-rag/README_EQUAL_SIZE.md`) needs ~3.5x
more candidates than plain k-means to reach the same recall. The reason is that
plain k-means is implicitly **adaptive**: dense regions get clusters with many
vectors, so a query landing in a crowded neighbourhood automatically pulls a bigger
candidate pool. Forcing every cluster to hold exactly `n` throws that away, and the
true neighbours end up scattered across many more clusters.

The idea: keep the naive clustering, and make it merely *look* uniform to PIR by
padding every list out to a common width `W`. PIR then fetches exactly `p * W`
records per query regardless of occupancy — the same uniformity property that
motivated equal-size in the first place — while retaining the adaptivity.

The catch is that padding is dead weight. Whether adaptivity is worth more than
the padding costs is empirical, which is what this trial settles.

## 2. Three policies

Choosing `W` below the maximum list length forces a decision about the long tail:

| policy | what it does | lossless? |
|---|---|---|
| `pad-max` | `W` = longest list. Everything fits, nothing is split. | yes |
| `split` | lists longer than `W` become `ceil(L/W)` chunks, each keeping a copy of the parent centroid | yes |
| `truncate` | lists longer than `W` keep only their `W` nearest members; the rest are dropped from the index | **no** |

Cost model per query, identical to the equal-size arm so the two are comparable:

```
PIR records fetched   = p * W                  (uniform -> no occupancy leak)
FHE centroid compares = number of rows         (splitting increases this)
total distance evals  = rows + p * W
```

## 3. Results

Real 65k index, 1,000 held-out queries, seed 1234 — **the same query set and the
same target as the equal-size arm**, verified programmatically. All configurations
matched at `recall@10 = 0.9519`.

### Head to head

| approach | centroid compares | PIR records | total |
|---|---|---|---|
| naive 4096, p=100 *(today; **not** uniform)* | 4,096 | ~1,562 *varies* | 5,658 |
| equal-size n=128, p=43 | **500** | 5,504 | 6,004 |
| equal-size n=64, p=85 | 1,000 | 5,440 | 6,440 |
| **split** nlist=512 W=64, p=74 | 1,249 | 4,736 | **5,985** |
| **split** nlist=4096 W=16, p=198 | 6,116 | **3,168** | 9,284 |
| pad-max nlist=4096 W=102, p=100 | 4,096 | 10,200 | 14,296 |
| pad-max nlist=512 W=509, p=27 | 512 | 13,743 | 14,255 |

### The Pareto frontier

Splitting trades PIR bandwidth against centroid comparisons, and every point below
is achievable at the same recall:

| config | centroid | PIR records | total |
|---|---|---|---|
| split nlist=4096 W=16 | 6,116 | 3,168 | 9,284 |
| split nlist=2048 W=16 | 5,008 | 3,264 | 8,272 |
| split nlist=1024 W=16 | 4,493 | 3,600 | 8,093 |
| split nlist=2048 W=32 | 3,097 | 3,744 | 6,841 |
| split nlist=1024 W=32 | 2,507 | 3,936 | 6,443 |
| split nlist=1024 W=48 | 1,857 | 4,272 | 6,129 |
| split nlist=1024 W=64 | 1,525 | 4,608 | 6,133 |
| **split nlist=512 W=64** | 1,249 | 4,736 | **5,985** |
| split nlist=512 W=96 | 925 | 5,184 | 6,109 |
| split nlist=512 W=128 | 753 | 5,632 | 6,385 |

Equal-size n=128 sits at (500, 5,504) — further along the cheap-centroid end than
any split configuration reaches, which is precisely its advantage.

### Findings

**`pad-max` fails, decisively.** Padding to the longest list wastes 75–85% of every
fetch and is worse than equal-size on *both* axes at *every* cluster count. If
"pad the clusters so they appear the same size" meant this, the answer is no.

**`truncate` is a dead end.** Dropping the overflow makes vectors unretrievable and
caps recall below target in most configurations — at nlist=512 W=16 it discards
55,880 of 64,000 vectors for a recall ceiling of 0.295. Where it is feasible it is
dominated by `split` anyway. It is retained in the sweep only to document that.

**`split` is the idea that works.** It keeps the naive centroids (so it keeps the
adaptivity) while bounding each row to `W`, and padding waste drops to 5–35%. It
reaches PIR fetches no equal-size configuration can match.

**Why `split` costs centroid comparisons.** Chunking emits one row per chunk, each
carrying a near-duplicate of the parent centroid, so the FHE centroid stage compares
against many almost-identical vectors. That is the price of the PIR saving, and it
is why the frontier slopes.

One thing worth correcting from earlier analysis: splitting was expected to *corrupt*
cluster selection, on the theory that probing several chunks of one dense cluster
wastes probes. The measurements do not support that. Those chunks hold genuinely
nearby vectors, so probing several of them is useful work — exactly the adaptive
behaviour the approach is trying to preserve. Splitting costs centroid comparisons,
not recall.

## 4. Which should you use

| if your binding constraint is… | choose | why |
|---|---|---|
| **PIR bandwidth / records fetched** | `split`, nlist=4096, W=16 | 3,168 records vs 5,504 — **42% less**. The FHE centroid saving of equal-size does nothing for PIR. |
| **Total homomorphic work** | either | 5,985 vs 6,004. A tie; pick on implementation cost. |
| **FHE centroid stage** | equal-size n=128 | 500 comparisons vs 1,249 at the best split point — 2.5x fewer, and 12x fewer than the PIR-optimal split. |
| **Simplicity** | equal-size | `nlist = ceil(N/n)` is known in advance; split's row count is data-dependent (`sum(ceil(L/W))`) and only known after building. |

Both give the properties that motivated this work: uniform fetch size, no occupancy
leak, and positional addressing (`row * W + index`, versus `cluster * n + index`).

**A balanced recommendation.** If both stages cost roughly the same per distance,
`split nlist=1024 W=32` is the most defensible middle point: 2,507 centroid
comparisons and 3,936 PIR records for 6,443 total — 29% less PIR than equal-size
n=128 for 7% more total work. If PIR records are more expensive than centroid
comparisons, which is typical since a PIR fetch moves an actual record while a
centroid comparison is one ciphertext operation, it dominates equal-size outright.

## 5. Running it

```bash
python -u pad_naive.py \
    --vectors-npy wiki65k.npy \
    --work-dir out \
    --nlists 512 1024 2048 4096 \
    --widths 16 32 48 64 96 128 \
    --policies pad-max split truncate \
    --baseline-nlist 4096 --baseline-nprobe 100 \
    --recall-mode holdout --n-test-queries 1000 --seed 1234
```

Use `-u`. Each configuration runs a full-database scan to establish whether the
target recall is reachable at all, so the sweep above takes about 70 minutes and a
buffered run shows nothing until it finishes.

To compare against the equal-size arm, the recall mode, query count and seed must
match — the query split is derived from `(N, mode, n_test, seed)`, so identical
values give an identical split and a valid comparison. The trial asserts nothing
about this, so check it.

## 6. Files

| file | what |
|---|---|
| `pad_naive.py` | The trial: naive k-means, the three padding policies, binary search for minimum `p` |
| `padded_naive_trial.json` | Full results, every configuration |
| `trial_run.log` | The run that produced the tables above |

Correctness of the layout builder is unit-tested across all three policies —
losslessness for `pad-max` and `split`, exact row counts, every vector id present
exactly once, and nearest-first ordering under `truncate`.

## 7. Caveats

- **Recall is `holdout`**, so exported ids index the 64,000-vector held-out subset,
  not the full corpus. These are measurement artifacts; regenerate with
  `--recall-mode perturbed` before deploying anything.
- The cost model counts **distance evaluations**, treating one centroid comparison
  and one candidate comparison as equal. They are not equal in wall-clock or in
  bandwidth, and the right weighting depends on your FHE parameters and PIR scheme.
  The Pareto table is given precisely so you can apply your own weights.
- `split` was swept over a grid of `(nlist, W)`. The frontier is not necessarily
  optimal — a finer grid, or splitting only the clusters that overflow while leaving
  the rest at their natural size, may do better.
- Only `ivf-flat` was tested. PQ/OPQ would change the constants but not the shape
  of the trade.
