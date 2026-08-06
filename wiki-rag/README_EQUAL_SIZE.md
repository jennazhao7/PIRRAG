# Equal-Size Clustering for FHE-Friendly ANN

Every cluster holds exactly `n` vectors, so the number of candidates a query
touches is `p * n` — the same for every query, always.

This document is meant to be read start to finish by someone who has not seen the
change. It covers what it does, why, what it costs, how to run it, and what you
need to know to build the PIR and FHE stages on top.

---

## 0. Both clustering modes are supported — pick one per run

**The original unequal clustering still works, unchanged and unpadded.** This is
not a migration; nothing was removed. One flag selects the mode, and they are
mutually exclusive so you cannot accidentally get a mix.

```bash
# ORIGINAL: fixed cluster count, variable sizes, NO padding.
python train_ivf.py --faiss-index-file .../index.faiss --k 4096 --output-dir out

# NEW: fixed cluster size n, count derived as ceil(N/n), lists padded to n.
python train_ivf.py --faiss-index-file .../index.faiss --cluster-size 128 --output-dir out
```

Omitting both flags gives you `--k 4096`, i.e. exactly the historical default.

|  | `--k K` *(original)* | `--cluster-size n` *(new)* |
|---|---|---|
| `sizing_mode` in metadata | `fixed-nlist` | `constant-size` |
| cluster count | `K`, chosen by you | `ceil(N / n)`, derived |
| cluster sizes | whatever k-means yields (**1–103** on the 65k index) | exactly `n`, always |
| padding | **none** | `-1` in unused slots (8–24 slots total) |
| `lists.json` | variable-length lists, no sentinel | every list exactly `n` long, `-1` padded |
| `cluster_slots.npy` | not written | written; the canonical artifact |
| candidates per query | `p ×` (varies per query) | exactly `p × n` |
| PIR addressing | `offsets[cluster] + index` | `cluster * n + index` |

Verified, not just asserted: fed the shipped centroids, the `--k` path reproduces
`prototype/data/lists.json` **exactly, 4096/4096 clusters**, and a legacy run's
`lists.json` contains **no `-1` anywhere**. The original behaviour is bit-identical,
with one deliberate exception noted in §6: `--niter` is now actually applied, having
previously been silently discarded, so a re-run is *better trained* than the shipped
artifact rather than identical to it.

Reading either mode is one call — never branch on list lengths:

```python
import ivf_io
c = ivf_io.load_clustering(path)        # a directory, or a bare lists.json
c.is_constant_size                       # False for --k, True for --cluster-size
c.members(cluster_id)                    # real vector ids, sentinel already stripped
```

There is also `--assignment faiss-nearest`, which gives plain nearest-centroid
variable-size assignment as an explicit diagnostic baseline. That is what the
4096/4096 oracle test above uses.

---

## 1. What changed

### Before

The retrieval stage is IVF: compare the query to every centroid, keep the top `p`
clusters, then rank the vectors inside them.

Clusters came from plain FAISS k-means. You chose the *number* of clusters (4096),
k-means placed 4096 centroids, and each vector went to whichever centroid was
nearest. Nothing constrained how many vectors landed in a cluster. On the 65k
database the result was:

```
4096 clusters, 65,000 vectors
sizes: min 1, max 103, mean 15.87, median 12   <- 100x spread
106 clusters held a single vector
```

So "fetch the top 100 clusters" meant fetching somewhere between a handful and
several thousand vectors, depending on where the query landed.

### After

k-means still places the centroids. The **assignment** step is replaced: instead
of "each vector goes to its nearest centroid," it is "each cluster gets exactly
`n` vectors, allocated as close to nearest-centroid as possible." The cluster
count is no longer a free parameter — it is forced to `ceil(N / n)`.

```
n = 128  ->  500 clusters x 128 slots, every one full   (508 on the full 65,000)
```

### How the assignment works

An auction. Each cluster carries a price. Each vector picks the cluster minimising
`distance + price`. Over-subscribed clusters get more expensive, and the process
repeats until the seats clear. Vectors that lose their first choice take their
second or third.

This is subgradient ascent on the dual of the transportation LP, which matters for
one practical reason: **every vector re-chooses at the current prices on every
round**, so there is no first-come-first-served bias. A simple greedy pass ("sort
all vector-cluster pairs by distance, assign until full") is also implemented and
is what repairs any leftovers, but on its own it is measurably worse — greedy lets
a vector lose a seat to a marginally closer competitor with no recourse.

Two guards make it robust:

- Only each vector's nearest `top_m` clusters are considered (default 32). The
  dense 65,000 x 4,063 distance matrix would be 1.06 GB; `(N, 32)` is 8.3 MB.
- Any vector whose entire candidate set filled up is placed by an exact scan over
  the clusters that still have room. This is why the algorithm always terminates
  with a complete, capacity-feasible assignment.

The whole thing takes a few seconds on 65k x 768.

---

## 2. What it costs, measured

Real numbers, on the real index (`royrin/wiki-rag`, `wiki_index__top_100000__2025-04-11`,
which holds 65,000 vectors of dim 768). 1,000 queries held out of the database, so
the recall figures are honest — see §6.

Matched at `recall@10 = 0.9519`:

| config | clusters | p | centroid dists | candidates | total | vs base | penalty |
|---|---|---|---|---|---|---|---|
| baseline (variable) | 4096 | 100 | 4,096 | ~1,562 *varies* | 5,658 | 1.00x | — |
| n=16 | 4000 | 419 | 4,000 | 6,704 | 10,704 | 1.89x | 1.149 |
| n=32 | 2000 | 163 | 2,000 | 5,216 | 7,216 | 1.28x | 1.103 |
| n=64 | 1000 | 85 | 1,000 | 5,440 | 6,440 | 1.14x | 1.081 |
| **n=128** | **500** | **43** | **500** | 5,504 | **6,004** | **1.06x** | 1.054 |

*penalty* = mean squared distance to the assigned centroid divided by the same to
the unconstrained nearest centroid. It is how much the equal-size constraint
degrades the clustering, and it is small: 5% at n=128.

### Read this part before adopting it

**The candidate stage gets ~3.5x more expensive.** Equal sizes need 5,504
candidates where the baseline needed ~1,562. That is not an implementation
weakness, it is intrinsic: variable cluster sizes were doing useful work.
k-means gives dense regions clusters with many vectors, so a query landing in a
crowded neighbourhood automatically pulled a bigger candidate pool. Uniform sizes
throw that adaptivity away, and the true neighbours end up scattered across more
clusters.

What makes n=128 worth it anyway is that this penalty is **nearly flat in `n`** —
5,216 at n=32, 5,504 at n=128, a 6% spread — while the centroid stage varies 8x
(2000 centroids vs 500). You pay the candidate penalty whichever `n` you pick, so
take the `n` that makes the centroid stage cheapest. Counting both stages, n=128
lands at 1.06x the baseline's total work.

**If PIR bandwidth rather than FHE compute is your binding constraint, this trade
is worse than the table makes it look**, because the 8x centroid saving does not
help PIR at all.

That is not hypothetical: the alternative in `../padded-naive/` has been measured
against this on the same query set, and **it wins on PIR bandwidth by 42%** (3,168
records vs 5,504 at the same recall), while tying on total homomorphic work (5,985
vs 6,004). It keeps naive k-means and splits each cluster into fixed-width chunks,
so it retains the adaptivity while still presenting uniform rows to PIR. Equal-size
still wins clearly on the FHE centroid stage — 500 comparisons versus 1,249 at the
best split point. **Read `../padded-naive/README.md` before committing to either.**

---

## 3. Artifacts and schema

`train_ivf.py` writes these into `--output-dir`:

| file | what it is |
|---|---|
| `ivf_metadata.json` | **Read this first.** Sizing mode, cluster count, cluster size, k-means params, diagnostics. Written by every run, legacy included. |
| `cluster_slots.npy` | Canonical. `(nlist, n)` int32. `slots[c][i]` is a global vector id, or `-1` for an unused slot. |
| `lists.json` | Compatibility view, `{"<cluster_id>": [ids...]}`. Under constant size every list has length exactly `n`, padded with `-1`. |
| `centroids.npy` | `(nlist, dim)` float32. |
| `cluster_balance.txt` | Human-readable diagnostics: size histogram, balance penalty, retrievability curve. |
| `db_ids.npy` | Only in `holdout` recall mode. See the warning in §6. |

### The one rule that matters

**Branch on `ivf_metadata.json["sizing_mode"]`. Never infer the layout from list
lengths.** A variable-size run can coincidentally produce equal-length lists on
small data, and code that guesses will silently do the wrong thing. Values are
`fixed-nlist` (legacy, variable) and `constant-size`.

### Reading it

Do not parse the files yourself. `ivf_io.load_clustering` handles both layouts and
strips the padding sentinel for you:

```python
import ivf_io

c = ivf_io.load_clustering("ivf_output_n128")
c.is_constant_size      # True
c.cluster_size          # 128
c.nlist                 # 500
c.members(cluster_id)   # real vector ids, sentinel already removed
c.slot_address(c=7, i=3)  # 7*128+3 = 899; raises on a variable-size clustering
```

It also accepts a bare path to a legacy `lists.json` (e.g.
`prototype/data/lists.json`), which has no metadata alongside it.

---

## 4. Running it

Everything needs numpy and `faiss-cpu`. Nothing in the new modules needs torch or
langchain — the embedding model was only ever required to satisfy
`FAISS.load_local`'s constructor, never to read stored vectors.

### Get the vectors

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('royrin/wiki-rag',
      'wiki_index__top_100000__2025-04-11/index.faiss', repo_type='model'))"
```

Only `index.faiss` (200 MB) is needed. Skip `index.pkl` (900 MB) unless you need
document text. Note the file holds **65,000** vectors despite the `top_100000`
name — the 65k in this repo is not the result of `truncate_databases.py`.

### Train one clustering

```bash
python train_ivf.py \
    --faiss-index-file .../wiki_index__top_100000__2025-04-11/index.faiss \
    --cluster-size 128 \
    --output-dir ivf_output_n128 \
    --test-recall --recall-mode holdout --n-test-queries 1000
```

`--k` and `--cluster-size` are mutually exclusive: `--k` is the legacy mode (fixed
count, variable sizes), `--cluster-size` is this one. Passing neither reproduces
the historical default of `--k 4096` exactly.

Useful flags: `--assignment {balanced,greedy}` (default `balanced`), `--assign-topm`
(raise it if the run warns about vectors placed outside their candidate set),
`--dual-bound` (certified optimality, one extra pass), `--lists-json {padded,trimmed,none}`.

### Find the right `n`

```bash
python sweep_cluster_size.py \
    --faiss-index-file .../index.faiss \
    --work-dir cluster_size_sweep \
    --cluster-sizes 16 32 64 128
```

Measures the baseline and every `n` in one process against one shared query set,
then binary-searches the smallest `p` reaching baseline recall. Writes
`cluster_size_sweep.{json,md}` and `recall_analysis_constant_size.txt`.
`--report-only` rebuilds the reports from saved JSON without re-measuring.

Binary search is valid because recall is monotonic in `p`: the top-`p+1` centroids
are a superset of the top-`p`, so candidates only grow and the exact rerank cannot
get worse.

---

## 5. Implementing downstream

### PIR

This is the main practical payoff. Because every cluster is exactly `n` wide, the
address of a record is arithmetic:

```
slot_address = cluster_id * n + index_in_cluster
```

Three consequences:

1. **The client no longer needs `lists.json`.** Today
   `batched_test/run_pir_single_query_loop.py:149-156` maps the PIR client's
   `(cluster_id, index)` back to a global vector id by indexing into `lists.json`,
   which means shipping the whole cluster map to the client. With constant `n`
   that lookup is `cid * n + idx`.
2. **Records are uniform**, so per-query cost stops revealing how full the probed
   clusters were.
3. **Fetch size is fixed** at `p * n` (5,504 at n=128, p=43) rather than varying
   per query.

**The database must be laid out cluster-major, and `prototype/pir_export.py` does
that for you.** It was not: `faiss_processor.py:76` emits one record per vector in
global FAISS row order and `pickle_processor.py` does the same in docstore order,
so a record's position said nothing about its cluster. The gap was bridged by
shipping `lists.json` to the client and translating before each fetch.

Reorder once, and the row index *becomes* the PIR address:

```bash
# vectors
python pir_export.py --clustering ../wiki-rag/ivf_output_n128 \
    --vectors-npy vectors.npy --output pir_vectors.npy --verify

# document records (chr(31)-separated fixed-width, from pickle_processor.py)
python pir_export.py --clustering ../wiki-rag/ivf_output_n128 \
    --records ../uniform_index.txt --output pir_db.txt --verify
```

It writes the reordered database plus a `.manifest.json` carrying the addressing
rule, row counts, and (for variable-size) the offsets table. **It works for both
clustering modes**, so cleaning up the PIR handoff does not require adopting
constant-size clusters:

| clustering | address | what the client needs |
|---|---|---|
| `--cluster-size n` | `cluster * n + index` | the single integer `n` |
| `--k K` *(original)* | `offsets[cluster] + index` | `K + 1` offsets — 32 KB at K=4096, vs 768 KB for `lists.json` |

Either way `lists.json` stops being a client dependency and becomes a pure
evaluation artifact. The offsets table is also a smaller disclosure than
`lists.json`: it reveals cluster *sizes* only, not which vector sits where.

`--verify` checks the property everything else rests on — that the documented
formula lands on the right vector — by resolving 2,000 random `(cluster, index)`
pairs against the clustering, plus confirming every vector appears at exactly one
row and only padding rows hold the sentinel. Measured on the real index:

```
constant-size n=128 : 508 clusters, 65,024 rows (24 padding)
                      row = cluster_id * 128 + index      [verified]
legacy    nlist=4096 : 65,000 rows (0 padding), 32 KB offsets
                      row = offsets[cluster_id] + index    [verified]
```

Padding rows are blank (records) or zero (vectors) so every row keeps the same
width; the client discards them on the sentinel recorded in the manifest. At n=128
there are 24 such rows out of 65,024.

> **The export refuses a holdout clustering.** In `--recall-mode holdout` the
> vector ids index the held-out subset, so reordering a full-corpus database by
> them would pair rows with the wrong payloads — and the result would look
> perfectly well-formed. `pir_export.py` errors out unless you pass
> `--allow-holdout-ids`. Build deployable artifacts with `--recall-mode perturbed`,
> which keeps the database whole.

What is still outside this repo: the `easypir` Go server takes a database file and
serves records positionally, so pointing it at the reordered file is all that is
needed — no Go changes. That last step is unverified here, since `easypir` is not
in this repository.

### FHE

**The existing OpenFHE code works unchanged.** Verified by reading the sources:

- `openfhe_compute_distances_centroid_batched.cpp:87` takes the centroid count
  from the file, not a constant.
- `:102` uses a ceiling division for the batch count and `:121` clamps the last
  batch with `std::min`, so 508 centroids (63.5 ciphertexts) is fine.
- `n_centroids` flows through metadata everywhere it is consumed
  (`openfhe_batched_workload.cpp:658`, `openfhe_decrypt_topk_centroid_batched.cpp:62`).
- `prototype/fhe_backend.py:404-406` derives it from `len(centroids)`.
- The CKKS slot constraint is `centroids_per_ciphertext * padded_dim <=
  poly_modulus_degree / 2`, which involves only packing parameters. The cluster
  count does not enter it.

So the only required step is regenerating `centroids.txt`, and the stage gets
proportionally cheaper:

| centroids | ciphertexts (at 8 per) |
|---|---|
| 4096 (baseline) | 512 |
| 500 (n=128, 64k holdout DB) | 63 |
| 508 (n=128, full 65,000 DB) | 64 — last batch partial, 4 of 8 lanes |

```bash
python fhe_prepare.py --ivf-dir ivf_output_n128 --out-dir openfhe_inputs
python fhe_prepare.py --self-test    # validates the kernel math, no OpenFHE needed
```

`fhe_prepare.py` exports `centroids.txt`, validates the CKKS layout, reports the
ciphertext count, and warns if the centroids are in OPQ-rotated space (in which
case the query must be rotated before encryption or every distance is wrong).

`--self-test` runs a numpy reproduction of the C++ kernel's exact slot arithmetic —
query packed at `slot[d*cpc+b]`, rotation-sum with stride `step*cpc`,
`dist = (|q|^2 + |c|^2) - 2*dot` — and checks it against direct squared L2 at
several centroid counts including the partial-batch case. It agrees to 2.7e-15.
Use it to sanity-check a layout before spending time on a real FHE run.

**What equal-size clustering does not do.** Being precise, because it is easy to
over-claim: the FHE layer here computes exactly one thing, distances from the
encrypted query to the plaintext centroids, which the client decrypts to learn its
top-`p` clusters. The per-cluster kNN that follows happens **client-side in
plaintext**, on records the client just fetched over PIR. There is no homomorphic
per-cluster kNN in this codebase, so uniform cluster size does not enable a new FHE
computation. What it buys is the cheaper centroid stage, uniform PIR records, and a
fixed-shape client-side rerank.

If a homomorphic per-cluster kNN is added later, the existing kernel is already the
right shape for it — it is generic "one encrypted query against M plaintext
vectors" and does not care that M happens to be a centroid count. That is a design
note, not something implemented.

### Plaintext consumers

`prototype/rag_operations/plaintext_rag_pipeline.py` and
`prototype/compute_ground_truth.py` already read either layout via
`ivf_io.load_clustering`. They are the reference oracle: use them to check any new
PIR or FHE path end to end.

---

## 6. Two measurement traps

Both were live bugs, both are fixed, and both matter if you compare against older
numbers.

**`--niter` was silently discarded.** It was parsed and printed but never applied —
nothing set `index.cp.niter`, so FAISS's default of 10 governed every run while the
log claimed 20. Now applied, with the seed pinned and `min_points_per_centroid`
lowered to 1. That last one matters for sweeps: points-per-centroid is roughly `n`,
so with FAISS's default of 39 the n=16 and n=32 runs would have tripped its
subsampling path while n=64 and n=128 would not, silently comparing two different
clustering regimes.

**Recall was measured on queries drawn from the database.** Every query was
guaranteed to retrieve itself as its own top hit. So **the 96.41% in
`recall_analysis.txt` is not comparable to anything here.** Use `--recall-mode`:

- `holdout` (default) — queries removed from both the database and the clustering.
  The honest number, and the same seeded split across every configuration.
- `perturbed` — database kept whole, query jittered off the stored vector. Use this
  when you want the artifact you measure to be the artifact you ship.
- `self` — the old behaviour, reported as `self_retrieval_recall@k`.

This is not a cosmetic distinction. Self-retrieval punishes exactly what balancing
does (moving a vector off its own nearest centroid), so measuring this change that
way would systematically overstate its cost.

> **Warning on `holdout` artifacts.** In holdout mode the database is a strict
> subset, so exported vector ids index the 64,000-vector subset, not the original
> corpus. `ivf_metadata.json` records `id_space: "holdout_subset"` and the run
> writes `db_ids.npy`. **Do not ship a holdout clustering to PIR.** Use
> `--recall-mode perturbed` (or `self`) for artifacts you intend to deploy.

---

## 7. Why you can trust the implementation

- **Oracle test against the real shipped artifact.** Fed the checked-in
  `openfhe_core/centroids.txt`, the new export path reproduces
  `prototype/data/lists.json` **exactly — 4096/4096 clusters**. That validates
  export, loader, and assignment before balancing enters as a variable.
- **Data lineage confirmed.** 100.0000% of the 65,000 vectors sit in the cluster
  that `lists.json` assigns them to, so the downloaded index is provably the one
  that produced the shipped clustering.
- **Invariants asserted, not hoped over.** Every run checks each list is exactly
  `n` wide, that every vector id appears exactly once, and that padding accounting
  balances. Violations raise.
- **`pack_slots` checked against a slow reference** on 200 randomised trials.
- **Certified near-optimality.** `--dual-bound` gives a Lagrangian lower bound on
  the best possible equal-size assignment, so the balance penalty comes with a
  duality gap rather than an assurance.
- **The numpy fallback matches faiss** exactly (ids identical, distances to 4e-7),
  so the assignment logic is testable without a faiss build.

Not verified: **anything requiring OpenFHE**. It is not installed in this
environment and there is no cmake, so the FHE claims above rest on reading the
sources plus the plaintext kernel oracle — not on a compiled run. Someone with a
working OpenFHE build should confirm before trusting the ciphertext counts.

---

## 8. Files

New:

| file | role |
|---|---|
| `ivf_io.py` | Schema, metadata, `load_clustering`, `pack_slots`, invariant checks |
| `balanced_ivf.py` | The assignment: candidate set, price balancing, greedy repair, exact tail, dual bound |
| `ivf_eval.py` | Plaintext IVF search, the three recall modes, balance diagnostics |
| `sweep_cluster_size.py` | Finds the `n` matching baseline recall at least total cost |
| `fhe_prepare.py` | Exports and validates FHE centroid inputs; kernel oracle |
| `cluster_size_sweep/` | Measured results |
| `../prototype/pir_export.py` | Cluster-major PIR database export (both modes) + address verification |

Modified: `train_ivf.py` (sizing flags, the `--niter` fix, `run_pipeline`,
constant-size export), plus the two plaintext consumers and
`prototype/data/README.md`.

The alternative approach — keep naive clustering and pad it to look uniform — is in
`../padded-naive/`.
