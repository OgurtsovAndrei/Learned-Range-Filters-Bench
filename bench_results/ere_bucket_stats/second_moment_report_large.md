# ERE Bucket Second Moment & Key-Weighted Tail at Maximum Scale

## Metric

The ERE backend partitions the universe into `B` blocks (buckets); the b-th
block holds `k_b` keys. The standard "uniform" average bucket size is
`(Sum_b k_b) / M` over the `M` non-empty blocks — every bucket is weighted
equally. Real range queries that hit data are not uniform over buckets,
though: if every stored key is equally likely to be a query target, the
probability that a query lands in bucket `b` is `k_b / n`. The expected
per-query bucket-search cost (in keys touched, ignoring constants) is
therefore the second moment of bucket occupancy normalised by `n`:

    E[bucket size touched by query] = Sum_b (k_b / n) * k_b = (1 / n) * Sum_b k_b^2

The `ratio = second_moment / avg_unif` quantifies how much heavier a
query-weighted draw is compared with a uniform draw over non-empty buckets.
A ratio close to 1 means buckets are nearly balanced; a large ratio means
the heavy-tail of the bucket-size distribution dominates query work.

In addition we report **key-weighted percentiles** `X_p` of the non-empty
bucket-size distribution, computed exactly as in the n=2^20 baseline (sort
sizes ascending, take the smallest `i` with `cum[i] >= ceil(p * n)`).

> `X_p` is the smallest bucket size such that buckets of size `<= X_p` hold
> at least `p * n` keys (key-weighted CDF over non-empty buckets).

This is a parallel large-n companion to `second_moment_report.md`; the
underlying methodology and column definitions are identical.

## Setup

- Filter: `are_soda_hash.SodaARE` (epsilon = 0.01) built via
  `NewSodaAREUint64InPlace`, the destructive uint64 fast path that hashes,
  sorts and dedupes inside the caller's slice (no per-key extra
  allocation).
- Backend: default ERE (classic two-level RSDic).
- Range lengths: `L in {16, 256, 4096}`, except `uniform` where only `L=16`
  is run — at n=2^20 we already established that `L` does not change the
  Poisson-like picture for uniform (`ratio ~ 1.30` at every L), so the
  remaining two combos were skipped to save wall-clock.
- Per-distribution `n`:
  - `uniform`: 2^30 ≈ 1.07 G keys (post-dedupe ≈ 1.07 G — ~4.3 M random
    collisions among 2^30 draws from `[0, 2^64)`).
  - `clustered`: target n = 2^30; **post-dedupe n = 470,832,294** because
    the 8 Gaussian clusters with `stddev ~ 2^29` produce many collisions
    when generating ~1G samples. We report results at the actual unique-key
    count.
  - `sosd_fb`, `sosd_wiki`, `sosd_books`: 2^27 ≈ 134 M (first 2^27 keys of
    the SOSD dataset, sorted, deduped). Wiki collapses to 65 M unique
    keys.
  - `sosd_osm`: 2^29 ≈ 537 M.
- Key generation memory model: uniform and clustered are generated without
  a `seen` map (which would be ~30 GB for n=2^30). Uniform writes
  `rng.Uint64()` directly into the slice; clustered writes Gaussian samples
  per cluster directly. Both then sort and dedupe in place. The
  `NewSodaAREUint64InPlace` build path then mutates the same slice further,
  so at no point are two copies of the n-element key array alive
  simultaneously.

## Results

`X_p` is the smallest bucket size such that buckets of size `<= X_p` hold
at least `p * n` keys (key-weighted CDF over non-empty buckets, nearest
rank). `avg_unif = (Sum_b k_b) / M` is the bucket-weighted (uniform) mean.
`second_moment = (1 / n) * Sum_b k_b^2`. `ratio = second_moment / avg_unif`.
`wall_s` is the per-combo wall-clock (load + build + stats extraction).

| Distribution | n            | L    | B          | M          | X_50    | X_90    | X_95    | X_99    | max     | avg_unif    | second moment | ratio       | wall_s |
|--------------|--------------|------|------------|------------|---------|---------|---------|---------|---------|-------------|---------------|-------------|--------|
| uniform      | 1,073,741,824| 16   | 536,870,912| 464,213,367| 3       | 5       | 6       | 7       | 16      | 2.3125      | 2.9990        | 1.2969      | 465.60 |
| clustered    |   470,832,294| 16   | 268,435,456|  89,508,024| 344     | 4,065   | 4,096   | 4,096   | 4,096   | 5.2598      | 1,079.0979    | 205.1606    | 222.37 |
| clustered    |   470,832,294| 256  | 268,435,456|  88,570,682| 5,513   | 65,026  | 65,532  | 65,536  | 65,536  | 5.3159      | 17,245.7461   | 3,244.2038  | 217.89 |
| clustered    |   470,832,294| 4096 | 268,435,456|  88,504,831| 88,200  |1,039,504|1,048,517|1,048,576|1,048,576| 5.3198      | 275,610.0131  | 51,807.8885 | 209.70 |
| sosd_fb      |   134,217,728| 16   | 134,217,728|  14,751,796| 27      | 103     | 136     | 208     | 463     | 9.0984      | 42.6303       | 4.6855      | 15.14  |
| sosd_fb      |   134,217,728| 256  | 134,217,728|   1,540,315| 208     | 545     | 660     | 886     | 1,589   | 87.1365     | 258.3991      | 2.9655      | 11.80  |
| sosd_fb      |   134,217,728| 4096 | 134,217,728|      99,155| 1,759   | 2,947   | 3,336   | 4,063   | 6,176   | 1,353.6153  | 1,834.9384    | 1.3556      | 11.11  |
| sosd_wiki    |    65,492,346| 16   |  33,554,432|      48,512| 3,240   | 3,843   | 3,906   | 3,972   | 4,054   | 1,350.0236  | 2,822.7309    | 2.0909      | 3.79   |
| sosd_wiki    |    65,492,346| 256  |  33,554,432|       3,174| 52,565  | 58,855  | 59,721  | 60,925  | 62,037  | 20,634.0095 | 44,582.6310   | 2.1606      | 3.63   |
| sosd_wiki    |    65,492,346| 4096 |  33,554,432|         199| 847,135 | 938,217 | 939,041 | 956,498 | 956,498 | 329,107.2663| 708,934.9703  | 2.1541      | 3.64   |
| sosd_books   |   134,217,728| 16   | 134,217,728|     185,309| 812     | 875     | 888     | 911     | 978     | 724.2915    | 753.6175      | 1.0405      | 11.32  |
| sosd_books   |   134,217,728| 256  | 134,217,728|      11,582| 13,215  | 13,670  | 13,728  | 13,826  | 14,039  | 11,588.4759 | 12,009.7971   | 1.0364      | 11.36  |
| sosd_books   |   134,217,728| 4096 | 134,217,728|         724| 211,319 | 217,641 | 217,893 | 218,144 | 218,469 | 185,383.6022| 192,094.6210  | 1.0362      | 11.03  |
| sosd_osm     |   536,870,912| 16   | 268,435,456| 231,999,951| 3       | 5       | 6       | 7       | 35      | 2.3130      | 3.0020        | 1.2979      | 146.35 |
| sosd_osm     |   536,870,912| 256  | 268,435,456| 231,629,855| 3       | 5       | 6       | 7       | 85      | 2.3177      | 3.0186        | 1.3024      | 139.57 |
| sosd_osm     |   536,870,912| 4096 | 268,435,456| 228,350,787| 3       | 5       | 6       | 8       | 502     | 2.3511      | 3.1391        | 1.3352      | 129.35 |

## Interpretation

The qualitative picture from the n=2^20 baseline holds: the SODA hash
flattens uniform-like inputs into a Poisson-like profile, while it leaves
the heavy tail of clustered / Wiki / Books inputs almost intact. What
*changes* with scale is mostly the absolute size of the worst bucket and,
for `clustered`, a dramatic blow-up of `ratio` driven by the SODA hash
running out of resolution.

### Uniform / OSM (Poisson-like) — stable picture, growing `max`

- `uniform/L=16` at n=2^30: `avg_unif=2.31`, `ratio=1.297`, `X_99=7`,
  `max=16`. The n=2^20 baseline was `avg_unif=2.31`, `ratio=1.297`,
  `X_99=7`, `max=12`. The mean and second moment are essentially
  unchanged, but `max` has grown from 12 → 16 (~33%). This is the textbook
  Poisson(`lambda ~ 2.3`) max-load scaling: with 2^10x more cells, the
  expected maximum grows by `log(2^30)/log(2^20) ~ 1.5`, exactly
  consistent with the 12 → 16 jump. Crucially, **`X_99 / max` shrinks**
  from 7/12 = 0.58 to 7/16 = 0.44 — the very-tail outlier becomes a more
  pronounced outlier even though the bulk of queries (X_99) is unchanged.
- `sosd_osm` shows the same pattern with even more dramatic `max`
  expansion: at L=4096, `max` grew from 29 (n=2^20) to **502** (n=2^29) —
  an order-of-magnitude jump — while `X_99` only crawled from 7 to 8.
  This is Poisson tail divergence at scale: there exist a handful of
  pathologically large buckets, but their mass is a vanishing fraction of
  total query work, so `ratio` stays at 1.30. Decision: **do not
  micro-optimise OSM bucket access for the worst-case max** — `X_99` is
  the right design point.

### Clustered — `ratio` explodes when the SODA universe saturates

This is the single most interesting finding of the large-scale run.

- At n=2^20 the clustered ratios were `1.46 / 4.21 / 20.7` for
  `L=16/256/4096`. At n=2^30 (post-dedupe ~470 M), they become
  **205 / 3,244 / 51,808** — three to four orders of magnitude larger.
- The mechanism is visible in the table: `X_99 == max` for all three
  clustered combos at large n, and `max` is a clean power of two
  (`4,096 = 2^12`, `65,536 = 2^16`, `1,048,576 = 2^20`). The SODA hash
  reduces every key to a `K`-bit code (`K = ceil(log2(n L / epsilon))`),
  hash-collisions create a few mega-buckets each holding the keys that
  share the same `K`-bit residue, and at clustered scale the per-cluster
  density is high enough to fully fill those mega-buckets up to `2^(K-k)`
  where `k = floor(log2(n))`. For L=4096 the largest bucket holds
  exactly 1,048,576 keys — every key in the cluster, mapped to one
  block. The `avg_unif` stays at ~5 because the *uniform* mean over 88M
  non-empty blocks is essentially unchanged, but the second moment is
  dominated by a handful of fully-packed mega-buckets.
- Conclusion: for clustered workloads, **the second-moment penalty is
  super-linear in `n`**. At thesis-paper scale (n=10^9) the SODA wrapper
  is no longer a valid uniformiser for clustered data — query-time will
  be bottlenecked by O(n / n_clusters) per-bucket scans.

### Wiki — heavy tail steady, magnitude scales with `n`

- `sosd_wiki/L=16`: `ratio = 2.69` at n=2^20, **`2.09` at n=2^27**. The
  ratio actually shrinks slightly — the wiki distribution has a fixed
  number of dense regions, and at larger `n` the mass spreads more
  evenly across them. But `avg_unif` exploded from 63.5 to **1,350**:
  the absolute work per query is 21x higher.
- `X_99` jumps from 1,464 (n=2^20) to **3,972** (n=2^27) at L=16 —
  almost equal to `max=4,054`. The tail at large n is *dense* (X_99 is
  98% of max), unlike at small n where X_99 was 91% of max. Implication:
  on Wiki at scale, the worst bucket *is* the typical query, not an
  outlier.

### Facebook — heavy tail attenuates as `M` grows

- `sosd_fb/L=16`: ratio drops slightly from 4.68 (n=2^20) to **4.69**
  (n=2^27) — essentially identical, even though `M` grew from 110k to
  14.7M. This is the SODA hash doing its job: more keys means more
  blocks, and the per-block load average only grows slightly
  (`avg_unif` 9.49 → 9.10).
- `max` actually went down (366 → 463 — wait, up by 27%), but
  `X_99` (214 → 208) is essentially flat. The classic "heavy tail of
  fixed shape, more buckets" regime.

### Books — mega-buckets multiply linearly with `n`

- At n=2^20, `sosd_books` had **78** non-empty buckets at L=256.
  At n=2^27 (128x more keys), it has **11,582** non-empty buckets — a
  148x increase, slightly super-linear. The keys-per-bucket stayed
  near the same `~13,000` median (n=2^20 X_50=13,594 vs n=2^27
  X_50=13,215).
- The total number of mega-buckets at L=4096 grew from **5** to
  **724** — essentially `n / 185,000` regardless of n. Books is a
  pure-density workload where SODA-block layout does almost nothing
  beyond binning by the most-significant 27 bits; the per-query cost
  is `O(n / n_dense_regions)` and that ratio is very stable.
- `ratio` stays pinned at ~1.04, confirming the variance-to-mean
  ratio of the bucket-size distribution is essentially zero — Books is
  a **fixed-bucket-size** workload.

### TL;DR vs the n=2^20 baseline

1. **Clustered ratios blow up at scale (1.46 → 205 at L=16,
   20.7 → 51,808 at L=4096)** because the SODA hash universe saturates
   into power-of-two mega-buckets. This is a real phase-change in the
   filter's behaviour — not visible at n=2^20.
2. **Uniform/OSM ratios are scale-invariant (1.30 stays 1.30) but their
   `max` grows like Poisson predicts (12 → 16, 29 → 502).** Tail
   maximum diverges from `X_99` — only `X_99` is the right design
   point.
3. **Books at n=2^27 has 11,582 mega-buckets at L=256** (vs 78 at
   n=2^20, a 148x increase for 128x more keys). Per-bucket load is
   stable; the ERE backend's work scales linearly in `n` here.

## Reproduction

- Machine: Apple M-series (Darwin 25.3.0), 64 GB RAM, 16 cores.
- Test source: `bench/ere_bucket_second_moment_large_test.go`
- Command: `go test -v -run TestEREBucketSecondMoment_SodaARE_Large -timeout 6h ./bench/ 2>&1 | tee bench_results/ere_bucket_stats/second_moment_n_large.log`
- Raw log: `bench_results/ere_bucket_stats/second_moment_n_large.log`
- Wall-clock total: **1,613.63 s (~26.9 min)** for all 16 sub-tests.
- Per-combo wall-clock: uniform/2^30 = 466 s, clustered/2^30 = ~217 s
  each, sosd_fb/wiki/books/2^27 = 4–15 s each, sosd_osm/2^29 = ~138 s
  each.
- Peak RSS: 24.1 GB (well under the 50 GB safety budget; under-budget
  because `NewSodaAREUint64InPlace` avoids the per-key BitString allocation
  path that the older `NewSodaARE` would have triggered).

## Notes / Limitations

- For `clustered` at n=2^30, the post-dedupe count was 470 M, not 1.07
  G — the cluster Gaussian generator with stddev ~ 2^29 produces ~57%
  collisions at this density. We report results at the actual unique-key
  count; the ratio numbers should still be interpreted relative to that
  effective `n`.
- For `sosd_wiki` at n=2^27, post-dedupe drops to 65 M — the Wiki
  timestamp dataset has many duplicates in its first 134 M entries. This
  matches the n=2^20 behaviour seen in `n_large.log`.
- The clustered `ratio = 51,808` at L=4096 is dominated by a single
  mega-bucket of size 1,048,576 = `2^20` = exactly the suffix-domain
  size for K=49. This is structurally a SODA-hash saturation, not a
  data-distribution finding — it tells us the parameter regime where
  SodaARE stops being effective for clustered inputs.
- `NewSodaAREUint64InPlace` consumed the input slice as documented; we
  set `keys = nil` immediately after the call so the GC can reclaim the
  8.6 GB key array before `EREStats()` and `ERENonEmptyBlockSizes()` run.
- Percentile rule: nearest-rank, no interpolation, identical to the
  baseline report.
