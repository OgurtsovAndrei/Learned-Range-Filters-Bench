# ERE Bucket Statistics — Results Summary

Bucket-occupancy data for the ERE inner layer of SodaARE, across distributions
and range lengths. All numbers measured on Apple M4 Max.

For full details see `second_moment_report.md` (n = 2²⁰) and
`second_moment_report_large.md` (large n — generated separately).

---

## 1. Bucket-fill statistics, n = 2²⁰ (1 048 576 keys)

ε = 0.01. `B` = total blocks, `M` = non-empty, `avg` over non-empty, `max` is the heaviest single bucket.

Source: `n_1M.log`.

| Distribution |    L |         B |       M |        avg |     max |
|--------------|-----:|----------:|--------:|-----------:|--------:|
| uniform      |   16 |   524 288 | 453 605 |       2.31 |      12 |
| uniform      |  256 |   524 288 | 453 104 |       2.31 |      12 |
| uniform      | 4096 | 1 048 576 | 662 451 |       1.58 |       9 |
| clustered    |   16 |   524 288 | 405 898 |       2.58 |      22 |
| clustered    |  256 |   524 288 | 170 408 |       6.15 |     137 |
| clustered    | 4096 | 1 048 576 | 115 097 |       9.11 |     864 |
| sosd_fb      |   16 | 1 048 576 | 110 470 |       9.49 |     366 |
| sosd_fb      |  256 | 1 048 576 |  11 190 |      93.71 |   1 294 |
| sosd_fb      | 4096 | 1 048 576 |     715 |   1 466.54 |   4 561 |
| sosd_wiki    |   16 |   524 288 |  15 572 |      63.50 |   1 613 |
| sosd_wiki    |  256 |   524 288 |   1 103 |     896.51 |   9 042 |
| sosd_wiki    | 4096 |   524 288 |      70 |  14 126.44 |  58 214 |
| sosd_osm     |   16 |   524 288 | 453 305 |       2.31 |      10 |
| sosd_osm     |  256 |   524 288 | 453 480 |       2.31 |      14 |
| sosd_osm     | 4096 |   524 288 | 451 126 |       2.32 |      29 |
| sosd_books   |   16 | 1 048 576 |   1 233 |     850.43 |     951 |
| sosd_books   |  256 | 1 048 576 |      78 |  13 443.28 |  13 924 |
| sosd_books   | 4096 | 1 048 576 |       5 | 209 715.20 | 218 469 |

---

## 2. Bucket-fill statistics, large n (full SOSD datasets)

`sosd_fb`, `sosd_wiki`, `sosd_books`: n = 2²⁷ (134 M keys).
`sosd_osm`: n = 2²⁹ (536 M keys).
ε = 0.01. Source: `n_large.log`.

| Distribution |    n |    L |           M |        avg |     max |
|--------------|-----:|-----:|------------:|-----------:|--------:|
| sosd_fb      | 134M |   16 |  14 751 796 |       9.10 |     463 |
| sosd_fb      | 134M |  256 |   1 540 315 |      87.14 |   1 589 |
| sosd_fb      | 134M | 4096 |      99 155 |   1 353.62 |   6 176 |
| sosd_wiki    | 65M¹ |   16 |      48 512 |   1 350.02 |   4 054 |
| sosd_wiki    | 65M¹ |  256 |       3 174 |  20 634.01 |  62 037 |
| sosd_wiki    | 65M¹ | 4096 |         199 | 329 107.27 | 956 498 |
| sosd_books   | 134M |   16 |     185 309 |     724.29 |     978 |
| sosd_books   | 134M |  256 |      11 582 |  11 588.48 |  14 039 |
| sosd_books   | 134M | 4096 |         724 | 185 383.60 | 218 469 |
| sosd_osm     | 536M |   16 | 231 999 951 |       2.31 |      35 |
| sosd_osm     | 536M |  256 | 231 629 855 |       2.32 |      85 |
| sosd_osm     | 536M | 4096 | 228 350 787 |       2.35 |     502 |

¹ wiki dataset deduplicates down to ~65M unique keys.

---

## 3. Time-weighted (key-weighted) bucket size, n = 2²⁰

`X_p` is the smallest bucket size such that buckets of size ≤ X_p collectively
hold at least p·n keys (key-weighted CDF over non-empty buckets, nearest-rank).

`second_moment = (1/n) · Σ_b k_b²` — the expected number of keys in the bucket
hit by a random query targeting real data (probability ∝ k_b/n).

`ratio = second_moment / avg` — how much heavier is a query-weighted draw vs
a uniform-bucket draw. Ratio close to 1 means buckets are balanced; large ratio
means a heavy tail dominates query work.

Source: `second_moment_report.md`.

| Distribution  |        L |    X_50 |    X_90 |    X_95 |    X_99 |     max |        avg | second moment | **ratio** |
|---------------|---------:|--------:|--------:|--------:|--------:|--------:|-----------:|--------------:|----------:|
| uniform       |       16 |       3 |       5 |       6 |       7 |      12 |       2.31 |          3.00 |      1.30 |
| uniform       |      256 |       3 |       5 |       6 |       7 |      12 |       2.31 |          3.00 |      1.30 |
| uniform       |     4096 |       2 |       3 |       4 |       5 |       9 |       1.58 |          2.00 |      1.27 |
| clustered     |       16 |       3 |       7 |       8 |      11 |      22 |       2.58 |          3.78 |      1.46 |
| clustered     |      256 |      20 |      57 |      64 |      92 |     137 |       6.15 |         25.88 |      4.21 |
| **clustered** | **4096** | **144** | **437** | **459** | **702** | **864** |   **9.11** |    **188.89** | **20.73** |
| sosd_fb       |       16 |      28 |     107 |     140 |     214 |     366 |       9.49 |         44.41 |      4.68 |
| sosd_fb       |      256 |     217 |     553 |     665 |     872 |   1 294 |      93.71 |        266.95 |      2.85 |
| sosd_fb       |     4096 |   1 847 |   3 069 |   3 468 |   4 124 |   4 561 |   1 466.54 |      1 935.72 |      1.32 |
| sosd_wiki     |       16 |     126 |     243 |     365 |   1 464 |   1 613 |      63.50 |        170.82 |      2.69 |
| sosd_wiki     |      256 |   1 950 |   3 101 |   5 227 |   7 307 |   9 042 |     896.51 |      2 039.10 |      2.27 |
| sosd_wiki     |     4096 |  32 284 |  55 444 |  58 214 |  58 214 |  58 214 |  14 126.44 |     29 198.82 |      2.07 |
| sosd_osm      |       16 |       3 |       5 |       6 |       7 |      10 |       2.31 |          3.00 |      1.30 |
| sosd_osm      |      256 |       3 |       5 |       6 |       7 |      14 |       2.31 |          3.00 |      1.30 |
| sosd_osm      |     4096 |       3 |       5 |       6 |       7 |      29 |       2.32 |          3.05 |      1.31 |
| sosd_books    |       16 |     851 |     886 |     901 |     918 |     951 |     850.43 |        851.42 |      1.00 |
| sosd_books    |      256 |  13 594 |  13 789 |  13 835 |  13 924 |  13 924 |  13 443.28 |     13 599.91 |      1.01 |
| sosd_books    |     4096 | 217 621 | 218 469 | 218 469 | 218 469 | 218 469 | 209 715.20 |    210 973.43 |      1.01 |

---

---

## 4. Time-weighted (key-weighted) bucket size, large n

Same metrics as Section 3, but at maximum-scale `n` per distribution
(uniform/clustered: 2³⁰; SOSD: dataset cap, see § 2). For uniform we
ran only `L = 16` (we already know `L` doesn't matter for Poisson-like
data). Source: `second_moment_report_large.md`.

| Distribution  |          n |        L |                    M |       X_50 |          X_90 |          X_95 |          X_99 |           max |        avg |  second moment |     **ratio** |
|---------------|-----------:|---------:|---------------------:|-----------:|--------------:|--------------:|--------------:|--------------:|-----------:|---------------:|--------------:|
| uniform       |     1.07 G |       16 |          464 213 367 |          3 |             5 |             6 |             7 |            16 |       2.31 |           3.00 |          1.30 |
| **clustered** | **471 M¹** |   **16** |       **89 508 024** |    **344** |     **4 065** |     **4 096** |     **4 096** |     **4 096** |   **5.26** |   **1 079.10** |    **205.16** |
| **clustered** | **471 M¹** |  **256** |       **88 570 682** |  **5 513** |    **65 026** |    **65 532** |    **65 536** |    **65 536** |   **5.32** |  **17 245.75** |  **3 244.20** |
| **clustered** | **471 M¹** | **4096** |       **88 504 831** | **88 200** | **1 039 504** | **1 048 517** | **1 048 576** | **1 048 576** |   **5.32** | **275 610.01** | **51 807.89** |
| sosd_fb       |      134 M |       16 |           14 751 796 |         27 |           103 |           136 |           208 |           463 |       9.10 |          42.63 |          4.69 |
| sosd_fb       |      134 M |      256 |            1 540 315 |        208 |           545 |           660 |           886 |         1 589 |      87.14 |         258.40 |          2.97 |
| sosd_fb       |      134 M |     4096 |               99 155 |      1 759 |         2 947 |         3 336 |         4 063 |         6 176 |   1 353.62 |       1 834.94 |          1.36 |
| sosd_wiki     |      65 M² |       16 |               48 512 |      3 240 |         3 843 |         3 906 |         3 972 |         4 054 |   1 350.02 |       2 822.73 |          2.09 |
| sosd_wiki     |      65 M² |      256 |                3 174 |     52 565 |        58 855 |        59 721 |        60 925 |        62 037 |  20 634.01 |      44 582.63 |          2.16 |
| sosd_wiki     |      65 M² |     4096 |                  199 |    847 135 |       938 217 |       939 041 |       956 498 |       956 498 | 329 107.27 |     708 934.97 |          2.15 |
| sosd_books    |      134 M |       16 |              185 309 |        812 |           875 |           888 |           911 |           978 |     724.29 |         753.62 |          1.04 |
| sosd_books    |      134 M |      256 | 134 217 728 → 11 582 |     13 215 |        13 670 |        13 728 |        13 826 |        14 039 |  11 588.48 |      12 009.80 |          1.04 |
| sosd_books    |      134 M |     4096 |                  724 |    211 319 |       217 641 |       217 893 |       218 144 |       218 469 | 185 383.60 |     192 094.62 |          1.04 |
| sosd_osm      |      537 M |       16 |          231 999 951 |          3 |             5 |             6 |             7 |            35 |       2.31 |           3.00 |          1.30 |
| sosd_osm      |      537 M |      256 |          231 629 855 |          3 |             5 |             6 |             7 |            85 |       2.32 |           3.02 |          1.30 |
| sosd_osm      |      537 M |     4096 |          228 350 787 |          3 |             5 |             6 |             8 |           502 |       2.35 |           3.14 |          1.34 |

¹ `clustered` target was 2³⁰; post-dedupe ~471 M unique keys due to
Gaussian-cluster collisions at high density. Numbers are at the actual
effective `n`.
² `sosd_wiki` deduplicates to ~65 M from the first 134 M entries.

Run wall-clock: 26.9 min total, peak RSS 24.1 GB.

---

## 5. Headline findings

### 4.1 Heavy-tail penalty: `clustered/L=4096` is the worst

`avg = 9` keys/bucket, but the **median key** lives in a bucket of **144** keys
(`X_50 = 144`), and `X_99 = 702`. **Ratio = 20.7×.**

→ Bucket-weighted average drastically understates query cost when the
distribution is non-uniform. A handful of heavy buckets absorb almost all query
mass.

### 4.2 SODA hash flattens uniform and OSM identically

`uniform` and `sosd_osm` are indistinguishable post-hash:

- avg ≈ 2.31, second_moment ≈ 3.00, ratio ≈ 1.30 across all L.
- `X_99 = 7`, max = 10–14 — distribution is tight, no heavy tail.
- This is exactly the SODA hash design point. OSM hashes uniform-like because
  S2 cell IDs are well-spread once SODA wraps them.

### 4.3 `sosd_books` collapses into mega-buckets

At L = 4096, only **5 non-empty buckets** survive — each holding ~200k keys.
Variance is tiny (`X_99 ≈ max ≈ 218k`), so `ratio ≈ 1.01`.
The problem here isn't a heavy tail; it's that the bucket *mean itself* is huge.
Per-query cost is bounded by raw bucket size, not variance.
Optimisation lever: change K/B, not de-clustering.

### 4.4 `sosd_wiki` shows a long heavy tail at every L

At L = 16: `X_99 = 1464`, but `X_95 = 365` — a **9× jump** between the 95th and
99th percentiles. A few buckets of size ~1.5k carry the last percentile of keys.

### 4.5 Bucket fill grows with L for non-uniform inputs

Larger L → larger w → larger block capacity 2^w → more of a cluster fits in one
bucket. For uniform/OSM (no clustering) L is irrelevant.

`clustered` max:    22 → 137 → 864          (L = 16 → 256 → 4096)
`sosd_fb`  max:    366 → 1 294 → 4 561
`sosd_wiki` max: 1 613 → 9 042 → 58 214
`sosd_books` max:  951 → 13 924 → 218 469

### 5.6 Scaling with n (large-n SOSD)

Going from n = 2²⁰ to dataset cap (~134 M for fb/wiki/books, 537 M for osm):

- `sosd_fb` max grows: 4 561 → 6 176 (n × 128, max × 1.4) — sub-linear, encouraging.
- `sosd_books` max essentially unchanged: 218 469 → 218 469. Cluster spans don't grow with `n`; the dataset has
  fixed-size dense regions.
- `sosd_osm` stays Poisson-like: max grows from 29 to 502 (17×) — still small relative to `n`; tail is light.
- `sosd_wiki` max blows up: 58 214 → 956 498 at L = 4096 — wiki has very few but very dense ranges, all keys collapse
  into them.

### 5.7 Phase change: clustered ratio explodes at large n

**Single most striking finding of the large-n run.**

At n = 2²⁰ the `clustered` ratios were `1.46 / 4.21 / 20.7` for L = 16/256/4096. At n ≈ 471 M they become **205 / 3
244 / 51 808** — **three to four orders of magnitude larger**.

Mechanism: at large `n` the SODA hash universe **saturates**. The largest bucket equals exactly
`2^(K - floor(log2 n))` — a clean power of 2 (4 096 = 2¹², 65 536 = 2¹⁶, 1 048 576 = 2²⁰), and `X_99 == max`. Hash
collisions create a few mega-buckets that fully fill up to capacity. `avg_unif` stays at ~5 because the *uniform* mean
over millions of non-empty blocks is unchanged, but the second moment is dominated by the saturated mega-buckets.

**Implication for thesis text**: at production scale (n ≈ 10⁹) the SODA wrapper is no longer a valid uniformiser for
clustered data. Query-time becomes bottlenecked by `O(n / n_clusters)` per-bucket scans. This is a real phase change,
not visible at thesis-default n = 10⁶.

### 5.8 X_99 vs max divergence at scale (uniform/OSM)

At small `n` the Poisson tail outlier (`max`) is close to `X_99`. At large `n` the tail becomes a genuine outlier:

- uniform: X_99/max = 7/12 = 0.58 (n=2²⁰) → 7/16 = 0.44 (n=2³⁰)
- sosd_osm/L=4096: X_99/max = 7/29 = 0.24 (n=2²⁰) → **8/502 = 0.016** (n=2²⁹)

X_99 stays essentially constant; max diverges. **`X_99` is the right design point at scale, not `max`** —
micro-optimising for max would optimise for a vanishing query mass.

### 5.9 Books: pure-density workload

At L = 256, `sosd_books` has 78 mega-buckets at n=2²⁰ → **11 582 at n=2²⁷** (148× for 128× more keys). Per-bucket size
stays constant at ~13 000 keys. `ratio` pinned at ~1.04 — variance-to-mean is essentially zero. ERE work scales linearly
in `n`; the lever for optimisation is `K`/`B`, not de-clustering.

---

## 6. Files

- `n_1M.log`                      — bucket stats at n = 2²⁰
- `n_large.log`                   — bucket stats at large n (SOSD only)
- `second_moment_n1M.log`         — raw log for the n = 2²⁰ second-moment + X_p run
- `second_moment_n_large.log`     — raw log for large-n second-moment run (in flight)
- `second_moment_report.md`       — full report at n = 2²⁰ with metric definitions
- `second_moment_report_large.md` — full report at large n (in flight)
- `bucket_search_crossover.log`   — separate experiment: linear vs binary search crossover
- `bucket_search_dense.log`       — separate experiment: binary search on dense buckets
- `ere_query_latency.log`         — separate experiment: end-to-end ERE query latency
