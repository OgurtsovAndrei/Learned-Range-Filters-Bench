# ERE Bucket Second Moment & Key-Weighted Tail (Query-Weighted Bucket Size)

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
bucket-size distribution. These are computed by sorting `sizes = [k_b for b in
non_empty_buckets]` ascending, summing into a cumulative `cum[i]` (so
`cum[-1] = n`), and picking the smallest `i` with `cum[i] >= ceil(p * n)`;
then `X_p = sizes[i]`. Empty buckets are excluded. Equivalently:

> `X_p` is the smallest bucket size such that buckets of size `<= X_p` hold
> at least `p * n` keys (key-weighted CDF over non-empty buckets).

This is the relevant quantity for query-time tail analysis: it answers
"what fraction of *keys* (and therefore queries that hit data) live in
buckets of size `<= X_p`", not "what fraction of buckets are smaller
than X". We pick the nearest-rank rule (no interpolation) for consistency.

## Setup

- Filter: `are_soda_hash.SodaARE` (epsilon = 0.01)
- Backend: default ERE (classic two-level RSDic)
- Keys per run: `n = 2^20` (1,048,576) — for SOSD datasets the first 1M keys
  are loaded, sorted, and de-duplicated; `M` is the number of non-empty
  blocks after the SODA hash assigns keys to blocks.
- Range lengths: `L in {16, 256, 4096}`
- Distributions: `uniform`, `clustered`, `sosd_fb`, `sosd_wiki`, `sosd_osm`,
  `sosd_books`.

## Results

`X_p` is the smallest bucket size such that buckets of size `<= X_p` hold
at least `p * n` keys (key-weighted CDF over non-empty buckets, nearest
rank). `avg_unif = (Sum_b k_b) / M` is the bucket-weighted (uniform) mean.
`second_moment = (1 / n) * Sum_b k_b^2`. `ratio = second_moment / avg_unif`.

| Distribution | L    | B        | M       | X_50   | X_90   | X_95   | X_99   | max     | avg (non-empty) | second moment | ratio   |
|--------------|------|----------|---------|--------|--------|--------|--------|---------|-----------------|---------------|---------|
| uniform      | 16   | 524288   | 453605  | 3      | 5      | 6      | 7      | 12      | 2.3111          | 2.9980        | 1.2972  |
| uniform      | 256  | 524288   | 453104  | 3      | 5      | 6      | 7      | 12      | 2.3142          | 2.9995        | 1.2961  |
| uniform      | 4096 | 1048576  | 662451  | 2      | 3      | 4      | 5      | 9       | 1.5829          | 2.0033        | 1.2656  |
| clustered    | 16   | 524288   | 405898  | 3      | 7      | 8      | 11     | 22      | 2.5829          | 3.7809        | 1.4638  |
| clustered    | 256  | 524288   | 170408  | 20     | 57     | 64     | 92     | 137     | 6.1532          | 25.8798       | 4.2059  |
| clustered    | 4096 | 1048576  | 115097  | 144    | 437    | 459    | 702    | 864     | 9.1104          | 188.8897      | 20.7335 |
| sosd_fb      | 16   | 1048576  | 110470  | 28     | 107    | 140    | 214    | 366     | 9.4920          | 44.4121       | 4.6789  |
| sosd_fb      | 256  | 1048576  | 11190   | 217    | 553    | 665    | 872    | 1294    | 93.7065         | 266.9459      | 2.8487  |
| sosd_fb      | 4096 | 1048576  | 715     | 1847   | 3069   | 3468   | 4124   | 4561    | 1466.5399       | 1935.7217     | 1.3199  |
| sosd_wiki    | 16   | 524288   | 15572   | 126    | 243    | 365    | 1464   | 1613    | 63.5019         | 170.8196      | 2.6900  |
| sosd_wiki    | 256  | 524288   | 1103    | 1950   | 3101   | 5227   | 7307   | 9042    | 896.5104        | 2039.1040     | 2.2745  |
| sosd_wiki    | 4096 | 524288   | 70      | 32284  | 55444  | 58214  | 58214  | 58214   | 14126.4429      | 29198.8174    | 2.0670  |
| sosd_osm     | 16   | 524288   | 453305  | 3      | 5      | 6      | 7      | 10      | 2.3121          | 2.9989        | 1.2971  |
| sosd_osm     | 256  | 524288   | 453480  | 3      | 5      | 6      | 7      | 14      | 2.3122          | 3.0042        | 1.2993  |
| sosd_osm     | 4096 | 524288   | 451126  | 3      | 5      | 6      | 7      | 29      | 2.3243          | 3.0538        | 1.3138  |
| sosd_books   | 16   | 1048576  | 1233    | 851    | 886    | 901    | 918    | 951     | 850.4266        | 851.4200      | 1.0012  |
| sosd_books   | 256  | 1048576  | 78      | 13594  | 13789  | 13835  | 13924  | 13924   | 13443.2821      | 13599.9119    | 1.0117  |
| sosd_books   | 4096 | 1048576  | 5       | 217621 | 218469 | 218469 | 218469 | 218469  | 209715.2000     | 210973.4282   | 1.0060  |

## Interpretation

The `ratio` column captures the gap between a uniform-bucket assumption and
what a query that lands on real data actually sees. Together with the
key-weighted percentiles `X_50 ... X_99` and `max` it tells us about the
tail shape of bucket occupancy:

- **`uniform` and `sosd_osm` look like Poisson(`lambda ~ 2`).** With
  `M / B ~ 0.86`, `avg_unif ~ 2.31`, `second_moment ~ 3.0`, and `ratio ~ 1.30`
  across all L. The key-weighted tail is also tight: `X_50 = 3`, `X_99 = 7`,
  `max = 10-14`. This is exactly the SODA hash's design point — `X_99` is
  within a small constant of `max`. OSM hashes look uniform-like because the
  SODA wrapper hashes block IDs so intra-block structure is washed out.
- **`clustered` and `sosd_wiki` show a strong heavy-tail penalty.**
  `clustered/L=4096` has `avg_unif ~ 9.1` and `X_50 = 144` — the median
  query (key-weighted) lands in a bucket holding 144 keys, while the
  uniform-bucket average is only 9. The ratio of 20.7x is mechanical
  evidence: the heavy buckets (max = 864, X_99 = 702) absorb almost all
  the query mass. Wiki is similar: at `L=16`, `X_99 = 1464` is nine times
  `X_95 = 365` — a few buckets of size ~1.5k carry the last percentile of
  keys.
- **`sosd_books` and large-L `sosd_fb` collapse to a few near-equal
  mega-buckets.** With only 5-78 non-empty blocks for books at large L,
  every block holds tens to hundreds of thousands of keys; the variance in
  k_b is small relative to the mean, so `X_50, X_99, max` are within ~0.5%
  of each other and `ratio ~ 1.01`. The absolute cost is huge (`X_99 ~ 2 *
  10^5` for books at `L=4096`), but the heavy-tail penalty is small — the
  problem there is the average bucket size, not its variance.

The TL;DR: `ratio` is most actionable on distributions where the SODA hash
fails to flatten the input. For clustered and Wiki workloads, query-time
optimisation should target the worst buckets — they dominate the expected
search cost by an order of magnitude even though they are a minority of the
buckets, and the key-weighted percentiles confirm that the median *query*
already sits well above `avg_unif`.

## Reproduction

- Machine: Apple Silicon (Darwin 25.3.0)
- Command: `go test -v -run TestEREBucketSecondMoment_SodaARE -timeout 10m ./bench/`
- Test source: `bench/ere_bucket_second_moment_test.go`
- Raw log: `bench_results/ere_bucket_stats/second_moment_n1M.log`
- Wall-clock: ~5 s for all 18 sub-tests (no combo exceeded 0.5 s).

## Notes

- The metric extraction reuses the existing `SodaARE.EREStats()` path. The
  underlying `Stats` struct in `Thesis/emptiness/ere`,
  `Thesis/emptiness/ere_one_d`, and the `Thesis/emptiness/exact` adapter
  were extended with a new `SumSquaredKeys uint64` field; the second moment
  is then `SumSquaredKeys / n`.
- A new `NonEmptyBlockSizes() []int` method was added to both ERE
  back-ends and surfaced through `exact.NonEmptyBlockSizesOf` and
  `SodaARE.ERENonEmptyBlockSizes`. The test sorts that slice once and
  reuses it for all four percentiles.
- All previously-existing fields on `Stats` are unchanged; existing tests
  still compile and pass.
- Percentile rule: nearest-rank, no interpolation. We pick the smallest
  index `i` with `cum[i] >= ceil(p * n)`.
