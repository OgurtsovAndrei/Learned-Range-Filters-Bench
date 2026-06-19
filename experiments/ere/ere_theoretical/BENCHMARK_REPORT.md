# ERE Variants — Benchmark Report

Comparison of three Exact Range Emptiness implementations on uniform random 60-bit keys.

**Setup:** macOS ARM64, Go 1.25, 3 runs averaged, 10K queries per (N, L) pair.

## Implementations

| Variant | In-block query | Key storage | Locator |
|---|---|---|---|
| **ERE** | Binary search on packed w-bit suffixes | Bit-packed uint64 array | None |
| **TheoreticalERE** | WeakPrefixSearch (O(1) per block) | Full BitString per block | Per-block CompactLocalExactRangeLocator |
| **GlobalERE** | WeakPrefixSearch (O(1) via global locator) | Full BitString array | Single global CompactLocalExactRangeLocator |

## Build Time (ns/key)

| Filter | N=65536 | N=262144 | N=1048576 | N=16777216 |
|---|---|---|---|---|
| ERE | **34** | **34** | **32** | **32** |
| TheoreticalERE | 11608 | 8539 | 8896 | 9221 |
| GlobalERE | 6533 | 4460 | 5384 | 12040 |

ERE build is ~250-350x faster than locator-based variants.
GlobalERE build scales worse at N=16M due to single large SHZFT + RangeLocator construction.

## Memory (bits/key)

| Filter | N=65536 | N=262144 | N=1048576 | N=16777216 |
|---|---|---|---|---|
| ERE | **47** | **45** | **43** | **39** |
| TheoreticalERE | 4852 | 4355 | 3745 | 3004 |
| GlobalERE | 143 | 143 | 141 | 143 |

Note: ERE counts packed suffixes in ByteSize. TheoreticalERE and GlobalERE count locator
overhead but not the full BitString key arrays (which add ~60 bits/key for 60-bit keys).

- TheoreticalERE: ~n/2 per-block locators with ~500B minimum each → catastrophic overhead
- GlobalERE: single locator → 30x more compact than TheoreticalERE, ~3x of ERE

## Query Time (ns/query)

### L=1

| Filter | N=65536 | N=262144 | N=1048576 | N=16777216 |
|---|---|---|---|---|
| ERE | **154** | **170** | **221** | **933** |
| TheoreticalERE | 945 | 598 | 720 | 2365 |
| GlobalERE | 1274 | 1287 | 1627 | 4390 |

### L=16

| Filter | N=65536 | N=262144 | N=1048576 | N=16777216 |
|---|---|---|---|---|
| ERE | **152** | **164** | **197** | **571** |
| TheoreticalERE | 508 | 475 | 500 | 2364 |
| GlobalERE | 1269 | 1288 | 1606 | 4761 |

### L=128

| Filter | N=65536 | N=262144 | N=1048576 | N=16777216 |
|---|---|---|---|---|
| ERE | **144** | **170** | **194** | **309** |
| TheoreticalERE | 454 | 506 | 490 | 1974 |
| GlobalERE | 1561 | 1528 | 1642 | 4467 |

### L=1024

| Filter | N=65536 | N=262144 | N=1048576 | N=16777216 |
|---|---|---|---|---|
| ERE | **184** | **186** | **199** | **322** |
| TheoreticalERE | 543 | 480 | 491 | 2105 |
| GlobalERE | 1307 | 1374 | 1724 | 4214 |

## Linear Scan Experiment

An additional `LinearIsEmpty` variant replaces binary search with inlined streaming
linear scan over packed suffixes. With `k = floor(log2(n))`, blocks average ~2 keys
(max 4-12), so binary search performs only 1-2 comparisons. Linear scan shows no
consistent speedup at these block sizes (±5-10%, within noise).

Linear scan would benefit from larger blocks (32-128 keys), which would require a
different `k` parameterization.

## Conclusions

1. **ERE with binary search is the optimal practical choice** — fastest build (32-34 ns/key),
   most compact (39-47 bpk), fastest queries (150-930 ns).

2. **TheoreticalERE** achieves O(1) in-block query via WeakPrefixSearch but pays
   catastrophic memory overhead (3000-4800 bpk) from per-block locators. Query time
   is 3-6x slower than ERE due to locator lookup overhead.

3. **GlobalERE** eliminates per-block locator overhead (143 bpk) but queries are
   even slower than TheoreticalERE (~2x) because one large SHZFT has deeper traversal.

4. **The bottleneck is not in-block search** — it's D1/D2 Rank/Select and block range
   computation. Binary search on 2-key blocks is effectively O(1) already.
