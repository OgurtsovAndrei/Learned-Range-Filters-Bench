# Analysis of O(N) Pathology in SUX SelectZero

This document describes the technical root cause of the performance degradation observed in the Grafite filter on clustered data distributions and the spatial-indexing fix implemented to resolve it.

## The Symptom
During benchmarking at $N=2^{24}$, Grafite's query latency on the `clustered` distribution was measured at **~13-88 microseconds** per query. In contrast, SODA and other ARE variants maintained **~100-200 nanoseconds**. 

Investigation revealed that the cost was dominated by the Elias-Fano `rank` operation, specifically the underlying `selectZero` call in the **SUX** library.

## Root Cause: Inventory-based Indexing vs. Gaps
Grafite uses the Elias-Fano structure from the SUX library. Elias-Fano stores sorted integers as a bitvector of "upper bits" and an array of "lower bits". Range queries rely on `rank` and `select` primitives on this bitvector.

### SUX Implementation: `SimpleSelectZeroHalf`
The SUX library implements `SelectZero` using an **inventory (value-based index)**:
1. It stores the physical bit-position of every $1024^{th}$ zero.
2. To find the $i^{th}$ zero, it jumps to the nearest stored position $p = \text{inventory}[i / 1024]$.
3. From position $p$, it performs a **linear scan** of the bitvector, word by word (64 bits at a time), using `popcount` to count the remaining zeros until it reaches the target.

### The Pathology
On **clustered distributions**, keys are densely packed in a small range of the universe. In Elias-Fano's bitvector, this creates a **massive run of ones** (up to $N$ ones) with no zeros in between.

If a query needs to find a zero located after such a cluster:
- The inventory jump only gets the pointer to the *start* of the cluster.
- The linear scan must then traverse millions of "all-ones" words.
- For a cluster of 16M keys, this involves scanning **250,000 words** ($16,000,000 / 64$).
- Even at 0.5 ns per word, this adds **125 microseconds** of latency to a single operation that should be $O(1)$ or $O(\log N)$.

## The Fix: Spatial Indexing (inspired by RSDic)
The pathology exists because the index is built on the *number of zeros* (value), not on the *bit-position* (space). Large gaps in the indexed value cause the search to fall back to a linear scan.

We implemented a **Spatial Index** (similar to the `rankBlocks` optimization in this project's `rsdic` implementation):

1. **`spatial_inventory`**: Added a secondary index that stores the cumulative number of zeros at fixed **bit intervals** (every 1024 bits).
2. **Accelerated Search**: In `selectZero`, if the gap between inventory entries is large, we perform a **binary search** on the `spatial_inventory`.
3. **Complexity reduction**: This allows us to "jump" over clusters of ones in $O(\log (\text{Universe} / 1024))$ time. The subsequent linear scan is now capped at a maximum of 1024 bits (16 words).

## Measured Impact
A micro-benchmark (`thirdparty/grafite/cpp/sux_pathology_test.cpp`) simulating a 16M-bit cluster gap showed:

| Implementation | Latency (per `selectZero`) | Complexity |
|----------------|----------------------------|------------|
| Original SUX  | **88,415.0 ns**            | $O(N)$     |
| Fixed SUX     | **30.9 ns**                | $O(\log N)$|

**End-to-end impact on Grafite ($N=2^{24}$, clustered):**
- **Before**: 64.6 seconds for the benchmark batch (~13,000 ns/op).
- **After**: 14.3 seconds for the benchmark batch (~300 ns/op).

This optimization proves that the Elias-Fano performance gap was not fundamental to the algorithm, but rather a specific implementation flaw in the SUX library's handling of non-uniform bit distributions.
