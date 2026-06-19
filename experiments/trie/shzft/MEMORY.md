# Memory Analysis & Reduction Strategies: Succinct Z-Fast Trie (SHZFT)

## 1. Executive Summary

This document investigates the memory footprint of the `SuccinctHZFastTrie` (SHZFT) independently of the `lerloc` module.

There was a prior hypothesis that SHZFT's high memory consumption (~34.6 bits/key) was an artifact of `lerloc` passing the expanded boundary set $P$ ($|P| \approx 5N$) to the trie. **Independent testing has disproven this hypothesis.** The `lerloc` module passes the pure original key set $N$ to SHZFT. 

The expansion happens entirely **internally** due to the mathematical requirements of the "Fat Binary Search" algorithm, which generates "Pseudo-descriptors" to map intermediate unbranched paths to $\infty$.

## 2. Independent Memory Breakdown ($N=32,768, L=64$)

A pure test was run inserting exactly 32,768 random keys (length 64) directly into a standalone `SuccinctHZFastTrie`. The results perfectly match the `lerloc` benchmarks, confirming the overhead is intrinsic to SHZFT.

| Metric | Value | Per-Key Cost |
| :--- | :--- | :--- |
| **Total Keys ($N$)** | 32,768 | - |
| **Total Internal Entries** | 179,298 | **5.47 entries / key** |
| - True Descriptors | 65,535 | 2.0 entries / key |
| - Pseudo-descriptors | 113,763 | 3.47 entries / key |
| **Total Memory Cost** | 141,801 Bytes | **34.62 bits / key** |

### Component Level Breakdown
The 34.62 bits/key are distributed among the three internal succinct data structures:

1. **`mph` (BoomPHF): ~19.69 bits/key**
   - Indexes all 179,298 entries.
   - Cost: $\approx 3.6$ bits per entry.
2. **`deltas_array`: ~8.00 bits/key**
   - Bit-packed array storing delta lengths (`extentLen - descriptorLen`) only for the 65,535 true descriptors.
   - Cost: Exactly 4 bits per true descriptor (`deltaBits = 4`).
3. **`shzft_bv` (RSDic Bitvector): ~6.92 bits/key**
   - A bitvector of length 179,298 marking `1` for true descriptors and `0` for pseudo.
   - Cost: $\approx 1.26$ bits per entry (0.26 bits of rank/select overhead).

## 3. Strategies for Memory Reduction

Since the 5.47x entry expansion is algorithmically mandatory to support Fat Binary Search, memory reduction must focus on compressing the three underlying data structures.

### A. Bitvector Overhead Reduction (High ROI, Low Effort)
Currently, SHZFT uses `rsdic.RSDic` which carries a ~26% metadata overhead to support both `Rank()` and `Select()` queries.
- **Observation:** `shzft.go` **never** calls `Select()`. It only requires `Bit(idx)` and `Rank(idx, true)`.
- **Action:** Replace `rsdic` with the custom `SuccinctBitVector` (referenced in Roadmap). This specialized vector is optimized for $N + o(N)$ space and only supports Rank, stripping away unnecessary `Select` metadata and potentially saving **~1-1.5 bits/key**.

### B. MPH Replacement (Highest ROI, Medium Effort)
The `boomphf.H` minimal perfect hash function consumes the majority of the memory (~20 bits/key) because it requires ~3.6 bits per entry.
- **Observation:** The keys inserted into the MPH are trie prefixes. These prefixes can be extracted and sorted lexicographically.
- **Action:** Replace `boomphf` with a learned monotone hash (`LeMonHash`). Learned monotone hashing has been proven in the `lemon_lerloc` module to achieve $\approx 1.5 - 2$ bits per entry. Applying this to SHZFT would reduce the MPH cost from ~19.69 bits/key down to **~8-10 bits/key**, achieving a massive **30% reduction** in total SHZFT size.

### C. Delta Encoding Optimization (Low ROI, High Effort)
The `deltas_array` is uniformly packed using a global `deltaBits` (determined by `maxDelta`).
- **Observation:** In the test, `deltaBits = 4`. This is already incredibly small ($4 \times 2N / N = 8$ bits/key).
- **Action:** While variable-byte encoding or Elias-Fano could theoretically compress this further (preventing short deltas from paying for the `maxDelta`), the absolute savings would be minimal (< 2 bits/key). Given the complexity of implementing fast random-access Elias-Fano, this should be the lowest priority.

## 4. Conclusion
The most effective path to sub-25 bits/key for SHZFT is:
1. Swap `rsdic` for `SuccinctBitVector`.
2. Swap `boomphf` for `LeMonHash`.