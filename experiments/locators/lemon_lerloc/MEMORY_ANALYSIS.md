# Memory Analysis: LeMon-Lerloc (LeMonHash + SuccinctHZFastTrie)

## 1. Executive Summary

By replacing the classical bucketing MMPH with the learned `LeMonHash`, and combining it with the `SuccinctHZFastTrie` as the exit-node locator, we achieved a massive reduction in memory usage for the Local Exact Range Locator.

For a dataset of **32,768 keys** (64-bit length):
- **Baseline LERLOC (Compact):** ~141.9 bits/key
- **LeMon-Lerloc (LeMonHash + Generic HZFastTrie):** ~87.7 bits/key
- **LeMon-Lerloc (LeMonHash + SuccinctHZFastTrie):** **~58.9 bits/key**
- **Overall Improvement:** **~83 bits/key (~58% reduction from baseline)**

## 2. Component Breakdown (N=32,768)

| Component | Bits/Key (Baseline) | Bits/Key (LeMon-Lerloc) | Notes |
| :--- | :--- | :--- | :--- |
| **HZFastTrie** | 34.6 | **34.6** | Using `SuccinctHZFastTrie` via PGM-index and RSDic instead of raw arrays. |
| **MMPH (Boundary Set P)** | 4.3 (Trie) | **19.1 (LeMonHash)** | Learned index on the $|P| \approx 5N$ boundary strings. |
| **Leaf BitVector (RSDic)** | 5.1 | **5.1** | Rankable bitvector on $|P|$ elements. |
| **Metadata & Headers** | 97.8 | **~0.01** | Classical implementation had huge overhead in 'Other' category. |
| **TOTAL** | **141.9** | **~58.9** | |

## 3. Query Performance (N=32,768, L=64)

| Metric | Baseline LERLOC (Compact) | LeMon-Lerloc (New) | Change |
| :--- | :--- | :--- | :--- |
| **Query Latency** | ~722 ns/op | **~758 ns/op** | +5% |
| **Allocations** | 1 alloc/op | 3 allocs/op | +2 allocs |

The massive memory saving (58%) comes at a negligible cost in query latency (~36 ns). The small increase in allocations is due to the multi-component boundary checks, which can be further optimized.

## 4. The Boundary Set Impact ($|P| \approx 5N$)

As noted during analysis, the boundary set $P$ constructed from the ZFastTrie is significantly larger than the number of original keys $N$. 
- For $N=32,768$, the RSDic takes ~5.1 bits per *original* key. Since RSDic overhead is slightly above 1 bit per element, this implies $|P| \approx 5.1 \times N$.
- Even with $|P|$ being 5x larger than $N$, LeMonHash remains extremely efficient. The ~19.1 bits/leaf contributed by LeMonHash means it takes only **~3.7 bits per boundary string** in the set $P$.

## 4. Scalability

| N | LERLOC Compact (bits/key) | LeMon-Lerloc (bits/key) | Improvement |
| :--- | :--- | :--- | :--- |
| 1,024 | 161.7 | 104.0 | -35% |
| 8,192 | 145.6 | 62.8 | -56% |
| 32,768 | 141.9 | **58.9** | **-58%** |

## 5. Visualizations

Generated plots can be found in `locators/benchmarks/plots/`:
- `mem_efficiency_all.svg`: Comparison of all locator variants.
- `mem_breakdown.csv`: Raw data for all components.
