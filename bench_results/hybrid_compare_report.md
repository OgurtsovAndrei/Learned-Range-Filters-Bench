# Hybrid ARE Comparison Report

**Date:** 2026-03-20
**Parameters:** L=128, N=262144 (60-bit keys), 3 seeds, `generateSmartQueries` for SOSD
**Theoretical lower bound:** BPK = log2(L/eps) = log2(128/0.001) = 17.0

## Algorithms

| Algorithm    | Package           | Segmentation                          | Fallback                       | Complexity |
|--------------|-------------------|---------------------------------------|--------------------------------|------------|
| Hybrid       | `are_hybrid`      | Gap-percentile (elbow)                | Trunc only                     | O(n log n) |
| Scan-ARE     | `are_hybrid_scan` | 1D DBSCAN                             | Trunc or SODA (Auto policy)    | O(n log n) |
| Greedy+Merge | `are_greedy_scan` | Spread-threshold + hierarchical merge | Trunc (for SODA-mode clusters) | O(n)       |
| DP-Optimal   | `are_dp_scan`     | Dynamic programming (min total bits)  | None                           | O(n^2)     |

## BPK at FPR <= 0.001 (L=128, N=262144)

| Distribution | Hybrid   | Scan-ARE | Greedy+Merge | Best        | Theoretical |
|--------------|----------|----------|--------------|-------------|-------------|
| sosd_books   | 5.0      | 5.0      | **4.3**      | Greedy      | 17.0        |
| zipfian      | 9.7      | **6.9**  | 9.7          | Scan        | 17.0        |
| sosd_wiki    | **10.1** | 13.1     | **10.1**     | Tie         | 17.0        |
| sosd_fb      | 13.0     | 11.7     | **11.2**     | Greedy      | 17.0        |
| uniform      | **13.0** | 45.0     | **13.0**     | Tie         | 17.0        |
| temporal     | 13.1     | 13.1     | **13.0**     | ~Tie        | 17.0        |
| spread       | **0.0**  | **0.0**  | 17.0         | Hybrid/Scan | 17.0        |
| clustered    | 15.2     | 16.3     | **14.5**     | Greedy      | 17.0        |
| sosd_osm     | >35      | 34.5     | **32.9**     | Greedy      | 17.0        |

## BPK at FPR = 0 (exact zero)

| Distribution | Hybrid   | Scan-ARE | Greedy+Merge | Best   |
|--------------|----------|----------|--------------|--------|
| sosd_books   | 5.0      | 5.0      | **4.3**      | Greedy |
| zipfian      | 9.7      | **6.9**  | 9.7          | Scan   |
| sosd_wiki    | **10.1** | 13.1     | **10.1**     | Tie    |
| sosd_fb      | 13.0     | 11.7     | **11.2**     | Greedy |
| temporal     | **13.1** | **13.1** | 13.2         | Tie    |
| clustered    | 18.2     | 16.3     | **15.1**     | Greedy |
| uniform      | **25.0** | 45.0     | **25.0**     | Tie    |
| spread       | **0.0**  | 25.0     | 25.0         | Hybrid |
| sosd_osm     | >35      | 34.5     | 35.2         | Scan   |

## Greedy+Merge Segmentation Quality (K=20, 10K clustered keys)

| Metric   | Greedy (raw) | Greedy+Merge | Greedy+Merge+Fallback    | DP-Optimal |
|----------|--------------|--------------|--------------------------|------------|
| Clusters | 2449         | 50           | 48 exact + 7622 fallback | 47         |
| BPK      | 48.06        | 13.17        | **6.57**                 | 13.08      |

## Build Time (ns/key)

| Distribution | Hybrid | Scan-ARE | Greedy+Merge          |
|--------------|--------|----------|-----------------------|
| Clustered    | ~100   | ~150     | ~200                  |
| Uniform      | ~90    | ~150     | **86** (all fallback) |
| Spread       | ~90    | ~150     | **86** (all fallback) |
| SOSD OSM     | ~100   | ~150     | ~300                  |

## Key Findings

1. **Greedy+Merge is the best or tied-best on 7/9 distributions** for FPR<=0.001
2. **Trunc fallback for SODA-mode clusters** fixes the uniform regression (21 BPK -> 13 BPK)
3. **Greedy+Merge ~= 99.3% of DP-optimal** at O(n) instead of O(n^2)
4. **Merge optimization** (index refs instead of key copying) reduced uniform build time from 205us/key to 86ns/key (
   2400x speedup)
5. **OSM remains hard** for all filters (>32 BPK vs theoretical 17). Root cause: near-key smart queries + large spread.
   SODA-based filters (Scan-ARE) do better here.
6. **Spread distribution**: Hybrid/Scan win (exact mode via SODA hash collisions). Greedy+Merge falls back to trunc
   which costs more.

## Weak Spots

- **Spread**: Greedy+Merge sends everything to trunc (17 BPK) while Hybrid/Scan achieve 0 BPK via SODA hash collisions
- **OSM**: All filters struggle. Near-key queries cause phantom FPs in trunc mode. SODA fallback helps but costs log2(L)
  extra bits
- **Zipfian**: Scan-ARE's DBSCAN finds better clusters than Greedy's spread-threshold for this distribution

## Files

- Plots: `bench_results/plots/hybrid_compare/N262144/{dist}/L{16,128,1024}.svg`
- Data: `bench_results/data/hybrid_compare/N262144/{dist}/L{16,128,1024}.json`
- Build time plots: `bench_results/plots/hybrid_compare/build_time/{dist}.svg`
