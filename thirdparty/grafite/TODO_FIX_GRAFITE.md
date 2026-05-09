# TODO: Fix Grafite Pathology & Forking Plan

## Findings
During benchmarking on clustered distributions (e.g., `clustered` synthetic data and SOSD OSM), Grafite query latency was found to degrade to **~13-88 microseconds** per query, while SODA/ARE maintained **~100-200 nanoseconds**.

### Root Cause: SUX SelectZero Pathology
Grafite depends on the **SUX** library for Elias-Fano encoding. The `SimpleSelectZeroHalf` implementation uses an inventory-based index (storing positions of every 1024th zero). 
- **Pathology**: When a large cluster of keys (e.g., 16M keys) is stored, it creates a massive run of ones in the Elias-Fano bitvector.
- **Result**: Finding the "next zero" after such a cluster triggers a **linear scan** of millions of bits using `popcount` on 64-bit words.
- **Complexity**: $O(N)$ instead of $O(\log N)$ or $O(1)$.

## Implemented Fixes (Local)
1. **Spatial Index in SUX**: Added `spatial_inventory` to `SimpleSelectZeroHalf.hpp` (counting zeros every 1024 bits) + binary search.
2. **Binary Search in Elias-Fano**: Replaced linear scan in `EliasFano.hpp::rank` with an optimized `rankv2` (binary search).
3. **Bug Fix**: Fixed `unsigned` underflow in `rankv2` logic.

**Measured Speedup**: `SelectZero` micro-operation accelerated **~2800x** (88,000 ns -> 30 ns). Clustered benchmark for $N=2^{24}$ accelerated **~30x** end-to-end.

## Future Plan: "Two Grafites" Comparison
To highlight this discovery and fix in the Thesis, we want to show both the original and the fixed versions on the same plots.

### Step 1: Create a Fork
- Fork `https://github.com/marcocosta97/grafite` to a personal/project account.
- Apply the fixes discovered here (SUX and Grafite headers).

### Step 2: Namespace Isolation (Avoid Linker Collisions)
Since we want to link both original and fixed versions into the same Go binary, we must rename C++ symbols in the fork:
- `namespace grafite` -> `namespace grafite_fixed`
- `namespace sux` -> `namespace sux_fixed`
- Prefix C-wrapper functions: `grafite_new` -> `grafite_fixed_new`, etc.

### Step 3: Multi-Submodule Integration
- Add original repo as `thirdparty/grafite`.
- Add the fork as `thirdparty/grafite_fixed`.
- Register both in `bench/b6_latency_test.go` as separate filters:
  - `Grafite (Original)`
  - `Grafite (Fixed)`

---
*Created on 2026-05-09*
