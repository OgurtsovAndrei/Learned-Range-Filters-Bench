# LeMonHash Performance Analysis (CGO)

## 1. Benchmarks Summary (Apple M4 Max)

The following metrics represent the "Zero-Allocation" performance where data preparation is done outside the query loop to measure pure CGO and C++ overhead.

| Metric | 1k keys | 1M keys | 16M keys |
| :--- | :--- | :--- | :--- |
| **Space (bits/key)** | 29.2 | 3.34 | **3.32** |
| **Rank (Single, Zero-Alloc)** | ~88 ns | ~175 ns | ~200 ns |
| **RankPair (2 keys/call)** | **~65 ns** | **~162 ns** | ~180 ns |
| **RankBatch (1024 keys)** | ~190 ns | ~290 ns | ~310 ns |

## 2. Evolution of Performance: From 600ns to 88ns

Initially, a single `Rank(key)` call took ~600ns. We reduced this by **85%** through structural optimizations.

### Where the time went (Overhead Breakdown):

| Component | Initial Latency | Current Latency | Optimization |
| :--- | :--- | :--- | :--- |
| **Go Allocation** | ~350 ns | **0 ns** | Avoided `key.Data()` allocation (benchmarked via `rankRaw`) |
| **C Allocation** | ~150 ns | **0 ns** | Removed `C.CBytes` and `C.free` (using `unsafe.Pointer`) |
| **CGO Transition** | ~70 ns | ~70 ns | Fixed cost of context switching |
| **C++ Logic** | ~30 ns | ~18 ns | Pure mathematical lookup + PGM search |
| **TOTAL** | **~600 ns** | **~88 ns** | |

## 3. Key Optimization Techniques

### A. Zero-Copy CGO
The most significant gain came from switching from `C.CBytes` to direct `unsafe.Pointer` passing. 
- **Old way:** `malloc` in C heap -> `memcpy` Go data to C -> call C -> `free` in C.
- **New way:** Pass pointer to existing Go slice directly to C. Go's runtime guarantees the memory is pinned during the call duration.

### B. Implicit vs. Manual Pinning
The "Batch Paradox" showed that `RankBatch` (1024 keys) is slower than single `Rank` calls.
- **Manual Pinning (`runtime.Pinner`):** Used in `RankBatch` to pin 1024 separate slices. The overhead of the Go runtime managing these pins exceeds the cost of 1024 CGO transitions.
- **Implicit Pinning:** Used in `Rank` and `RankPair`. Go pins arguments passed directly to CGO functions at the compiler level with near-zero overhead. 
- **The Result:** `RankPair` (2 keys per call) is the fastest method because it amortizes the transition cost across two queries without incurring the `runtime.Pinner` penalty.

## 4. Path to Production Excellence

To achieve these ~88ns speeds in a real application (not just benchmarks), we must eliminate the allocation in the `bits` module:

1. **`BitString` Refactoring:** Currently, `key.Data()` creates a new `[]byte` slice. We need to transition from Little-Endian `[]uint64` to **Big-Endian `[]byte`** storage. This will enable true zero-copy CGO.
    * See [Detailed Refactoring Plan](../../bits/BITSTRING_REFACTOR_PLAN.md) for more info.
2. **Flat Buffers for Batching:** For high-throughput batching, keys should be stored in a single contiguous `[]byte`. This would allow pinning 1000s of keys with a **single** `Pin()` call, likely bringing batch latency down to **< 50ns/key**.
