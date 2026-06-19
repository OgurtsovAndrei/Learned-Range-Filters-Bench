# BBHash (MPHF) Optimizations for BitStrings

This module provides a fast Minimal Perfect Hash Function (MPHF) based on the BBHash algorithm, adapted for `BitString`
keys.

## Implementations

We have developed and compared three versions of the algorithm:

1. **Original (`/original`)**: A port of the `dgryski/go-boomphf` implementation. It stores each level as a separate
   slice of `uint64`.
2. **Flat (`/flat`)**: Optimizes memory by storing all levels in a single contiguous `uint64` array. Level offsets are
   calculated dynamically to save space.
3. **Inline (`/inline`)**: Further optimizes the flat structure by inlining level sizes into the main array and ensuring
   **8-word (512-bit) alignment** for every level's data.

## Performance Comparison (Gamma=2.0, Keys=262,144)

| Metric           | Original |   Flat   |  **Inline**  | Improvement (vs Orig) |
|:-----------------|:--------:|:--------:|:------------:|:---------------------:|
| **Build Time**   | 3.17 ms  | 3.09 ms  |   3.38 ms    |           -           |
| **Lookup Time**  | 15.00 ns | 11.34 ns | **10.92 ns** |      **-27.2%**       |
| **Bits per Key** |  2.250   |  2.125   |    2.127     |       **-5.5%**       |

*Benchmarks conducted on an Intel Core i9-13900H, pinned to a single core.*

## Why Inline Implementation?

The **Inline** implementation was selected as the preferred version for the following reasons:

1. **Maximum Lookup Speed**: It is the fastest version, achieving a lookup time of ~11ns. The 8-word alignment ensures
   that each level's bitvector starts on a clean cache line boundary, maximizing CPU cache efficiency.
2. **Cache Locality**: By inlining level sizes and keeping all data in one contiguous block, we minimize pointer chasing
   and extra memory fetches from separate metadata slices.
3. **Significant Space Savings**: Compared to the original version, it reduces memory consumption by **~6%** (saving
   roughly 0.12 bits per key).
4. **Hardware-Friendly**: It uses the `bits.RotateLeft32` instruction, which maps directly to a single CPU instruction (
   `ROR`/`ROL`), significantly speeding up the multi-level hashing process.


## Conclusion

The **Inline** implementation represents the optimal trade-off for high-performance applications, providing superior
lookup speeds and a highly compact memory footprint with negligible construction overhead.
