# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git & Submodule Rules

- **Submodule Updates:** After committing inside `Thesis/`, check when the root repository last updated the submodule pointer (`git log -1 --format=%ci -- Thesis`). If more than 24 hours have passed since that commit, stage and commit the updated pointer: `git add Thesis && git commit -m "chore: bump Thesis submodule"`. Otherwise leave it — the user will batch-update before pushing.
- **Commit Style:** Use conventional prefixes: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `bench:`. Scope in parens when helpful, e.g. `feat(bench):`, `fix(are_hybrid):`.
- **Commit Messages:** Do NOT add `Co-Authored-By` signatures.

## Repository Structure

This is a **two-module Go workspace** for benchmarking range emptiness filter structures:

- **`Thesis/`** — Git submodule (`module Thesis`). Contains all filter implementations, unit tests, and shared utilities. This is the research code.
- **Root** (`module Thesis-bench-industry`) — Benchmark harness comparing Thesis filters against industry baselines (Grafite, SNARF, SuRF) via CGo wrappers.

The root module depends on Thesis via `replace Thesis => ./Thesis` in go.mod.

## Build & Test

### Prerequisites

CGo wrappers require pre-built C++ libraries. Build once:
```bash
for lib in grafite snarf surf; do
  cd thirdparty/$lib && mkdir -p build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
  make -j$(nproc)
  cd ../../..
done
```

SOSD datasets (for real-world benchmarks): `bash bench/sosd_data/download.sh`

### Running tests

```bash
# Thesis submodule unit tests (no CGo needed)
cd Thesis && go test ./...

# Single package
cd Thesis && go test -v ./emptiness/are_hybrid/

# Single test
cd Thesis && go test -v -run TestHybridARE_NoFN_Clustered ./emptiness/are_hybrid/

# Industry benchmarks (requires CGo builds)
go test -v -run TestBuildThroughput -timeout 60m ./bench/

# Vet both modules
go vet ./bench/ && cd Thesis && go vet ./...
```

### Benchmark execution rules

- Run benchmark tests **one at a time**, never in parallel — they measure wall-clock performance.
- FPR/BPK tradeoff tests can parallelize internally; query-time and build-time benchmarks must not.
- Many benchmarks have long timeouts (30-60m). Use `-timeout` flag.

## Architecture

### Filter Implementations (Thesis/emptiness/)

Six ARE (Approximate Range Emptiness) packages, each implementing a different approach:

| Package | Key type | Constructor | Notes |
|---|---|---|---|
| `are_trunc` | `bits.BitString` | `NewApproximateRangeEmptiness(keys, eps)` | Prefix truncation, no rangeLen param |
| `are_adaptive` | `bits.BitString` | `NewAdaptiveARE(keys, rangeLen, eps, threshold)` | Adaptive with threshold parameter |
| `are_hybrid` | `bits.BitString` | `NewHybridARE(keys, rangeLen, eps)` | Cluster detection + per-segment ARE |
| `are_soda_hash` | `uint64` | `NewApproximateRangeEmptinessSoda(keys, rangeLen, eps)` | 2-universal hashing, FPR is distribution-independent |
| `are_pgm` | `uint64` | `NewPGMApproximateRangeEmptiness(keys, rangeLen, eps, pgmEps)` | CDF-based with PGM index, O(n²) build — guarded at N>1M |
| `are_bloom` | `uint64` | `NewBloomARE(keys, rangeLen, eps)` | Bloom filter baseline |

Plus `ere/` (Exact Range Emptiness — O(n log(U/n)) bits, O(1) query) and `ere_theoretical/` (theoretical baseline).

Key split: `bits.BitString`-based packages use trie representation (MSB-first ordering); `uint64`-based packages work with raw keys.

### Foundational Packages (Thesis/)

The ARE/ERE filters are built on these lower-level structures:
- `bits/` — `BitString` type (arbitrary-length binary keys with trie-consistent Compare, Prefix, Suffix, arithmetic)
- `succinct_bit_vector/` — Rank/Select in O(1) time
- `mmph/` — Monotone Minimal Perfect Hashing
- `trie/` — Z-Fast Trie, hollow tries
- `locators/` — Range locators (MMPH-based, Z-Fast Trie-based)

### CGo Wrappers (thirdparty/{grafite,snarf,surf,rosetta}/)

Each wraps a C++ range filter library. ~50-200ns overhead per CGo call.

**Key Bit-Width Logic:**
- **Native Data**: SOSD and NYC Taxi datasets are used in their native bit-width (up to 64-bit). No masking is applied by default to avoid artificial collisions and dataset shrinkage.
- **Synthetic Data**: Generators (`uniform`, `zipfian`, etc.) produce 60-bit keys by default to provide a consistent "universe" for comparison and ensure out-of-the-box compatibility with all filters.
- **Filter-Specific Limits**: SNARF internally masks to 60 bits as it cannot handle wider keys. Other filters (SODA, Rosetta, Grafite, SuRF) operate on the full bits provided.

### Shared Utilities (Thesis/testutils/)

- `distribution.go` — Key generators: `GenerateClusterDistribution`, uniform, spread, zipfian
- `plot.go` — SVG chart generation (`GeneratePerformanceSVG`, `GenerateTradeoffSVG`)
- `keys.go` — Cached benchmark key sets (`GetBenchKeys`)
- `metrics.go` — FPR/BPK measurement helpers
- `convert.go` — `TrieBS(uint64)` converts uint64 to BitString (trie-ready binary representation)

### Benchmark Tests (bench/)

- `comparison_test.go` — FPR vs BPK tradeoff for synthetic + SOSD distributions
- `throughput_test.go` — Build throughput (M keys/sec) across N sizes, with synthetic key generation
- `performance_test.go` — Build time per key, query time vs range length, scalability
- `sosd_test.go` — SOSD real-dataset benchmarks (Facebook, Wiki, OSM)
- `sosd_fb_dist_test.go` — Distribution histograms

Synthetic keys are pre-generated in SOSD binary format (`[uint64 count LE][count × uint64 keys LE]`) and saved to `bench/synthetic_data/`.

All plots are **SVG format**. Use log scales for asymptotic analysis (key counts on X, time/throughput on Y).

## Known Issues

- `are_hybrid` cluster detection (`detectClusters`) fails on sequential/evenly-spaced distributions — all gaps equal, elbow detector returns 0 clusters, falls back to plain Truncation ARE.
- `are_pgm` build is O(n²) due to PGM hull construction. Constructor returns error for N > 2^20.
- Benchmark outputs (`bench_results/plots/`, `bench_results/data/`) are gitignored — regenerate by running tests.
