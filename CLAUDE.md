# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. For comprehensive multi-agent details, see [AGENTS.md](file:///Users/andrei.ogurtsov/Thesis-Bench-industry/AGENTS.md) and Cursor rules under `.cursor/rules/`.

## Git & Submodule Rules

- **Submodule Updates:** After committing inside `Thesis/`, check when the root repository last updated the submodule pointer (`git log -1 --format=%ci -- Thesis`). If more than 24 hours have passed since that commit, stage and commit the updated pointer: `git add Thesis && git commit -m "chore: bump Thesis submodule"`. Otherwise leave it — the user will batch-update before pushing.
- **Commit Style:** Use conventional prefixes: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `bench:`. Scope in parens when helpful, e.g. `feat(bench):`, `fix(are_hybrid):`.
- **Commit Messages:** Do NOT add `Co-Authored-By` signatures.

## Repository Structure

This is a **two-module Go workspace** for benchmarking range emptiness filter structures:

- **`Thesis/`** — Git submodule (`module Thesis`). Contains all filter implementations, unit tests, and shared utilities. This is the research code.
  - **Dissertation Text**: Located in [Thesis/text/](file:///Users/andrei.ogurtsov/Thesis-Bench-industry/Thesis/text/).
- **Root** (`module Thesis-bench-industry`) — Benchmark harness comparing Thesis filters against industry baselines (Grafite, SNARF, SuRF, Rosetta) via CGo wrappers.

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

Always prefer the **unified Makefile targets** in the project root:

```bash
# Run fast unit tests (both root and submodule, completes in < 1s)
make test

# Run all tests, including heavy benchmarks and large dataset sweeps
make test-all

# Vet and lint syntax across root and submodule
make vet

# Tidy dependencies in both modules
make tidy
```

### Benchmark execution & Consistency rules

- **Exclusion Tag**: Heavy performance tests, tradeoff sweeps, and cache sweeps MUST start with the `//go:build heavy` build tag on line 1 so they are excluded from the fast default `make test` run.
- **CPU Pinning**: Pin benchmarks to a single fixed core (`taskset -c 0`) to achieve consistent, reproducible latency and throughput figures.
- **Results Caching**: Results are stored under `bench_results/data/`.
- **Plot-Only Mode**: Set `PLOT_ONLY=1` to skip running benchmarks and instantly redraw plots from cached JSON files.
- **Fast Operating Point Finder (`bisect` runner)**: Use [bench/bisect_runner_test.go](file:///Users/andrei.ogurtsov/Thesis-Bench-industry/bench/bisect_runner_test.go) (`TestB6BisectOperatingPoint`) to run a binary search for the exact BPK needed to achieve a target FPR $\epsilon$ (runs ~3× faster than full grid sweeps).

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

### Visualizations (`plotter`)

Centralized in [Thesis/testutils/plot.go](file:///Users/andrei.ogurtsov/Thesis-Bench-industry/Thesis/testutils/plot.go). Generates scalability and tradeoff curves in **SVG format only**. Uses logarithmic scaling with Unicode superscript decade ticks (e.g. `10⁻⁷`), a dashed observation limit line (representing 0 FP observed), and a secondary threshold sub-chart indicating operating BPK points for $10^{-2}$ and $10^{-3}$ FPR.

## Known Issues

- `are_hybrid` cluster detection (`detectClusters`) fails on sequential/evenly-spaced distributions — all gaps equal, elbow detector returns 0 clusters, falls back to plain Truncation ARE.
- `are_pgm` build is O(n²) due to PGM hull construction. Constructor returns error for N > 2^20.
- Benchmark outputs (`bench_results/plots/`, `bench_results/data/`) are gitignored — regenerate by running tests.
