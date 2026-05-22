# AI Onboarding & Project Standards Handbook (AGENTS.md)

Welcome! This document outlines the codebase architecture, benchmarking protocols, and development workflows established for this repository. Read this file to understand how to build, test, and contribute correctly.

---

## 1. Codebase Architecture

This project is structured as a **two-module Go workspace** designed to benchmark range emptiness filter structures:

1. **`Thesis/` (Git Submodule)**:
   - Contains all approximate/exact range emptiness implementations (`are_trunc`, `are_adaptive`, `are_hybrid`, `are_soda_hash`, `are_pgm`, `are_bloom`, `ere`, `ere_theoretical`).
   - Contains foundational succinct structures, locators, MMSH, bit-vector operations, and shared testing/plotting utilities.
   - **Dissertation Text**: Located in [Thesis/text/](file:///Users/andrei.ogurtsov/Thesis-Bench-industry/Thesis/text/) (drafts, plans, and technical writing).
2. **Root Workspace**:
   - Contains CGo benchmarking wrappers (`thirdparty/grafite`, `thirdparty/snarf`, `thirdparty/surf`, `thirdparty/rosetta`) and the main benchmarking harness (`bench/`).
   - The root module depends on the submodule via `replace Thesis => ./Thesis` in `go.mod`.

---

## 2. Quick Start: Build & Test Workflow

Always use the **unified root `Makefile`** to execute commands:

* **`make test`**: Runs only fast unit tests (completes in < 1s).
* **`make test-all`**: Runs all tests, including heavy performance tests, large SOSD dataset sweeps, and multi-hour tradeoff benchmarks (marked with the `//go:build heavy` tag).
* **`make vet`**: Standard Go syntax and lint verification across both the root module and the `Thesis/` submodule.
* **`make tidy`**: Cleans up and runs `go mod tidy` in both modules to keep dependencies healthy.

---

## 3. Benchmarking & Testing Protocols (`bench/`)

When working with or editing files under [bench/](file:///Users/andrei.ogurtsov/Thesis-Bench-industry/bench/), adhere to these strict rules:

### A. Performance Consistency & CPU Pinning
* **Wall-Clock Latency**: Benchmarks measure wall-clock latency (nanoseconds per query) and build throughput. CPU migrations, context switches, thermal throttling, and competing processes cause huge noise.
* **CPU Pinning**: To obtain consistent, reproducible results, benchmarks **MUST** be pinned to a single, fixed CPU core.
  - *Linux*: Run using `taskset -c 0 make test-all` or `taskset -c 0 go test ...`
  - *macOS*: Ensure the environment is stable, all heavy background apps are closed, and compile using native flags.
* **Serialization**: Run query-time and build-time benchmarks one-at-a-time (`B6_PARALLELISM=1`). Never parallelize latency benchmarks internally as it degrades wall-clock accuracy.

### B. Results Caching & Data Formats
* **JSON Cache**: Results are automatically saved to `bench_results/data/L<rangeLen>.json`.
* **Fast Plot-Only Runs**: If you only need to regenerate or refine SVG charts from existing runs without executing the full benchmark suite, set `PLOT_ONLY=1` before running tests (e.g. `PLOT_ONLY=1 go test -tags=heavy -run TestComparison ...`).

### C. Fast Operating-Point Finder: The `bisect` Runner
* Located in [bench/bisect_runner_test.go](file:///Users/andrei.ogurtsov/Thesis-Bench-industry/bench/bisect_runner_test.go).
* Performs a **binary search** over the parameter space (e.g. key threshold $K$) to find the exact Bits-Per-Key (BPK) needed to hit a target False Positive Rate ($\epsilon$), instead of performing a brute-force sweep.
* **Speedup**: Runs **~3× faster** than standard sweeps when only operating points are needed.
* **Tuning**: Configured via environment variables:
  - `B6_BISECT_EPS`: Target FPR (default `0.001`).
  - `B6_BISECT_ITERS`: Maximum binary search iterations (default `8`).
  - `B6_BISECT_TOL`: Binary search stopping tolerance (default `0.5`).

---

## 4. SVG Visualizations (`plotter`)

Centralized plotting utilities are located in [Thesis/testutils/plot.go](file:///Users/andrei.ogurtsov/Thesis-Bench-industry/Thesis/testutils/plot.go) (`GeneratePerformanceSVG` and `GenerateTradeoffSVG`).

All generated plots must follow these standards:
1. **SVG Format Only**: Scalable, vector-based graphics for infinite quality at small size.
2. **Logarithmic Axis Scales**: Used to show asymptotic behavior ($O(N)$ vs $O(\log N)$).
3. **Unicode Pow-10 Exponents**: Super-script notation (e.g. `10⁻⁷` instead of `10^-7`) is programmatically formatted to avoid Y-axis label overlaps.
4. **Dashed Measurement Floor**: Represents the limit of observable False Positives based on sample size (e.g., "0 FP observed").
5. **Threshold Indicators Sub-Chart**: A secondary panel rendering BPK tick lines where each filter crosses $10^{-2}$ and $10^{-3}$ FPR.

---

## 5. Git & Submodule Rules

- **Submodule Commits**: Do not push out-of-sync submodule pointers. After committing changes in `Thesis/`, check if the submodule was updated in the root repository recently:
  `git log -1 --format=%ci -- Thesis`
  If more than 24 hours have passed, commit the pointer: `git add Thesis && git commit -m "chore: bump Thesis submodule"`.
- **Commit Format**: Follow Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `bench:`). Scope is encouraged (e.g. `feat(bench):`).
- **No Co-Authored Signatures**: Keep commits clean without metadata additions.
