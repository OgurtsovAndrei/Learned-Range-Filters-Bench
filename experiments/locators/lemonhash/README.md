# LeMonHash (CGO Wrapper)

This module provides a Go wrapper for the [ByteHamster/LeMonHash](https://github.com/ByteHamster/LeMonHash) C++ library using CGO. LeMonHash is a state-of-the-art **Learned Monotone Minimal Perfect Hash Function (MMPHF)** introduced in 2023.

## Features
- **Compact**: Achieving ~3.3 bits per key for large datasets.
- **Fast**: Mathematical piecewise-linear model (PGM-index) instead of tree/trie traversal.
- **Batch Support**: Includes a `RankBatch` method to reduce CGO overhead when querying multiple keys.
- **Cross-Platform**: Patched to support both Linux (x86_64) and macOS (Apple Silicon/ARM64).

## Performance (Apple M4 Max)
- **Space**: ~3.32 bits/key (at 16M keys)
- **Single Query**: ~600ns (including CGO overhead and memory allocation)
- **Batch Query**: See `PERFORMANCE.md` for a detailed analysis of CGO overhead and optimization plans.

## Building
This module requires a C++20 compiler (GCC 11+ or Clang 13+) and CMake.

To build the static libraries required for the Go wrapper:
```bash
make build
```

To run tests:
```bash
make test
```

To run benchmarks:
```bash
make bench
```

## Structure
- `lemonhash.go`: Go API and CGO bridge.
- `cpp/wrapper.cpp`: C++ implementation of the C-compatible bridge.
- `ext/LeMonHash`: Vendored and patched source code of the original library.
- `VENDORED_DEPENDENCIES.md`: Documentation of applied patches and original libraries.
- `PERFORMANCE.md`: Deep dive into query latency and CGO bottlenecks.
