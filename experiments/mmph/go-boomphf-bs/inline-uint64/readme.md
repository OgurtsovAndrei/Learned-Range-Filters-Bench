# go-boomphf

This repository contains a Go implementation of a fast perfect hash function, designed for massive key sets.

## Origin and License

The foundational code for this project is adapted from the excellent work by Damian Gryski:
[https://github.com/dgryski/go-boomphf](https://github.com/dgryski/go-boomphf)

This project is distributed under the [MIT License](LICENSE).

## Modifications

This version of `go-boomphf` has been enhanced by Ogurtsov Andrei with the following additions:

*   **Serialization:** Functionality to serialize and deserialize the hash function structure has been integrated (`serialize.go`).
*   **Extended Testing:** New test suites have been added, including:
    *   Tests for serialization integrity (`serialize_test.go`).
    *   Comprehensive benchmark tests to evaluate performance (`bench_test.go`).
*   **Benchmark Results:** Detailed performance metrics from the benchmarks are documented in `bench_results.md`.