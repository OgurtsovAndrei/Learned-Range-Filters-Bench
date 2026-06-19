# rbtz

This repository contains a Go implementation of minimal perfect hashing using the "Hash, displace, and compress" algorithm.

## Origin and License

The foundational code for this project is adapted from the excellent work by:

- [https://github.com/SaveTheRbtz/mph](https://github.com/SaveTheRbtz/mph)
- Original work by Caleb Spare and Alexey Ivanov

This project is distributed under the [MIT License](LICENSE).

## Modifications

This version of `rbtz` has been enhanced by Ogurtsov Andrei with the following additions:

* **Serialization:** Functionality to serialize and deserialize the hash function structure has been integrated (`serialize.go`).
* **Extended Testing:** New test suites have been added, including:
    * Tests for serialization integrity (`serialize_test.go`).
    * Tests for monotone properties (`monotone_test.go`).
    * Comprehensive benchmark tests to evaluate performance (`bench_test.go`).

## References:

See [Hash, displace, and compress](https://cmph.sourceforge.net/papers/esa09.pdf)