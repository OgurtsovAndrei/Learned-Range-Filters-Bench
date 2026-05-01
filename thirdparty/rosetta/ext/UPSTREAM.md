# Rosetta upstream provenance

- **Source repository:** [github.com/marcocosta97/grafite](https://github.com/marcocosta97/grafite)
- **Source path within repo:** `bench/include/Rosetta/`
- **Files imported:** `dst.h`, `dst.cpp`, `MurmurHash3.h`, `MurmurHash3.cpp`
- **Last sync:** 2026-05-01

## Original Rosetta paper

Luo, S., Chatterjee, S., Ketsetsidis, R., Dayan, N., Qin, W., Idreos, S.
"Rosetta: A Robust Space-Time Optimized Range Filter for Key-Value Stores."
SIGMOD 2020. <https://dl.acm.org/doi/10.1145/3318464.3389731>

## License

The vendored files originate from a GPLv3-licensed project (Grafite). The
`LICENSE` file in this directory is the verbatim Grafite GPLv3 text. The
MurmurHash3 implementation by Austin Appleby is in the public domain (see
header comment in `MurmurHash3.cpp`).

## Local modifications

None — files are byte-identical with upstream as of the sync date. All glue
lives in `../cpp/wrapper.cpp`.
