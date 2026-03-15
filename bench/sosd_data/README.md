# SOSD Benchmark Datasets

Real-world datasets from [SOSD: A Benchmark for Learned Indexes](https://github.com/learnedsystems/SOSD).

## Download

```bash
./download.sh
```

Requires `curl` and `zstd`. On macOS: `brew install zstd`.

## Datasets

| File                      | Size   | Keys        | Description               |
|---------------------------|--------|-------------|---------------------------|
| `fb_200M_uint64`          | 1.5 GB | 200M uint64 | Facebook user IDs         |
| `wiki_ts_200M_uint64`     | 1.5 GB | 200M uint64 | Wikipedia edit timestamps |
| `osm_cellids_800M_uint64` | 6.0 GB | 800M uint64 | OpenStreetMap S2 CellIDs  |
| `books_200M_uint32`       | 0.8 GB | 200M uint32 | Amazon book popularity    |

## Format

Binary files. First 8 bytes: `uint64` key count (little-endian).
Remaining bytes: sorted array of keys (`uint64` or `uint32`, little-endian).

```go
var count uint64
binary.Read(f, binary.LittleEndian, &count)
keys := make([]uint64, count)
binary.Read(f, binary.LittleEndian, keys)
```

## Reference

Kipf et al., "SOSD: A Benchmark for Learned Indexes", 2019.
https://arxiv.org/abs/1911.13014
