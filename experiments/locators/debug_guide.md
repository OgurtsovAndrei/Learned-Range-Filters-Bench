# MMPH Debugging Guide

This guide explains how to use the debug tests to capture and analyze MMPH (Monotone Minimal Perfect Hash Function) build failures.

## Overview

When `RangeLocator` is constructed, it builds an MMPH to hash the delimiter set. Sometimes this build fails after exhausting rebuild attempts. The debug tests help:

1. **Capture failures** - Identify which key sets cause MMPH build failures
2. **Save to JSON** - Persist failing key sets for reproducible debugging
3. **Replay failures** - Test the same key sets again to verify fixes

## Running the Debug Tests

### Step 1: Capture MMPH Failures

Run the capture test to find failing key sets:

```bash
cd /Users/andrei.ogurtsov/Thesis
go test ./rloc -v -run TestRangeLocator_CaptureMMPHFailures
```

This test:
- Runs 100 test iterations with random key sets
- For each MMPH build failure, records:
  - The seed used to generate the keys
  - The actual BitString data (hex + bit length)
  - The error message
  - The timestamp

- **Output**: Saves results to `/tmp/mmph_failures.json`

### Step 2: Review the Failures

Check what failed:

```bash
cat /tmp/mmph_failures.json | jq '.'
```

Example output structure:
```json
[
  {
    "seed": 1707014234567890123,
    "keys": [
      {"hex": "abc123", "bit_length": 12},
      {"hex": "def456", "bit_length": 12}
    ],
    "key_count": 256,
    "max_bit_length": 16,
    "error_message": "failed to build working approximate z-fast trie after 100 attempts...",
    "trie_rebuild_attempts": -1,
    "timestamp": "2025-02-04T12:34:56Z"
  }
]
```

### Step 3: Replay Failures

Once you've fixed the MMPH or Z-Fast Trie code, replay the failing key sets:

```bash
go test ./rloc -v -run TestRangeLocator_LoadAndReplayFailures
```

This test:
- Loads the failures from `/tmp/mmph_failures.json`
- Attempts to rebuild `RangeLocator` for each failing key set
- Reports which ones now succeed and which still fail
- Verifies that successful builds pass queries

## Understanding the Failures

### Common Issues

1. **Z-Fast Trie Seed Sensitivity**: The approximate Z-Fast Trie might fail with certain key distributions and random seeds. The retry logic tries up to 100 times with different seeds.

2. **Validation Failure**: The MMPH validation checks that all keys can be correctly located in their buckets using the approximate trie. If the trie can't reliably distinguish between buckets for a key set, validation fails.

3. **Type Parameter Constraints**: The type parameters `E`, `S`, and `I` in `MonotoneHashWithTrie[E, S, I]` have specific constraints:
   - `E`: must be large enough for key bit lengths
   - `S`: signature bits (should be ≥ log(log n) + log(log w) - log(eps))
   - `I`: delimiter index bits (should be ≥ log(n/bucketSize))

## Debugging Steps

1. **Analyze the key patterns**: Look at the hex values in the failures - are there patterns? Specific bit distributions?

2. **Test with smaller sets**: Try building `RangeLocator` with just the first few keys from a failing set.

3. **Check Z-Fast Trie alone**: Test if the Z-Fast Trie can be built from just the delimiters:
   ```go
   // Extract delimiters from the failing key set
   // Try: zfasttrie.NewApproxZFastTrie[E, S, I](delimiters, false)
   ```

4. **Increase retry attempts**: In `hash.go`, temporarily increase `maxTrieRebuilds` from 100 to 1000+ to see if more retries help.

5. **Adjust type parameters**: Try increasing `S` or `I` in the `uint16` type parameter to give more bits for disambiguation.

## Modifying MMPH for Better Diagnostics

To get better error information, consider modifying `mmph/relative_trie/hash.go`:

1. **Return TrieRebuildAttempts in error**:
```go
// Instead of just returning an error, include the attempt count
type BuildError struct {
    Attempts int
    Msg string
}
```

2. **Add detailed validation logging**: When `validateAllKeys` fails, log which keys failed validation.

3. **Export the delimiter trie separately**: Allow testing the trie in isolation.

## JSON Format

The `MMPHFailureRecord` JSON structure:

```go
type BitStringData struct {
    Hex       string // Hex representation of the key data
    BitLength int    // Number of bits in the key
}

type MMPHFailureRecord struct {
    Seed             int64            // Random seed for reproducibility
    Keys             []BitStringData  // The failing key set
    KeyCount         int              // Number of keys
    MaxBitLength     int              // Maximum bit length in the set
    ErrorMessage     string           // The error from MMPH build
    TrieRebuildCount int              // Number of rebuild attempts (-1 if unknown)
    Timestamp        string           // When this failure was recorded
}
```

## Recovery

Once you've identified and fixed the issue:

1. Commit the fix
2. Re-run `TestRangeLocator_CaptureMMPHFailures` with a fresh run to verify no new failures
3. Run `TestRangeLocator_LoadAndReplayFailures` to confirm all previously-failing keys now work

---

For questions about the implementation, see:
- `/Users/andrei.ogurtsov/Thesis/mmph/relative_trie/hash.go` - MMPH implementation
- `/Users/andrei.ogurtsov/Thesis/locators/locators.go` - RangeLocator implementation
- `/Users/andrei.ogurtsov/Thesis/zfasttrie/` - Z-Fast Trie implementation
