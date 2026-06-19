# MMPH Debugging Setup - Quick Start

## What Was Created

Two new files to help debug MMPH (Monotone Minimal Perfect Hash Function) build failures:

### 1. **rloc_test_debug.go** - Debug test file

Contains two test functions:

- `TestRangeLocator_CaptureMMPHFailures` - Captures failing cases to JSON
- `TestRangeLocator_LoadAndReplayFailures` - Replays failures to test fixes

### 2. **DEBUG_GUIDE.md** - Comprehensive debugging guide

Full instructions on how to use the debug tests and understand the failures.

## Quick Start

### Step 1: Set up Go environment

```bash
# Make sure Go is installed and in your PATH
# Check: go version
```

If Go is not installed, install it from https://golang.org/doc/install

### Step 2: Run the capture test

```bash
cd /Users/andrei.ogurtsov/Thesis
go test ./rloc -v -run TestRangeLocator_CaptureMMPHFailures -timeout 30m
```

This will run 100 test iterations and save any MMPH build failures to `/tmp/mmph_failures.json`.

### Step 3: Check the failures

```bash
# View the JSON file
cat /tmp/mmph_failures.json | jq '.'

# Or pretty-print it
cat /tmp/mmph_failures.json | python3 -m json.tool
```

### Step 4: After fixing MMPH code, replay failures

```bash
go test ./rloc -v -run TestRangeLocator_LoadAndReplayFailures
```

This tests if your fixes resolve the previously-failing key sets.

## Key Features

✅ **Reproducible failures** - Saves seed + exact BitString data
✅ **Full bit-level fidelity** - Stores hex representation + bit lengths
✅ **Query verification** - Replayed tests verify the structure actually works
✅ **JSON format** - Easy to analyze and share with team
✅ **Detailed logging** - Each test logs what's happening

## File Locations

- **Test file**: `/Users/andrei.ogurtsov/Thesis/locators/rloc_test_debug.go`
- **Debug guide**: `/Users/andrei.ogurtsov/Thesis/locators/DEBUG_GUIDE.md`
- **Failure output**: `/tmp/mmph_failures.json` (created by tests)

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  TestRangeLocator_CaptureMMPHFailures                       │
│  - Generates random key sets (100 iterations)              │
│  - Tries to build RangeLocator with each                   │
│  - On failure: saves to JSON                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
         /tmp/mmph_failures.json
         (BitString data + metadata)
                   │
                   ▼
┌──────────────────┴──────────────────────────────────────────┐
│  TestRangeLocator_LoadAndReplayFailures                     │
│  - Reads JSON file                                          │
│  - Reconstructs BitStrings                                 │
│  - Tests if builds now succeed                             │
│  - Verifies queries work correctly                         │
└──────────────────────────────────────────────────────────────┘
```

## What to Look For

When analyzing failures in `/tmp/mmph_failures.json`:

1. **Error patterns** - Do all failures mention "approximate z-fast trie"?
2. **Key counts** - Do failures happen with certain key set sizes?
3. **Bit lengths** - Are there patterns in bit length distributions?
4. **Timestamps** - Were failures clustered in time?

## Debugging Tips

- Run `go test ./rloc -v` to see all the existing tests pass
- Modify `debugTestRuns` in `rloc_test_debug.go` to run more/fewer iterations
- Check `/Users/andrei.ogurtsov/Thesis/mmph/relative_trie/hash.go` for the MMPH implementation
- The issue is likely in `buildValidatedTrieWithIndices` or `validateAllKeys`

## JSON Structure Example

```json
{
  "seed": 1707014234567890,
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
```

---

For detailed debugging information, see `DEBUG_GUIDE.md`
