package rloc

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"
)

const (
	debugTestRuns  = 10_000 // Fewer runs for faster debugging
	debugMaxKeys   = 1024
	debugMaxBitLen = 16
)

// BitStringData represents a BitString in serializable form
type BitStringData struct {
	Hex       string `json:"hex"`        // Hex representation of data
	BitLength int    `json:"bit_length"` // Number of bits in this string
}

// MMPHFailureRecord represents a single failure case
type MMPHFailureRecord struct {
	Seed             int64           `json:"seed"`      // Seed for key generation
	MMPHSeed         uint64          `json:"mmph_seed"` // Seed used for MMPH construction
	Keys             []BitStringData `json:"keys"`      // Full BitString data
	KeyCount         int             `json:"key_count"`
	MaxBitLength     int             `json:"max_bit_length"`
	ErrorMessage     string          `json:"error_message"`
	TrieRebuildCount int             `json:"trie_rebuild_attempts"`
	Timestamp        string          `json:"timestamp"`
}

// TestRangeLocator_CaptureMMPHFailures captures keys that fail to build MMPH to a JSON file
func TestRangeLocator_CaptureMMPHFailures(t *testing.T) {
	var failures []MMPHFailureRecord
	outputFile := "/tmp/mmph_failures.json"

	for run := 0; run < debugTestRuns; run++ {
		t.Run(fmt.Sprintf("Run %d", run), func(t *testing.T) {

			seed := time.Now().UnixNano() + int64(run)
			keys := GenUniqueBitStringsDebug(seed, debugMaxKeys, debugMaxBitLen)

			// Use a deterministic MMPH seed for this test run
			mmphSeed := uint64(seed) * 31337 // Derive from key seed

			zt := zft.Build(keys)
			_, err := NewRangeLocatorSeeded(zt, mmphSeed)

			if err != nil {
				t.Logf("MMPH Build failed (seed: %d, mmphSeed: %d): %v", seed, mmphSeed, err)

				// Immediately retry with THE SAME keys and mmphSeed to verify reproducibility
				zt2 := zft.Build(keys)
				_, err2 := NewRangeLocatorSeeded(zt2, mmphSeed)

				if err2 == nil {
					t.Errorf("INCONSISTENT FAILURE (seed: %d, mmphSeed: %d): Original build failed but retry succeeded! This should not happen with deterministic seeds.", seed, mmphSeed)
					t.Logf("  Original error: %v", err)
					t.Logf("  Retry error: nil (succeeded)")
					t.Logf("  Retry succeeded - THIS IS A BUG in the determinism")
					return
				}

				// Verify it's the same error
				if err.Error() != err2.Error() {
					t.Logf("WARNING (seed: %d, mmphSeed: %d): Different errors on retry", seed, mmphSeed)
					t.Logf("  Original: %v", err)
					t.Logf("  Retry: %v", err2)
				}

				// Serialize the keys for JSON storage
				serializedKeys := keysToData(keys)

				// Save the reproducible failure
				record := MMPHFailureRecord{
					Seed:             seed,
					MMPHSeed:         mmphSeed,
					Keys:             serializedKeys,
					KeyCount:         len(keys),
					MaxBitLength:     inferBitLength(keys),
					ErrorMessage:     err.Error(),
					TrieRebuildCount: -1, // Unknown without modifying MMPH error
					Timestamp:        time.Now().Format(time.RFC3339),
				}

				failures = append(failures, record)
				t.Logf("Reproducible failure confirmed and saved (seed: %d, mmphSeed: %d)", seed, mmphSeed)
			}
		})
	}

	// Write failures to JSON file
	if len(failures) > 0 {
		data, err := json.MarshalIndent(failures, "", "  ")
		if err != nil {
			t.Fatalf("Failed to marshal failures to JSON: %v", err)
		}

		err = os.WriteFile(outputFile, data, 0644)
		if err != nil {
			t.Fatalf("Failed to write failures to file %s: %v", outputFile, err)
		}

		t.Logf("Saved %d MMPH failures to %s", len(failures), outputFile)
	} else {
		t.Logf("No MMPH failures detected in %d runs", debugTestRuns)
	}
}

// TestRangeLocator_LoadAndReplayFailures loads failures from JSON and attempts to rebuild
func TestRangeLocator_LoadAndReplayFailures(t *testing.T) {
	inputFile := "/tmp/mmph_failures.json"

	// Read the JSON file
	data, err := os.ReadFile(inputFile)
	if err != nil {
		t.Skipf("Could not read failure file %s: %v", inputFile, err)
	}

	var failures []MMPHFailureRecord
	err = json.Unmarshal(data, &failures)
	if err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	t.Logf("Loaded %d failure records to replay", len(failures))

	successCount := 0
	failCount := 0

	for i, record := range failures {
		t.Run(fmt.Sprintf("replay_seed_%d_mmphSeed_%d", record.Seed, record.MMPHSeed), func(t *testing.T) {
			// Regenerate keys using the original seed
			keys := GenUniqueBitStringsDebug(record.Seed, debugMaxKeys, debugMaxBitLen)
			t.Logf("Regenerated %d keys from seed %d (expected: %d keys)",
				len(keys), record.Seed, record.KeyCount)

			// Verify key count matches (catches bugs in key generation)
			if len(keys) != record.KeyCount {
				t.Fatalf("BUG: Key count mismatch! Regenerated: %d, Expected: %d",
					len(keys), record.KeyCount)
			}

			// Note: We use regenerated keys instead of saved keys because the
			// BitString serialization (Data() -> hex -> NewFromUint64) is lossy.
			// The seeds are the source of truth for perfect reproducibility.

			zt := zft.Build(keys)
			rl, err := NewRangeLocatorSeeded(zt, record.MMPHSeed)

			if err != nil {
				t.Logf("Replay %d: MMPH build still fails (seed: %d, mmphSeed: %d): %v", i, record.Seed, record.MMPHSeed, err)
				failCount++
			} else {
				t.Logf("Replay %d: MMPH build now succeeds! (seed: %d, mmphSeed: %d)", i, record.Seed, record.MMPHSeed)
				successCount++

				// Try to query the structure to verify it works
				for _, key := range keys {
					_, _, queryErr := rl.Query(key)
					if queryErr != nil {
						t.Errorf("Query failed for key: %v", queryErr)
						failCount++
						return
					}
				}
			}
		})
	}

	t.Logf("Replay Results: %d successes, %d failures", successCount, failCount)
}

// Helper functions

func keysToData(keys []bits.BitString) []BitStringData {
	data := make([]BitStringData, len(keys))
	for i, key := range keys {
		data[i] = BitStringData{
			Hex:       fmt.Sprintf("%x", key.Data()),
			BitLength: int(key.Size()),
		}
	}
	return data
}

func dataToKeys(data []BitStringData) []bits.BitString {
	keys := make([]bits.BitString, len(data))
	for i, d := range data {
		var val uint64
		fmt.Sscanf(d.Hex, "%x", &val)
		bs := bits.NewFromUint64(val)
		// Adjust to the correct bit length if needed
		if d.BitLength < int(bs.Size()) {
			bs = bs.Prefix(d.BitLength)
		}
		keys[i] = bs
	}
	return keys
}

func inferBitLength(keys []bits.BitString) int {
	if len(keys) == 0 {
		return 0
	}
	maxBits := 0
	for _, k := range keys {
		if int(k.Size()) > maxBits {
			maxBits = int(k.Size())
		}
	}
	return maxBits
}
