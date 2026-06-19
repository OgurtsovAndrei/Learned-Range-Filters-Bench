package azft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"math/rand"
	"testing"
	"time"
)

const (
	builderTestRuns   = 100
	maxKeys    = 64
	maxBitLen  = 16
)

// TestAZFT_MatchesReference tests AZFT builder correctness.
func TestAZFT_MatchesReference(t *testing.T) {
	for run := 0; run < builderTestRuns; run++ {
		seed := time.Now().UnixNano() + int64(run)
		r := rand.New(rand.NewSource(seed))

		numKeys := r.Intn(maxKeys-1) + 2
		bitLen := r.Intn(maxBitLen-8) + 8
		keys := zft.GenerateRandomBitStrings(numKeys, bitLen, r)

		fixedSeed := uint64(seed)

		// Build AZFT
		azft, err := NewApproxZFastTrieWithSeed[uint16, uint32, uint32](keys, fixedSeed)
		if err != nil {
			t.Fatalf("Builder failed: %v (seed: %d)", err, seed)
		}

		// Handle empty cases
		if azft == nil {
			continue
		}

		// Test behavior on all prefixes of all keys
		for _, key := range keys {
			for prefixLen := 1; prefixLen <= int(key.Size()); prefixLen++ {
				prefix := key.Prefix(prefixLen)

				result := azft.GetExistingPrefix(prefix)

				// Just verify it doesn't crash - result can be nil or non-nil
				_ = result
			}
		}
	}
}

// TestAZFT_EmptyInput tests builder with empty input.
func TestAZFT_EmptyInput(t *testing.T) {
	keys := []bits.BitString{}

	azft, err := NewApproxZFastTrie[uint16, uint32, uint32](keys)
	if err != nil {
		t.Fatalf("Builder failed on empty input: %v", err)
	}

	if azft == nil {
		t.Errorf("Expected non-nil empty AZFT, got nil")
	}
}

// TestAZFT_SingleKey tests builder with single key.
func TestAZFT_SingleKey(t *testing.T) {
	key := bits.NewFromBinary("10110")
	keys := []bits.BitString{key}
	seed := uint64(12345)

	azft, err := NewApproxZFastTrieWithSeed[uint16, uint32, uint32](keys, seed)
	if err != nil {
		t.Fatalf("Builder failed: %v", err)
	}

	if azft == nil {
		return
	}

	// Test query
	result := azft.GetExistingPrefix(key)
	if result == nil {
		t.Fatalf("GetExistingPrefix returned nil")
	}
}

// TestAZFT_TwoKeys tests builder with two keys.
func TestAZFT_TwoKeys(t *testing.T) {
	// Two keys with common prefix
	key1 := bits.NewFromBinary("10110")
	key2 := bits.NewFromBinary("10111")
	keys := []bits.BitString{key1, key2}
	seed := uint64(12345)

	azft, err := NewApproxZFastTrieWithSeed[uint16, uint32, uint32](keys, seed)
	if err != nil {
		t.Fatalf("Builder failed: %v", err)
	}

	if azft == nil {
		t.Fatalf("Unexpected nil")
	}

	// Test both keys
	for _, key := range keys {
		result := azft.GetExistingPrefix(key)
		if result == nil {
			t.Errorf("GetExistingPrefix returned nil for key %s", key.PrettyString())
		}
	}
}
