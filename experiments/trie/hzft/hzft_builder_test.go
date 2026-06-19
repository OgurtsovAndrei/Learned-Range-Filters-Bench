package hzft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"math/rand"
	"testing"
	"time"
)

// TestStreamingBuilder_MatchesHeavyBuilder verifies streaming builder produces
// identical results to the heavy ZFT-based builder.
func TestStreamingBuilder_MatchesHeavyBuilder(t *testing.T) {
	t.Parallel()

	const runs = 500
	const maxKeys = 64
	const maxBitLen = 32

	for run := 0; run < runs; run++ {
		seed := time.Now().UnixNano() + int64(run)
		r := rand.New(rand.NewSource(seed))

		numKeys := r.Intn(maxKeys-1) + 2
		bitLen := r.Intn(maxBitLen-8) + 8
		keys := zft.GenerateRandomBitStrings(numKeys, bitLen, r)

		// Build using heavy ZFT
		heavyHZFT, err := NewHZFastTrieFromIterator[uint32](bits.NewSliceBitStringIterator(keys))
		if err != nil {
			t.Fatalf("Heavy builder failed: %v (seed: %d)", err, seed)
		}

		// Build using streaming
		streamingHZFT, err := NewHZFastTrieFromIteratorStreaming[uint32](bits.NewSliceBitStringIterator(keys))
		if err != nil {
			t.Fatalf("Streaming builder failed: %v (seed: %d)", err, seed)
		}

		// Handle empty cases
		if heavyHZFT == nil && streamingHZFT == nil {
			continue
		}
		if heavyHZFT == nil || streamingHZFT == nil {
			t.Fatalf("One builder returned nil, other didn't (seed: %d)", seed)
		}

		// Compare data lengths
		if len(heavyHZFT.data) != len(streamingHZFT.data) {
			t.Fatalf("Data length mismatch: heavy=%d, streaming=%d (seed: %d)",
				len(heavyHZFT.data), len(streamingHZFT.data), seed)
		}

		// Test behavior on all prefixes of all keys
		for _, key := range keys {
			for prefixLen := 1; prefixLen <= int(key.Size()); prefixLen++ {
				prefix := key.Prefix(prefixLen)

				heavyResult := heavyHZFT.GetExistingPrefix(prefix)
				streamingResult := streamingHZFT.GetExistingPrefix(prefix)

				if heavyResult != streamingResult {
					t.Errorf("GetExistingPrefix mismatch for prefix %s: heavy=%d, streaming=%d (seed: %d)",
						prefix.PrettyString(), heavyResult, streamingResult, seed)
				}
			}
		}
	}
}

// TestStreamingBuilder_EmptyInput tests streaming builder with empty input.
func TestStreamingBuilder_EmptyInput(t *testing.T) {
	keys := []bits.BitString{}

	hzft, err := NewHZFastTrieFromIteratorStreaming[uint32](bits.NewSliceBitStringIterator(keys))
	if err != nil {
		t.Fatalf("Streaming builder failed on empty input: %v", err)
	}

	if hzft != nil {
		t.Errorf("Expected nil for empty input, got %v", hzft)
	}
}

// TestStreamingBuilder_SingleKey tests streaming builder with single key.
func TestStreamingBuilder_SingleKey(t *testing.T) {
	key := bits.NewFromBinary("10110")
	keys := []bits.BitString{key}

	heavyHZFT, _ := NewHZFastTrieFromIterator[uint32](bits.NewSliceBitStringIterator(keys))
	streamingHZFT, err := NewHZFastTrieFromIteratorStreaming[uint32](bits.NewSliceBitStringIterator(keys))

	if err != nil {
		t.Fatalf("Streaming builder failed: %v", err)
	}

	if heavyHZFT == nil && streamingHZFT == nil {
		return // Both nil is ok for single key if that's expected behavior
	}

	if (heavyHZFT == nil) != (streamingHZFT == nil) {
		t.Fatalf("Nil mismatch: heavy=%v, streaming=%v", heavyHZFT == nil, streamingHZFT == nil)
	}

	// Test query
	result := streamingHZFT.GetExistingPrefix(key)
	heavyResult := heavyHZFT.GetExistingPrefix(key)

	if result != heavyResult {
		t.Errorf("Single key query mismatch: heavy=%d, streaming=%d", heavyResult, result)
	}
}

// TestStreamingBuilder_TwoKeys tests streaming builder with two keys.
func TestStreamingBuilder_TwoKeys(t *testing.T) {
	// Two keys with common prefix
	key1 := bits.NewFromBinary("10110")
	key2 := bits.NewFromBinary("10111")
	keys := []bits.BitString{key1, key2}

	heavyHZFT, _ := NewHZFastTrieFromIterator[uint32](bits.NewSliceBitStringIterator(keys))
	streamingHZFT, err := NewHZFastTrieFromIteratorStreaming[uint32](bits.NewSliceBitStringIterator(keys))

	if err != nil {
		t.Fatalf("Streaming builder failed: %v", err)
	}

	if heavyHZFT == nil || streamingHZFT == nil {
		t.Fatalf("Unexpected nil")
	}

	// Test all prefixes
	for _, key := range keys {
		for prefixLen := 1; prefixLen <= int(key.Size()); prefixLen++ {
			prefix := key.Prefix(prefixLen)

			heavyResult := heavyHZFT.GetExistingPrefix(prefix)
			streamingResult := streamingHZFT.GetExistingPrefix(prefix)

			if heavyResult != streamingResult {
				t.Errorf("Mismatch for prefix %s: heavy=%d, streaming=%d",
					prefix.PrettyString(), heavyResult, streamingResult)
			}
		}
	}
}

// BenchmarkStreamingVsHeavy benchmarks memory and time for both approaches.
func BenchmarkStreamingVsHeavy(b *testing.B) {
	r := rand.New(rand.NewSource(42))
	keys := zft.GenerateRandomBitStrings(1000, 64, r)

	b.Run("Heavy", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			NewHZFastTrieFromIterator[uint32](bits.NewSliceBitStringIterator(keys))
		}
	})

	b.Run("Streaming", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			NewHZFastTrieFromIteratorStreaming[uint32](bits.NewSliceBitStringIterator(keys))
		}
	})
}
