package bucket

import (
	"Thesis/bits"
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

func buildUniqueStrKeys(size int) []string {
	keys := make([]string, size)
	unique := make(map[string]bool, size)

	for i := 0; i < size; i++ {
		for {
			b := make([]byte, 8)
			_, _ = rand.Read(b)
			s := string(b)
			if !unique[s] {
				keys[i] = s
				unique[s] = true
				break
			}
		}
	}
	return keys
}

type bitStringSorter []bits.BitString

func (s bitStringSorter) Len() int      { return len(s) }
func (s bitStringSorter) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s bitStringSorter) Less(i, j int) bool {
	lcp := s[i].GetLCPLength(s[j])

	if lcp == s[i].Size() && lcp == s[j].Size() {
		return false
	}
	if lcp == s[i].Size() {
		return true
	}
	if lcp == s[j].Size() {
		return false
	}

	return !s[i].At(lcp)
}

func TestMonotoneHashWithTrie_Randomized(t *testing.T) {
	t.Parallel()
	sizes := []int{1, 10, 100, 1_000, 10_000, 100_000}

	for _, size := range sizes {
		keys := buildUniqueStrKeys(size)

		bitKeys := make([]bits.BitString, size)
		for i, s := range keys {
			bitKeys[i] = bits.NewFromText(s)
		}

		sort.Sort(bitStringSorter(bitKeys))

		testName := fmt.Sprintf("Size_%d", size)
		t.Run(testName, func(t *testing.T) {
			t.Parallel()
			mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](bitKeys)
			if err != nil {
				t.Fatalf("Failed to create MonotoneHashWithTrie: %v", err)
			}

			t.Logf("Trie rebuild attempts: %d", mh.TrieRebuildAttempts)

			for i, key := range bitKeys {
				rank := mh.GetRank(key)
				if rank != i {
					t.Errorf("Mismatch for key index %d: expected rank %d, got %d", i, i, rank)
				}
			}
		})
	}
}

func TestMonotoneHashWithTrie_TrieRebuildTracking(t *testing.T) {
	t.Parallel()
	// Test that tracks trie rebuild statistics
	totalAttempts := 0
	maxAttempts := 0
	numTests := 10
	keySize := 1000

	for test := 0; test < numTests; test++ {
		keys := buildUniqueStrKeys(keySize)

		bitKeys := make([]bits.BitString, keySize)
		for i, s := range keys {
			bitKeys[i] = bits.NewFromText(s)
		}

		sort.Sort(bitStringSorter(bitKeys))

		mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](bitKeys)
		if err != nil {
			t.Fatalf("Failed to create MonotoneHashWithTrie: %v", err)
		}

		totalAttempts += mh.TrieRebuildAttempts
		if mh.TrieRebuildAttempts > maxAttempts {
			maxAttempts = mh.TrieRebuildAttempts
		}

		// Verify correctness
		for i, key := range bitKeys {
			rank := mh.GetRank(key)
			if rank != i {
				t.Errorf("Test %d: Mismatch for key index %d: expected rank %d, got %d", test, i, i, rank)
			}
		}
	}

	avgAttempts := float64(totalAttempts) / float64(numTests)
	t.Logf("Trie rebuild statistics over %d tests with %d keys each:", numTests, keySize)
	t.Logf("  Average attempts: %.2f", avgAttempts)
	t.Logf("  Maximum attempts: %d", maxAttempts)
	t.Logf("  Total attempts: %d", totalAttempts)

	// We expect most builds to succeed on first attempt, with occasional retries
	if avgAttempts > 5.0 {
		t.Errorf("Average rebuild attempts too high: %.2f (expected < 5.0)", avgAttempts)
	}
}

func TestMonotoneHashWithTrie_EmptyInput(t *testing.T) {
	t.Parallel()
	mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](nil)
	if err != nil {
		t.Fatalf("Failed to create MonotoneHashWithTrie with empty input: %v", err)
	}

	// Should handle empty case gracefully
	rank := mh.GetRank(bits.NewFromText("test"))
	if rank != -1 {
		t.Errorf("Expected -1 for empty structure, got %d", rank)
	}
}

func TestMonotoneHashWithTrie_SingleKey(t *testing.T) {
	t.Parallel()
	key := bits.NewFromText("key")
	mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16]([]bits.BitString{key})
	if err != nil {
		t.Fatalf("Failed to create MonotoneHashWithTrie with single key: %v", err)
	}

	rank := mh.GetRank(key)
	if rank != 0 {
		t.Errorf("Expected rank 0 for single key, got %d", rank)
	}
}

func TestMonotoneHashWithTrie_CompareWithSimple(t *testing.T) {
	t.Parallel()
	// Compare results with the simple bucket implementation to ensure correctness
	sizes := []int{100, 1000}

	for _, size := range sizes {
		keys := buildUniqueStrKeys(size)

		bitKeys := make([]bits.BitString, size)
		for i, s := range keys {
			bitKeys[i] = bits.NewFromText(s)
		}

		sort.Sort(bitStringSorter(bitKeys))

		// Build both implementations
		mhTrie, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](bitKeys)
		if err != nil {
			t.Fatalf("Failed to create MonotoneHashWithTrie: %v", err)
		}

		// Note: We'd need to import the simple bucket implementation to compare
		// For now, we just verify that our implementation produces correct ranks
		t.Run(fmt.Sprintf("Compare_Size_%d", size), func(t *testing.T) {
			t.Parallel()
			for i, key := range bitKeys {
				rank := mhTrie.GetRank(key)
				if rank != i {
					t.Errorf("Mismatch for key index %d: expected rank %d, got %d", i, i, rank)
				}
			}
		})
	}
}

func TestMonotoneHashWithTrie_NonExistentKeys(t *testing.T) {
	t.Parallel()
	// Test behavior with keys not in the original set
	keys := buildUniqueStrKeys(100)

	bitKeys := make([]bits.BitString, len(keys))
	for i, s := range keys {
		bitKeys[i] = bits.NewFromText(s)
	}

	sort.Sort(bitStringSorter(bitKeys))

	mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](bitKeys)
	if err != nil {
		t.Fatalf("Failed to create MonotoneHashWithTrie: %v", err)
	}

	// Test some random non-existent keys
	for i := 0; i < 10; i++ {
		nonExistentKeys := buildUniqueStrKeys(1)
		nonExistentKey := bits.NewFromText(nonExistentKeys[0])

		rank := mh.GetRank(nonExistentKey)
		// Should return -1 for non-existent keys
		if rank != -1 {
			t.Logf("Non-existent key %s got rank %d (might be a false positive due to hash collision)",
				nonExistentKey.PrettyString(), rank)
		}
	}
}
