package bucket

import (
	"Thesis/bits"
	"sort"
	"testing"
)

func TestByteSizeMethods(t *testing.T) {
	t.Parallel()
	// Test that ByteSize methods work correctly
	keys := buildUniqueStrKeys(100)

	bitKeys := make([]bits.BitString, len(keys))
	for i, s := range keys {
		bitKeys[i] = bits.NewFromText(s)
	}

	// Sort the keys as required by the implementation
	sort.Sort(bitStringSorter(bitKeys))

	mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](bitKeys)
	if err != nil {
		t.Fatalf("Failed to create MonotoneHashWithTrie: %v", err)
	}

	// Test that Size() and ByteSize() return the same value
	size := mh.Size()
	byteSize := mh.ByteSize()
	if size != byteSize {
		t.Errorf("Size() and ByteSize() should return same value: got %d vs %d", size, byteSize)
	}

	// Test that we can get exact trie size
	if mh.delimiterTrie != nil {
		trieSize := mh.delimiterTrie.ByteSize()
		if trieSize <= 0 {
			t.Errorf("Trie size should be positive, got %d", trieSize)
		}

		t.Logf("Total structure size: %d bytes", size)
		t.Logf("Exact trie size: %d bytes (%.2f%% of total)", trieSize, float64(trieSize)*100/float64(size))
		t.Logf("Trie rebuild attempts: %d", mh.TrieRebuildAttempts)

		// Verify the trie size is reasonable (not just a placeholder)
		if trieSize > size {
			t.Errorf("Trie size (%d) should not exceed total size (%d)", trieSize, size)
		}
	}

	// Test that individual components have reasonable sizes
	delimiterSize := 0
	bucketSize := 0
	for _, bucket := range mh.buckets {
		if bucket != nil {
			// MPHF + ranks array + delimiter
			bucketSize += bucket.mphf.Size()
			bucketSize += len(bucket.ranks)
			delimiterSize += int(bucket.delimiter.Size())/8 + 1
		}
	}

	t.Logf("Component sizes:")
	t.Logf("  Delimiters: %d bytes", delimiterSize)
	t.Logf("  Buckets: %d bytes", bucketSize)

	if delimiterSize <= 0 {
		t.Errorf("Delimiter size should be positive, got %d", delimiterSize)
	}

	if bucketSize <= 0 {
		t.Errorf("Bucket size should be positive, got %d", bucketSize)
	}
}

func TestTrieExcludesDebugData(t *testing.T) {
	t.Parallel()
	// Create a small but robust trie to test that debug data is excluded from size calculation
	rawKeys := buildUniqueStrKeys(50) // Use more keys for robustness
	keys := make([]bits.BitString, len(rawKeys))
	for i, s := range rawKeys {
		keys[i] = bits.NewFromText(s)
	}

	// Sort the keys as required by the implementation
	sort.Sort(bitStringSorter(keys))

	mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](keys)
	if err != nil {
		t.Fatalf("Failed to create MonotoneHashWithTrie: %v", err)
	}

	if mh.delimiterTrie != nil {
		trieSize := mh.delimiterTrie.ByteSize()

		// The size should not include the debug trie (which would be much larger)
		// We expect a relatively small size for just the MPH + NodeData
		t.Logf("Trie size (excluding debug trie): %d bytes", trieSize)

		// Rough sanity check - should be reasonable for a small trie
		if trieSize > 10000 { // Arbitrary upper bound for small test
			t.Errorf("Trie size seems too large (%d bytes), might include debug data", trieSize)
		}

		if trieSize <= 0 {
			t.Errorf("Trie size should be positive, got %d", trieSize)
		}
	}
}
