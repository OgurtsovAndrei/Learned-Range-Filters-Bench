package bucket

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/azft"
	"fmt"
	"sort"
	"testing"
)

func TestO1BucketLookup(t *testing.T) {
	t.Parallel()
	// Test that we're actually using O(1) bucket lookup via rank
	// rather than linear search through delimiters

	// Create a dataset with many buckets to make linear search expensive
	keys := buildUniqueStrKeys(10000)
	bitKeys := make([]bits.BitString, len(keys))
	for i, s := range keys {
		bitKeys[i] = bits.NewFromText(s)
	}
	sort.Sort(bitStringSorter(bitKeys))

	mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](bitKeys)
	if err != nil {
		t.Fatalf("Failed to create MonotoneHashWithTrie: %v", err)
	}

	t.Logf("Created structure with %d buckets (bucket size: %d)", len(mh.buckets), mh.bucketSize)

	// Verify all keys can be found correctly
	for i, key := range bitKeys {
		rank := mh.GetRank(key)
		if rank != i {
			t.Errorf("Key %d: expected rank %d, got %d", i, i, rank)
		}
	}

	// Test that the rank field is properly set in trie nodes
	if mh.delimiterTrie != nil {
		// Use the approximate z-fast trie to get candidates for the bucket
		cand1, cand2, cand3, cand4, cand5, cand6 := mh.delimiterTrie.LowerBound(bitKeys[len(bitKeys)/2])

		maxDelimiterIndex := uint16(^uint16(0))
		foundValidIndex := false

		// Check if any candidate has a valid rank
		candidates := []*azft.NodeData[uint8, uint16, uint16]{cand1, cand2, cand3, cand4, cand5, cand6}

		for _, candidate := range candidates {
			if candidate != nil && candidate.Rank != maxDelimiterIndex {
				foundValidIndex = true
				bucketIdx := int(candidate.Rank)
				t.Logf("Found candidate with rank: %d (max possible: %d)",
					candidate.Rank, maxDelimiterIndex)

				// Verify the bucket index is valid
				if bucketIdx >= len(mh.buckets) {
					t.Errorf("rank %d is out of range (max: %d)",
						bucketIdx, len(mh.buckets)-1)
				}
				break
			}
		}

		if !foundValidIndex {
			t.Error("No candidate had a valid rank - O(1) lookup not working")
		}
	}
}

func BenchmarkBucketLookupScaling(b *testing.B) {
	// Benchmark to show that lookup time doesn't scale with number of buckets
	// (which would happen with linear search)

	sizes := []int{1000, 5000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Keys=%d", size), func(b *testing.B) {
			keys := buildUniqueStrKeys(size)
			bitKeys := make([]bits.BitString, len(keys))
			for i, s := range keys {
				bitKeys[i] = bits.NewFromText(s)
			}
			sort.Sort(bitStringSorter(bitKeys))

			mh, err := NewMonotoneHashWithTrie[uint8, uint16, uint16](bitKeys)
			if err != nil {
				b.Fatalf("Failed to create MonotoneHashWithTrie: %v", err)
			}

			// Test key from middle of dataset
			testKey := bitKeys[len(bitKeys)/2]

			b.ReportMetric(float64(len(mh.buckets)), "num_buckets")
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_ = mh.GetRank(testKey)
			}
		})
	}
}

// Helpers (buildUniqueStrKeys, bitStringSorter) assumed to be in other files in package bucket
// ... they are likely in prop_test.go or similar.