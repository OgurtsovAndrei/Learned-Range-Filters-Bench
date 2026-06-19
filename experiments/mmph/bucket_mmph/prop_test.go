package bucket

import (
	"Thesis/bits"
	"encoding/hex"
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
			s := hex.EncodeToString(b)
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

func TestMonotoneHash_Randomized(t *testing.T) {
	t.Parallel()
	sizes := []int{1, 10, 100, 1_000, 10_000, 100_000, 1_000_000}

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
			mh := NewMonotoneHash(bitKeys)

			for i, key := range bitKeys {
				rank := mh.GetRank(key)
				if rank != i {
					t.Errorf("Mismatch for key index %d: expected rank %d, got %d", i, i, rank)
				}
			}
		})
	}
}
