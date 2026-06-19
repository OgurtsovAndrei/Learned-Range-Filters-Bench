package ere_theoretical

import (
	"Thesis/bits"
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

const (
	testRuns    = 1000
	minN        = 50
	maxExtraN   = 500
	fixedBitLen = 64
)

func setupTheoreticalData(rng *rand.Rand, n, bl int) ([]bits.BitString, *TheoreticalExactRangeEmptiness, error) {
	keySet := make(map[uint64]bool)
	sortedKeys := make([]bits.BitString, 0, n)
	for len(keySet) < n {
		val := rng.Uint64()
		if !keySet[val] {
			keySet[val] = true
			sortedKeys = append(sortedKeys, bits.NewFromUint64(val))
		}
	}
	sort.Slice(sortedKeys, func(i, j int) bool {
		return sortedKeys[i].Compare(sortedKeys[j]) < 0
	})

	universe := bits.NewBitString(uint32(bl))
	ere, err := NewTheoreticalExactRangeEmptiness(sortedKeys, universe)
	return sortedKeys, ere, err
}

func TestTheoreticalERE_Property_PointInclusion(t *testing.T) {
	t.Parallel()
	runParallelTheoretical(t, func(t *testing.T, rng *rand.Rand, keys []bits.BitString, ere *TheoreticalExactRangeEmptiness) {
		for j := 0; j < 10; j++ {
			key := keys[rng.Intn(len(keys))]
			if ere.IsEmpty(key, key) {
				t.Errorf("Key %v not found", key)
			}
		}
	})
}

func TestTheoreticalERE_Property_TightOverhang(t *testing.T) {
	t.Parallel()
	runParallelTheoretical(t, func(t *testing.T, rng *rand.Rand, keys []bits.BitString, ere *TheoreticalExactRangeEmptiness) {
		for j := 0; j < 10; j++ {
			key := keys[rng.Intn(len(keys))]
			if !key.IsAllZeros() {
				prev := key.Predecessor()
				if ere.IsEmpty(prev, key) {
					t.Errorf("Range [%v, %v] failed", prev, key)
				}
			}
			if !key.IsAllOnes() {
				next := key.Successor()
				if ere.IsEmpty(key, next) {
					t.Errorf("Range [%v, %v] failed", key, next)
				}
			}
		}
	})
}

func TestTheoreticalERE_Property_MassiveSpan(t *testing.T) {
	t.Parallel()
	runParallelTheoretical(t, func(t *testing.T, rng *rand.Rand, keys []bits.BitString, ere *TheoreticalExactRangeEmptiness) {
		if ere.IsEmpty(keys[0], keys[len(keys)-1]) {
			t.Errorf("Massive span failed")
		}
	})
}

func runParallelTheoretical(t *testing.T, testFn func(t *testing.T, rng *rand.Rand, keys []bits.BitString, ere *TheoreticalExactRangeEmptiness)) {
	for i := 0; i < testRuns; i++ {
		i := i
		t.Run(fmt.Sprintf("Iter/%d", i), func(t *testing.T) {
			t.Parallel()
			rng := rand.New(rand.NewSource(int64(i + 300)))
			keys, ere, err := setupTheoreticalData(rng, minN+rng.Intn(maxExtraN), fixedBitLen)
			if err != nil {
				t.Fatalf("Setup failed: %v", err)
			}
			testFn(t, rng, keys, ere)
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
