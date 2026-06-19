package rloc

import (
	"Thesis/bits"
	"Thesis/testutils"
	"math"
	"math/rand"
	"sort"
)

var (
	BenchKeyCounts  = testutils.DefaultBenchKeyCounts
	BenchBitLengths = testutils.DefaultBenchBitLengths
)

func GetBenchKeys(bitLen int, count int) []bits.BitString {
	return testutils.GetBenchKeys(bitLen, count)
}

func InitBenchKeys() {
	// No-op for compatibility
}

func GenUniqueBitStrings(seed int64, maxKeys int, maxBitLen int) []bits.BitString {
	r := rand.New(rand.NewSource(seed))

	numKeys := r.Intn(maxKeys) + 1
	minSize := int(math.Log2(float64(maxKeys))) + 1
	bitLen := minSize + r.Intn(maxBitLen-minSize)

	uniqueUints := make(map[uint64]bool)
	mask := uint64(0)
	if bitLen == 64 {
		mask = 0xFFFFFFFFFFFFFFFF
	} else {
		mask = (uint64(1) << uint(bitLen)) - 1
	}

	for len(uniqueUints) < numKeys {
		uniqueUints[r.Uint64()&mask] = true
	}

	keys := make([]bits.BitString, 0, len(uniqueUints))
	for val := range uniqueUints {
		bs := bits.NewFromUint64(val)
		if uint32(bitLen) < bs.Size() {
			bs = bs.Prefix(bitLen)
		}
		keys = append(keys, bs)
	}

	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Compare(keys[j]) < 0
	})
	return keys
}

func GenUniqueBitStringsDebug(seed int64, maxKeys int, maxBitLen int) []bits.BitString {
	// Re-using same logic for now
	return GenUniqueBitStrings(seed, maxKeys, maxBitLen)
}

func FindRange(keys []bits.BitString, prefix bits.BitString) (int, int) {
	start := sort.Search(len(keys), func(i int) bool {
		return keys[i].Compare(prefix) >= 0
	})

	end := start
	for end < len(keys) {
		if !keys[end].HasPrefix(prefix) {
			break
		}
		end++
	}
	return start, end
}
