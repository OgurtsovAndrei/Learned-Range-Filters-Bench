package paramselect

import (
	"Thesis/utils/errutil"
	"math"
)

const (
	Width8  = 8
	Width16 = 16
	Width32 = 32
	Width64 = 64
)

var supportedWidths = []int{Width8, Width16, Width32, Width64}

// WidthForMaxValue returns the minimum unsigned integer width (in bits)
// required to represent values in [0..maxInclusive].
func WidthForMaxValue(maxInclusive uint64) int {
	switch {
	case maxInclusive <= 0xFF:
		return Width8
	case maxInclusive <= 0xFFFF:
		return Width16
	case maxInclusive <= 0xFFFFFFFF:
		return Width32
	default:
		return Width64
	}
}

// WidthForCountWithSentinel returns the minimum width (in bits) for indices
// [0..count-1] plus one sentinel value.
//
// The sentinel is represented by value `count`, so this is equivalent to
// WidthForMaxValue(count).
func WidthForCountWithSentinel(count int) int {
	errutil.BugOn(count < 0, "count must be non-negative, got %d", count)
	return WidthForMaxValue(uint64(count))
}

// WidthForBitLength returns the minimum width (in bits) required to store a
// bit-length up to maxBitLen.
func WidthForBitLength(maxBitLen int) int {
	errutil.BugOn(maxBitLen < 0, "maxBitLen must be non-negative, got %d", maxBitLen)
	return WidthForMaxValue(uint64(maxBitLen))
}

// BucketCount returns ceil(totalKeys / bucketSize).
func BucketCount(totalKeys, bucketSize int) int {
	errutil.BugOn(totalKeys < 0, "totalKeys must be non-negative, got %d", totalKeys)
	errutil.BugOn(bucketSize <= 0, "bucketSize must be positive, got %d", bucketSize)
	if totalKeys == 0 {
		return 0
	}
	return (totalKeys + bucketSize - 1) / bucketSize
}

// DelimiterTrieNodeUpperBound returns an upper bound on the number of nodes in
// a binary trie built over numDelimiters leaves.
//
// For a binary tree with m leaves, nodes <= 2m-1, so 2m is a safe bound.
func DelimiterTrieNodeUpperBound(numDelimiters int) int {
	errutil.BugOn(numDelimiters < 0, "numDelimiters must be non-negative, got %d", numDelimiters)
	return 2 * numDelimiters
}

// WidthForDelimiterTrieIndex returns width (in bits) for delimiter-trie node
// indices plus one sentinel.
func WidthForDelimiterTrieIndex(numDelimiters int) int {
	nodesUpper := DelimiterTrieNodeUpperBound(numDelimiters)
	return WidthForCountWithSentinel(nodesUpper)
}

// SignatureBitsProbabilisticTrie returns required PSig width S (in bits) from
// Theorem 4.1 (probabilistic trie):
//
//	S >= log2(log2(w)) + log2(1/epsilonQuery)
//
// where w is max key length in bits and epsilonQuery is per-query failure
// probability target.
func SignatureBitsProbabilisticTrie(maxKeyBits int, epsilonQuery float64) int {
	errutil.BugOn(maxKeyBits <= 0, "maxKeyBits must be positive, got %d", maxKeyBits)
	errutil.BugOn(epsilonQuery <= 0 || epsilonQuery >= 1, "epsilonQuery must be in (0,1), got %f", epsilonQuery)

	loglogW := 0.0
	if maxKeyBits > 2 {
		loglogW = math.Log2(math.Log2(float64(maxKeyBits)))
	}
	required := math.Ceil(loglogW + math.Log2(1.0/epsilonQuery))
	if required < 1 {
		return 1
	}
	return int(required)
}

// SignatureBitsRelativeTrie returns PSig width S (in bits) for the relative
// trie setup of Theorem 5.2 by substituting epsilonQuery = m/n:
//
//	S >= log2(log2(w)) + log2(n/m)
//
// where n is totalKeys and m is numDelimiters.
func SignatureBitsRelativeTrie(maxKeyBits, totalKeys, numDelimiters int) int {
	errutil.BugOn(totalKeys <= 0, "totalKeys must be positive, got %d", totalKeys)
	errutil.BugOn(numDelimiters <= 0, "numDelimiters must be positive, got %d", numDelimiters)
	errutil.BugOn(numDelimiters > totalKeys, "numDelimiters must be <= totalKeys, got %d > %d", numDelimiters, totalKeys)

	epsilonQuery := float64(numDelimiters) / float64(totalKeys)
	return SignatureBitsProbabilisticTrie(maxKeyBits, epsilonQuery)
}

// WidthCandidates returns supported widths >= minBits.
func WidthCandidates(minBits int) []int {
	errutil.BugOn(minBits <= 0, "minBits must be positive, got %d", minBits)

	out := make([]int, 0, len(supportedWidths))
	for _, w := range supportedWidths {
		if w >= minBits {
			out = append(out, w)
		}
	}
	return out
}
