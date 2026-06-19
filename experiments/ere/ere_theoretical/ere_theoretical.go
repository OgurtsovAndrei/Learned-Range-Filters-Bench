package ere_theoretical

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/locators/lerloc"
	"fmt"
	"math"
	"unsafe"

	"github.com/hillbig/rsdic"
)

// TheoreticalExactRangeEmptiness implements the 1D range emptiness structure
// using the full theoretical approach from SODA 2015, Section 3.2.
// It uses a Weak Prefix Search structure (compact LocalExactRangeLocator with
// SuccinctHZFastTrie + pure-Go RangeLocator) inside each block to achieve
// O(1) worst-case query time.
type TheoreticalExactRangeEmptiness struct {
	D1        *rsdic.RSDic
	D2        *rsdic.RSDic
	locators  []lerloc.LocalExactRangeLocator
	blockKeys [][]bits.BitString

	n         int
	numBlocks int
	L         uint32
	k         uint32
}

func NewTheoreticalExactRangeEmptiness(keys []bits.BitString, universe bits.BitString) (*TheoreticalExactRangeEmptiness, error) {
	n := len(keys)
	if n == 0 {
		return &TheoreticalExactRangeEmptiness{n: 0}, nil
	}

	for i := 1; i < n; i++ {
		if keys[i-1].Compare(keys[i]) > 0 {
			return nil, fmt.Errorf("keys must be sorted")
		}
	}

	k := uint32(math.Floor(math.Log2(float64(n))))
	if k == 0 {
		k = 1
	}

	numBlocks := 1 << k
	L := universe.Size()
	if L < k {
		L = k
	}

	D1 := rsdic.New()
	D2 := rsdic.New()
	locators := make([]lerloc.LocalExactRangeLocator, 0)
	blockKeys := make([][]bits.BitString, 0)

	i := 0
	for b := 0; b < numBlocks; b++ {
		var currentBlockKeys []bits.BitString
		for i < n && getBlockIndex(keys[i], k) == uint64(b) {
			currentBlockKeys = append(currentBlockKeys, keys[i])
			i++
		}

		if len(currentBlockKeys) > 0 {
			D1.PushBack(true)
			D2.PushBack(true)
			for c := 0; c < len(currentBlockKeys); c++ {
				D2.PushBack(false)
			}

			loc, err := lerloc.NewCompactLocalExactRangeLocator(currentBlockKeys)
			if err != nil {
				return nil, fmt.Errorf("failed to build locator for block %d: %w", b, err)
			}
			locators = append(locators, loc)
			blockKeys = append(blockKeys, currentBlockKeys)
		} else {
			D1.PushBack(false)
		}
	}
	D2.PushBack(true) // sentinel

	return &TheoreticalExactRangeEmptiness{
		D1:        D1,
		D2:        D2,
		locators:  locators,
		blockKeys: blockKeys,
		n:         n,
		numBlocks: numBlocks,
		L:         L,
		k:         k,
	}, nil
}

func getBlockIndex(x bits.BitString, k uint32) uint64 {
	return x.Prefix(int(k)).TrieUint64()
}

func (ere *TheoreticalExactRangeEmptiness) IsEmpty(a, b bits.BitString) bool {
	if ere.n == 0 || a.Compare(b) > 0 {
		return true
	}

	blockA := getBlockIndex(a, ere.k)
	blockB := getBlockIndex(b, ere.k)

	if blockA >= uint64(ere.numBlocks) {
		return true
	}
	if blockB >= uint64(ere.numBlocks) {
		blockB = uint64(ere.numBlocks - 1)
	}

	// 1. Check intermediate full blocks
	if blockB > blockA+1 {
		if ere.D1.Rank(blockB, true) > ere.D1.Rank(blockA+1, true) {
			return false
		}
	}

	// 2. Check boundary blocks
	blockMax := bits.NewFromUint64WithLength(^uint64(0), a.Size())
	blockMin := bits.NewBitString(b.Size())
	if blockA == blockB {
		if ere.D1.Bit(blockA) {
			if !ere.isRangeEmptyInBlock(blockA, a, b) {
				return false
			}
		}
	} else {
		if ere.D1.Bit(blockA) {
			if !ere.isRangeEmptyInBlock(blockA, a, blockMax) {
				return false
			}
		}
		if ere.D1.Bit(blockB) {
			if !ere.isRangeEmptyInBlock(blockB, blockMin, b) {
				return false
			}
		}
	}

	return true
}

// isRangeEmptyInBlock implements the Section 3.2 query algorithm using
// WeakPrefixSearch for O(1) in-block queries. Given query [a, b], it computes
// the longest common prefix p of a and b, then checks:
//   - The largest key prefixed by p◦0 (if ≥ a, range is non-empty)
//   - The smallest key prefixed by p◦1 (if ≤ b, range is non-empty)
func (ere *TheoreticalExactRangeEmptiness) isRangeEmptyInBlock(blockIdx uint64, a, b bits.BitString) bool {
	numNonEmptyBefore := int(ere.D1.Rank(blockIdx, true))
	keys := ere.blockKeys[numNonEmptyBefore]
	loc := ere.locators[numNonEmptyBefore]
	nKeys := len(keys)

	lcp := a.GetLCPLength(b)

	// p◦0: the largest key with this prefix might be ≥ a
	prefix0 := a.Prefix(int(lcp)).AppendBit(false)
	_, hi0, _ := loc.WeakPrefixSearch(prefix0)
	if hi0 > 0 && hi0 <= nKeys {
		candidate := keys[hi0-1]
		if candidate.Compare(a) >= 0 && candidate.Compare(b) <= 0 {
			return false
		}
	}

	// p◦1: the smallest key with this prefix might be ≤ b
	prefix1 := a.Prefix(int(lcp)).AppendBit(true)
	lo1, _, _ := loc.WeakPrefixSearch(prefix1)
	if lo1 >= 0 && lo1 < nKeys {
		candidate := keys[lo1]
		if candidate.Compare(a) >= 0 && candidate.Compare(b) <= 0 {
			return false
		}
	}

	return true
}

func (ere *TheoreticalExactRangeEmptiness) ByteSize() int {
	if ere == nil || ere.n == 0 {
		return 0
	}
	size := int(unsafe.Sizeof(*ere))
	size += ere.D1.AllocSize()
	size += ere.D2.AllocSize()
	for _, loc := range ere.locators {
		size += loc.ByteSize()
	}
	return size
}
