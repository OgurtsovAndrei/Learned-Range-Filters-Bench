package ere_global

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/locators/lerloc"
	"fmt"
	"math"
	"unsafe"

	"github.com/hillbig/rsdic"
)

// GlobalExactRangeEmptiness implements 1D range emptiness using a single global
// LocalExactRangeLocator (SuccinctHZFastTrie + RangeLocator) instead of
// per-block locators. Block partitioning and D1/D2 bitvectors are identical
// to ERE and TheoreticalERE; the difference is that WeakPrefixSearch calls go
// through one shared locator built on all n keys.
type GlobalExactRangeEmptiness struct {
	D1      *rsdic.RSDic
	D2      *rsdic.RSDic
	locator lerloc.LocalExactRangeLocator
	keys    []bits.BitString

	n         int
	numBlocks int
	keySize   uint32
	k         uint32
}

func NewGlobalExactRangeEmptiness(keys []bits.BitString, universe bits.BitString) (*GlobalExactRangeEmptiness, error) {
	n := len(keys)
	if n == 0 {
		return &GlobalExactRangeEmptiness{n: 0}, nil
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
	keySize := universe.Size()
	if keySize < k {
		keySize = k
	}

	D1 := rsdic.New()
	D2 := rsdic.New()

	i := 0
	for b := 0; b < numBlocks; b++ {
		countInBlock := 0
		for i < n && getBlockIndex(keys[i], k) == uint64(b) {
			countInBlock++
			i++
		}
		if countInBlock > 0 {
			D1.PushBack(true)
			D2.PushBack(true)
			for c := 0; c < countInBlock; c++ {
				D2.PushBack(false)
			}
		} else {
			D1.PushBack(false)
		}
	}
	D2.PushBack(true) // sentinel

	loc, err := lerloc.NewCompactLocalExactRangeLocator(keys)
	if err != nil {
		return nil, fmt.Errorf("failed to build global locator: %w", err)
	}

	return &GlobalExactRangeEmptiness{
		D1:        D1,
		D2:        D2,
		locator:   loc,
		keys:      keys,
		n:         n,
		numBlocks: numBlocks,
		keySize:   keySize,
		k:         k,
	}, nil
}

func getBlockIndex(x bits.BitString, k uint32) uint64 {
	return x.Prefix(int(k)).TrieUint64()
}

func (ere *GlobalExactRangeEmptiness) IsEmpty(a, b bits.BitString) bool {
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

func (ere *GlobalExactRangeEmptiness) getBlockRange(blockIdx uint64) (int, int) {
	numNonEmptyBefore := int(ere.D1.Rank(blockIdx, true))
	posInD2 := ere.D2.Select(uint64(numNonEmptyBefore), true)
	startIndex := int(posInD2 - uint64(numNonEmptyBefore))
	posEndInD2 := ere.D2.Select(uint64(numNonEmptyBefore+1), true)
	endIndex := int(posEndInD2 - uint64(numNonEmptyBefore+1))
	return startIndex, endIndex
}

// isRangeEmptyInBlock uses the global locator's WeakPrefixSearch to find
// candidates. The returned global ranks are intersected with the block's key
// range [blockStart, blockEnd) to identify block-local candidates.
func (ere *GlobalExactRangeEmptiness) isRangeEmptyInBlock(blockIdx uint64, a, b bits.BitString) bool {
	blockStart, blockEnd := ere.getBlockRange(blockIdx)
	if blockEnd <= blockStart {
		return true
	}

	lcp := a.GetLCPLength(b)

	// p◦0: the largest key in this block prefixed by LCP◦0
	prefix0 := a.Prefix(int(lcp)).AppendBit(false)
	lo0, hi0, _ := ere.locator.WeakPrefixSearch(prefix0)
	effLo0 := max(lo0, blockStart)
	effHi0 := min(hi0, blockEnd)
	if effHi0 > effLo0 {
		candidate := ere.keys[effHi0-1]
		if candidate.Compare(a) >= 0 && candidate.Compare(b) <= 0 {
			return false
		}
	}

	// p◦1: the smallest key in this block prefixed by LCP◦1
	prefix1 := a.Prefix(int(lcp)).AppendBit(true)
	lo1, hi1, _ := ere.locator.WeakPrefixSearch(prefix1)
	effLo1 := max(lo1, blockStart)
	effHi1 := min(hi1, blockEnd)
	if effHi1 > effLo1 {
		candidate := ere.keys[effLo1]
		if candidate.Compare(a) >= 0 && candidate.Compare(b) <= 0 {
			return false
		}
	}

	return true
}

func (ere *GlobalExactRangeEmptiness) ByteSize() int {
	if ere == nil || ere.n == 0 {
		return 0
	}
	size := int(unsafe.Sizeof(*ere))
	size += ere.D1.AllocSize()
	size += ere.D2.AllocSize()
	size += ere.locator.ByteSize()
	return size
}
