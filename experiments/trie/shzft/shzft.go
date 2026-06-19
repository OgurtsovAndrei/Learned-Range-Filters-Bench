package shzft

import (
	"Thesis/bits"
	"Thesis/utils/errutil"
	boomphf "Thesis-bench-industry/experiments/mmph/go-boomphf-bs"
	"Thesis/utils"
	"fmt"
	"strings"
	"unsafe"

	"github.com/hillbig/rsdic"
)

// SuccinctHZFastTrie is an asymptotically optimal O(N log log L) space
// implementation of the Heavy Z-Fast Trie. It uses a Relative Dictionary
// (Bitvector + Rank/Select) and Delta-encoding to eliminate the memory overhead
// of pseudo-descriptors.
type SuccinctHZFastTrie struct {
	mph       *boomphf.H   // Indexes all descriptors and pseudo-descriptors
	bv        *rsdic.RSDic // Bitvector: 1 = true descriptor, 0 = pseudo-descriptor
	deltas    []uint64     // Bit-packed array of Delta values (extentLen - descriptorLen)
	deltaBits int          // Number of bits used to pack each Delta
	rootId    uint64       // MPH index of the root descriptor
}

func areSorted(keys []bits.BitString) bool {
	for i := 0; i < len(keys)-1; i++ {
		if keys[i].Compare(keys[i+1]) > 0 {
			return false
		}
	}
	return true
}

func NewSuccinctHZFastTrieFromIterator(iter bits.BitStringIterator) (*SuccinctHZFastTrie, error) {
	// Use streaming builder for reduced memory usage
	return NewSHZFastTrieFromIteratorStreaming(iter)
}

func NewSuccinctHZFastTrie(keys []bits.BitString) *SuccinctHZFastTrie {
	errutil.BugOn(!areSorted(keys), "Keys should be sorted")

	shzft, err := NewSuccinctHZFastTrieFromIterator(bits.NewSliceBitStringIterator(keys))
	if err != nil {
		panic(err)
	}
	return shzft
}

// GetExistingPrefix implements Fat Binary Search to find the longest existing prefix.
func (shzft *SuccinctHZFastTrie) GetExistingPrefix(pattern bits.BitString) int64 {
	if shzft == nil || pattern.IsEmpty() {
		return 0
	}

	patternLength := uint32(pattern.Size())
	rootExtentLen := shzft.queryT(shzft.rootId, 0) // root descriptor length is 0

	if rootExtentLen >= patternLength {
		return 0
	}

	l := uint64(rootExtentLen)
	r := uint64(patternLength)
	maxI := bits.MostSignificantBit(uint64(patternLength))

	for i := maxI; i >= 0; i-- {
		if r-l <= 1 {
			break
		}
		f := uint64((r-1)>>uint(i)) << uint(i)

		// Check if f is in (l, r)
		if f > l && f < r {
			queryPrefix := pattern.Prefix(int(f))
			query := shzft.mph.Query(queryPrefix)
			if query == 0 {
				continue // Should not happen if the prefix exists, but safe
			}

			idx := query - 1
			g := shzft.queryT(idx, uint32(f))

			if g >= patternLength {
				r = f
			} else {
				l = uint64(g)
			}
		}
	}

	return int64(l + 1)
}

// queryT implements the function T from the paper.
// Maps true descriptors to extent lengths, pseudo-descriptors to infinity.
func (shzft *SuccinctHZFastTrie) queryT(idx uint64, descriptorLen uint32) uint32 {
	if idx >= shzft.bv.Num() {
		return ^uint32(0) // infinity - out of bounds
	}

	// Check if it's a pseudo-descriptor
	if !shzft.bv.Bit(idx) {
		return ^uint32(0) // infinity
	}

	// It's a true descriptor, get its rank
	rank := shzft.bv.Rank(idx, true)

	// Unpack the Delta value
	var delta uint64 = 0
	if shzft.deltaBits > 0 {
		delta = unpackBit(shzft.deltas, int(rank), shzft.deltaBits)
	}

	// Calculate absolute extent length
	return descriptorLen + uint32(delta)
}

func (shzft *SuccinctHZFastTrie) String() string {
	var sb strings.Builder
	sb.WriteString("SuccinctHZFastTrie:\n")
	sb.WriteString(fmt.Sprintf("| rootId: %d\n", shzft.rootId))
	sb.WriteString(fmt.Sprintf("| total descriptors (BV length): %d\n", shzft.bv.Num()))
	sb.WriteString(fmt.Sprintf("| true descriptors: %d\n", shzft.bv.Rank(shzft.bv.Num(), true)))
	sb.WriteString(fmt.Sprintf("| deltaBits: %d\n", shzft.deltaBits))

	return sb.String()
}

// ByteSize returns the resident size estimate in bytes.
func (shzft *SuccinctHZFastTrie) ByteSize() int {
	if shzft == nil {
		return 0
	}

	size := 0

	if shzft.mph != nil {
		size += shzft.mph.Size()
	}

	if shzft.bv != nil {
		size += shzft.bv.AllocSize()
	}

	size += len(shzft.deltas) * 8
	size += int(unsafe.Sizeof(*shzft))

	return size
}

// MemDetailed returns a detailed memory usage report for SuccinctHZFastTrie.
func (shzft *SuccinctHZFastTrie) MemDetailed() utils.MemReport {
	if shzft == nil {
		return utils.MemReport{Name: "hzft", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*shzft))
	mphSize := 0
	if shzft.mph != nil {
		mphSize = shzft.mph.Size()
	}
	bvSize := 0
	if shzft.bv != nil {
		bvSize = shzft.bv.AllocSize()
	}
	deltaSize := len(shzft.deltas) * 8

	return utils.MemReport{
		Name:       "hzft",
		TotalBytes: shzft.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			{Name: "mph", TotalBytes: mphSize},
			{Name: "shzft_bv", TotalBytes: bvSize},
			{Name: "deltas_array", TotalBytes: deltaSize},
		},
	}
}

// GetNumEntries returns the number of true descriptors + pseudo-descriptors.
func (shzft *SuccinctHZFastTrie) GetNumEntries() int {
	if shzft == nil || shzft.bv == nil {
		return 0
	}
	return int(shzft.bv.Num())
}

// GetTrueEntries returns the number of true descriptors.
func (shzft *SuccinctHZFastTrie) GetTrueEntries() int {
	if shzft == nil || shzft.bv == nil {
		return 0
	}
	return int(shzft.bv.Rank(shzft.bv.Num(), true))
}
