package hzft

import (
	"Thesis/bits"
	"Thesis/utils/errutil"
	boomphf "Thesis-bench-industry/experiments/mmph/go-boomphf-bs"
	"Thesis/utils"
	"fmt"
	"strings"
	"unsafe"
)

// see https://arxiv.org/abs/1804.04720

type UNumber interface {
	~uint8 | ~uint16 | ~uint32 | ~uint64
}

type HNodeData[E UNumber] struct {
	extentLen E // Length of the extent (prefix) represented by this node, should be at least log w, -1 stands for inf
}

type HZFastTrie[E UNumber] struct {
	mph    *boomphf.H     // Minimal Perfect Hash function mapping prefixes to indices.
	data   []HNodeData[E] // Flat array of node data, indexed via the MPH.
	rootId uint64
}

func areSorted(keys []bits.BitString) bool {
	for i := 0; i < len(keys)-1; i++ {
		if keys[i].Compare(keys[i+1]) > 0 {
			return false
		}
	}
	return true
}

func NewHZFastTrieFromIterator[E UNumber](iter bits.BitStringIterator) (*HZFastTrie[E], error) {
	// Use streaming builder for reduced memory usage
	return NewHZFastTrieFromIteratorStreaming[E](iter)
}

func NewHZFastTrie[E UNumber](keys []bits.BitString) *HZFastTrie[E] {
	errutil.BugOn(!areSorted(keys), "Keys should be sorted")

	hzft, err := NewHZFastTrieFromIterator[E](bits.NewSliceBitStringIterator(keys))
	if err != nil {
		panic(err)
	}
	return hzft
}

func (hzft *HZFastTrie[E]) GetExistingPrefix(pattern bits.BitString) int64 {
	if hzft == nil || len(hzft.data) == 0 || pattern.IsEmpty() {
		return 0
	}

	patternLength := uint32(pattern.Size())
	rootData := hzft.data[hzft.rootId]
	rootextentLen := uint32(rootData.extentLen)

	if rootextentLen >= patternLength {
		return 0
	}

	l := uint64(rootextentLen)
	r := uint64(patternLength)
	maxI := bits.MostSignificantBit(uint64(patternLength))

	for i := maxI; i >= 0; i-- {
		if r-l <= 1 {
			break
		}
		f := uint64((r-1)>>uint(i)) << uint(i)

		// Check if f is in (l, r)
		if f > l && f < r {
			g := hzft.queryT(pattern.Prefix(int(f)))
			if g >= patternLength {
				r = f
			} else {
				l = uint64(g)
			}
		}
	}

	return int64(l + 1)
}

// queryT implements the function T from the paper
// Maps descriptors to extent lengths, pseudo-descriptors to infinity
func (hzft *HZFastTrie[E]) queryT(prefix bits.BitString) uint32 {
	nodeData := hzft.getNodeData(prefix)
	if nodeData == nil {
		return ^uint32(0) // infinity - not found
	}
	if uint64(nodeData.extentLen) == uint64(^E(0)) {
		return ^uint32(0) // infinity - pseudo-descriptor
	}
	return uint32(nodeData.extentLen)
}

func (hzft *HZFastTrie[E]) getNodeData(bitString bits.BitString) *HNodeData[E] {
	// Query return values from 1 to n, 0 used for no Entry
	query := hzft.mph.Query(bitString)
	if query == 0 {
		return nil
	}
	id := query - 1
	return &hzft.data[id]
}

func (hzft *HZFastTrie[E]) String() string {
	var sb strings.Builder
	sb.WriteString("HZFastTrie:\n")
	sb.WriteString(fmt.Sprintf("| rootId: %d\n", hzft.rootId))
	sb.WriteString(fmt.Sprintf("| data length: %d\n", len(hzft.data)))

	sb.WriteString("| MPH mappings:\n")
	for i, nodeData := range hzft.data {
		sb.WriteString(fmt.Sprintf("  [%d] extentLen=%d", i, nodeData.extentLen))
		if uint64(nodeData.extentLen) == uint64(^E(0)) {
			sb.WriteString(" (pseudo-descriptor)")
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// ByteSize returns the total size of the structure in bytes.
func (hzft *HZFastTrie[E]) ByteSize() int {
	if hzft == nil {
		return 0
	}

	size := 0

	if hzft.mph != nil {
		size += hzft.mph.Size()
	}

	nodeDataSize := len(hzft.data) * int(unsafe.Sizeof(*new(E)))
	size += nodeDataSize

	size += 8

	return size
}

// MemDetailed returns a detailed memory usage report for HZFastTrie.
func (hzft *HZFastTrie[E]) MemDetailed() utils.MemReport {
	if hzft == nil {
		return utils.MemReport{Name: "HZFastTrie", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*hzft))
	mphSize := 0
	if hzft.mph != nil {
		mphSize = hzft.mph.Size()
	}
	dataSize := len(hzft.data) * int(unsafe.Sizeof(*new(E)))

	return utils.MemReport{
		Name:       "hzft",
		TotalBytes: hzft.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			{Name: "mph", TotalBytes: mphSize},
			{Name: "data", TotalBytes: dataSize},
		},
	}
}