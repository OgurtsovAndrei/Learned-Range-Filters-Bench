package lerloc

import (
	"Thesis/bits"
	"Thesis/utils/errutil"
	"Thesis-bench-industry/experiments/locators/rloc"
	"Thesis-bench-industry/experiments/trie/hzft"
	"Thesis-bench-industry/experiments/trie/shzft"
	"Thesis-bench-industry/experiments/trie/zft"
	"Thesis/utils"
	"fmt"
	"unsafe"
)

type TrieType int

const (
	FastTrie    TrieType = iota // Standard HZFT (Fast)
	CompactTrie                 // Succinct HZFT (Compact)
)

type hzftAccessor interface {
	GetExistingPrefix(pattern bits.BitString) int64
	ByteSize() int
	MemDetailed() utils.MemReport
}

type autoLocalExactRangeLocator struct {
	hzft     hzftAccessor
	rl       rloc.RangeLocator
	widths   rloc.TypeWidths
	trieType TrieType
}

// NewLocalExactRangeLocator constructs a LocalExactRangeLocator with standard
// HZFT (Fast mode) and automatically selected bit-widths.
func NewLocalExactRangeLocator(keys []bits.BitString) (LocalExactRangeLocator, error) {
	return NewLocalExactRangeLocatorWithType(keys, FastTrie)
}

// NewCompactLocalExactRangeLocator constructs a LocalExactRangeLocator with
// Succinct HZFT (Compact mode) and automatically selected bit-widths.
func NewCompactLocalExactRangeLocator(keys []bits.BitString) (LocalExactRangeLocator, error) {
	return NewLocalExactRangeLocatorWithType(keys, CompactTrie)
}

// NewLocalExactRangeLocatorWithType constructs a LocalExactRangeLocator with
// the specified Trie type and automatically selected bit-widths.
func NewLocalExactRangeLocatorWithType(keys []bits.BitString, trieType TrieType) (LocalExactRangeLocator, error) {
	zt := zft.Build(keys)

	rl, err := rloc.NewRangeLocator(zt)
	if err != nil {
		return nil, err
	}

	widths := rl.TypeWidths()
	hzftComp, err := buildHZFastTrieWithWidth(keys, widths.E, trieType)
	if err != nil {
		return nil, err
	}

	return &autoLocalExactRangeLocator{
		hzft:     hzftComp,
		rl:       rl,
		widths:   widths,
		trieType: trieType,
	}, nil
}

func (lerl *autoLocalExactRangeLocator) WeakPrefixSearch(prefix bits.BitString) (int, int, error) {
	if lerl == nil || lerl.hzft == nil || lerl.rl == nil {
		return 0, 0, nil
	}

	exitNodeLength := lerl.hzft.GetExistingPrefix(prefix)

	var exitNode bits.BitString
	if exitNodeLength == 0 {
		exitNode = bits.NewFromText("")
	} else {
		exitNode = prefix.Prefix(int(exitNodeLength))
	}

	return lerl.rl.Query(exitNode)
}

func (lerl *autoLocalExactRangeLocator) ByteSize() int {
	if lerl == nil {
		return 0
	}

	size := 0
	if lerl.hzft != nil {
		size += lerl.hzft.ByteSize()
	}
	if lerl.rl != nil {
		size += lerl.rl.ByteSize()
	}
	return size
}

// MemDetailed returns a detailed memory usage report for autoLocalExactRangeLocator.
func (lerl *autoLocalExactRangeLocator) MemDetailed() utils.MemReport {
	if lerl == nil {
		return utils.MemReport{Name: "autoLocalExactRangeLocator", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*lerl))
	hzftReport := lerl.hzft.MemDetailed()
	rlReport := lerl.rl.MemDetailed()

	return utils.MemReport{
		Name:       "autoLocalExactRangeLocator",
		TotalBytes: lerl.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			hzftReport,
			rlReport,
		},
	}
}

func (lerl *autoLocalExactRangeLocator) TypeWidths() rloc.TypeWidths {
	if lerl == nil {
		return rloc.TypeWidths{}
	}
	return lerl.widths
}

func buildHZFastTrieWithWidth(keys []bits.BitString, eWidth int, trieType TrieType) (hzftAccessor, error) {
	if trieType == CompactTrie {
		// SHZFT uses E width internally for delta calculation,
		// but its interface is non-generic.
		return shzft.NewSuccinctHZFastTrie(keys), nil
	}

	switch eWidth {
	case 8:
		return hzft.NewHZFastTrie[uint8](keys), nil
	case 16:
		return hzft.NewHZFastTrie[uint16](keys), nil
	case 32:
		return hzft.NewHZFastTrie[uint32](keys), nil
	default:
		errutil.Bug("unsupported HZFastTrie E width %d", eWidth)
		return nil, fmt.Errorf("unsupported HZFastTrie E width %d", eWidth)
	}
}
