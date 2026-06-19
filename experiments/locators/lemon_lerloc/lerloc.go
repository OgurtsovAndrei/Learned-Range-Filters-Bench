package lemon_lerloc

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/locators/lemon_rloc"
	"Thesis-bench-industry/experiments/locators/rloc"
	"Thesis-bench-industry/experiments/trie/shzft"
	"Thesis-bench-industry/experiments/trie/zft"
	"Thesis/utils"
	"unsafe"
)

// LeMonLocalExactRangeLocator supports weak prefix search and returns exact
// rank intervals in the sorted key set. It uses the highly compressed
// LeMonRangeLocator based on LeMonHash, significantly reducing the memory
// footprint compared to the classical generic bucketing approach.
// It also uses SuccinctHZFastTrie for the exit-node locator to minimize memory.
type LeMonLocalExactRangeLocator struct {
	shzft *shzft.SuccinctHZFastTrie
	rl    *lemon_rloc.LeMonRangeLocator
}

// NewLeMonLocalExactRangeLocator creates a new LeMonLocalExactRangeLocator.
// It composes a SuccinctHZFastTrie (to map a query prefix to an exit node) with a
// LeMonRangeLocator (to map that exit node to a rank interval).
//
// It assumes the keys provided are sorted. If not, construction
// will panic.
func NewLeMonLocalExactRangeLocator(keys []bits.BitString) (*LeMonLocalExactRangeLocator, error) {
	// Build RangeLocator first
	// We need to build a ZFastTrie to pass to NewLeMonRangeLocator
	zt := zft.Build(keys)
	rl, err := lemon_rloc.NewLeMonRangeLocator(zt)
	if err != nil {
		return nil, err
	}

	// Build SuccinctHZFastTrie
	shz := shzft.NewSuccinctHZFastTrie(keys)

	return &LeMonLocalExactRangeLocator{
		shzft: shz,
		rl:    rl,
	}, nil
}

// WeakPrefixSearch returns the half-open interval [i, j) of ranks of all
// elements in the original set that share the provided prefix.
func (lerl *LeMonLocalExactRangeLocator) WeakPrefixSearch(prefix bits.BitString) (int, int, error) {
	if lerl == nil || lerl.shzft == nil || lerl.rl == nil {
		return 0, 0, nil
	}

	// 1. Find the exit node using SuccinctHZFastTrie
	exitNodeLength := lerl.shzft.GetExistingPrefix(prefix)

	var exitNode bits.BitString
	if exitNodeLength == 0 {
		exitNode = bits.NewFromText("")
	} else {
		exitNode = prefix.Prefix(int(exitNodeLength))
	}

	// 2. Query the Range Locator with the exit node name
	return lerl.rl.Query(exitNode)
}

// ByteSize returns the estimated resident size in bytes.
func (lerl *LeMonLocalExactRangeLocator) ByteSize() int {
	if lerl == nil {
		return 0
	}

	size := int(unsafe.Sizeof(*lerl))

	if lerl.shzft != nil {
		size += lerl.shzft.ByteSize()
	}

	if lerl.rl != nil {
		size += lerl.rl.ByteSize()
	}

	return size
}

// MemDetailed returns a detailed memory usage report.
func (lerl *LeMonLocalExactRangeLocator) MemDetailed() utils.MemReport {
	if lerl == nil {
		return utils.MemReport{Name: "LeMonLocalExactRangeLocator", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*lerl))
	shzftSize := 0
	if lerl.shzft != nil {
		shzftSize = lerl.shzft.ByteSize()
	}

	var rlReport utils.MemReport
	if lerl.rl != nil {
		rlReport = lerl.rl.MemDetailed()
	}

	return utils.MemReport{
		Name:       "LeMonLocalExactRangeLocator",
		TotalBytes: lerl.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			{Name: "hzft", TotalBytes: shzftSize}, // Keep "hzft" name for benchmark parsing
			rlReport,
		},
	}
}

// TypeWidths returns bit-widths. S and I are hardcoded to 0 since LeMonHash
// resolves types dynamically in C++. E is 0 because SuccinctHZFastTrie is not generic.
func (lerl *LeMonLocalExactRangeLocator) TypeWidths() rloc.TypeWidths {
	return rloc.TypeWidths{
		E: 0,
		S: 0,
		I: 0,
	}
}
