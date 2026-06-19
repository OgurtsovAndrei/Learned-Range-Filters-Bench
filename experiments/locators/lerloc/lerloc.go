package lerloc

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/locators/rloc"
	"Thesis-bench-industry/experiments/trie/hzft"
	"Thesis-bench-industry/experiments/trie/zft"
	"Thesis/utils"
	"unsafe"
)

// LocalExactRangeLocator supports weak prefix search and returns exact rank
// intervals in the sorted key set.
//
// This is the runtime-facing interface. The generic implementation is
// GenericLocalExactRangeLocator[E, S, I].
type LocalExactRangeLocator interface {
	WeakPrefixSearch(prefix bits.BitString) (int, int, error)
	ByteSize() int
	MemDetailed() utils.MemReport
	TypeWidths() rloc.TypeWidths
}

// GenericLocalExactRangeLocator supports weak prefix search and returns exact
// rank intervals in the sorted key set.
//
// Memory analysis:
//
//   - Theoretical asymptotics (from "Fast Prefix Search in Little Space, with
//     Applications"): this structure is the sum of an exit-node locator
//     (implemented here as HZFastTrie) and a RangeLocator.
//     Space(LocalExactRangeLocator) = Space(HZFastTrie) + Space(RangeLocator).
//     Using paper-level bounds for these components gives
//     O(n log l) + O(n log log l) = O(n log l) bits total.
//
//   - Concrete field-level resident memory in this implementation (64-bit):
//     struct payload is 16 bytes (hzft pointer + rl pointer), plus the pointed
//     component memory.
//
//   - Practical estimate from fields:
//     16 + hzft.ByteSize() + rl.ByteSize() bytes.
//
//   - Empirical resident-size range from recent BenchmarkMemoryComparison runs
//     (see mmph/relative_trie/study/memory_bench_v2.txt):
//     about 115.3..320.0 bits/key, with ~116..246 bits/key in the larger-key
//     regime (keys >= 8192 in that run).
//
// References:
//   - papers/Fast Prefix Search.pdf (Local Exact Range Locator decomposition)
//   - papers/Hollow-Z-Fast-Trie (Fast Prefix Search)/Section-5.md
//   - papers/Hollow-Z-Fast-Trie (Fast Prefix Search)/Section-6.md
type GenericLocalExactRangeLocator[E zft.UNumber, S zft.UNumber, I zft.UNumber] struct {
	hzft *hzft.HZFastTrie[E]
	rl   *rloc.GenericRangeLocator[E, S, I]
}

// NewGenericLocalExactRangeLocator creates a new GenericLocalExactRangeLocator.
// It composes an HZFastTrie (to map a query prefix to an exit node) with a
// GenericRangeLocator (to map that exit node to a rank interval).
//
// The type parameters E, S, I determine the bit-widths used for internal
// storage. E is used for both HZFastTrie and RangeLocator.
//
// It assumes the keys provided are sorted. If not, HZFastTrie construction
// will panic.
//
// Note: This constructor does not perform automatic parameter selection. It uses
// the provided types E, S, I directly. For automatic selection, use
// NewLocalExactRangeLocator.
func NewGenericLocalExactRangeLocator[E zft.UNumber, S zft.UNumber, I zft.UNumber](keys []bits.BitString) (*GenericLocalExactRangeLocator[E, S, I], error) {
	// Build RangeLocator first
	// We need to build a ZFastTrie to pass to NewGenericRangeLocator
	zt := zft.Build(keys)
	rl, err := rloc.NewGenericRangeLocator[E, S, I](zt)
	if err != nil {
		return nil, err
	}

	// Build HZFastTrie
	// Note: We could reuse zt if HZFastTrie supported construction from ZFastTrie directly,
	// but currently NewHZFastTrie takes keys.
	// Actually NewHZFastTrie builds its own internal trie anyway.
	hzft := hzft.NewHZFastTrie[E](keys)

	return &GenericLocalExactRangeLocator[E, S, I]{
		hzft: hzft,
		rl:   rl,
	}, nil
}

func (lerl *GenericLocalExactRangeLocator[E, S, I]) WeakPrefixSearch(prefix bits.BitString) (int, int, error) {
	if lerl == nil || lerl.hzft == nil || lerl.rl == nil {
		return 0, 0, nil
	}

	// 1. Find the exit node using HZFastTrie
	// The paper says: "Let u be the node of T that is the exit node of P."
	// HZFastTrie returns the length of the exit node extent.
	exitNodeLength := lerl.hzft.GetExistingPrefix(prefix)

	var exitNode bits.BitString
	if exitNodeLength == 0 {
		exitNode = bits.NewFromText("")
	} else {
		exitNode = prefix.Prefix(int(exitNodeLength))
	}

	// 2. Query the Range Locator with the exit node name
	// The paper says: "Locate(u)"
	return lerl.rl.Query(exitNode)
}

func (lerl *GenericLocalExactRangeLocator[E, S, I]) ByteSize() int {
	if lerl == nil {
		return 0
	}

	size := 0

	// Size of the struct itself
	size += int(unsafe.Sizeof(*lerl))

	// It is the sum of HZFastTrie and RangeLocator sizes and excludes temporary
	// structures used during construction.

	// Size of the HZFastTrie
	if lerl.hzft != nil {
		size += lerl.hzft.ByteSize()
	}

	// Size of the RangeLocator
	if lerl.rl != nil {
		size += lerl.rl.ByteSize()
	}

	return size
}

// MemDetailed returns a detailed memory usage report for GenericLocalExactRangeLocator.
func (lerl *GenericLocalExactRangeLocator[E, S, I]) MemDetailed() utils.MemReport {
	if lerl == nil {
		return utils.MemReport{Name: "GenericLocalExactRangeLocator", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*lerl))
	hzftSize := 0
	if lerl.hzft != nil {
		hzftSize = lerl.hzft.ByteSize()
	}
	rlReport := lerl.rl.MemDetailed()

	return utils.MemReport{
		Name:       "GenericLocalExactRangeLocator",
		TotalBytes: lerl.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			{Name: "hzft", TotalBytes: hzftSize},
			rlReport,
		},
	}
}

func (lerl *GenericLocalExactRangeLocator[E, S, I]) TypeWidths() rloc.TypeWidths {
	// Reconstruct TypeWidths from the type parameters
	var e E
	var s S
	var i I
	return rloc.TypeWidths{
		E: int(unsafe.Sizeof(e)) * 8,
		S: int(unsafe.Sizeof(s)) * 8,
		I: int(unsafe.Sizeof(i)) * 8,
	}
}