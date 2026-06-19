package rloc

import (
	"Thesis/bits"
	"Thesis/bits/maps"
	"Thesis/utils/errutil"
	bucket "Thesis-bench-industry/experiments/mmph/relative_trie"
	"Thesis-bench-industry/experiments/trie/zft"
	"Thesis/utils"
	"fmt"
	"math/rand"
	"sort"
	"time"
	"unsafe"

	"github.com/hillbig/rsdic"
)

// TypeWidths stores bit-widths selected for generic integer parameters.
type TypeWidths struct {
	E int
	S int
	I int
}

// RangeLocator maps trie node names to leaf-rank intervals.
//
// This is the runtime-facing interface. The generic implementation is
// GenericLocalExactRangeLocator[E, S, I].
type RangeLocator interface {
	Query(nodeName bits.BitString) (int, int, error)
	ByteSize() int
	MemDetailed() utils.MemReport
	TypeWidths() TypeWidths
}

// GenericRangeLocator maps trie node names to leaf-rank intervals.
//
// Memory analysis:
//
//   - Theoretical asymptotics (from "Fast Prefix Search in Little Space, with
//     Applications", Section 4, and MMPH results):
//     for n keys with average length l, the construction stores a boundary set
//     P with |P| = O(n) (up to three boundary strings per internal trie node
//     before deduplication).
//
//     The leaf-marker bitvector with rank support is
//     O(|P|) = O(n) bits (paper-level bound: <= 3n + o(n) for this component),
//     and the MMPH on P is O(|P| log log l) bits (or O(|P| log l) with the
//     simpler variant).
//
//     Total RangeLocator asymptotic space is therefore
//     O(n log log l) bits (or O(n log l) bits in the simpler variant).
//
//   - Concrete field-level resident memory in this implementation (64-bit):
//     struct payload is 24 bytes (mmph pointer + bv pointer + int totalLeaves).
//     Real resident usage is dominated by pointed objects:
//     mmph memory + rsdic memory. The rsdic object has fixed in-struct metadata
//     (~200 bytes: 6 slice headers + 7 uint64 counters) plus backing arrays
//     reported by bv.AllocSize().
//
//   - Practical estimate from fields:
//     24 + mmph.ByteSize() + 200 + bv.AllocSize() bytes.
//
//   - Empirical resident-size range from recent BenchmarkMemoryComparison runs
//     (see mmph/relative_trie/study/memory_bench_v2.txt):
//     about 51.27..106.00 bits/key, with ~51..59 bits/key in the larger-key
//     regime (keys >= 8192 in that run).
//
// References:
//   - papers/Fast Prefix Search.pdf (range-locator idea and space bounds)
//   - papers/MMPH/Section-4-Relative-Ranking.md
//   - papers/MMPH/Section-5-Relative-Trie.md
type GenericRangeLocator[E zft.UNumber, S zft.UNumber, I zft.UNumber] struct {
	mmph        *bucket.MonotoneHashWithTrie[E, S, I]
	bv          *rsdic.RSDic
	totalLeaves int
}

type pItem struct {
	bs     bits.BitString
	isLeaf bool
}

// NewRangeLocator builds a RangeLocator using a random seed for MMPH
// construction.
//
// The constructor chooses the smallest practical type widths for E/S/I from
// input data and escalates to wider types if needed.
func NewRangeLocator(zt *zft.ZFastTrie[bool]) (RangeLocator, error) {
	// Use a random seed for MMPH construction
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	return NewRangeLocatorSeeded(zt, rng.Uint64())
}

// NewGenericRangeLocator builds a generic RangeLocator using a random seed.
func NewGenericRangeLocator[E zft.UNumber, S zft.UNumber, I zft.UNumber](zt *zft.ZFastTrie[bool]) (*GenericRangeLocator[E, S, I], error) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	return NewGenericRangeLocatorSeeded[E, S, I](zt, rng.Uint64())
}

// NewRangeLocatorSeeded builds a RangeLocator from a compacted trie and a seed.
//
// Construction follows the range-locator idea from "Fast Prefix Search in Little
// Space, with Applications": build a boundary set P from trie extents, index P
// with a monotone minimal perfect hash, and store leaf markers in a rankable
// bitvector.
//
// The constructor chooses deterministic widths:
//   - E from max key bit-length;
//   - I from delimiter count upper bound (2*m with m buckets);
//   - S from relative-trie formula, with escalation only across {8,16,32} if
//     construction still fails for the chosen seed.
func NewRangeLocatorSeeded(zt *zft.ZFastTrie[bool], mmphSeed uint64) (RangeLocator, error) {
	if zt == nil {
		return &GenericRangeLocator[uint8, uint8, uint8]{totalLeaves: 0}, nil
	}

	sortedItems, maxBitLen := collectPSortedItems(zt)
	plan, err := makeAutoBuildPlan(len(sortedItems), maxBitLen)
	if err != nil {
		return nil, fmt.Errorf("failed to create auto build plan for P size %d: %w", len(sortedItems), err)
	}

	var lastErr error
	for _, sBits := range plan.sCandidates {
		rl, err := buildWithFixedWidths(sortedItems, mmphSeed, plan.eBits, sBits, plan.iBits)
		if err == nil {
			return rl, nil
		}
		lastErr = err
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("no width choices available")
	}
	return nil, fmt.Errorf(
		"failed to build auto RangeLocator for P size %d (E=%d I=%d S=%v): %w",
		len(sortedItems),
		plan.eBits,
		plan.iBits,
		plan.sCandidates,
		lastErr,
	)
}

// NewGenericRangeLocatorSeeded builds a generic RangeLocator from a compacted
// trie and an explicit seed.
func NewGenericRangeLocatorSeeded[E zft.UNumber, S zft.UNumber, I zft.UNumber](zt *zft.ZFastTrie[bool], mmphSeed uint64) (*GenericRangeLocator[E, S, I], error) {
	if zt == nil {
		return &GenericRangeLocator[E, S, I]{totalLeaves: 0}, nil
	}

	sortedItems, _ := collectPSortedItems(zt)
	return newGenericRangeLocatorFromItems[E, S, I](sortedItems, mmphSeed)
}

func collectPSortedItems(zt *zft.ZFastTrie[bool]) ([]pItem, int) {
	pMap := maps.NewBitMap[bool]()
	maxBitLen := 0

	addToMap := func(bs bits.BitString, isLeaf bool) {
		if existingIsLeaf, exists := pMap.Get(bs); exists {
			// Prioritize leaf status
			if isLeaf {
				pMap.Put(bs, true)
			} else {
				pMap.Put(bs, existingIsLeaf)
			}
		} else {
			pMap.Put(bs, isLeaf)
		}
		if int(bs.Size()) > maxBitLen {
			maxBitLen = int(bs.Size())
		}
	}

	it := zft.NewIterator(zt)
	for it.Next() {
		node := it.Node()
		errutil.BugOn(node == nil, "node should not be nil")

		extent := node.Extent

		// Use TrimTrailingZeros instead of string conversion and trimming
		eArrowBs := extent.TrimTrailingZeros()
		addToMap(eArrowBs, node.IsLeaf)

		// Use AppendBit instead of string concatenation
		e1Bs := extent.AppendBit(true)
		addToMap(e1Bs, false)

		if !extent.IsAllOnes() {
			successor := extent.AppendBit(true).Successor()
			// Use TrimTrailingZeros instead of string conversion and trimming
			succArrowBs := successor.TrimTrailingZeros()
			addToMap(succArrowBs, false)
		}
	}

	sortedItems := make([]pItem, 0, pMap.Len())
	pMap.Range(func(bs bits.BitString, isLeaf bool) bool {
		sortedItems = append(sortedItems, pItem{
			bs:     bs,
			isLeaf: isLeaf,
		})
		return true
	})

	sort.Slice(sortedItems, func(i, j int) bool {
		return sortedItems[i].bs.Compare(sortedItems[j].bs) < 0
	})

	return sortedItems, maxBitLen
}

func newGenericRangeLocatorFromItems[E zft.UNumber, S zft.UNumber, I zft.UNumber](sortedItems []pItem, mmphSeed uint64) (*GenericRangeLocator[E, S, I], error) {
	bv := rsdic.New()
	keysForMMPH := make([]bits.BitString, len(sortedItems))

	for i, item := range sortedItems {
		bv.PushBack(item.isLeaf)
		keysForMMPH[i] = item.bs
	}
	bits.BugIfNotSortedOrHaveDuplicates(keysForMMPH)

	// Build MMPH - data is already sorted in TrieCompare order
	mmph, err := bucket.NewMonotoneHashWithTrieSeeded[E, S, I](keysForMMPH, mmphSeed)
	if err != nil {
		return nil, fmt.Errorf("failed to build MMPH for P set of size %d: %w", len(keysForMMPH), err)
	}

	totalLeaves := 0
	if bv.Num() > 0 {
		totalLeaves = int(bv.Rank(bv.Num(), true))
	}

	return &GenericRangeLocator[E, S, I]{
		mmph:        mmph,
		bv:          bv,
		totalLeaves: totalLeaves,
	}, nil
}

// Query returns the half-open interval [i, j) of leaf ranks under nodeName.
//
// If nodeName is empty, the method returns the full range [0, number_of_leaves).
// For non-empty names, bounds are computed using rank(h(x<-)) and
// rank(h((x1+)<-)) as in the range-locator construction.
func (rl *GenericRangeLocator[E, S, I]) Query(nodeName bits.BitString) (int, int, error) {
	if nodeName.Size() == 0 {
		return 0, rl.totalLeaves, nil
	}

	if rl.mmph == nil {
		return 0, 0, fmt.Errorf("MMPH not initialized")
	}

	xArrowBs := nodeName.TrimTrailingZeros()
	lexRankLeft := rl.mmph.GetRank(xArrowBs)

	if lexRankLeft == -1 {
		return 0, 0, fmt.Errorf("key not found in structure")
	}

	i := int(rl.bv.Rank(uint64(lexRankLeft), true))

	var j int

	if nodeName.IsAllOnes() {
		j = rl.totalLeaves
	} else {
		xSucc := nodeName.AppendBit(true).Successor()
		xSuccArrowBs := xSucc.TrimTrailingZeros()

		lexRankRight := rl.mmph.GetRank(xSuccArrowBs)
		if lexRankRight == -1 {
			return i, i, nil
		}

		j = int(rl.bv.Rank(uint64(lexRankRight), true))
	}

	return i, j, nil
}

// ByteSize returns the estimated resident size of RangeLocator in bytes.
func (rl *GenericRangeLocator[E, S, I]) ByteSize() int {
	if rl == nil {
		return 0
	}

	size := 0

	if rl.mmph != nil {
		size += rl.mmph.ByteSize()
	}

	if rl.bv != nil {
		size += rl.bv.AllocSize()
	}

	size += 8

	return size
}

// MemDetailed returns a detailed memory usage report for GenericRangeLocator.
func (rl *GenericRangeLocator[E, S, I]) MemDetailed() utils.MemReport {
	if rl == nil {
		return utils.MemReport{Name: "GenericRangeLocator", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*rl))
	mmphReport := rl.mmph.MemDetailed()
	bvSize := 0
	if rl.bv != nil {
		bvSize = rl.bv.AllocSize()
	}

	return utils.MemReport{
		Name:       "GenericRangeLocator",
		TotalBytes: rl.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			mmphReport,
			{Name: "rsdic_bv", TotalBytes: bvSize},
		},
	}
}

// TypeWidths returns bit-widths of generic integer parameters used by this
// concrete instance.
func (rl *GenericRangeLocator[E, S, I]) TypeWidths() TypeWidths {
	return TypeWidths{
		E: int(unsafe.Sizeof(*new(E))) * 8,
		S: int(unsafe.Sizeof(*new(S))) * 8,
		I: int(unsafe.Sizeof(*new(I))) * 8,
	}
}
