package lemon_rloc

import (
	"Thesis/bits"
	"Thesis/bits/maps"
	"Thesis/utils/errutil"
	"Thesis-bench-industry/experiments/locators/lemonhash"
	"Thesis-bench-industry/experiments/trie/zft"
	"Thesis/utils"
	"fmt"
	"sort"
	"unsafe"

	"github.com/hillbig/rsdic"
)

// LeMonRangeLocator maps trie node names to leaf-rank intervals.
// It is heavily optimized for space by using LeMonHash (a learned Monotone Minimal Perfect Hash)
// instead of classical bucketing MMPH methods.
type LeMonRangeLocator struct {
	lh          *lemonhash.LeMonHash
	bv          *rsdic.RSDic
	totalLeaves int
}

type pItem struct {
	bs     bits.BitString
	isLeaf bool
}

// NewLeMonRangeLocator builds a LeMonRangeLocator from a compacted trie.
func NewLeMonRangeLocator(zt *zft.ZFastTrie[bool]) (*LeMonRangeLocator, error) {
	if zt == nil {
		return &LeMonRangeLocator{totalLeaves: 0}, nil
	}

	sortedItems := collectPSortedItems(zt)
	return newLeMonRangeLocatorFromItems(sortedItems)
}

func collectPSortedItems(zt *zft.ZFastTrie[bool]) []pItem {
	pMap := maps.NewBitMap[bool]()

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

	// Sort mathematically by lexicographical order.
	// Since our lemonhash wrapper uses bit-reversal, BitString.Compare order
	// is exactly what LeMonHash needs (byte-lexicographical after reversal).
	sort.Slice(sortedItems, func(i, j int) bool {
		return sortedItems[i].bs.Compare(sortedItems[j].bs) < 0
	})

	return sortedItems
}

func newLeMonRangeLocatorFromItems(sortedItems []pItem) (*LeMonRangeLocator, error) {
	bv := rsdic.New()
	keysForMMPH := make([]bits.BitString, len(sortedItems))

	for i, item := range sortedItems {
		bv.PushBack(item.isLeaf)
		keysForMMPH[i] = item.bs
	}

	// Data must be strictly sorted and deduplicated
	bits.BugIfNotSortedOrHaveDuplicates(keysForMMPH)

	// Build the Learned MMPH
	// Since our lemonhash wrapper reverses bits, keys sorted by TrieCompare
	// will automatically be in correct memcmp order for LeMonHash.
	lh := lemonhash.New(keysForMMPH)
	if lh == nil && len(keysForMMPH) > 1 {
		return nil, fmt.Errorf("failed to build LeMonHash")
	}

	totalLeaves := 0
	if bv.Num() > 0 {
		totalLeaves = int(bv.Rank(bv.Num(), true))
	}

	return &LeMonRangeLocator{
		lh:          lh,
		bv:          bv,
		totalLeaves: totalLeaves,
	}, nil
}

// Query returns the half-open interval [i, j) of leaf ranks under nodeName.
func (rl *LeMonRangeLocator) Query(nodeName bits.BitString) (int, int, error) {
	if nodeName.Size() == 0 {
		return 0, rl.totalLeaves, nil
	}

	if rl.lh == nil {
		return 0, 0, fmt.Errorf("MMPH not initialized")
	}

	xArrowBs := nodeName.TrimTrailingZeros()
	lexRankLeft := rl.lh.Rank(xArrowBs)

	// LeMonHash returns a rank. Unlike classical HDC which might return an invalid rank for non-existent keys,
	// PGM guarantees a mapped rank. If the key exists, it's the exact rank.
	// Note: In an exact range locator, the queried keys (the boundaries) are guaranteed to be in the set P.

	i := int(rl.bv.Rank(uint64(lexRankLeft), true))

	var j int

	if nodeName.IsAllOnes() {
		j = rl.totalLeaves
	} else {
		xSucc := nodeName.AppendBit(true).Successor()
		xSuccArrowBs := xSucc.TrimTrailingZeros()

		lexRankRight := rl.lh.Rank(xSuccArrowBs)
		j = int(rl.bv.Rank(uint64(lexRankRight), true))
	}

	return i, j, nil
}

// ByteSize returns the estimated resident size of LeMonRangeLocator in bytes.
func (rl *LeMonRangeLocator) ByteSize() int {
	if rl == nil {
		return 0
	}

	size := 0

	if rl.lh != nil {
		size += rl.lh.ByteSize()
	}

	if rl.bv != nil {
		size += rl.bv.AllocSize()
	}

	size += 8

	return size
}

// MemDetailed returns a detailed memory usage report.
func (rl *LeMonRangeLocator) MemDetailed() utils.MemReport {
	if rl == nil {
		return utils.MemReport{Name: "LeMonRangeLocator", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*rl))

	lhSize := 0
	if rl.lh != nil {
		lhSize = rl.lh.ByteSize()
	}

	bvSize := 0
	if rl.bv != nil {
		bvSize = rl.bv.AllocSize()
	}

	return utils.MemReport{
		Name:       "LeMonRangeLocator",
		TotalBytes: rl.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			{Name: "lemonhash", TotalBytes: lhSize},
			{Name: "rsdic_bv", TotalBytes: bvSize},
		},
	}
}
