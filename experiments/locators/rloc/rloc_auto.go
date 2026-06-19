package rloc

import (
	"Thesis/utils/errutil"
	"Thesis-bench-industry/experiments/mmph/paramselect"
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
)

const rlocBucketSize = 256

type autoBuildPlan struct {
	eBits       int
	iBits       int
	sCandidates []int
}

func makeAutoBuildPlan(pSize int, maxBitLen int) (autoBuildPlan, error) {
	errutil.BugOn(pSize < 0, "pSize must be non-negative, got %d", pSize)
	errutil.BugOn(maxBitLen < 0, "maxBitLen must be non-negative, got %d", maxBitLen)

	if pSize == 0 {
		return autoBuildPlan{
			eBits:       8,
			iBits:       8,
			sCandidates: []int{8},
		}, nil
	}

	numBuckets := paramselect.BucketCount(pSize, rlocBucketSize)
	eBits := paramselect.WidthForBitLength(maxBitLen)
	iBits := paramselect.WidthForDelimiterTrieIndex(numBuckets)

	// Relative-trie setup from the paper: epsilon_query = m/n.
	sMin := paramselect.SignatureBitsRelativeTrie(maxBitLen, pSize, numBuckets)
	candidates := filterWidthsUpTo32(paramselect.WidthCandidates(sMin))
	if len(candidates) == 0 {
		return autoBuildPlan{}, fmt.Errorf("unsupported signature width: need >= %d bits", sMin)
	}

	return autoBuildPlan{
		eBits:       eBits,
		iBits:       iBits,
		sCandidates: candidates,
	}, nil
}

func filterWidthsUpTo32(widths []int) []int {
	out := make([]int, 0, len(widths))
	for _, w := range widths {
		if w <= 32 {
			out = append(out, w)
		}
	}
	return out
}

func buildWithFixedWidths(sortedItems []pItem, mmphSeed uint64, eBits int, sBits int, iBits int) (RangeLocator, error) {
	switch eBits {
	case 8:
		return buildWithFixedWidthsE8(sortedItems, mmphSeed, sBits, iBits)
	case 16:
		return buildWithFixedWidthsE16(sortedItems, mmphSeed, sBits, iBits)
	case 32:
		return buildWithFixedWidthsE32(sortedItems, mmphSeed, sBits, iBits)
	default:
		errutil.Bug("unsupported E width %d", eBits)
		return nil, fmt.Errorf("unsupported E width %d", eBits)
	}
}

func buildWithFixedWidthsE8(sortedItems []pItem, mmphSeed uint64, sBits int, iBits int) (RangeLocator, error) {
	switch iBits {
	case 8:
		return buildWithFixedWidthsSI[uint8, uint8](sortedItems, mmphSeed, sBits)
	case 16:
		return buildWithFixedWidthsSI[uint8, uint16](sortedItems, mmphSeed, sBits)
	case 32:
		return buildWithFixedWidthsSI[uint8, uint32](sortedItems, mmphSeed, sBits)
	default:
		errutil.Bug("unsupported I width %d for E=%d", iBits, 8)
		return nil, fmt.Errorf("unsupported I width %d", iBits)
	}
}

func buildWithFixedWidthsE16(sortedItems []pItem, mmphSeed uint64, sBits int, iBits int) (RangeLocator, error) {
	switch iBits {
	case 8:
		return buildWithFixedWidthsSI[uint16, uint8](sortedItems, mmphSeed, sBits)
	case 16:
		return buildWithFixedWidthsSI[uint16, uint16](sortedItems, mmphSeed, sBits)
	case 32:
		return buildWithFixedWidthsSI[uint16, uint32](sortedItems, mmphSeed, sBits)
	default:
		errutil.Bug("unsupported I width %d for E=%d", iBits, 16)
		return nil, fmt.Errorf("unsupported I width %d", iBits)
	}
}

func buildWithFixedWidthsE32(sortedItems []pItem, mmphSeed uint64, sBits int, iBits int) (RangeLocator, error) {
	switch iBits {
	case 8:
		return buildWithFixedWidthsSI[uint32, uint8](sortedItems, mmphSeed, sBits)
	case 16:
		return buildWithFixedWidthsSI[uint32, uint16](sortedItems, mmphSeed, sBits)
	case 32:
		return buildWithFixedWidthsSI[uint32, uint32](sortedItems, mmphSeed, sBits)
	default:
		errutil.Bug("unsupported I width %d for E=%d", iBits, 32)
		return nil, fmt.Errorf("unsupported I width %d", iBits)
	}
}

func buildWithFixedWidthsSI[E zft.UNumber, I zft.UNumber](sortedItems []pItem, mmphSeed uint64, sBits int) (RangeLocator, error) {
	switch sBits {
	case 8:
		return safeBuildGenericRangeLocatorFromItems[E, uint8, I](sortedItems, mmphSeed)
	case 16:
		return safeBuildGenericRangeLocatorFromItems[E, uint16, I](sortedItems, mmphSeed)
	case 32:
		return safeBuildGenericRangeLocatorFromItems[E, uint32, I](sortedItems, mmphSeed)
	default:
		errutil.Bug("unsupported S width %d", sBits)
		return nil, fmt.Errorf("unsupported S width %d", sBits)
	}
}

func safeBuildGenericRangeLocatorFromItems[E zft.UNumber, S zft.UNumber, I zft.UNumber](sortedItems []pItem, mmphSeed uint64) (rl *GenericRangeLocator[E, S, I], err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic during generic build: %v", r)
			rl = nil
		}
	}()
	return newGenericRangeLocatorFromItems[E, S, I](sortedItems, mmphSeed)
}
