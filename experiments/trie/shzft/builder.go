package shzft

import (
	"Thesis/bits"
	"Thesis/bits/maps"
	"Thesis/utils/errutil"
	boomphf "Thesis-bench-industry/experiments/mmph/go-boomphf-bs"
	"fmt"
	"github.com/hillbig/rsdic"
	math_bits "math/bits"
)

type nodeData struct {
	isTrueDescriptor bool
	extentLen        uint64
}

// streamingBuilder builds SHZFT. It processes sorted keys one at a time,
// emitting handle->extentLen pairs on-the-fly.
type streamingBuilder struct {
	kv *maps.BitMap[nodeData]
}

func newStreamingBuilder() *streamingBuilder {
	return &streamingBuilder{
		kv: maps.NewBitMap[nodeData](),
	}
}

// emit adds a node with given extent depth and parent depth to the builder.
// It computes the handle (descriptor) and pseudo-descriptors automatically.
func (b *streamingBuilder) emit(depth, parentDepth int, key bits.BitString) {
	a := uint64(parentDepth)
	extentLen := uint64(depth)

	// Compute the 2-fattest number in (a, extentLen] to get handle length
	original := bits.TwoFattest(a, extentLen)

	desc := key.Prefix(int(original))

	b.kv.Put(desc, nodeData{
		isTrueDescriptor: true,
		extentLen:        extentLen,
	})

	if original == 0 {
		return
	}

	// Add pseudo-descriptors for all 2-fattest numbers in (a, original)
	b_pseudo := original - 1
	for a < b_pseudo {
		ftst := bits.TwoFattest(a, b_pseudo)
		descPseudo := key.Prefix(int(ftst))

		b.kv.Put(descPseudo, nodeData{
			isTrueDescriptor: false,
			extentLen:        ^uint64(0), // infinity - marks pseudo-descriptor
		})
		b_pseudo = ftst - 1
	}
}

// NewSHZFastTrieFromIteratorStreaming creates an SHZFT without building heavy ZFastTrie.
// The iterator MUST provide keys in sorted order.
func NewSHZFastTrieFromIteratorStreaming(iter bits.BitStringIterator) (*SuccinctHZFastTrie, error) {
	b := newStreamingBuilder()

	type stackItem struct {
		depth int
	}
	stack := []stackItem{{depth: 0}}

	var prevKey bits.BitString
	var firstKey bits.BitString
	isFirst := true

	for iter.Next() {
		key := iter.Value()
		if isFirst {
			firstKey = key
			isFirst = false
			prevKey = key
			continue
		}

		lcp := int(prevKey.GetLCPLength(key))
		d := int(prevKey.Size())

		// Close nodes that are no longer on the path
		for d > lcp {
			topDepth := -1
			if len(stack) > 0 {
				topDepth = stack[len(stack)-1].depth
			}

			p_d := lcp
			if topDepth > lcp {
				p_d = topDepth
			}

			b.emit(d, p_d, prevKey)

			d = p_d

			if len(stack) > 0 && stack[len(stack)-1].depth == d {
				stack = stack[:len(stack)-1]
			}
		}

		// Push new branching point if needed
		if lcp > 0 {
			topDepth := -1
			if len(stack) > 0 {
				topDepth = stack[len(stack)-1].depth
			}

			if topDepth < lcp {
				stack = append(stack, stackItem{depth: lcp})
			}
		}

		prevKey = key
	}

	if err := iter.Error(); err != nil {
		return nil, err
	}

	if !firstKey.HasValue() {
		return nil, nil
	}

	// Close remaining nodes after last key
	d := int(prevKey.Size())
	lcp := 0

	for d > lcp {
		topDepth := -1
		if len(stack) > 0 {
			topDepth = stack[len(stack)-1].depth
		}

		p_d := lcp
		if topDepth > lcp {
			p_d = topDepth
		}

		b.emit(d, p_d, prevKey)

		d = p_d

		if len(stack) > 0 && stack[len(stack)-1].depth == d {
			stack = stack[:len(stack)-1]
		}
	}

	// Emit root node
	rootExtentLen := int(firstKey.GetLCPLength(prevKey))
	b.emit(rootExtentLen, -1, firstKey)

	// Phase 1: Build MPH from collected handles
	keysForMPH := make([]bits.BitString, 0, b.kv.Len())
	b.kv.Range(func(k bits.BitString, _ nodeData) bool {
		keysForMPH = append(keysForMPH, k)
		return true
	})

	mph := boomphf.NewDefault(keysForMPH)

	// Phase 2: Create temporary arrays mapped by MPH index
	totalEntries := len(keysForMPH)
	isTrueDescriptorArray := make([]bool, totalEntries)
	extentLenArray := make([]uint64, totalEntries)
	descriptorLenArray := make([]uint32, totalEntries)

	b.kv.Range(func(key bits.BitString, value nodeData) bool {
		idx := mph.Query(key) - 1
		errutil.BugOn(idx >= uint64(totalEntries), "Out of bounds")
		isTrueDescriptorArray[idx] = value.isTrueDescriptor
		extentLenArray[idx] = value.extentLen
		descriptorLenArray[idx] = key.Size()
		return true
	})

	// Phase 3: Build Bitvector (RSDic)
	bv := rsdic.New()
	for i := 0; i < totalEntries; i++ {
		bv.PushBack(isTrueDescriptorArray[i])
	}
	// rsdic doesn't need explicit Build() call after PushBack in this library

	// Phase 4: Delta-Encoding for True Descriptors
	numTrueDescriptors := int(bv.Rank(uint64(totalEntries), true))
	var maxDelta uint64 = 0
	deltas := make([]uint64, numTrueDescriptors)

	for i := 0; i < totalEntries; i++ {
		if isTrueDescriptorArray[i] {
			rank := int(bv.Rank(uint64(i), true))
			delta := extentLenArray[i] - uint64(descriptorLenArray[i])
			deltas[rank] = delta
			if delta > maxDelta {
				maxDelta = delta
			}
		}
	}

	// Determine required bits for Delta
	deltaBits := 0
	if maxDelta > 0 {
		deltaBits = math_bits.Len64(maxDelta)
	}

	// Pack Deltas
	packedDeltas := packBits(deltas, deltaBits)

	// Phase 5: Root Id
	rootA := uint64(^uint64(0)) // Use -1 as conceptual parent for root to get descriptor length 0
	rootB := uint64(rootExtentLen)
	rootOriginal := bits.TwoFattest(rootA, rootB)
	rootHandle := firstKey.Prefix(int(rootOriginal))

	rootQuery := mph.Query(rootHandle)
	if rootQuery == 0 {
		return nil, fmt.Errorf("Root not found in MPH")
	}
	rootId := rootQuery - 1

	return &SuccinctHZFastTrie{
		mph:       mph,
		bv:        bv,
		deltas:    packedDeltas,
		deltaBits: deltaBits,
		rootId:    rootId,
	}, nil
}
