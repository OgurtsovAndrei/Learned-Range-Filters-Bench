package hzft

import (
	"Thesis/bits"
	"Thesis/bits/maps"
	"Thesis/utils/errutil"
	boomphf "Thesis-bench-industry/experiments/mmph/go-boomphf-bs"
	"fmt"
)

// streamingBuilder builds HZFT without materializing the full heavy ZFastTrie.
// It processes sorted keys one at a time, emitting handle→extentLen pairs on-the-fly.
// Memory usage: O(n × log L) instead of O(n × L) where L is average key length.
type streamingBuilder[E UNumber] struct {
	kv *maps.BitMap[HNodeData[E]]
}

func newStreamingBuilder[E UNumber]() *streamingBuilder[E] {
	return &streamingBuilder[E]{
		kv: maps.NewBitMap[HNodeData[E]](),
	}
}

// emit adds a node with given extent depth and parent depth to the builder.
// It computes the handle (descriptor) and pseudo-descriptors automatically.
//
// Parameters:
//   - depth: the extent length of the node (how far the extent extends)
//   - parentDepth: the name length - 1 of the node (depth of parent's extent)
//   - key: a key that has this node's extent as prefix (used to extract the handle)
func (b *streamingBuilder[E]) emit(depth, parentDepth int, key bits.BitString) {
	a := uint64(parentDepth)

	// My parentDepth logic relies on stack, which starts at 0.
	// For Root, parentDepth is 0 (conceptual parent).
	// ZFastTrie Root has NameLength 0.
	// If NameLength > 0, a = NameLength - 1.
	// If NameLength == 0, a = 0 (per my previous analysis of HZFT code).
	// So passing 0 is consistent.

	extentLen := uint64(depth)

	// Compute the 2-fattest number in (a, extentLen] to get handle length
	original := bits.TwoFattest(a, extentLen)

	desc := key.Prefix(int(original))

	b.kv.Put(desc, HNodeData[E]{
		extentLen: E(extentLen),
	})

	if original == 0 {
		return
	}

	// Add pseudo-descriptors for all 2-fattest numbers in (a, original)
	b_pseudo := original - 1
	for a < b_pseudo {
		ftst := bits.TwoFattest(a, b_pseudo)
		descPseudo := key.Prefix(int(ftst))

		b.kv.Put(descPseudo, HNodeData[E]{
			extentLen: ^E(0), // infinity - marks pseudo-descriptor
		})
		b_pseudo = ftst - 1
	}
}

// NewHZFastTrieFromIteratorStreaming creates an HZFT without building heavy ZFastTrie.
// The iterator MUST provide keys in sorted order (use CheckedSortedIterator if unsure).
//
// Algorithm:
// 1. Process keys one at a time, computing LCP with previous key
// 2. Use a stack to track the path of "open" internal nodes
// 3. When LCP decreases, "close" nodes by emitting their handles
// 4. Build MPH from collected handles
//
// Memory: Only stores handles (short prefixes) and one stack of depths.
// Does NOT store full keys in memory.
func NewHZFastTrieFromIteratorStreaming[E UNumber](iter bits.BitStringIterator) (*HZFastTrie[E], error) {
	b := newStreamingBuilder[E]()

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

	// Build MPH from collected handles
	keysForMPH := make([]bits.BitString, 0, b.kv.Len())
	b.kv.Range(func(k bits.BitString, _ HNodeData[E]) bool {
		keysForMPH = append(keysForMPH, k)
		return true
	})

	mph := boomphf.NewDefault(keysForMPH)

	data := make([]HNodeData[E], len(keysForMPH))
	b.kv.Range(func(key bits.BitString, value HNodeData[E]) bool {
		idx := mph.Query(key) - 1
		errutil.BugOn(idx >= uint64(len(data)), "Out of bounds")
		data[idx] = value
		return true
	})

	// Compute root handle
	rootA := uint64(^uint64(0)) // Use -1 as conceptual parent for root to get descriptor length 0
	rootB := uint64(rootExtentLen)
	rootOriginal := bits.TwoFattest(rootA, rootB)
	rootHandle := firstKey.Prefix(int(rootOriginal))

	rootQuery := mph.Query(rootHandle)
	if rootQuery == 0 {
		return nil, fmt.Errorf("Root not found in MPH")
	}
	rootId := rootQuery - 1

	return &HZFastTrie[E]{
		mph:    mph,
		data:   data,
		rootId: rootId,
	}, nil
}
