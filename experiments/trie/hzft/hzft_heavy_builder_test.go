package hzft

import (
	"Thesis/bits"
	"Thesis/bits/maps"
	"Thesis/utils/errutil"
	boomphf "Thesis-bench-industry/experiments/mmph/go-boomphf-bs"
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
)

// NewHZFastTrieFromIteratorHeavy creates an HZFT by first building a full ZFastTrie.
//
// Deprecated: This is the "old" implementation that the streaming version aims to replace.
// It is kept in test files only for performance comparison in benchmarks.
func NewHZFastTrieFromIteratorHeavy[E UNumber](iter bits.BitStringIterator) (*HZFastTrie[E], error) {
	zt, err := zft.BuildFromIterator(iter)
	if err != nil {
		return nil, err
	}
	if zt.Root == nil {
		return nil, nil
	}

	kv := maps.NewBitMap[HNodeData[E]]()
	
	// Traverse ZFT to collect handles and pseudo-descriptors
	var traverse func(n *zft.Node[bool])
	traverse = func(n *zft.Node[bool]) {
		if n == nil {
			return
		}

		a := uint64(n.NameLength - 1)
		if n.NameLength == 0 {
			a = 0
		}
		extentLen := uint64(n.ExtentLength())

		// Main descriptor
		original := bits.TwoFattest(a, extentLen)
		desc := n.Extent.Prefix(int(original))
		kv.Put(desc, HNodeData[E]{extentLen: E(extentLen)})

		// Pseudo-descriptors
		if original > 0 {
			b_pseudo := original - 1
			for a < b_pseudo {
				ftst := bits.TwoFattest(a, b_pseudo)
				descPseudo := n.Extent.Prefix(int(ftst))
				kv.Put(descPseudo, HNodeData[E]{extentLen: ^E(0)})
				b_pseudo = ftst - 1
			}
		}

		traverse(n.LeftChild)
		traverse(n.RightChild)
	}
	traverse(zt.Root)

	// Build MPH from collected handles
	keysForMPH := make([]bits.BitString, 0, kv.Len())
	kv.Range(func(k bits.BitString, _ HNodeData[E]) bool {
		keysForMPH = append(keysForMPH, k)
		return true
	})

	mph := boomphf.NewDefault(keysForMPH)

	data := make([]HNodeData[E], len(keysForMPH))
	kv.Range(func(key bits.BitString, value HNodeData[E]) bool {
		idx := mph.Query(key) - 1
		errutil.BugOn(idx >= uint64(len(data)), "Out of bounds")
		data[idx] = value
		return true
	})

	// Compute root handle
	rootA := uint64(0)
	rootB := uint64(zt.Root.ExtentLength())
	rootOriginal := bits.TwoFattest(rootA, rootB)
	rootHandle := zt.Root.Extent.Prefix(int(rootOriginal))

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
