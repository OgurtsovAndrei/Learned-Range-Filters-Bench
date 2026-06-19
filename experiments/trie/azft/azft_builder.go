package azft

import (
	"Thesis/bits"
	"Thesis/bits/maps"
	"Thesis/utils/errutil"
	boomphf "Thesis-bench-industry/experiments/mmph/go-boomphf-bs"
	"Thesis-bench-industry/experiments/trie/zft"
	"sort"
)

// NewApproxZFastTrieFromIteratorStreaming creates AZFT with reduced memory overhead.
// Instead of keeping the full ZFT in memory after building, we extract what we need
// and discard it immediately.
//
// The iterator MUST provide keys in sorted order.
//
// Note: This still builds a ZFT temporarily, but:
// 1. We don't keep debug references to nodes
// 2. We discard the ZFT as soon as we extract the data
//
// For truly streaming construction that avoids the ZFT entirely, see the HZFT
// streaming builder which only needs extentLen per node.
func NewApproxZFastTrieFromIteratorStreaming[E UNumber, S UNumber, I UNumber](
	iter bits.BitStringIterator,
	seed uint64,
) (*ApproxZFastTrie[E, S, I], error) {

	checkedIter := bits.NewCheckedSortedIterator(iter)
	trie, err := zft.BuildFromIterator(checkedIter)
	if err != nil {
		return nil, err
	}
	if trie == nil || trie.Root == nil {
		return &ApproxZFastTrie[E, S, I]{seed: seed}, nil
	}

	// Extract handles for MPH
	keysForMPH := make([]bits.BitString, 0, trie.Handle2NodeMap.Len())
	trie.Handle2NodeMap.Range(func(handle bits.BitString, _ *zft.Node[bool]) bool {
		keysForMPH = append(keysForMPH, handle)
		return true
	})
	// Sort keysForMPH to ensure deterministic order
	sort.Slice(keysForMPH, func(i, j int) bool {
		return keysForMPH[i].Compare(keysForMPH[j]) < 0
	})

	if len(keysForMPH) == 0 {
		return &ApproxZFastTrie[E, S, I]{seed: seed}, nil
	}

	mph := boomphf.NewDefault(keysForMPH)
	data := make([]NodeData[E, S, I], len(keysForMPH))

	// Create mapping from keys to their delimiter indices
	keyToDelimiterIdx := maps.NewBitMap[int]()

	// Reconstruct ranks by traversing Trie in-order
	trieIter := zft.NewSortedIterator(trie)
	rank := 0
	for trieIter.Next() {
		node := trieIter.Node()
		if node.Value {
			keyToDelimiterIdx.Put(node.Extent, rank)
			rank++
		}
	}

	maxDelimiterIndex := I(^I(0))

	trie.Handle2NodeMap.Range(func(handle bits.BitString, node *zft.Node[bool]) bool {
		idx := mph.Query(handle) - 1
		errutil.BugOn(idx >= uint64(len(data)), "Out of bounds")

		mostLeft := node
		for mostLeft.LeftChild != nil {
			mostLeft = mostLeft.LeftChild
		}
		errutil.BugOn(!mostLeft.Value, "mostLeft should have a value")
		minChildHandle := mostLeft.Handle()
		minChildIdx := mph.Query(minChildHandle) - 1
		errutil.BugOn(minChildIdx >= uint64(len(data)), "Out of bounds")
		minChild := I(minChildIdx)
		errutil.BugOn(uint64(minChild) != minChildIdx, "Data loss on minChild index")

		var minGreaterChild = maxDelimiterIndex
		if node.RightChild != nil {
			mostLeft := node.RightChild
			for mostLeft.LeftChild != nil {
				mostLeft = mostLeft.LeftChild
			}
			errutil.BugOn(!mostLeft.Value, "mostLeft in right subtree should have a value")
			lmcHandle := mostLeft.Handle()
			lmcIdx := mph.Query(lmcHandle) - 1
			errutil.BugOn(lmcIdx >= uint64(len(data)), "Out of bounds")
			minGreaterChild = I(lmcIdx)
			errutil.BugOn(uint64(minGreaterChild) != lmcIdx, "Data loss on minGreaterChild index")
		}

		sig := S(hashBitString(node.Extent, seed))
		extentLength := E(node.ExtentLength())
		errutil.BugOn(uint32(extentLength) != node.ExtentLength(), "Data loss")

		// Determine delimiter index for this node
		delimiterIdx := maxDelimiterIndex
		if delimIdx, exists := keyToDelimiterIdx.Get(node.Extent); exists {
			delimiterIdx = I(delimIdx)
		}

		// Set rightChild index
		var rightChildIdx I = maxDelimiterIndex
		if node.RightChild != nil {
			rcHandle := node.RightChild.Handle()
			rcIdx := mph.Query(rcHandle) - 1
			errutil.BugOn(rcIdx >= uint64(len(data)), "Out of bounds")
			rightChildIdx = I(rcIdx)
			errutil.BugOn(uint64(rightChildIdx) != rcIdx, "Data loss on rightChild index")
		}

		data[idx] = NodeData[E, S, I]{
			extentLen:       extentLength,
			PSig:            sig,
			parent:          maxDelimiterIndex, // will be set correctly in the next loop
			minChild:        minChild,
			minGreaterChild: minGreaterChild,
			rightChild:      rightChildIdx,
			Rank:            delimiterIdx,
		}
		return true
	})

	// Set up parent relationships - find first ancestor where node is in left subtree
	var setParentRecursive func(*zft.Node[bool], I)
	setParentRecursive = func(node *zft.Node[bool], leftAncestor I) {
		if node == nil {
			return
		}

		nodeHandle := node.Handle()
		nodeIdx := mph.Query(nodeHandle) - 1
		if nodeIdx < uint64(len(data)) {
			data[nodeIdx].parent = leftAncestor
		}

		setParentRecursive(node.LeftChild, I(nodeIdx))
		setParentRecursive(node.RightChild, leftAncestor)
	}

	var rootIdx I
	rootHandle := trie.Root.Handle()
	rootQuery := mph.Query(rootHandle)
	if rootQuery == 0 {
		errutil.Bug("Root is empty")
	} else {
		rootIdx = I(rootQuery - 1)
		setParentRecursive(trie.Root, maxDelimiterIndex)
	}

	// Trie is discarded here (goes out of scope) - no debug reference kept
	return &ApproxZFastTrie[E, S, I]{
		mph:    mph,
		data:   data,
		seed:   seed,
		rootId: rootIdx,
	}, nil
}
