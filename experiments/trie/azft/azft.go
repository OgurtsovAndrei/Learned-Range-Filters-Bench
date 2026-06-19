package azft

import (
	"Thesis/bits"
	"Thesis/utils/errutil"
	boomphf "Thesis-bench-industry/experiments/mmph/go-boomphf-bs"
	"Thesis/utils"
	"math/rand"
	"unsafe"
)

type UNumber interface {
	~uint8 | ~uint16 | ~uint32 | ~uint64
}

// NodeData contains the packed data for a Trie node.
type NodeData[E UNumber, S UNumber, I UNumber] struct {
	PSig            S
	parent          I
	minChild        I
	minGreaterChild I
	rightChild      I
	Rank            I
	extentLen       E
	// should be at ((log log n) + (log log w) - (log eps)) bits
}

// ApproxZFastTrie is a compact probabilistic implementation of a Z-Fast Trie,
// utilizing Minimal Perfect Hashing (MPH) instead of explicit pointers.
type ApproxZFastTrie[E UNumber, S UNumber, I UNumber] struct {
	mph    *boomphf.H          // Minimal Perfect Hash function mapping prefixes to indices.
	data   []NodeData[E, S, I] // Flat array of node data, indexed via the MPH.
	seed   uint64              // Seed used for computing PSig signatures.
	rootId I
}

// NewApproxZFastTrie initializes a compact Trie with delimiter index information.
// This is used for bucketing where we need to map Trie nodes to bucket indices.
// Uses a random seed from the global rand package.
func NewApproxZFastTrie[E UNumber, S UNumber, I UNumber](keys []bits.BitString) (*ApproxZFastTrie[E, S, I], error) {
	return NewApproxZFastTrieWithSeed[E, S, I](keys, rand.Uint64())
}

// NewApproxZFastTrieFromIterator initializes a compact Trie from an iterator.
func NewApproxZFastTrieFromIterator[E UNumber, S UNumber, I UNumber](iter bits.BitStringIterator) (*ApproxZFastTrie[E, S, I], error) {
	return NewApproxZFastTrieWithSeedFromIterator[E, S, I](iter, rand.Uint64())
}

// NewApproxZFastTrieWithSeed initializes a compact Trie with delimiter index information using a specified seed.
// This is used for bucketing where we need to map Trie nodes to bucket indices.
// seed is the value used for computing PSig signatures, allowing deterministic construction.
func NewApproxZFastTrieWithSeed[E UNumber, S UNumber, I UNumber](keys []bits.BitString, seed uint64) (*ApproxZFastTrie[E, S, I], error) {
	errutil.BugOn(!areSorted(keys), "Keys should be sorted")
	return NewApproxZFastTrieWithSeedFromIterator[E, S, I](bits.NewSliceBitStringIterator(keys), seed)
}

// NewApproxZFastTrieWithSeedFromIterator initializes a compact Trie from an iterator with a specified seed.
func NewApproxZFastTrieWithSeedFromIterator[E UNumber, S UNumber, I UNumber](iter bits.BitStringIterator, seed uint64) (*ApproxZFastTrie[E, S, I], error) {
	// Use streaming builder (no debug references)
	return NewApproxZFastTrieFromIteratorStreaming[E, S, I](iter, seed)
}

// GetExistingPrefix finds node the longest existing extent, which is a prefix for a given pattern.
// It implements the "Fat Binary Search" algorithm using 2-fattest numbers.
func (azft *ApproxZFastTrie[E, S, I]) GetExistingPrefix(pattern bits.BitString) *NodeData[E, S, I] {
	if len(azft.data) == 0 {
		return nil
	}
	//todo: azft.stat.GetExitNodeCnt++
	patternLength := int32(pattern.Size())
	a := int32(0)
	b := patternLength
	var result = &azft.data[azft.rootId]

	for 0 < (b - a) { // is <= ok?
		//todo: azft.stat.GetExitNodeInnerLoopCnt++
		fFast := bits.TwoFattest(uint64(a), uint64(b))

		handle := pattern.Prefix(int(fFast))
		node := azft.getNodeData(handle)

		if node != nil && pattern.Size() >= uint32(node.extentLen) && S(hashBitString(pattern.Prefix(int(node.extentLen)), azft.seed)) == node.PSig {
			if uint64(node.extentLen) < fFast {
				// collision, see getexistingprefix_collision_filter.md
				b = int32(fFast) - 1
			} else {
				a = int32(node.extentLen)
				result = node
			}
		} else {
			b = int32(fFast) - 1
		}
	}

	return result
}

// LowerBound returns a set of 6 candidate nodes that might be the lexicographical lower bound
// (the smallest delimiter >= pattern) for the given pattern.
//
// In a compacted Z-Fast Trie, the Exit Node (longest existing prefix) is insufficient to
// determine the exact lower bound because the trie lacks full keys. For MMPH, we need to
// identify the bucket delimiter that follows the query key.
//
// todo: !!! DOC with pictures is REALLY REQUIRED !!!
// We return all possible nodes where the lower bound could reside relative to the Exit Node (6 candidates).
//
// The caller (e.g., MonotoneHashWithTrie) must verify these candidates against actual keys.
// See trie/azft/lower_bound_candidates.md for a detailed theoretical explanation.
func (azft *ApproxZFastTrie[E, S, I]) LowerBound(pattern bits.BitString) (*NodeData[E, S, I], *NodeData[E, S, I], *NodeData[E, S, I], *NodeData[E, S, I], *NodeData[E, S, I], *NodeData[E, S, I]) {
	node := azft.GetExistingPrefix(pattern)
	if node == nil {
		return nil, nil, nil, nil, nil, nil
	}

	parentNode := (*NodeData[E, S, I])(nil)
	if node.parent != I(^I(0)) {
		parentNode = &azft.data[node.parent]
	}

	cand2 := (*NodeData[E, S, I])(nil)
	if node.minGreaterChild != I(^I(0)) {
		cand2 = &azft.data[node.minGreaterChild]
	}
	cand5 := (*NodeData[E, S, I])(nil)
	if node.rightChild != I(^I(0)) {
		cand5 = &azft.data[node.rightChild]
	}

	// work with parents
	cand3 := azft.getMinGreaterFromParent(parentNode)
	cand4 := azft.getGreaterFromParent(parentNode)

	cand6 := (*NodeData[E, S, I])(node)
	return &azft.data[node.minChild], cand2, cand3, cand4, cand5, cand6
}

func (azft *ApproxZFastTrie[E, S, I]) getMinGreaterFromParent(parentNode *NodeData[E, S, I]) *NodeData[E, S, I] {
	if parentNode == nil {
		return nil
	}
	for parentNode.minGreaterChild == I(^I(0)) && parentNode.parent != I(^I(0)) {
		parentId := parentNode.parent
		parentNode = &azft.data[parentId]
	}
	cand := (*NodeData[E, S, I])(nil)
	if parentNode.minGreaterChild != I(^I(0)) {
		cand = &azft.data[parentNode.minGreaterChild]
	}
	return cand
}

func (azft *ApproxZFastTrie[E, S, I]) getGreaterFromParent(parentNode *NodeData[E, S, I]) *NodeData[E, S, I] {
	if parentNode == nil {
		return nil
	}
	for parentNode.rightChild == I(^I(0)) && parentNode.parent != I(^I(0)) {
		parentId := parentNode.parent
		parentNode = &azft.data[parentId]
	}
	cand4 := (*NodeData[E, S, I])(nil)
	if parentNode.rightChild != I(^I(0)) {
		cand4 = &azft.data[parentNode.rightChild]
	}
	return cand4
}

func (azft *ApproxZFastTrie[E, S, I]) getNodeData(bitString bits.BitString) *NodeData[E, S, I] {
	query := azft.mph.Query(bitString)
	// Query return values from 1 to n, 0 used for no Entry
	if query == 0 {
		return nil
	}
	id := query - 1
	return &azft.data[id]
}

// ByteSize returns the resident size estimate of ApproxZFastTrie in bytes.
//
// Approximate memory model (bits):
//   - Let u be the number of trie nodes materialized in data[] (typically u=O(m),
//     and for a binary trie with m leaves, u <= 2m-1).
//   - AZFT_bits ~= O(1) + MPH_u_bits + u*(E + S + 5*I).
//   - Current implementation keeps originalNode pointer in NodeData layout, so
//     practical model is:
//     AZFT_bits ~= O(1) + MPH_u_bits + u*(E + S + 5*I + 64).
//
// It includes:
//   - top-level struct header (pointers/slice header/scalars),
//   - backing storage of MPH (via mph.Size()),
//   - backing storage of data[] (len * sizeof(NodeData)).
//
// Note: NodeData always includes originalNode pointer field in layout (nil in
// production mode). The debug Trie object itself is not included.
func (azft *ApproxZFastTrie[E, S, I]) ByteSize() int {
	if azft == nil {
		return 0
	}

	// Include struct header: mph pointer, data slice header, seed, rootId, Trie pointer.
	size := int(unsafe.Sizeof(*azft))

	// Size of the MPH (Minimal Perfect Hash function)
	if azft.mph != nil {
		size += azft.mph.Size()
	}

	// Size of data backing array (real NodeData layout including padding/all fields).
	size += len(azft.data) * int(unsafe.Sizeof(NodeData[E, S, I]{}))

	return size
}

// MemDetailed returns a detailed memory usage report for ApproxZFastTrie.
func (azft *ApproxZFastTrie[E, S, I]) MemDetailed() utils.MemReport {
	if azft == nil {
		return utils.MemReport{Name: "ApproxZFastTrie", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*azft))
	mphSize := 0
	if azft.mph != nil {
		mphSize = azft.mph.Size()
	}
	dataSize := len(azft.data) * int(unsafe.Sizeof(NodeData[E, S, I]{}))

	return utils.MemReport{
		Name:       "ApproxZFastTrie",
		TotalBytes: azft.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			{Name: "mph", TotalBytes: mphSize},
			{Name: "data", TotalBytes: dataSize},
		},
	}
}

func hashBitString(bs bits.BitString, seed uint64) uint64 {
	return bs.HashWithSeed(seed)
}

func areSorted(keys []bits.BitString) bool {
	for i := 1; i < len(keys); i++ {
		if keys[i-1].Compare(keys[i]) > 0 {
			return false
		}
	}
	return true
}
