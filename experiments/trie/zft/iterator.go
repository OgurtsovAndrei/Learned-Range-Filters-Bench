package zft

import "Thesis/bits"

type NodeInfo struct {
	Extent bits.BitString
	IsLeaf bool
	Value  bool
}

type Iterator struct {
	trie     *ZFastTrie[bool]
	stack    []*Node[bool]
	finished bool
}

func (it *Iterator) Next() bool {
	if it.finished {
		return false
	}

	if len(it.stack) == 0 {
		if it.trie.Root == nil {
			it.finished = true
			return false
		}
		it.stack = append(it.stack, it.trie.Root)
		it.stack = append(it.stack, mostLeft(it.trie.Root)...)
		return true
	}

	node := it.stack[len(it.stack)-1]
	if node.RightChild != nil {
		it.stack = append(it.stack, node.RightChild)
		it.stack = append(it.stack, mostLeft(node.RightChild)...)
		return true
	}

	for len(it.stack) > 1 && it.stack[len(it.stack)-1] == it.stack[len(it.stack)-2].RightChild {
		it.stack = it.stack[:len(it.stack)-1]
	}
	if len(it.stack) > 1 {
		it.stack = it.stack[:len(it.stack)-1]
		return true
	}
	it.finished = true
	return false
}

func mostLeft(node *Node[bool]) (path []*Node[bool]) {
	path = make([]*Node[bool], 0)
	for node.LeftChild != nil {
		node = node.LeftChild
		path = append(path, node)
	}
	return path
}

func (it *Iterator) Node() *NodeInfo {
	if len(it.stack) == 0 {
		return nil
	}
	node := it.stack[len(it.stack)-1]
	return &NodeInfo{
		Extent: node.Extent,
		IsLeaf: node.IsLeaf(),
		Value:  node.Value,
	}
}

func NewIterator(zt *ZFastTrie[bool]) *Iterator {
	return &Iterator{
		trie:     zt,
		stack:    []*Node[bool]{},
		finished: false,
	}
}

// SortedIterator traverses the Trie in lexicographical order (Pre-Order).
type SortedIterator struct {
	stack []*Node[bool]
	curr  *Node[bool]
}

func NewSortedIterator(zt *ZFastTrie[bool]) *SortedIterator {
	stack := []*Node[bool]{}
	if zt.Root != nil {
		stack = append(stack, zt.Root)
	}
	return &SortedIterator{
		stack: stack,
	}
}

func (it *SortedIterator) Next() bool {
	if len(it.stack) == 0 {
		return false
	}
	node := it.stack[len(it.stack)-1]
	it.stack = it.stack[:len(it.stack)-1]

	// Push Right then Left so Left is popped first
	if node.RightChild != nil {
		it.stack = append(it.stack, node.RightChild)
	}
	if node.LeftChild != nil {
		it.stack = append(it.stack, node.LeftChild)
	}

	it.curr = node
	return true
}

func (it *SortedIterator) Node() *NodeInfo {
	if it.curr == nil {
		return nil
	}
	return &NodeInfo{
		Extent: it.curr.Extent,
		IsLeaf: it.curr.IsLeaf(),
		Value:  it.curr.Value,
	}
}