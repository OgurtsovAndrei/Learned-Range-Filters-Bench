package zft

import (
	"Thesis/bits"
	"fmt"
)

// Node represents a Node in the ZFastTrie.
type Node[V comparable] struct {
	Value      V
	Extent     bits.BitString
	NameLength int32
	LeftChild  *Node[V]
	RightChild *Node[V]
}

func NewNode[V comparable](value V, extent bits.BitString) *Node[V] {
	return &Node[V]{
		Value:      value,
		Extent:     extent,
		NameLength: 0,
	}
}

func NewNodeWithNameLength[V comparable](value V, extent bits.BitString, nameLength int32) *Node[V] {
	return &Node[V]{
		Value:      value,
		Extent:     extent,
		NameLength: nameLength,
	}
}

func (n *Node[V]) Set(value V, extent bits.BitString, nameLength int32) {
	n.Value = value
	n.Extent = extent
	n.NameLength = nameLength
}

func (n *Node[V]) SetExtent(extent bits.BitString) {
	if extent.IsEmpty() {
		n.NameLength = 0
	}
	n.Extent = extent
}

func (n *Node[V]) ExtentLength() uint32 {
	return n.Extent.Size()
}

func (n *Node[V]) HandleLength() uint32 {
	// C++: Fast::twoFattest(nameLength_ - 1, extentLength());
	aFast := uint64(n.NameLength - 1)
	if aFast == ^uint64(0) {
		aFast = 0
	}
	bFast := uint64(n.ExtentLength())

	return uint32(bits.TwoFattest(aFast, bFast))
}

func (n *Node[V]) Handle() bits.BitString {
	if n.ExtentLength() <= 0 {
		return bits.NewFromText("")
	}
	return n.Extent.Prefix(int(n.HandleLength()))
}

func (n *Node[V]) IsLeaf() bool {
	return n.LeftChild == nil && n.RightChild == nil
}

func (n *Node[V]) Key() bool {
	// C++: extent_.at(nameLength_ - 1)
	return n.Extent.At(uint32(n.NameLength - 1))
}

func (n *Node[V]) SizeChildren() uint32 {
	size := uint32(0)
	if n.LeftChild != nil {
		size++
	}
	if n.RightChild != nil {
		size++
	}
	return size
}

func (n *Node[V]) InsertChild(child *Node[V], lcpLength uint32) {
	// C++: child->extent_.at(lcpLength)
	if child.Extent.At(lcpLength) {
		n.RightChild = child
	} else {
		n.LeftChild = child
	}
}

func (n *Node[V]) EraseChild(key bool) {
	if key {
		n.RightChild = nil
	} else {
		n.LeftChild = nil
	}
}

func (n *Node[V]) GetChild() *Node[V] {
	if n.LeftChild != nil {
		return n.LeftChild
	}
	return n.RightChild
}

func (n *Node[V]) String() string {
	if n == nil {
		return "<nil>"
	}
	return fmt.Sprintf("Node{Value: %v, Extent: \"%v\", NameLength: %d, ExtentLen: %d,  LeftChild: %t, RightChild: %t}",
		n.Value, n.Extent, n.NameLength, n.ExtentLength(), n.LeftChild != nil, n.RightChild != nil)
}