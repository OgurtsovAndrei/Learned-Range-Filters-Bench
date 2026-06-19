package zft

import (
	"Thesis/bits"
	"Thesis/bits/maps"
	"Thesis/utils/errutil"
	"fmt"
	"log"
	"strings"
)

import (
	"os"
)

var debug bool

func init() {
	if os.Getenv("DEBUG") == "1" {
		debug = true
	} else {
		debug = false
	}
}

type statistics struct {
	GetExitNodeCnt          int
	GetExitNodeInnerLoopCnt int
}

type UNumber interface {
	~uint8 | ~uint16 | ~uint32 | ~uint64
}

// ZFastTrie implementation in Go.
// Ported from ZFastTrie.hpp
type ZFastTrie[V comparable] struct {
	size           int32
	Root           *Node[V]
	Handle2NodeMap *maps.BitMap[*Node[V]]
	emptyValue     V

	stat statistics
}

// NewZFastTrie creates a new ZFastTrie.
// emptyValue is the value designated as empty (e.g., nil or 0 in Go).
func NewZFastTrie[V comparable](emptyValue V) *ZFastTrie[V] {
	return &ZFastTrie[V]{
		size:           0,
		Root:           nil,
		Handle2NodeMap: maps.NewBitMap[*Node[V]](),
		emptyValue:     emptyValue,
	}
}

func (zt *ZFastTrie[V]) Insert(newText string, value V) {
	zt.checkTrie()
	zt.InsertBitString(bits.NewFromText(newText), value)
	zt.checkTrie()
}

func (zt *ZFastTrie[V]) Erase(targetText string) {
	zt.checkTrie()
	zt.EraseBitString(bits.NewFromText(targetText))
	zt.checkTrie()
}

func (zt *ZFastTrie[V]) InsertBitString(newText bits.BitString, value V) {
	if value == zt.emptyValue {
		panic("cannot insert empty value")
	}

	if zt.size == 0 {
		zt.Root = NewNode(value, newText)
		zt.insertHandle2NodeMap(zt.Root)
	} else {
		exitNode := zt.GetExitNode(newText)
		if exitNode == nil {
			panic("internal error: exitNode is nil")
		}

		lcpLength := exitNode.Extent.GetLCPLength(newText)

		if lcpLength < exitNode.ExtentLength() {
			exitNodeExtent := exitNode.Extent
			oldExitValue := exitNode.Value

			var newExtent bits.BitString = bits.NewFromText("")
			if lcpLength > 0 {
				newExtent = exitNode.Extent.Prefix(int(lcpLength))
			}

			if exitNode.IsLeaf() {
				zt.eraseHandle2NodeMap(exitNodeExtent)
				exitNode.SetExtent(newExtent)
				exitNode.Value = zt.emptyValue
				zt.insertHandle2NodeMap(exitNode)
			} else {
				// C++: const uint &newHandleLength = Fast::twoFattest(lcpLength, newExtent.size());
				// newExtent.size() == lcpLength.
				// newHandleLength = twoFattest(lcpLength, lcpLength) == 0.
				// C++: bool isChangeHandle = (exitNode->handleLength() != newHandleLength);
				// isChangeHandle = (exitNode->handleLength() != 0)

				oldHandle := exitNode.Handle()
				isChangeHandle := !oldHandle.IsEmpty()

				if isChangeHandle {
					zt.eraseHandle2NodeMap(oldHandle)
					exitNode.SetExtent(newExtent)
					exitNode.Value = zt.emptyValue
					zt.insertHandle2NodeMap(exitNode)
				} else {
					exitNode.SetExtent(newExtent)
				}
			}

			newNode := NewNodeWithNameLength(oldExitValue, exitNodeExtent, int32(lcpLength+1))
			swapChildren(exitNode, newNode)
			exitNode.InsertChild(newNode, lcpLength)
			zt.insertHandle2NodeMap(newNode)

			if lcpLength < newText.Size() {
				newTextNode := NewNodeWithNameLength(value, newText, int32(lcpLength+1))
				errutil.BugOn(exitNode.Value != zt.emptyValue, "value already exists, but how?")
				exitNode.InsertChild(newTextNode, lcpLength)
				zt.insertHandle2NodeMap(newTextNode)
			} else {
				errutil.BugOn(exitNode.Value != zt.emptyValue, "value already exists, but how?")
				exitNode.Value = value
			}

		} else { // lcpLength >= exitNode.ExtentLength()
			if lcpLength == newText.Size() { // lcpLength == exitNode.ExtentLength() == newText.Size()
				if exitNode.Value == zt.emptyValue {
					exitNode.Value = value
				} else {
					if debug {
						log.Println("Warning: new text already exist.")
					}
					exitNode.Value = value
				}
			} else { // lcpLength == exitNode.ExtentLength() < newText.Size()
				if exitNode.IsLeaf() {
					zt.insertHandle2NodeMap(exitNode)
				}
				newTextNode := NewNodeWithNameLength(value, newText, int32(lcpLength+1))
				exitNode.InsertChild(newTextNode, lcpLength)
				zt.insertHandle2NodeMap(newTextNode)
			}
		}
	}

	zt.size++
}

func (zt *ZFastTrie[V]) EraseBitString(targetText bits.BitString) {
	tarGetNode := zt.GetExitNode(targetText)
	if tarGetNode == nil || !tarGetNode.Extent.Equal(targetText) || tarGetNode.Value == zt.emptyValue {
		if debug {
			log.Println("Warning: trying to erase non-existent key")
		}
		return
	}

	if zt.size <= 1 {
		if zt.Root == tarGetNode {
			zt.eraseHandle2NodeMap(tarGetNode.Handle())
			zt.Root = nil
		}
	} else {
		if tarGetNode.IsLeaf() {
			if tarGetNode == zt.Root {
				zt.eraseHandle2NodeMap(tarGetNode.Handle())
				zt.Root = nil
			} else {
				var parentNode *Node[V]
				if tarGetNode.NameLength <= 1 {
					parentNode = zt.Root
				} else {
					parentPrefix := tarGetNode.Extent.Prefix(int(tarGetNode.NameLength - 1))
					parentNode = zt.GetExitNode(parentPrefix)
				}

				if parentNode == nil {
					panic("internal error: parentNode not found during erase")
				}
				errutil.BugOn(parentNode.LeftChild != tarGetNode && parentNode.RightChild != tarGetNode, "wrong parent:\n%s\n%s", parentNode, tarGetNode)

				parentNode.EraseChild(tarGetNode.Key())
				zt.eraseHandle2NodeMap(tarGetNode.Handle())

				if parentNode.SizeChildren() == 1 && parentNode.Value == zt.emptyValue {
					zt.eraseHandle2NodeMap(parentNode.Handle())
					childNode := parentNode.GetChild()

					parentNode.EraseChild(childNode.Key())
					swapChildren(parentNode, childNode)

					parentNode.Set(childNode.Value, childNode.Extent, parentNode.NameLength)

					zt.eraseHandle2NodeMap(childNode.Handle())
					zt.insertHandle2NodeMap(parentNode)

				}
			}
		} else if tarGetNode.SizeChildren() == 1 {
			childNode := tarGetNode.GetChild()
			tarGetNode.EraseChild(childNode.Key())
			zt.eraseHandle2NodeMap(tarGetNode.Handle())
			//zt.eraseHandle2NodeMap(childNode.Handle())

			tarGetNode.Set(childNode.Value, childNode.Extent, tarGetNode.NameLength)
			swapChildren(tarGetNode, childNode)

			zt.eraseHandle2NodeMap(childNode.Handle())
			zt.insertHandle2NodeMap(tarGetNode)
		} else {
			tarGetNode.Value = zt.emptyValue
		}
	}
	zt.size--
}

func (zt *ZFastTrie[V]) Contains(pattern string) bool {
	return zt.ContainsBitString(bits.NewFromText(pattern))
}

func (zt *ZFastTrie[V]) ContainsBitString(pattern bits.BitString) bool {
	exitNode := zt.GetExitNode(pattern)
	if exitNode == nil {
		return false
	}
	return exitNode.Extent.Equal(pattern) && exitNode.Value != zt.emptyValue
}

func (zt *ZFastTrie[V]) GetBitString(pattern bits.BitString) (value V) {
	exitNode := zt.GetExitNode(pattern)

	if exitNode == nil {
		return zt.emptyValue
	}
	if exitNode.Extent.Equal(pattern) {
		return exitNode.Value
	}
	return zt.emptyValue
}

func swapChildren[V comparable](a, b *Node[V]) {
	a.LeftChild, b.LeftChild = b.LeftChild, a.LeftChild
	a.RightChild, b.RightChild = b.RightChild, a.RightChild
}

func (zt *ZFastTrie[V]) checkTrie() {
	if debug {
		cnt := zt.checkTrieRec(zt.Root)
		errutil.BugOn(cnt != zt.Handle2NodeMap.Len(), "%d != %d\n%s", cnt, zt.Handle2NodeMap.Len(), zt)
	}
}

func (zt *ZFastTrie[V]) checkTrieRec(node *Node[V]) (notEmptyNodesInTrie int) {
	if node == nil {
		return 0
	}

	if node.NameLength != 0 {
		fFast := node.HandleLength()
		f := int32(fFast)
		handle := node.Extent.Prefix(int(f))
		handleNode, ok := zt.Handle2NodeMap.Get(handle)
		errutil.BugOn(!ok, "on %q, %d != %d\n%s\n%s\n%s", handle, zt.size, f, node, handleNode, zt)
		errutil.BugOn(node != handleNode, "%s\n%s\n%s\n%s", handle, node, handleNode, zt)
	}

	if !node.Extent.IsEmpty() {
		notEmptyNodesInTrie = 1
	}

	return notEmptyNodesInTrie + zt.checkTrieRec(node.LeftChild) + zt.checkTrieRec(node.RightChild)
}

func (zt *ZFastTrie[V]) insertHandle2NodeMap(n *Node[V]) {
	//if n.ExtentLength() == 0 {
	//	return
	//}
	handle := n.Handle()
	if handle.IsEmpty() {
		_, exists := zt.Handle2NodeMap.Get(handle)
		errutil.BugOn(exists, "root already in Handle2NodeMap")
		//return
	}
	zt.Handle2NodeMap.Put(handle, n)
}

func (zt *ZFastTrie[V]) eraseHandle2NodeMap(handle bits.BitString) {
	//if handle.IsEmpty() {
	//	return
	//}
	zt.Handle2NodeMap.Delete(handle)
}

// ContainsPrefix checks if the string is a prefix of any entry in the Trie.
func (zt *ZFastTrie[V]) ContainsPrefix(pattern string) bool {
	return zt.ContainsPrefixBitString(bits.NewFromText(pattern))
}

func (zt *ZFastTrie[V]) ContainsPrefixBitString(pattern bits.BitString) bool {
	node := zt.GetExitNode(pattern)
	if node == nil {
		return false
	}
	return node.Extent.HasPrefix(pattern)
}

func (zt *ZFastTrie[V]) GetExitNode(pattern bits.BitString) *Node[V] {
	result := zt.GetExistingPrefix(pattern)

	if result != nil {
		lcpLength := result.Extent.GetLCPLength(pattern)
		if lcpLength == result.ExtentLength() && lcpLength < pattern.Size() {
			var next *Node[V]
			if pattern.At(lcpLength) {
				next = result.RightChild
			} else {
				next = result.LeftChild
			}
			if next != nil {
				result = next
			}
		}
	}

	return result
}

func (zt *ZFastTrie[V]) GetExistingPrefix(pattern bits.BitString) *Node[V] {
	zt.stat.GetExitNodeCnt++
	patternLength := int32(pattern.Size())
	a := int32(0)
	b := patternLength
	var node *Node[V]
	result := zt.Root

	for 0 < (b - a) {
		zt.stat.GetExitNodeInnerLoopCnt++
		// C++: f = Fast::twoFattest(a, b);
		fFast := bits.TwoFattest(uint64(a), uint64(b))

		handle := pattern.Prefix(int(fFast))
		node = zt.GetNode(handle)

		if node != nil {
			a = int32(node.ExtentLength())
			result = node
		} else {
			b = int32(fFast) - 1
		}
	}
	return result
}

func (zt *ZFastTrie[V]) GetNode(handle bits.BitString) *Node[V] {
	node, _ := zt.Handle2NodeMap.Get(handle)
	return node
}

func (zt *ZFastTrie[V]) String() string {
	var sb strings.Builder
	sb.WriteString("ZFastTrie:\n")
	sb.WriteString(fmt.Sprintf("| size: %d\n", zt.size))
	sb.WriteString(fmt.Sprintf("| emptyValue: %v\n", zt.emptyValue))

	sb.WriteString("| Root:")
	if zt.Root == nil {
		sb.WriteString("nil")
	} else {
		// Pass 2 for the initial indentation of the root node's fields (value, extent, etc.)
		sb.WriteString(zt.stringNode(zt.Root, "| | "))
	}

	zt.Handle2NodeMap.Range(func(bitString bits.BitString, z *Node[V]) bool {
		sb.WriteString(bitString.PrettyString())
		sb.WriteString(": ")
		sb.WriteString(z.String())
		sb.WriteString("\n")
		return true
	})
	if zt.Handle2NodeMap.Len() == 0 {
		sb.WriteString("<empty handle>")
	}
	sb.WriteString("\n")

	return sb.String()
}

func (zt *ZFastTrie[V]) stringNode(node *Node[V], prefix string) string {
	if node == nil {
		return " nil\n"
	}

	var sb strings.Builder

	sb.WriteString("\n")

	sb.WriteString(fmt.Sprintf("%svalue: %v\n", prefix, node.Value))
	sb.WriteString(fmt.Sprintf("%sextent: %q\n", prefix, node.Extent.PrettyString()))
	sb.WriteString(fmt.Sprintf("%sNameLength: %d\n", prefix, node.NameLength))
	sb.WriteString(fmt.Sprintf("%sleftChild:", prefix))
	sb.WriteString(zt.stringNode(node.LeftChild, prefix+"| "))
	sb.WriteString(fmt.Sprintf("%srightChild:", prefix))
	sb.WriteString(zt.stringNode(node.RightChild, prefix+"| "))

	return sb.String()
}

// BuildFromIterator creates a standard Z-Fast Trie from a sequence of bit strings.
// It serves as an intermediate step during the construction of the compact version.
func BuildFromIterator(iter bits.BitStringIterator) (*ZFastTrie[bool], error) {
	trie := NewZFastTrie[bool](false)
	for iter.Next() {
		trie.InsertBitString(iter.Value(), true)
	}
	if err := iter.Error(); err != nil {
		return nil, err
	}
	return trie, nil
}

// Build creates a standard Z-Fast Trie from a set of bit strings.
// It serves as an intermediate step during the construction of the compact version.
func Build(keys []bits.BitString) *ZFastTrie[bool] {
	trie, err := BuildFromIterator(bits.NewSliceBitStringIterator(keys))
	if err != nil {
		panic(err) // Should not happen with slice iterator
	}
	return trie
}
