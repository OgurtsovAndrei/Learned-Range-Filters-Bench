package zft

import (
	"Thesis/bits"
	"math/rand"
	"sort"
	"testing"
	"time"
)

const iteratorCharset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

func generateIteratorBitString(bitLen int, r *rand.Rand) bits.BitString {
	if bitLen <= 64 {
		k := r.Uint64()
		s := bits.NewFromUint64(k)
		if uint32(bitLen) < s.Size() {
			s = s.Prefix(bitLen)
		}
		return s
	} else {
		byteLen := bitLen / 8
		if bitLen%8 != 0 {
			byteLen++
		}
		b := make([]byte, byteLen)
		for j := range b {
			b[j] = iteratorCharset[r.Intn(len(iteratorCharset))]
		}
		s := bits.NewFromText(string(b))
		if uint32(bitLen) < s.Size() {
			s = s.Prefix(bitLen)
		}
		return s
	}
}

func TestIterator(t *testing.T) {
	t.Parallel()
	trie := NewZFastTrie[bool](false)

	// Test empty Trie
	it := NewIterator(trie)
	if it.Next() {
		t.Error("Expected no next element in empty Trie")
	}

	// Add some elements
	keys := []string{"a", "ab", "abc", "b", "bc"}
	for _, key := range keys {
		trie.Insert(key, true)
	}

	// Test iteration
	it = NewIterator(trie)
	var visited []string
	for it.Next() {
		node := it.Node()
		if node != nil && !node.Extent.IsEmpty() {
			visited = append(visited, node.Extent.PrettyString())
		}
	}

	t.Logf("Visited nodes: %v", visited)

	// Should visit all nodes in the Trie
	if len(visited) == 0 {
		t.Error("Expected to visit some nodes")
	}
}

func TestIteratorRandomSortedOrder(t *testing.T) {
	t.Parallel()
	for testId := 0; testId < 100; testId++ {
		t.Run("iteration", func(t *testing.T) {
			t.Parallel()
			seed := time.Now().UnixNano()
			r := rand.New(rand.NewSource(seed))

			trie := NewZFastTrie[bool](false)

			numKeys := r.Intn(1024) // 1000-2000 keys
			maxBitLen := 64

			insertedKeys := []bits.BitString{}
			keySet := make(map[string]bool)

			for len(insertedKeys) < numKeys {
				bitLen := r.Intn(maxBitLen) + 1
				key := generateIteratorBitString(bitLen, r)
				keyStr := key.PrettyString()

				if !keySet[keyStr] {
					keySet[keyStr] = true
					insertedKeys = append(insertedKeys, key)
					trie.InsertBitString(key, true)
				}
			}

			sort.Slice(insertedKeys, func(i, j int) bool {
				return bitStringLess(insertedKeys[i], insertedKeys[j])
			})

			it := NewIterator(trie)
			var iteratedBitStrings []bits.BitString
			var iteratedKeys []string
			var leafCount int
			for it.Next() {
				node := it.Node()
				if node != nil && !node.Extent.IsEmpty() && trie.ContainsBitString(node.Extent) {
					iteratedBitStrings = append(iteratedBitStrings, node.Extent)
					iteratedKeys = append(iteratedKeys, node.Extent.PrettyString())
					if node.IsLeaf {
						leafCount++
					}
				}
			}

			var expectedKeys []string
			for _, key := range insertedKeys {
				expectedKeys = append(expectedKeys, key.PrettyString())
			}
			if len(iteratedKeys) != int(trie.size) {
				t.Errorf("Total keys count mismatch: expected %d, got %d (leafCount: %d)", trie.size, len(iteratedKeys), leafCount)
			}

			iteratedSet := make(map[string]bool)
			for _, key := range iteratedKeys {
				iteratedSet[key] = true
			}

			for _, expectedKey := range expectedKeys {
				if !iteratedSet[expectedKey] {
					t.Errorf("Missing key in iteration: %s", expectedKey)
				}
			}

			for i := 0; i < len(iteratedBitStrings)-1; i++ {
				if !bitStringLess(iteratedBitStrings[i], iteratedBitStrings[i+1]) {
					t.Errorf("Iterator returned keys not in sorted order: %s vs %s",
						iteratedBitStrings[i].PrettyString(), iteratedBitStrings[i+1].PrettyString())
					t.Logf("First 10 keys: %v", iteratedKeys[:min(10, len(iteratedKeys))])
					break
				}
			}
		})
	}
}

func bitStringLess(a, b bits.BitString) bool {
	lcp := a.GetLCPLength(b)
	if lcp == a.Size() && lcp == b.Size() {
		return false
	}
	if lcp < a.Size() && lcp < b.Size() {
		return !a.At(lcp) && b.At(lcp)
	}
	if lcp == a.Size() {
		return b.At(lcp)
	}
	return !a.At(lcp)
}
