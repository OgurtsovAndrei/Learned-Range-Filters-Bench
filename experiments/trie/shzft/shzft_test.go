package shzft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/hzft"
	"math/rand"
	"sort"
	"testing"
)

func genRandomKeys(n, l int, seed int64) []bits.BitString {
	unique := make(map[string]bool)
	keys := make([]bits.BitString, 0, n)

	rng := rand.New(rand.NewSource(seed))

	for len(keys) < n {
		byteLen := (l + 7) / 8
		b := make([]byte, byteLen)
		rng.Read(b)

		bs := bits.NewFromDataAndSize(b, uint32(l))
		str := bs.PrettyString()
		if !unique[str] {
			unique[str] = true
			keys = append(keys, bs)
		}
	}

	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Compare(keys[j]) < 0
	})
	return keys
}

func TestSHZFastTrie_Correctness(t *testing.T) {
	n := 1024
	l := 64
	seed := int64(42)

	keys := genRandomKeys(n, l, seed)

	// Build old HZFT (reference)
	refTrie := hzft.NewHZFastTrie[uint8](keys)

	// Build new SHZFT
	shzftTrie := NewSuccinctHZFastTrie(keys)

	// Verify all keys match
	for i, key := range keys {
		refExt := refTrie.GetExistingPrefix(key)
		shzExt := shzftTrie.GetExistingPrefix(key)

		if refExt != shzExt {
			t.Fatalf("Mismatch on key %d: %s. Ref=%d, SHZFT=%d", i, key.PrettyString(), refExt, shzExt)
		}
	}

	// Blind tries like HZFT have undefined/random behavior for keys not in the trie.
	// Therefore, we do not test miss keys against a reference trie, as their
	// underlying Minimal Perfect Hash functions will yield different random collisions.

	t.Logf("SHZFT Memory (L=64): %d bytes (%.2f bits/key)", shzftTrie.ByteSize(), float64(shzftTrie.ByteSize())*8.0/float64(n))
	t.Logf("HZFT Memory (L=64): %d bytes (%.2f bits/key)", refTrie.ByteSize(), float64(refTrie.ByteSize())*8.0/float64(n))

	// Test L=1024
	l1024 := 1024
	keys1024 := genRandomKeys(n, l1024, seed)
	refTrie1024 := hzft.NewHZFastTrie[uint16](keys1024)
	shzftTrie1024 := NewSuccinctHZFastTrie(keys1024)
	for i, key := range keys1024 {
		refExt := refTrie1024.GetExistingPrefix(key)
		shzExt := shzftTrie1024.GetExistingPrefix(key)
		if refExt != shzExt {
			t.Fatalf("Mismatch on L=1024 key %d: Ref=%d, SHZFT=%d", i, refExt, shzExt)
		}
	}
	t.Logf("SHZFT Memory (L=1024): %d bytes (%.2f bits/key)", shzftTrie1024.ByteSize(), float64(shzftTrie1024.ByteSize())*8.0/float64(n))
	t.Logf("HZFT Memory (L=1024): %d bytes (%.2f bits/key)", refTrie1024.ByteSize(), float64(refTrie1024.ByteSize())*8.0/float64(n))
}
