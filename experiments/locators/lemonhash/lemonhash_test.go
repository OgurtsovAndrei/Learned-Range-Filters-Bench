package lemonhash

import (
	"Thesis/bits"
	"fmt"
	"sort"
	"testing"
)

func TestLeMonHash(t *testing.T) {
	keysStr := []string{
		"apple",
		"banana",
		"cherry",
		"date",
		"elderberry",
		"fig",
		"grape",
	}

	keys := make([]bits.BitString, len(keysStr))
	for i, s := range keysStr {
		keys[i] = bits.NewFromText(s)
	}

	lh := New(keys)

	for i, k := range keys {
		rank := lh.Rank(k)
		if rank != i {
			t.Errorf("Expected rank %d for key %s, got %d", i, keysStr[i], rank)
		}
	}
}

func TestLeMonHashEmpty(t *testing.T) {
	lh := New([]bits.BitString{})
	if lh.Rank(bits.NewFromText("a")) != 0 {
		t.Errorf("Expected rank 0 for empty hash")
	}
}

func TestLeMonHashSingle(t *testing.T) {
	keys := []bits.BitString{bits.NewFromText("alone")}
	lh := New(keys)
	rank := lh.Rank(keys[0])
	if rank != 0 {
		t.Errorf("Expected rank 0, got %d", rank)
	}
}

func TestLeMonHashBatch(t *testing.T) {
	keysStr := []string{"apple", "banana", "cherry", "date"}
	keys := make([]bits.BitString, len(keysStr))
	for i, s := range keysStr {
		keys[i] = bits.NewFromText(s)
	}

	lh := New(keys)
	results := make([]int, len(keys))
	lh.RankBatch(keys, results)

	for i, r := range results {
		if r != i {
			t.Errorf("Expected rank %d for key %s, got %d", i, keysStr[i], r)
		}
	}
}

func TestLeMonHashPair(t *testing.T) {
	keysStr := []string{"apple", "banana"}
	keys := make([]bits.BitString, len(keysStr))
	for i, s := range keysStr {
		keys[i] = bits.NewFromText(s)
	}

	lh := New(keys)
	r1, r2 := lh.RankPair(keys[0], keys[1])

	if r1 != 0 || r2 != 1 {
		t.Errorf("Expected ranks (0, 1), got (%d, %d)", r1, r2)
	}
}

func TestLeMonHashLargeKeys(t *testing.T) {
	// Generate sorted keys
	var keys []bits.BitString
	for i := 0; i < 1000; i++ {
		s := fmt.Sprintf("key_%05d", i)
		keys = append(keys, bits.NewFromText(s))
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].TrieCompare(keys[j]) < 0
	})
	
	lh := New(keys)
	
	for i, k := range keys {
		rank := lh.Rank(k)
		if rank != i {
			t.Errorf("Expected rank %d, got %d", i, rank)
		}
	}
}
