package original

import (
	"fmt"
	"testing"
	tbits "Thesis/bits"
)

func TestH(t *testing.T) {
	counts := []int{10, 100, 1000, 10000}
	for _, count := range counts {
		t.Run(fmt.Sprintf("Keys=%d", count), func(t *testing.T) {
			keys := make([]tbits.BitString, count)
			for i := 0; i < count; i++ {
				keys[i] = tbits.NewFromUint64(uint64(i))
			}
			h := New(2.0, keys)
			
			seen := make(map[uint64]bool)
			for _, k := range keys {
				idx := h.Query(k)
				if idx == 0 {
					t.Fatalf("key %v not found", k)
				}
				if seen[idx] {
					t.Fatalf("collision at index %d", idx)
				}
				seen[idx] = true
			}
		})
	}
}

func TestReproducible(t *testing.T) {
	keys := make([]tbits.BitString, 1000)
	for i := 0; i < 1000; i++ {
		keys[i] = tbits.NewFromUint64(uint64(i))
	}
	h1 := New(2.0, keys)
	h2 := New(2.0, keys)
	for _, k := range keys {
		if h1.Query(k) != h2.Query(k) {
			t.Fatal("results not reproducible")
		}
	}
}
