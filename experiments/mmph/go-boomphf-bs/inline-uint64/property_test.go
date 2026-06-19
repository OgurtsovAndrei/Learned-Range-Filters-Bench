package inline_uint64

import (
	"math/rand"
	"sort"
	"testing"
	"testing/quick"
)

func genUniqueKeys(n int) []uint64 {
	m := make(map[uint64]struct{}, n)
	out := make([]uint64, 0, n)

	for len(out) < n {
		v := rand.Uint64()
		if _, ok := m[v]; ok {
			continue
		}
		m[v] = struct{}{}
		out = append(out, v)
	}
	return out
}

func TestMPHRangeProperty(t *testing.T) {
	t.Parallel()
	f := func(n uint8) bool {
		size := int(n)%100 + 1
		keys := genUniqueKeys(size)

		h := New(2.0, keys)

		seen := make(map[uint64]struct{}, size)

		for _, k := range keys {
			v := h.Query(k)
			if v < 1 || v > uint64(size) {
				t.Errorf("Query(%d) out of range: got %d, want [1..%d]", k, v, size)
				return false
			}
			if _, exists := seen[v]; exists {
				t.Errorf("Duplicate index %d for key %d", v, k)
				return false
			}
			seen[v] = struct{}{}
		}

		if len(seen) != size {
			t.Errorf("MPH range missing values, seen=%d expected=%d", len(seen), size)
			return false
		}

		return true
	}

	cfg := &quick.Config{MaxCount: 50}
	if err := quick.Check(f, cfg); err != nil {
		t.Fatalf("Range property failed: %v", err)
	}
}

func TestMPHReproducible(t *testing.T) {
	t.Parallel()
	f := func(n uint8) bool {
		size := int(n)%100 + 1
		keys := genUniqueKeys(size)

		h1 := New(2.0, keys)
		h2 := New(2.0, keys)

		for _, k := range keys {
			v1 := h1.Query(k)
			v2 := h2.Query(k)
			if v1 != v2 {
				t.Errorf("Non-reproducible Query(%d): %d != %d", k, v1, v2)
				return false
			}
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 50}
	if err := quick.Check(f, cfg); err != nil {
		t.Fatalf("Reproducibility property failed: %v", err)
	}
}

func TestMPHOrderIndependence(t *testing.T) {
	t.Parallel()
	keys := genUniqueKeys(200)

	h1 := New(2.0, append([]uint64{}, keys...))

	shuffled := append([]uint64{}, keys...)
	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	h2 := New(2.0, shuffled)

	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	for _, k := range keys {
		v1 := h1.Query(k)
		v2 := h2.Query(k)
		if v1 != v2 {
			t.Fatalf("Order independence broken for key=%d: %d != %d", k, v1, v2)
		}
	}
}
