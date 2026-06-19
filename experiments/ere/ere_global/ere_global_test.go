package ere_global

import (
	"Thesis/bits"
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

func TestGlobalExactRangeEmptiness(t *testing.T) {
	keys := []bits.BitString{
		bits.NewFromUint64(10),
		bits.NewFromUint64(20),
		bits.NewFromUint64(30),
		bits.NewFromUint64(100),
		bits.NewFromUint64(105),
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Compare(keys[j]) < 0
	})

	universe := bits.NewBitString(64)

	ere, err := NewGlobalExactRangeEmptiness(keys, universe)
	if err != nil {
		t.Fatalf("Failed to build: %v", err)
	}

	tests := []struct {
		a, b     uint64
		expected bool
	}{
		{5, 9, true},
		{10, 10, false},
		{11, 19, true},
		{20, 25, false},
		{21, 29, true},
		{30, 30, false},
		{31, 99, true},
		{100, 105, false},
		{106, 200, true},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("[%d,%d]", tc.a, tc.b), func(t *testing.T) {
			res := ere.IsEmpty(bits.NewFromUint64(tc.a), bits.NewFromUint64(tc.b))
			if res != tc.expected {
				t.Errorf("Expected IsEmpty(%d, %d) = %v, got %v", tc.a, tc.b, tc.expected, res)
			}
		})
	}
}

func TestGlobalERE_Random(t *testing.T) {
	const (
		nKeys   = 1000
		bitLen  = 60
		mask60  = (uint64(1) << 60) - 1
		nChecks = 5000
	)

	rng := rand.New(rand.NewSource(42))
	seen := make(map[uint64]bool, nKeys)
	raw := make([]uint64, 0, nKeys)
	for len(raw) < nKeys {
		v := rng.Uint64() & mask60
		if !seen[v] {
			seen[v] = true
			raw = append(raw, v)
		}
	}
	sort.Slice(raw, func(i, j int) bool { return raw[i] < raw[j] })

	keysBS := make([]bits.BitString, nKeys)
	for i, v := range raw {
		keysBS[i] = bits.NewFromTrieUint64(v, uint32(bitLen))
	}
	universe := bits.NewBitString(uint32(bitLen))

	ere, err := NewGlobalExactRangeEmptiness(keysBS, universe)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	for i := 0; i < nChecks; i++ {
		lo := rng.Uint64() & mask60
		hi := lo + uint64(rng.Intn(1000))
		if hi > mask60 {
			hi = mask60
		}

		a := bits.NewFromTrieUint64(lo, uint32(bitLen))
		b := bits.NewFromTrieUint64(hi, uint32(bitLen))
		got := ere.IsEmpty(a, b)

		// Ground truth via binary search
		want := true
		idx := sort.Search(len(raw), func(j int) bool { return raw[j] >= lo })
		if idx < len(raw) && raw[idx] <= hi {
			want = false
		}

		if got != want {
			t.Fatalf("IsEmpty(%d, %d): got %v, want %v", lo, hi, got, want)
		}
	}
}

func TestGlobalERE_Empty(t *testing.T) {
	ere, err := NewGlobalExactRangeEmptiness(nil, bits.NewBitString(64))
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	if !ere.IsEmpty(bits.NewFromUint64(0), bits.NewFromUint64(100)) {
		t.Error("Empty structure should always return true")
	}
}
