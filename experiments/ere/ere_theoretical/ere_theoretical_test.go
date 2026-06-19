package ere_theoretical

import (
	"Thesis/bits"
	"fmt"
	"sort"
	"testing"
)

func TestTheoreticalExactRangeEmptiness(t *testing.T) {
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

	ere, err := NewTheoreticalExactRangeEmptiness(keys, universe)
	if err != nil {
		t.Fatalf("Failed to build ERE: %v", err)
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
				t.Errorf("Expected IsEmpty(%d, %d) to be %v, got %v", tc.a, tc.b, tc.expected, res)
			}
		})
	}
}
