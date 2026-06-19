package shzft

import (
	"testing"
)

func TestBitPack(t *testing.T) {
	values := []uint64{0, 1, 2, 3, 4, 15, 6, 7}
	bitWidth := 4

	packed := packBits(values, bitWidth)

	for i, v := range values {
		unpacked := unpackBit(packed, i, bitWidth)
		if unpacked != v {
			t.Errorf("Mismatch at %d: expected %d, got %d", i, v, unpacked)
		}
	}
}

func TestBitPackCrossWord(t *testing.T) {
	values := []uint64{0x3F, 0x1A, 0x2B, 0x3C, 0x05, 0x3F}
	bitWidth := 6

	// Let's create an array that spans across the 64-bit boundary
	// We need > 64/6 = 10 values
	values = make([]uint64, 20)
	for i := range values {
		values[i] = uint64(i % 64)
	}

	packed := packBits(values, bitWidth)

	for i, v := range values {
		unpacked := unpackBit(packed, i, bitWidth)
		if unpacked != v {
			t.Errorf("Mismatch at %d: expected %d, got %d", i, v, unpacked)
		}
	}
}
