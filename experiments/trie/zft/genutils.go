package zft

import (
	"Thesis/bits"
	"math/rand"
	"sort"
)

const benchmarkCharset = "abcdefghijklmnopqrstuvwxyz0123456789"

func GenerateRandomBitStrings(n, bitLen int, r *rand.Rand) []bits.BitString {
	if bitLen <= 0 {
		bitLen = 1
	}
	keys := make([]bits.BitString, n)
	for i := 0; i < n; i++ {
		keys[i] = GenerateBitString(bitLen, r)
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Compare(keys[j]) < 0
	})
	return keys
}

func GenerateBitString(bitLen int, r *rand.Rand) bits.BitString {
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
			b[j] = benchmarkCharset[r.Intn(len(benchmarkCharset))]
		}
		s := bits.NewFromText(string(b))
		if uint32(bitLen) < s.Size() {
			s = s.Prefix(bitLen)
		}
		return s
	}
}
