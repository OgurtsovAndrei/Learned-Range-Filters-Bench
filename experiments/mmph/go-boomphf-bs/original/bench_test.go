package original

import (
	"fmt"
	"testing"
	tbits "Thesis/bits"
)

var benchKeyCounts = []int{1 << 13, 1 << 18}

func BenchmarkBuild(b *testing.B) {
	for _, count := range benchKeyCounts {
		keys := make([]tbits.BitString, count)
		for i := 0; i < count; i++ { keys[i] = tbits.NewFromUint64(uint64(i)) }
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ { _ = New(2.0, keys) }
		})
	}
}

func BenchmarkLookup(b *testing.B) {
	for _, count := range benchKeyCounts {
		keys := make([]tbits.BitString, count)
		for i := 0; i < count; i++ { keys[i] = tbits.NewFromUint64(uint64(i)) }
		h := New(2.0, keys)
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ { _ = h.Query(keys[i%count]) }
		})
	}
}
