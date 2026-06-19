package bucket

import (
	"Thesis/testutils"
	"fmt"
	"testing"
)

var (
	benchKeyCounts = []int{1 << 5, 1 << 8, 1 << 10, 1 << 13, 1 << 15, 1 << 18, 1 << 20, 1 << 22, 1 << 24}
)

func BenchmarkMonotoneHashBuild(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeys(64, count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				table := NewMonotoneHash(keys)
				b.ReportMetric(float64(table.Size())*8/float64(count), "bits/key_in_mem")
				b.ReportMetric(float64(table.Size()), "bytes_in_mem")
			}
		})
	}
}

func BenchmarkMonotoneHashLookup(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeys(64, count)
			mh := NewMonotoneHash(keys)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Запрашиваем ключи циклически
				_ = mh.GetRank(keys[i%count])
			}
		})
	}
}
