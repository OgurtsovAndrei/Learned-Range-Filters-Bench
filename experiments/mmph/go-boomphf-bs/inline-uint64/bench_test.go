package inline_uint64

import (
	"fmt"
	"testing"
)

var (
	benchUintKeys  map[int][]uint64
	testGammas     = []float64{1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0}
	benchKeyCounts = []int{1 << 5, 1 << 8, 1 << 10, 1 << 13, 1 << 15, 1 << 18, 1 << 20, 1 << 22, 1 << 24}
)

func init() {
	benchUintKeys = make(map[int][]uint64)
	for _, count := range benchKeyCounts {
		keys := make([]uint64, count)
		for i := 0; i < count; i++ {
			keys[i] = uint64(i)
		}
		benchUintKeys[count] = keys
	}
}

func BenchmarkBBHashBuild(b *testing.B) {
	for _, gamma := range testGammas {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("Gamma=%.1f/Keys=%d", gamma, count), func(b *testing.B) {
				keys := benchUintKeys[count]
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = New(gamma, keys)
				}
			})
		}
	}
}

func BenchmarkBBHashLookup(b *testing.B) {
	for _, gamma := range testGammas {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("Gamma=%.1f/Keys=%d", gamma, count), func(b *testing.B) {
				keys := benchUintKeys[count]
				h := New(gamma, keys)

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = h.Query(keys[i%count])
				}
			})
		}
	}
}

func BenchmarkBBHashMemory(b *testing.B) {
	for _, gamma := range testGammas {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("Gamma=%.1f/Keys=%d", gamma, count), func(b *testing.B) {
				keys := benchUintKeys[count]
				h := New(gamma, keys)
				b.ReportMetric(float64(h.Size()*8)/float64(count), "bits/key_in_mem")
				b.ReportMetric(float64(h.Size()), "bytes_in_mem")
			})
		}
	}
}
