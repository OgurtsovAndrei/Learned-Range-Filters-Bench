package rloc

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
	"testing"
)

// Benchmark RangeLocator construction
func BenchmarkRangeLocatorBuild(b *testing.B) {
	InitBenchKeys()

	for _, bitLen := range BenchBitLengths {
		for _, count := range BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					zt := zft.Build(keys)
					rl, err := NewRangeLocator(zt)
					if err != nil {
						b.Fatalf("NewRangeLocator failed: %v", err)
					}

					if rl == nil {
						b.Fatal("Failed to build RangeLocator")
					}

					// Report memory metrics
					size := rl.ByteSize()
					widths := rl.TypeWidths()
					b.ReportMetric(float64(widths.E), "E_bits")
					b.ReportMetric(float64(widths.S), "S_bits")
					b.ReportMetric(float64(widths.I), "I_bits")
					b.ReportMetric(float64(size), "total_bytes")
					b.ReportMetric(float64(size)*8/float64(count), "bits_per_key")
				}
			})
		}
	}
}

// Benchmark RangeLocator query performance
func BenchmarkRangeLocatorQuery(b *testing.B) {
	InitBenchKeys()

	for _, bitLen := range BenchBitLengths {
		for _, count := range BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := GetBenchKeys(bitLen, count)
				zt := zft.Build(keys)
				rl, err := NewRangeLocator(zt)
				if err != nil {
					b.Fatalf("NewRangeLocator failed: %v", err)
				}

				if rl == nil {
					b.Fatal("Failed to build RangeLocator")
				}

				// Collect node extents from the trie
				var nodeExtents []bits.BitString
				it := zft.NewIterator(zt)
				for it.Next() {
					node := it.Node()
					if node != nil {
						nodeExtents = append(nodeExtents, node.Extent)
					}
				}

				if len(nodeExtents) == 0 {
					b.Skip("No trie nodes found")
				}

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					extent := nodeExtents[i%len(nodeExtents)]
					_, _, err := rl.Query(extent)
					if err != nil {
						b.Fatalf("Query failed: %v", err)
					}
				}
			})
		}
	}
}

// Benchmark detailed memory breakdown
func BenchmarkMemoryDetailed(b *testing.B) {
	InitBenchKeys()

	for _, bitLen := range BenchBitLengths {
		for _, count := range BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := GetBenchKeys(bitLen, count)
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					zt := zft.Build(keys)
					rl, _ := NewRangeLocator(zt)
					if i == 0 {
						b.Logf("JSON_MEM_REPORT: %s", rl.MemDetailed().JSON())
					}
				}
			})
		}
	}
}
