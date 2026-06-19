package lerloc

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/locators/rloc"
	"fmt"
	"testing"
)

// Benchmark LocalExactRangeLocator construction (Fast mode)
func BenchmarkLocalExactRangeLocatorBuildFast(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					lerl, err := NewLocalExactRangeLocator(keys)
					if err != nil {
						b.Fatalf("Failed to build LocalExactRangeLocator: %v", err)
					}

					// Report memory metrics
					size := lerl.ByteSize()
					b.ReportMetric(float64(size)*8/float64(count), "bits_per_key")
				}
			})
		}
	}
}

// Benchmark LocalExactRangeLocator construction (Compact mode)
func BenchmarkLocalExactRangeLocatorBuildCompact(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					lerl, err := NewCompactLocalExactRangeLocator(keys)
					if err != nil {
						b.Fatalf("Failed to build LocalExactRangeLocator: %v", err)
					}

					// Report memory metrics
					size := lerl.ByteSize()
					b.ReportMetric(float64(size)*8/float64(count), "bits_per_key")
				}
			})
		}
	}
}

// Benchmark query performance (Fast mode)
func BenchmarkLocalExactRangeLocatorQueryFast(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)
				lerl, _ := NewLocalExactRangeLocator(keys)

				queryPrefixes := generateQueryPrefixes(keys)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					prefix := queryPrefixes[i%len(queryPrefixes)]
					_, _, _ = lerl.WeakPrefixSearch(prefix)
				}
			})
		}
	}
}

// Benchmark query performance (Compact mode)
func BenchmarkLocalExactRangeLocatorQueryCompact(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)
				lerl, _ := NewCompactLocalExactRangeLocator(keys)

				queryPrefixes := generateQueryPrefixes(keys)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					prefix := queryPrefixes[i%len(queryPrefixes)]
					_, _, _ = lerl.WeakPrefixSearch(prefix)
				}
			})
		}
	}
}

func generateQueryPrefixes(keys []bits.BitString) []bits.BitString {
	queryPrefixes := make([]bits.BitString, 0, len(keys))
	for _, key := range keys {
		if key.Size() > 2 {
			prefixLen := 1 + (int(key.Size()) / 3)
			queryPrefixes = append(queryPrefixes, key.Prefix(prefixLen))
		} else {
			queryPrefixes = append(queryPrefixes, key)
		}
	}
	return queryPrefixes
}

// Benchmark memory usage comparison
func BenchmarkMemoryComparison(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					fastLerl, _ := NewLocalExactRangeLocator(keys)
					compactLerl, _ := NewCompactLocalExactRangeLocator(keys)

					b.ReportMetric(float64(fastLerl.ByteSize())*8/float64(count), "fast_bits_key")
					b.ReportMetric(float64(compactLerl.ByteSize())*8/float64(count), "compact_bits_key")
					b.ReportMetric(float64(fastLerl.ByteSize())/float64(compactLerl.ByteSize()), "fast_vs_compact_ratio")
				}
			})
		}
	}
}

// Benchmark detailed memory breakdown
func BenchmarkMemoryDetailed(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			// Fast Mode
			b.Run(fmt.Sprintf("Fast/KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					lerl, _ := NewLocalExactRangeLocator(keys)
					if i == 0 {
						b.Logf("JSON_MEM_REPORT: %s", lerl.MemDetailed().JSON())
					}
				}
			})

			// Compact Mode
			b.Run(fmt.Sprintf("Compact/KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					lerl, _ := NewCompactLocalExactRangeLocator(keys)
					if i == 0 {
						b.Logf("JSON_MEM_REPORT: %s", lerl.MemDetailed().JSON())
					}
				}
			})
		}
	}
}
