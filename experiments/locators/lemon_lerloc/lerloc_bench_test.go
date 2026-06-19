package lemon_lerloc

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/locators/lerloc"
	"Thesis-bench-industry/experiments/locators/rloc"
	"fmt"
	"testing"
)

func BenchmarkLeMonLocalExactRangeLocator_Build(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					lerl, err := NewLeMonLocalExactRangeLocator(keys)
					if err != nil {
						b.Fatalf("Failed to build locator: %v", err)
					}

					// Report memory metrics
					size := lerl.ByteSize()
					b.ReportMetric(float64(size)*8/float64(count), "bits_per_key")
				}
			})
		}
	}
}

func BenchmarkLeMonLocalExactRangeLocator_Query(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)
				lerl, _ := NewLeMonLocalExactRangeLocator(keys)

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

func BenchmarkMemoryComparison(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					original, _ := lerloc.NewLocalExactRangeLocator(keys)
					compact, _ := NewLeMonLocalExactRangeLocator(keys)

					b.ReportMetric(float64(original.ByteSize())*8/float64(count), "orig_bits_key")
					b.ReportMetric(float64(compact.ByteSize())*8/float64(count), "lemon_bits_key")
					b.ReportMetric(float64(original.ByteSize())/float64(compact.ByteSize()), "improvement_ratio")
				}
			})
		}
	}
}

func BenchmarkMemoryDetailed(b *testing.B) {
	rloc.InitBenchKeys()

	for _, bitLen := range rloc.BenchBitLengths {
		for _, count := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, count)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					lerl, _ := NewLeMonLocalExactRangeLocator(keys)
					if i == 0 {
						// Use MemDetailed directly
						report := lerl.MemDetailed()
						b.Logf("JSON_MEM_REPORT: %s", report.JSON())
					}
				}
			})
		}
	}
}
