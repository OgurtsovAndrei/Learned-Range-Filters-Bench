package lemon_rloc

import (
	"Thesis-bench-industry/experiments/locators/rloc"
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
	"testing"
)

func BenchmarkLeMonRangeLocator_MemoryAndBuild(b *testing.B) {
	for _, bitLen := range rloc.BenchBitLengths {
		for _, n := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, n), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, n)
				zt := zft.Build(keys)

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					rl, err := NewLeMonRangeLocator(zt)
					if err != nil {
						b.Fatalf("build failed: %v", err)
					}
					b.ReportMetric(float64(rl.ByteSize())*8/float64(n), "bits/key")
				}
			})
		}
	}
}

func BenchmarkLeMonRangeLocator_Query(b *testing.B) {
	for _, bitLen := range rloc.BenchBitLengths {
		for _, n := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, n), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, n)
				zt := zft.Build(keys)

				rl, err := NewLeMonRangeLocator(zt)
				if err != nil {
					b.Fatalf("build failed: %v", err)
				}

				// Pre-collect nodes for querying
				var queries []zft.NodeInfo
				it := zft.NewIterator(zt)
				for it.Next() {
					if it.Node() != nil {
						queries = append(queries, *it.Node())
					}
				}

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					node := queries[i%len(queries)]
					_, _, _ = rl.Query(node.Extent)
				}
			})
		}
	}
}

func BenchmarkMemoryDetailed(b *testing.B) {
	for _, bitLen := range rloc.BenchBitLengths {
		for _, n := range rloc.BenchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, n), func(b *testing.B) {
				keys := rloc.GetBenchKeys(bitLen, n)
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					zt := zft.Build(keys)
					rl, _ := NewLeMonRangeLocator(zt)
					if i == 0 {
						b.Logf("JSON_MEM_REPORT: %s", rl.MemDetailed().JSON())
					}
				}
			})
		}
	}
}
