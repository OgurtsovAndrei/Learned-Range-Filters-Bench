package shzft

import (
	"Thesis-bench-industry/experiments/locators/rloc"
	"Thesis-bench-industry/experiments/trie/hzft"
	"fmt"
	"math/rand"
	"testing"
)

func BenchmarkMemory(b *testing.B) {
	rloc.InitBenchKeys()

	// Production-realistic key counts (matching rloc.BenchKeyCounts)
	benchCounts := []int{1024, 8192, 32768, 262144}

	for _, l := range rloc.BenchBitLengths {
		for _, n := range benchCounts {
			// Skip extreme combinations to save time
			if n > 65536 && l > 1024 {
				continue
			}

			b.Run(fmt.Sprintf("HZFT/L=%d/N=%d", l, n), func(b *testing.B) {
				keys := rloc.GetBenchKeys(l, n)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					trie := hzft.NewHZFastTrie[uint32](keys)
					if i == 0 {
						b.Logf("JSON_MEM_REPORT: %s", trie.MemDetailed().JSON())
					}
					b.ReportMetric(float64(trie.ByteSize())*8.0/float64(n), "bits/key")
				}
			})

			b.Run(fmt.Sprintf("SHZFT/L=%d/N=%d", l, n), func(b *testing.B) {
				keys := rloc.GetBenchKeys(l, n)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					trie := NewSuccinctHZFastTrie(keys)
					if i == 0 {
						b.Logf("JSON_MEM_REPORT: %s", trie.MemDetailed().JSON())
					}
					b.ReportMetric(float64(trie.ByteSize())*8.0/float64(n), "bits/key")
				}
			})
		}
	}
}

func BenchmarkQuery(b *testing.B) {
	rloc.InitBenchKeys()
	benchCounts := []int{1024, 8192, 32768, 262144}

	for _, l := range []int{64, 256, 1024, 4096} {
		for _, n := range benchCounts {
			if n > 65536 && l > 1024 {
				continue
			}
			keys := rloc.GetBenchKeys(l, n)
			hzftTrie := hzft.NewHZFastTrie[uint32](keys)
			shzftTrie := NewSuccinctHZFastTrie(keys)

			// Shuffle query indices to avoid sequential access patterns
			queryIndices := make([]int, n)
			for i := range queryIndices {
				queryIndices[i] = i
			}
			r := rand.New(rand.NewSource(42))
			r.Shuffle(len(queryIndices), func(i, j int) {
				queryIndices[i], queryIndices[j] = queryIndices[j], queryIndices[i]
			})

			b.Run(fmt.Sprintf("HZFT/Query/L=%d/N=%d", l, n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = hzftTrie.GetExistingPrefix(keys[queryIndices[i%n]])
				}
			})

			b.Run(fmt.Sprintf("SHZFT/Query/L=%d/N=%d", l, n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = shzftTrie.GetExistingPrefix(keys[queryIndices[i%n]])
				}
			})
		}
	}
}
