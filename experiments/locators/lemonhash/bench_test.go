package lemonhash

import (
	"Thesis/bits"
	"Thesis/testutils"
	"fmt"
	"sort"
	"testing"
)

var (
	benchKeyCounts = []int{1 << 10, 1 << 15, 1 << 20}
)

func BenchmarkMMPHBuild(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keysStr := testutils.GetBenchKeysAsStrings(64, count)
			keys := make([]bits.BitString, len(keysStr))
			for i, s := range keysStr {
				keys[i] = bits.NewFromText(s)
			}
			sort.Slice(keys, func(i, j int) bool {
				return keys[i].TrieCompare(keys[j]) < 0
			})
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				table := New(keys)
				b.ReportMetric(float64(table.ByteSize())*8/float64(count), "bits/key_in_mem")
				b.ReportMetric(float64(table.ByteSize()), "bytes_in_mem")
			}
		})
	}
}

func BenchmarkMMPHQuery(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keysStr := testutils.GetBenchKeysAsStrings(64, count)
			keys := make([]bits.BitString, len(keysStr))
			for i, s := range keysStr {
				keys[i] = bits.NewFromText(s)
			}
			sort.Slice(keys, func(i, j int) bool {
				return keys[i].TrieCompare(keys[j]) < 0
			})
			table := New(keys)

			// Pre-extract data to avoid allocation in benchmark loop
			keysData := make([][]byte, len(keys))
			for i, k := range keys {
				keysData[i] = k.Data()
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = table.rankRaw(keysData[i%count])
			}
		})
	}
}

func BenchmarkMMPHQueryBatch(b *testing.B) {
	batchSize := 1024
	for _, count := range benchKeyCounts {
		if count < batchSize {
			continue
		}
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keysStr := testutils.GetBenchKeysAsStrings(64, count)
			keys := make([]bits.BitString, len(keysStr))
			for i, s := range keysStr {
				keys[i] = bits.NewFromText(s)
			}
			sort.Slice(keys, func(i, j int) bool {
				return keys[i].TrieCompare(keys[j]) < 0
			})
			table := New(keys)
			results := make([]int, batchSize)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				start := 0
				if count > batchSize {
					start = (i * batchSize) % (count - batchSize)
				}
				table.RankBatch(keys[start:start+batchSize], results)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*batchSize), "ns/key_avg")
		})
	}
}

func BenchmarkMMPHQueryPair(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keysStr := testutils.GetBenchKeysAsStrings(64, count)
			keys := make([]bits.BitString, len(keysStr))
			for i, s := range keysStr {
				keys[i] = bits.NewFromText(s)
			}
			sort.Slice(keys, func(i, j int) bool {
				return keys[i].TrieCompare(keys[j]) < 0
			})
			table := New(keys)

			keysData := make([][]byte, len(keys))
			for i, k := range keys {
				keysData[i] = k.Data()
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				k1 := keysData[(i*2)%count]
				k2 := keysData[(i*2+1)%count]
				_, _ = table.rankPairRaw(k1, k2)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*2), "ns/key_avg")
		})
	}
}
