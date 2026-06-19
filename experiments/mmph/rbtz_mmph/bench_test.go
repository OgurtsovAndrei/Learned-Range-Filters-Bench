package rbtz

import (
	"Thesis/testutils"
	"fmt"
	"testing"
)

var (
	benchKeyCounts = []int{1 << 5, 1 << 8, 1 << 10, 1 << 13, 1 << 15, 1 << 18, 1 << 20, 1 << 22, 1 << 24}
)

func BenchmarkBuild(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeysAsStrings(64, count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				table := Build(keys)
				b.ReportMetric(float64(table.ByteSize())*8/float64(count), "bits/key_in_mem")
				b.ReportMetric(float64(table.ByteSize()), "bytes_in_mem")
			}
		})
	}
}

func BenchmarkLookup(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeysAsStrings(64, count)
			table := Build(keys)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = table.Lookup(keys[i%count])
			}
		})
	}
}

func BenchmarkSerialize(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeysAsStrings(64, count)
			table := Build(keys)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				serialized, _ := table.Serialize()
				serializedSize := len(serialized)
				b.ReportMetric(float64(serializedSize)*8/float64(count), "bits/key_serialized")
				b.ReportMetric(float64(serializedSize), "bytes_serialized")
			}
		})
	}
}

func BenchmarkDeserialize(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeysAsStrings(64, count)
			table := Build(keys)
			data, _ := table.Serialize()
			var newTable Table

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = Deserialize(data, &newTable)
			}
		})
	}
}
