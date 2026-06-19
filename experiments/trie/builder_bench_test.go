package trie

import (
	"Thesis-bench-industry/experiments/locators/lerloc"
	"Thesis-bench-industry/experiments/locators/rloc"
	"Thesis/testutils"
	"Thesis-bench-industry/experiments/trie/azft"
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
	"testing"
)

var (
	benchKeyCounts  = []int{1 << 10, 1 << 13, 1 << 15}
	benchBitLengths = []int{64, 128, 256}
)

// Benchmark ZFT construction
func BenchmarkZFTBuild(b *testing.B) {
	for _, bitLen := range benchBitLengths {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := testutils.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					zt := zft.Build(keys)
					if zt == nil {
						b.Fatal("Failed to build ZFastTrie")
					}
				}
			})
		}
	}
}

// Benchmark AZFT construction
func BenchmarkAZFTBuild(b *testing.B) {
	for _, bitLen := range benchBitLengths {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := testutils.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					// Build AZFT (now all use streaming internally)
					azft, err := azft.NewApproxZFastTrie[uint16, uint32, uint32](keys)
					if err != nil {
						b.Fatalf("Failed to build AZFT: %v", err)
					}
					if azft == nil {
						b.Fatal("Failed to build AZFT")
					}
				}
			})
		}
	}
}

// Benchmark RangeLocator construction
func BenchmarkRangeLocatorBuild(b *testing.B) {
	for _, bitLen := range benchBitLengths {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := testutils.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					zt := zft.Build(keys)
					rl, err := rloc.NewRangeLocator(zt)
					if err != nil {
						b.Fatalf("NewRangeLocator failed: %v", err)
					}
					if rl == nil {
						b.Fatal("Failed to build RangeLocator")
					}
				}
			})
		}
	}
}

// Benchmark LocalExactRangeLocator construction
func BenchmarkLocalExactRangeLocatorBuild(b *testing.B) {
	for _, bitLen := range benchBitLengths {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := testutils.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					lerl, err := lerloc.NewLocalExactRangeLocator(keys)
					if err != nil {
						b.Fatalf("Failed to build LocalExactRangeLocator: %v", err)
					}
					if lerl == nil {
						b.Fatal("Failed to build LocalExactRangeLocator")
					}
				}
			})
		}
	}
}
