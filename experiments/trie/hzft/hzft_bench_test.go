package hzft

import (
	"Thesis/bits"
	"Thesis/testutils"
	"fmt"
	"testing"
)

var (
	benchKeyCounts  = []int{1 << 10, 1 << 13, 1 << 15}
	benchBitLengths = []int{64, 128, 256}
)

// Benchmark HZFT construction (old/heavy - builds full ZFT first)
func BenchmarkHZFTBuild_Heavy(b *testing.B) {
	for _, bitLen := range benchBitLengths {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := testutils.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					// Old path: Build full ZFT first
					hzft, err := NewHZFastTrieFromIteratorHeavy[uint32](bits.NewSliceBitStringIterator(keys))
					if err != nil {
						b.Fatalf("Failed to build heavy HZFT: %v", err)
					}
					if hzft == nil {
						b.Fatal("Failed to build heavy HZFT")
					}
				}
			})
		}
	}
}

// Benchmark HZFT construction (new/streaming - processes keys on-the-fly)
func BenchmarkHZFTBuild_Streaming(b *testing.B) {
	for _, bitLen := range benchBitLengths {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := testutils.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					hzft, err := NewHZFastTrieFromIteratorStreaming[uint32](bits.NewSliceBitStringIterator(keys))
					if err != nil {
						b.Fatalf("Failed to build streaming HZFT: %v", err)
					}
					if hzft == nil {
						b.Fatal("Failed to build streaming HZFT")
					}
				}
			})
		}
	}
}
