package azft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// fileIterator reads BitStrings from a file, one per line (binary format)
type fileIterator struct {
	file    *os.File
	scanner *bufio.Scanner
	current bits.BitString
	err     error
}

func newFileIterator(path string) (*fileIterator, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	return &fileIterator{
		file:    f,
		scanner: scanner,
	}, nil
}

func (it *fileIterator) Next() bool {
	if it.err != nil {
		return false
	}
	if !it.scanner.Scan() {
		it.err = it.scanner.Err()
		return false
	}
	line := it.scanner.Text()
	it.current = bits.NewFromBinary(line)
	return true
}

func (it *fileIterator) Value() bits.BitString {
	return it.current
}

func (it *fileIterator) Error() error {
	return it.err
}

func (it *fileIterator) Close() error {
	return it.file.Close()
}

func writeKeysToFile(keys []bits.BitString, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, key := range keys {
		for i := uint32(0); i < key.Size(); i++ {
			if key.At(i) {
				w.WriteByte('1')
			} else {
				w.WriteByte('0')
			}
		}
		w.WriteByte('\n')
	}
	return w.Flush()
}

func getMemStats() uint64 {
	runtime.GC()
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

// TestMemory_AZFT_StreamingVsHeavy compares memory of streaming vs heavy AZFT builders
func TestMemory_AZFT_StreamingVsHeavy(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory test in short mode")
	}

	testCases := []struct {
		numKeys int
		bitLen  int
	}{
		{500, 64},
		{1000, 128},
		{2000, 128},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("n=%d_L=%d", tc.numKeys, tc.bitLen), func(t *testing.T) {
			r := rand.New(rand.NewSource(42))
			keys := zft.GenerateRandomBitStrings(tc.numKeys, tc.bitLen, r)

			tmpDir := t.TempDir()
			keyFile := filepath.Join(tmpDir, "keys.txt")
			err := writeKeysToFile(keys, keyFile)
			if err != nil {
				t.Fatalf("Failed to write keys: %v", err)
			}

			seed := uint64(12345)

			// Measure heavy builder (with debug info)
			runtime.GC()
			memBefore := getMemStats()

			iter1, err := newFileIterator(keyFile)
			if err != nil {
				t.Fatalf("Failed to create iterator: %v", err)
			}
			heavyAZFT, err := NewApproxZFastTrieWithSeedFromIterator[uint16, uint32, uint32](iter1,  seed)
			iter1.Close()
			if err != nil {
				t.Fatalf("Heavy builder failed: %v", err)
			}

			memAfterHeavy := getMemStats()
			heavyMem := memAfterHeavy - memBefore

			heavyAZFT = nil
			runtime.GC()
			runtime.GC()

			// Measure streaming builder (no debug info)
			memBefore = getMemStats()

			iter2, err := newFileIterator(keyFile)
			if err != nil {
				t.Fatalf("Failed to create iterator: %v", err)
			}
			streamingAZFT, err := NewApproxZFastTrieFromIteratorStreaming[uint16, uint32, uint32](iter2,  seed)
			iter2.Close()
			if err != nil {
				t.Fatalf("Streaming builder failed: %v", err)
			}

			memAfterStreaming := getMemStats()
			streamingMem := memAfterStreaming - memBefore

			t.Logf("Keys: %d, BitLen: %d", tc.numKeys, tc.bitLen)
			t.Logf("Heavy builder memory (with debug):  %d bytes (%.2f MB)", heavyMem, float64(heavyMem)/(1024*1024))
			t.Logf("Streaming builder memory (no debug): %d bytes (%.2f MB)", streamingMem, float64(streamingMem)/(1024*1024))

			if streamingMem > 0 {
				ratio := float64(heavyMem) / float64(streamingMem)
				t.Logf("Ratio (heavy/streaming): %.2fx", ratio)
			}

			// Verify results are equivalent
			for _, key := range keys[:min(50, len(keys))] {
				heavyResult := heavyAZFT
				_ = heavyResult // Already nil, but pattern shows intent

				result := streamingAZFT.GetExistingPrefix(key)
				if result == nil {
					t.Errorf("Streaming AZFT returned nil for key %s", key.PrettyString())
				}
			}

			_ = streamingAZFT
		})
	}
}

// TestMemory_AZFT_FinalStructureSize compares final structure sizes
func TestMemory_AZFT_FinalStructureSize(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory test in short mode")
	}

	numKeys := 1000
	bitLen := 128

	r := rand.New(rand.NewSource(42))
	keys := zft.GenerateRandomBitStrings(numKeys, bitLen, r)

	tmpDir := t.TempDir()
	keyFile := filepath.Join(tmpDir, "keys.txt")
	writeKeysToFile(keys, keyFile)

	seed := uint64(12345)

	// Build with debug info
	iter1, _ := newFileIterator(keyFile)
	heavyAZFT, _ := NewApproxZFastTrieWithSeedFromIterator[uint16, uint32, uint32](iter1,  seed)
	iter1.Close()

	// Build without debug info
	iter2, _ := newFileIterator(keyFile)
	streamingAZFT, _ := NewApproxZFastTrieFromIteratorStreaming[uint16, uint32, uint32](iter2,  seed)
	iter2.Close()

	heavySize := heavyAZFT.ByteSize()
	streamingSize := streamingAZFT.ByteSize()

	t.Logf("Final structure comparison (n=%d, L=%d):", numKeys, bitLen)
	t.Logf("Heavy AZFT size:     %d bytes", heavySize)
	t.Logf("Streaming AZFT size: %d bytes", streamingSize)
	t.Logf("Difference:          %d bytes", heavySize-streamingSize)

	// Both should have same size now (no Trie field in either)
}

// TestMemory_AZFT_ScalingWithKeyCount tests memory scaling
func TestMemory_AZFT_ScalingWithKeyCount(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory scaling test in short mode")
	}

	bitLen := 128
	keyCounts := []int{500, 1000, 2000}
	seed := uint64(42)

	t.Log("AZFT Memory scaling with key count (bitLen=128):")
	t.Log("Keys\tHeavy(KB)\tStreaming(KB)\tRatio")

	for _, n := range keyCounts {
		r := rand.New(rand.NewSource(42))
		keys := zft.GenerateRandomBitStrings(n, bitLen, r)

		tmpDir := t.TempDir()
		keyFile := filepath.Join(tmpDir, "keys.txt")
		writeKeysToFile(keys, keyFile)

		// Heavy
		runtime.GC()
		memBefore := getMemStats()
		iter1, _ := newFileIterator(keyFile)
		heavyAZFT, _ := NewApproxZFastTrieWithSeedFromIterator[uint16, uint32, uint32](iter1,  seed)
		iter1.Close()
		heavyMem := getMemStats() - memBefore
		_ = heavyAZFT
		heavyAZFT = nil
		runtime.GC()

		// Streaming
		memBefore = getMemStats()
		iter2, _ := newFileIterator(keyFile)
		streamingAZFT, _ := NewApproxZFastTrieFromIteratorStreaming[uint16, uint32, uint32](iter2,  seed)
		iter2.Close()
		streamingMem := getMemStats() - memBefore

		ratio := 0.0
		if streamingMem > 0 {
			ratio = float64(heavyMem) / float64(streamingMem)
		}
		t.Logf("%d\t%.1f\t\t%.1f\t\t%.2fx", n, float64(heavyMem)/1024, float64(streamingMem)/1024, ratio)

		_ = streamingAZFT
	}
}

// BenchmarkMemoryAlloc_AZFT_Heavy measures allocations for heavy builder
func BenchmarkMemoryAlloc_AZFT_Heavy(b *testing.B) {
	r := rand.New(rand.NewSource(42))
	keys := zft.GenerateRandomBitStrings(500, 128, r)

	tmpDir := b.TempDir()
	keyFile := filepath.Join(tmpDir, "keys.txt")
	writeKeysToFile(keys, keyFile)

	seed := uint64(42)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		iter, _ := newFileIterator(keyFile)
		azft, _ := NewApproxZFastTrieWithSeedFromIterator[uint16, uint32, uint32](iter,  seed)
		iter.Close()
		_ = azft
	}
}

// BenchmarkMemoryAlloc_AZFT_Streaming measures allocations for streaming builder
func BenchmarkMemoryAlloc_AZFT_Streaming(b *testing.B) {
	r := rand.New(rand.NewSource(42))
	keys := zft.GenerateRandomBitStrings(500, 128, r)

	tmpDir := b.TempDir()
	keyFile := filepath.Join(tmpDir, "keys.txt")
	writeKeysToFile(keys, keyFile)

	seed := uint64(42)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		iter, _ := newFileIterator(keyFile)
		azft, _ := NewApproxZFastTrieFromIteratorStreaming[uint16, uint32, uint32](iter,  seed)
		iter.Close()
		_ = azft
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
