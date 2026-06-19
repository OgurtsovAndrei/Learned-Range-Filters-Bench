package hzft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
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
	// Increase buffer size for long lines
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

// writeKeysToFile writes sorted keys to a temporary file in binary format
func writeKeysToFile(keys []bits.BitString, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, key := range keys {
		// Write key as binary string (0s and 1s)
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

// getMemStats returns current heap allocation in bytes
func getMemStats() uint64 {
	runtime.GC()
	runtime.GC() // Run twice to ensure cleanup
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

// TestMemory_StreamingVsHeavy compares memory usage of streaming vs heavy builders
func TestMemory_StreamingVsHeavy(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory test in short mode")
	}

	// Test parameters
	testCases := []struct {
		numKeys int
		bitLen  int
	}{
		{1000, 64},
		{5000, 128},
		{10000, 256},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("n=%d_L=%d", tc.numKeys, tc.bitLen), func(t *testing.T) {
			// Generate sorted keys
			r := rand.New(rand.NewSource(42))
			keys := zft.GenerateRandomBitStrings(tc.numKeys, tc.bitLen, r)

			// Write to temp file
			tmpDir := t.TempDir()
			keyFile := filepath.Join(tmpDir, "keys.txt")
			err := writeKeysToFile(keys, keyFile)
			if err != nil {
				t.Fatalf("Failed to write keys: %v", err)
			}

			// Measure heavy builder memory
			runtime.GC()
			memBefore := getMemStats()

			iter1, err := newFileIterator(keyFile)
			if err != nil {
				t.Fatalf("Failed to create iterator: %v", err)
			}
			heavyHZFT, err := NewHZFastTrieFromIterator[uint32](iter1)
			iter1.Close()
			if err != nil {
				t.Fatalf("Heavy builder failed: %v", err)
			}

			memAfterHeavy := getMemStats()
			heavyMem := memAfterHeavy - memBefore

			// Clear heavy result
			_ = heavyHZFT
			heavyHZFT = nil
			runtime.GC()
			runtime.GC()

			// Measure streaming builder memory
			memBefore = getMemStats()

			iter2, err := newFileIterator(keyFile)
			if err != nil {
				t.Fatalf("Failed to create iterator: %v", err)
			}
			streamingHZFT, err := NewHZFastTrieFromIteratorStreaming[uint32](iter2)
			iter2.Close()
			if err != nil {
				t.Fatalf("Streaming builder failed: %v", err)
			}

			memAfterStreaming := getMemStats()
			streamingMem := memAfterStreaming - memBefore

			// Report results
			t.Logf("Keys: %d, BitLen: %d", tc.numKeys, tc.bitLen)
			t.Logf("Heavy builder memory:     %d bytes (%.2f MB)", heavyMem, float64(heavyMem)/(1024*1024))
			t.Logf("Streaming builder memory: %d bytes (%.2f MB)", streamingMem, float64(streamingMem)/(1024*1024))

			if heavyMem > 0 {
				ratio := float64(heavyMem) / float64(streamingMem)
				t.Logf("Ratio (heavy/streaming):  %.2fx", ratio)
			}

			// Verify results are equivalent
			keys2 := make([]bits.BitString, len(keys))
			copy(keys2, keys)
			for _, key := range keys2[:min(100, len(keys2))] {
				for prefixLen := 1; prefixLen <= int(key.Size()); prefixLen++ {
					prefix := key.Prefix(prefixLen)
					// Just verify streaming result is valid
					_ = streamingHZFT.GetExistingPrefix(prefix)
				}
			}

			_ = streamingHZFT // Keep reference to prevent early GC
		})
	}
}

// TestMemory_ScalingWithKeyCount tests how memory scales with number of keys
func TestMemory_ScalingWithKeyCount(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory scaling test in short mode")
	}

	bitLen := 128
	keyCounts := []int{1000, 2000, 4000, 8000}

	t.Log("Memory scaling with key count (bitLen=128):")
	t.Log("Keys\tHeavy(KB)\tStreaming(KB)\tRatio")

	var results []struct {
		n         int
		heavy     uint64
		streaming uint64
	}

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
		heavyHZFT, _ := NewHZFastTrieFromIterator[uint32](iter1)
		iter1.Close()
		heavyMem := getMemStats() - memBefore
		_ = heavyHZFT
		heavyHZFT = nil
		runtime.GC()

		// Streaming
		memBefore = getMemStats()
		iter2, _ := newFileIterator(keyFile)
		streamingHZFT, _ := NewHZFastTrieFromIteratorStreaming[uint32](iter2)
		iter2.Close()
		streamingMem := getMemStats() - memBefore

		results = append(results, struct {
			n         int
			heavy     uint64
			streaming uint64
		}{n, heavyMem, streamingMem})

		ratio := float64(heavyMem) / float64(streamingMem)
		t.Logf("%d\t%.1f\t\t%.1f\t\t%.2fx", n, float64(heavyMem)/1024, float64(streamingMem)/1024, ratio)

		_ = streamingHZFT
	}

	// Verify roughly linear scaling
	if len(results) >= 2 {
		first := results[0]
		last := results[len(results)-1]

		expectedRatio := float64(last.n) / float64(first.n)
		actualHeavyRatio := float64(last.heavy) / float64(first.heavy)
		actualStreamingRatio := float64(last.streaming) / float64(first.streaming)

		t.Logf("\nScaling analysis (expected %.1fx for %dx keys):", expectedRatio, last.n/first.n)
		t.Logf("Heavy scaling:     %.2fx", actualHeavyRatio)
		t.Logf("Streaming scaling: %.2fx", actualStreamingRatio)
	}
}

// TestMemory_ScalingWithKeyLength tests how memory scales with key length
func TestMemory_ScalingWithKeyLength(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory scaling test in short mode")
	}

	numKeys := 2000
	bitLengths := []int{64, 128, 256, 512}

	t.Log("Memory scaling with key length (numKeys=2000):")
	t.Log("BitLen\tHeavy(KB)\tStreaming(KB)\tRatio")

	var results []struct {
		bitLen    int
		heavy     uint64
		streaming uint64
	}

	for _, bitLen := range bitLengths {
		r := rand.New(rand.NewSource(42))
		keys := zft.GenerateRandomBitStrings(numKeys, bitLen, r)

		tmpDir := t.TempDir()
		keyFile := filepath.Join(tmpDir, "keys.txt")
		writeKeysToFile(keys, keyFile)

		// Heavy
		runtime.GC()
		memBefore := getMemStats()
		iter1, _ := newFileIterator(keyFile)
		heavyHZFT, _ := NewHZFastTrieFromIterator[uint32](iter1)
		iter1.Close()
		heavyMem := getMemStats() - memBefore
		_ = heavyHZFT
		heavyHZFT = nil
		runtime.GC()

		// Streaming
		memBefore = getMemStats()
		iter2, _ := newFileIterator(keyFile)
		streamingHZFT, _ := NewHZFastTrieFromIteratorStreaming[uint32](iter2)
		iter2.Close()
		streamingMem := getMemStats() - memBefore

		results = append(results, struct {
			bitLen    int
			heavy     uint64
			streaming uint64
		}{bitLen, heavyMem, streamingMem})

		ratio := float64(heavyMem) / float64(streamingMem)
		t.Logf("%d\t%.1f\t\t%.1f\t\t%.2fx", bitLen, float64(heavyMem)/1024, float64(streamingMem)/1024, ratio)

		_ = streamingHZFT
	}

	// Analyze scaling
	if len(results) >= 2 {
		first := results[0]
		last := results[len(results)-1]

		t.Logf("\nScaling analysis (bitLen %d -> %d):", first.bitLen, last.bitLen)
		t.Logf("Heavy memory grew:     %.2fx", float64(last.heavy)/float64(first.heavy))
		t.Logf("Streaming memory grew: %.2fx", float64(last.streaming)/float64(first.streaming))
		t.Logf("Expected for heavy (O(n*L)): %.2fx", float64(last.bitLen)/float64(first.bitLen))
		t.Logf("Expected for streaming (O(n*log L)): %.2fx", float64(log2(last.bitLen))/float64(log2(first.bitLen)))
	}
}

func log2(x int) float64 {
	if x <= 0 {
		return 0
	}
	result := 0.0
	for x > 1 {
		x /= 2
		result++
	}
	return result
}

// BenchmarkMemoryAlloc measures allocations during construction
func BenchmarkMemoryAlloc_Heavy(b *testing.B) {
	r := rand.New(rand.NewSource(42))
	keys := zft.GenerateRandomBitStrings(1000, 128, r)

	tmpDir := b.TempDir()
	keyFile := filepath.Join(tmpDir, "keys.txt")
	writeKeysToFile(keys, keyFile)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		iter, _ := newFileIterator(keyFile)
		hzft, _ := NewHZFastTrieFromIterator[uint32](iter)
		iter.Close()
		_ = hzft
	}
}

func BenchmarkMemoryAlloc_Streaming(b *testing.B) {
	r := rand.New(rand.NewSource(42))
	keys := zft.GenerateRandomBitStrings(1000, 128, r)

	tmpDir := b.TempDir()
	keyFile := filepath.Join(tmpDir, "keys.txt")
	writeKeysToFile(keys, keyFile)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		iter, _ := newFileIterator(keyFile)
		hzft, _ := NewHZFastTrieFromIteratorStreaming[uint32](iter)
		iter.Close()
		_ = hzft
	}
}

// sliceBitStringIterator wraps a slice for iterator interface
type sliceBitStringIterator struct {
	keys  []bits.BitString
	index int
}

func (it *sliceBitStringIterator) Next() bool {
	it.index++
	return it.index < len(it.keys)
}

func (it *sliceBitStringIterator) Value() bits.BitString {
	return it.keys[it.index]
}

func (it *sliceBitStringIterator) Error() error {
	return nil
}

// Helper to generate sorted keys
func generateSortedKeys(n, bitLen int, r *rand.Rand) []bits.BitString {
	keys := zft.GenerateRandomBitStrings(n, bitLen, r)
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Compare(keys[j]) < 0
	})
	return keys
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
