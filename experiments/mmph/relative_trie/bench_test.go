package bucket

import (
	"Thesis/testutils"
	"fmt"
	"testing"
)

var (
	benchKeyCounts = []int{1 << 10, 1 << 13, 1 << 15, 1 << 18, 1 << 20}
)

func BenchmarkMonotoneHashWithTrieBuild(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeys(64, count) // Using default 64-bit length

			var totalTrieAttempts int64
			var maxTrieAttempts int

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				table, err := NewMonotoneHashWithTrie[uint8, uint32, uint16](keys)
				if err != nil {
					b.Fatalf("Failed to build: %v", err)
				}

				totalTrieAttempts += int64(table.TrieRebuildAttempts)
				if table.TrieRebuildAttempts > maxTrieAttempts {
					maxTrieAttempts = table.TrieRebuildAttempts
				}

				b.ReportMetric(float64(table.Size())*8/float64(count), "bits/key_in_mem")
				b.ReportMetric(float64(table.Size()), "bytes_in_mem")
			}

			b.ReportMetric(float64(totalTrieAttempts)/float64(b.N), "avg_trie_attempts")
			b.ReportMetric(float64(maxTrieAttempts), "max_trie_attempts")
		})
	}
}

func BenchmarkMonotoneHashWithTrieLookup(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeys(64, count)
			mh, err := NewMonotoneHashWithTrie[uint8, uint32, uint16](keys)
			if err != nil {
				b.Fatalf("Failed to build: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Query keys cyclically
				_ = mh.GetRank(keys[i%count])
			}
		})
	}
}

func BenchmarkMonotoneHashWithTrieLookupMiss(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeys(64, count)
			mh, err := NewMonotoneHashWithTrie[uint8, uint32, uint16](keys)
			if err != nil {
				b.Fatalf("Failed to build: %v", err)
			}

			// Generate some keys that are likely not in the set
			missKeys := testutils.GetBenchKeys(64, 100)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Query non-existent keys cyclically
				_ = mh.GetRank(missKeys[i%len(missKeys)])
			}
		})
	}
}

// Benchmark to measure trie construction overhead specifically
func BenchmarkTrieRebuilds(b *testing.B) {
	// Focus on medium-sized datasets where trie rebuilds are more likely
	testCounts := []int{1 << 10, 1 << 13, 1 << 15}

	for _, count := range testCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeys(64, count)

			successfulBuilds := 0
			totalAttempts := int64(0)
			failedBuilds := 0

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				table, err := NewMonotoneHashWithTrie[uint8, uint32, uint16](keys)
				if err != nil {
					failedBuilds++
					continue
				}

				successfulBuilds++
				totalAttempts += int64(table.TrieRebuildAttempts)
			}

			if successfulBuilds > 0 {
				b.ReportMetric(float64(totalAttempts)/float64(successfulBuilds), "avg_attempts_per_success")
			}
			b.ReportMetric(float64(failedBuilds), "failed_builds")
			b.ReportMetric(float64(successfulBuilds), "successful_builds")
		})
	}
}

// Memory usage benchmark
func BenchmarkMonotoneHashWithTrieMemory(b *testing.B) {
	for _, count := range benchKeyCounts {
		b.Run(fmt.Sprintf("Keys=%d", count), func(b *testing.B) {
			keys := testutils.GetBenchKeys(64, count)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				table, err := NewMonotoneHashWithTrie[uint8, uint32, uint16](keys)
				if err != nil {
					b.Fatalf("Failed to build: %v", err)
				}

				size := table.Size()

				// Report various memory metrics
				b.ReportMetric(float64(size), "total_bytes")
				b.ReportMetric(float64(size)*8/float64(count), "bits_per_key")
				b.ReportMetric(float64(len(table.buckets)), "num_buckets")
				b.ReportMetric(float64(table.bucketSize), "bucket_size")

				// Calculate exact trie size and overhead
				trieSize := 0
				if table.delimiterTrie != nil {
					trieSize = table.delimiterTrie.ByteSize()
				}

				bucketOverhead := 0
				delimiterOverhead := 0
				for _, bucket := range table.buckets {
					if bucket != nil {
						// MPHF + ranks array + delimiter
						bucketOverhead += bucket.mphf.Size()
						bucketOverhead += len(bucket.ranks)
						delimiterOverhead += int(bucket.delimiter.Size())/8 + 1
					}
				}

				// Report exact measurements
				b.ReportMetric(float64(trieSize), "exact_trie_bytes")
				b.ReportMetric(float64(trieSize)*100/float64(size), "trie_percent_of_total")
				b.ReportMetric(float64(delimiterOverhead), "delimiter_storage_bytes")
				b.ReportMetric(float64(bucketOverhead), "bucket_overhead_bytes")
			}
		})
	}
}

// Benchmark detailed memory breakdown for MMPH
func BenchmarkMemoryDetailed(b *testing.B) {
	for _, bitLen := range testutils.DefaultBenchBitLengths {
		for _, count := range benchKeyCounts {
			b.Run(fmt.Sprintf("KeySize=%d/Keys=%d", bitLen, count), func(b *testing.B) {
				keys := testutils.GetBenchKeys(bitLen, count)

				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					table, err := NewMonotoneHashWithTrie[uint8, uint32, uint16](keys)
					if err != nil {
						b.Fatalf("Failed to build: %v", err)
					}

					// Log the detailed report for the analyzer
					if i == 0 {
						report := table.MemDetailed()
						b.Logf("JSON_MEM_REPORT: %s", report.JSON())
					}
				}
			})
		}
	}
}
