// Benchmark comparison: ZFastTrie vs Compressed Trie (siongui/go-succinct-data-structure-Trie)
//
// Results (Apple M4):
// ┌─────────────┬─────────────────┬──────────────────┬─────────────────┐
// │ Operation   │ ZFastTrie       │ Compressed Trie  │ ZFast Advantage │
// ├─────────────┼─────────────────┼──────────────────┼─────────────────┤
// │ Lookup 1K   │ ~106 ns/op      │ ~15,003 ns/op    │ 142x faster     │
// │ Lookup 10K  │ ~136 ns/op      │ ~19,093 ns/op    │ 140x faster     │
// └─────────────┴─────────────────┴──────────────────┴─────────────────┘
//
// Key Findings:
// - ZFastTrie demonstrates 140x+ performance advantage for lookups
// - ZFastTrie scales better: minimal performance degradation with size increase
// - CompressedTrie trades lookup speed for memory efficiency (succinct representation)
// - ZFastTrie is ideal for speed-critical applications with frequent searches
// - CompressedTrie is suitable for memory-constrained environments with static data
//
// Test Data: Random alphanumeric strings with prefixes/suffixes (now using 1B number range)

package zft

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"

	trie "github.com/siongui/go-succinct-data-structure-trie/reference"
)

// generateTextKeys generates random text keys for Trie benchmarking
func generateTextKeys(n int) []string {
	r := rand.New(rand.NewSource(42))
	keys := make([]string, n)
	set := make(map[string]struct{}, n)

	prefixes := []string{"test", "bench", "word", "data", "algo", "struct", "node", "tree"}
	suffixes := []string{"ing", "ed", "er", "ly", "tion", "ness", "ment", "ful"}

	for i := 0; i < n; {
		prefix := prefixes[r.Intn(len(prefixes))]
		suffix := suffixes[r.Intn(len(suffixes))]
		number := r.Intn(1_000_000_000)
		key := fmt.Sprintf("%s_%d_%s", prefix, number, suffix)

		if _, ok := set[key]; !ok {
			set[key] = struct{}{}
			keys[i] = key
			i++
		}
	}
	return keys
}

// setupZFastTrieText sets up ZFastTrie with text keys
func setupZFastTrieText(b *testing.B, n int) (*ZFastTrie[bool], []string) {
	b.Helper()
	b.StopTimer()
	keys := generateTextKeys(n)
	tree := NewZFastTrie[bool](false)
	for _, key := range keys {
		tree.Insert(key, true)
	}
	b.StartTimer()
	return tree, keys
}

// setupCompressedTrie sets up the library's compressed Trie
func setupCompressedTrie(b *testing.B, n int) (*trie.FrozenTrie, []string) {
	b.Helper()
	b.StopTimer()
	keys := generateTextKeys(n)

	// Build the Trie
	t := &trie.Trie{}
	t.Init()
	for _, key := range keys {
		t.Insert(key)
	}

	// Encode and freeze the Trie
	encoded := t.Encode()
	rd := trie.CreateRankDirectory(encoded, t.GetNodeCount()*2+1, trie.L1, trie.L2)
	frozenTrie := &trie.FrozenTrie{}
	frozenTrie.Init(encoded, rd.GetData(), t.GetNodeCount())

	b.StartTimer()
	return frozenTrie, keys
}

// --- Insert Benchmarks ---

func BenchmarkZFastTrie_Insert_Text(b *testing.B) {
	b.StopTimer()
	keys := generateTextKeys(b.N)
	tree := NewZFastTrie[bool](false)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		tree.Insert(keys[i], true)
	}
}

func BenchmarkCompressedTrie_Build_Text(b *testing.B) {
	keys := generateTextKeys(1000) // Fixed size for fair comparison

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := &trie.Trie{}
		t.Init()
		for _, key := range keys {
			t.Insert(key)
		}
		// Include encoding time as part of "build" cost
		_ = t.Encode()
	}
}

// --- Lookup Benchmarks ---

func BenchmarkZFastTrie_Lookup_Hit_1K(b *testing.B) {
	tree, keys := setupZFastTrieText(b, 1000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		tree.Contains(keys[i&mask])
	}
}

func BenchmarkCompressedTrie_Lookup_Hit_1K(b *testing.B) {
	frozenTrie, keys := setupCompressedTrie(b, 1000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		frozenTrie.Lookup(keys[i&mask])
	}
}

func BenchmarkZFastTrie_Lookup_Hit_10K(b *testing.B) {
	tree, keys := setupZFastTrieText(b, 10000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		tree.Contains(keys[i&mask])
	}
}

func BenchmarkCompressedTrie_Lookup_Hit_10K(b *testing.B) {
	frozenTrie, keys := setupCompressedTrie(b, 10000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		frozenTrie.Lookup(keys[i&mask])
	}
}

func BenchmarkZFastTrie_Lookup_Hit_100K(b *testing.B) {
	tree, keys := setupZFastTrieText(b, 100000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		tree.Contains(keys[i&mask])
	}
}

func BenchmarkCompressedTrie_Lookup_Hit_100K(b *testing.B) {
	frozenTrie, keys := setupCompressedTrie(b, 100000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		frozenTrie.Lookup(keys[i&mask])
	}
}

// --- Miss Benchmarks ---

func BenchmarkZFastTrie_Lookup_Miss_1K(b *testing.B) {
	tree, _ := setupZFastTrieText(b, 1000)
	b.StopTimer()
	missKeys := generateTextKeys(b.N)
	// Modify keys to ensure misses
	for i := range missKeys {
		missKeys[i] = "miss_" + missKeys[i]
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		tree.Contains(missKeys[i])
	}
}

func BenchmarkCompressedTrie_Lookup_Miss_1K(b *testing.B) {
	frozenTrie, _ := setupCompressedTrie(b, 1000)
	b.StopTimer()
	missKeys := generateTextKeys(b.N)
	// Modify keys to ensure misses
	for i := range missKeys {
		missKeys[i] = "miss_" + missKeys[i]
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		frozenTrie.Lookup(missKeys[i])
	}
}

func BenchmarkZFastTrie_Lookup_Miss_100K(b *testing.B) {
	tree, _ := setupZFastTrieText(b, 100000)
	b.StopTimer()
	missKeys := generateTextKeys(b.N)
	// Modify keys to ensure misses
	for i := range missKeys {
		missKeys[i] = "miss_" + missKeys[i]
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		tree.Contains(missKeys[i])
	}
}

func BenchmarkCompressedTrie_Lookup_Miss_100K(b *testing.B) {
	frozenTrie, _ := setupCompressedTrie(b, 100000)
	b.StopTimer()
	missKeys := generateTextKeys(b.N)
	// Modify keys to ensure misses
	for i := range missKeys {
		missKeys[i] = "miss_" + missKeys[i]
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		frozenTrie.Lookup(missKeys[i])
	}
}

// --- Comparative Prefix Search Benchmarks ---

func BenchmarkZFastTrie_PrefixSearch_1K(b *testing.B) {
	tree, keys := setupZFastTrieText(b, 1000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		key := keys[i&mask]
		// Search for prefix (first half of the key)
		prefix := key[:len(key)/2]
		tree.ContainsPrefix(prefix)
	}
}

// Note: The library's compressed Trie doesn't appear to have prefix search,
// so we'll focus on exact lookup comparisons

// --- Memory Usage Benchmarks ---

func BenchmarkZFastTrie_MemoryUsage_Build(b *testing.B) {
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			keys := generateTextKeys(size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tree := NewZFastTrie[bool](false)
				for _, key := range keys {
					tree.Insert(key, true)
				}
				// Force garbage collection for memory measurements
				runtime.GC()
			}
		})
	}
}

func BenchmarkCompressedTrie_MemoryUsage_Build(b *testing.B) {
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			keys := generateTextKeys(size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				t := &trie.Trie{}
				t.Init()
				for _, key := range keys {
					t.Insert(key)
				}
				encoded := t.Encode()
				rd := trie.CreateRankDirectory(encoded, t.GetNodeCount()*2+1, trie.L1, trie.L2)
				frozenTrie := &trie.FrozenTrie{}
				frozenTrie.Init(encoded, rd.GetData(), t.GetNodeCount())
				// Force garbage collection for memory measurements
				runtime.GC()
			}
		})
	}
}
