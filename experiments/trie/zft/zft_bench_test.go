package zft

import (
	"Thesis/bits"
	"Thesis/bits/maps"
	"math/rand"
	"testing"

	iradix "github.com/hashicorp/go-immutable-radix"
)

func GenerateBitStringKeys(n int) []bits.BitString {
	r := rand.New(rand.NewSource(42))
	keys := make([]bits.BitString, n)
	set := make(map[uint64]struct{}, n)

	for i := 0; i < n; {
		k := r.Uint64()
		s := bits.NewFromUint64(k)
		if _, ok := set[k]; !ok {
			set[k] = struct{}{}
			keys[i] = s
			i++
		}
	}
	return keys
}

func setupBitStringTrie(b *testing.B, n int) (*ZFastTrie[bool], []bits.BitString) {
	b.Helper()
	b.StopTimer()
	keys := GenerateBitStringKeys(n)
	tree := NewZFastTrie[bool](false)
	for _, s := range keys {
		tree.InsertBitString(s, true)
	}
	b.StartTimer()
	return tree, keys
}

func setupStdMap(b *testing.B, n int) (*maps.BitMap[bool], []bits.BitString) {
	b.Helper()
	b.StopTimer()
	keys := GenerateBitStringKeys(n)
	m := maps.NewBitMap[bool]()
	for _, s := range keys {
		m.Put(s, true)
	}
	b.StartTimer()
	return m, keys
}

func setupiradixTrie(b *testing.B, n int) (*iradix.Tree, []bits.BitString) {
	b.Helper()
	b.StopTimer()
	keys := GenerateBitStringKeys(n)
	r := iradix.New()
	for _, s := range keys {
		r, _, _ = r.Insert(s.Data(), true)
	}
	b.StartTimer()
	return r, keys
}

func BenchmarkTrie_BitString_Insert(b *testing.B) {
	b.StopTimer()
	keys := GenerateBitStringKeys(b.N)
	tree := NewZFastTrie[bool](false)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		tree.InsertBitString(keys[i], true)
	}
}

func Benchmark_BitMap_BitString_Insert(b *testing.B) {
	b.StopTimer()
	keys := GenerateBitStringKeys(b.N)
	m := maps.NewBitMap[bool]()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		m.Put(keys[i], true)
	}
}

func Benchmark_iradix_BitString_Insert(b *testing.B) {
	b.StopTimer()
	keys := GenerateBitStringKeys(b.N)
	r := iradix.New()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		r, _, _ = r.Insert(keys[i].Data(), true)
	}
}

func BenchmarkTrie_BitString_Contains_Hit_100k(b *testing.B) {
	tree, keys := setupBitStringTrie(b, 100_000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		tree.ContainsBitString(keys[i&mask])
	}
}

func Benchmark_StdMap_BitString_Contains_Hit_100k(b *testing.B) {
	m, keys := setupStdMap(b, 100_000)
	mask := len(keys) - 1
	var ok bool
	for i := 0; i < b.N; i++ {
		_, ok = m.Get(keys[i&mask])
		_ = ok
	}
}

func Benchmark_iradix_BitString_Contains_Hit_100k(b *testing.B) {
	r, keys := setupiradixTrie(b, 100_000)
	mask := len(keys) - 1

	for i := 0; i < b.N; i++ {
		r.Get(keys[i&mask].Data())
	}
}

func BenchmarkTrie_BitString_Contains_Miss_100k(b *testing.B) {
	tree, _ := setupBitStringTrie(b, 100_000)
	b.StopTimer()
	missKeys := GenerateBitStringKeys(b.N)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		tree.ContainsBitString(missKeys[i])
	}
}

func Benchmark_StdMap_BitString_Contains_Miss_100k(b *testing.B) {
	m, _ := setupStdMap(b, 100_000)
	b.StopTimer()
	missKeys := GenerateBitStringKeys(b.N)
	b.StartTimer()
	var ok bool
	for i := 0; i < b.N; i++ {
		_, ok = m.Get(missKeys[i])
		_ = ok
	}
}

func Benchmark_iradix_BitString_Contains_Miss_100k(b *testing.B) {
	r, _ := setupiradixTrie(b, 100_000)
	b.StopTimer()
	missKeys := GenerateBitStringKeys(b.N)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		r.Get(missKeys[i].Data())
	}
}

func BenchmarkTrie_BitString_Erase_Hit_100k(b *testing.B) {
	b.StopTimer()

	keys := GenerateBitStringKeys(100_000 + b.N)
	tree := NewZFastTrie[bool](false)

	for i := 0; i < 100_000; i++ {
		tree.InsertBitString(keys[i], true)
	}

	eraseKeys := keys[100_000:]
	for i := 0; i < b.N; i++ {
		tree.InsertBitString(eraseKeys[i], true)
	}

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		tree.EraseBitString(eraseKeys[i])
	}
}

func BenchmarkTrie_BitString_Insert_Erase_100k(b *testing.B) {
	tree, _ := setupBitStringTrie(b, 100_000)
	b.StopTimer()
	keys := GenerateBitStringKeys(b.N)
	mask := len(keys) - 1
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		s := keys[i&mask]
		tree.InsertBitString(s, true)
		tree.EraseBitString(s)
	}
}

func Benchmark_StdMap_BitString_Insert_Erase_100k(b *testing.B) {
	m, _ := setupStdMap(b, 100_000)
	b.StopTimer()
	keys := GenerateBitStringKeys(b.N)
	mask := len(keys) - 1
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		s := keys[i&mask]
		m.Put(s, true)
		m.Delete(s)
	}
}

func Benchmark_iradix_BitString_Insert_Erase_100k(b *testing.B) {
	r, _ := setupiradixTrie(b, 100_000)
	b.StopTimer()
	keys := GenerateBitStringKeys(b.N)
	mask := len(keys) - 1
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		s := keys[i&mask].Data()
		r, _, _ = r.Insert(s, true)
		r, _, _ = r.Delete(s)
	}
}
