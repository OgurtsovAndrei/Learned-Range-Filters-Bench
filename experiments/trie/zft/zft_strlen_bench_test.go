package zft

import (
	"fmt"
	"math/rand"
	"testing"
)

var lengths = []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}

func BenchmarkTrie_ByStrLen_Insert(b *testing.B) {
	r := rand.New(rand.NewSource(42))

	for _, L := range lengths {
		b.Run(fmt.Sprintf("Len%d", L), func(b *testing.B) {
			b.StopTimer()
			keys := GenerateRandomBitStrings(b.N, L, r)
			tree := NewZFastTrie[bool](false)
			b.StartTimer()

			for i := 0; i < b.N; i++ {
				tree.InsertBitString(keys[i], true)
			}
		})
	}
}

func BenchmarkTrie_ByStrLen_GetExitNode_Hit(b *testing.B) {
	numSetupKeys := 100_000
	r := rand.New(rand.NewSource(42))

	for _, L := range lengths {
		b.Run(fmt.Sprintf("Len%d", L), func(b *testing.B) {
			b.StopTimer()
			keys := GenerateRandomBitStrings(numSetupKeys, L, r)
			tree := NewZFastTrie[bool](false)
			for _, k := range keys {
				tree.InsertBitString(k, true)
			}
			mask := numSetupKeys - 1
			b.StartTimer()

			for i := 0; i < b.N; i++ {
				exitNode := tree.GetExitNode(keys[i&mask])
				if exitNode == nil {
					b.Fatalf("GetExitNode returned nil for existing key")
				}
			}
		})
	}
}

func BenchmarkTrie_ByStrLen_GetExitNode_Miss(b *testing.B) {
	numSetupKeys := 100_000
	r := rand.New(rand.NewSource(42))
	rMiss := rand.New(rand.NewSource(43))

	for _, L := range lengths {
		b.Run(fmt.Sprintf("Len%d", L), func(b *testing.B) {
			b.StopTimer()
			keys := GenerateRandomBitStrings(numSetupKeys, L, r)
			tree := NewZFastTrie[bool](false)
			for _, k := range keys {
				tree.InsertBitString(k, true)
			}
			missKeys := GenerateRandomBitStrings(b.N, L, rMiss)
			b.StartTimer()

			for i := 0; i < b.N; i++ {
				exitNode := tree.GetExitNode(missKeys[i])
				_ = exitNode
			}
		})
	}
}
