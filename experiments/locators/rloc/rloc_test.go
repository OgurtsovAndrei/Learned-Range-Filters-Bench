package rloc

import (
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
	"testing"
	"time"
)

const (
	testRuns  = 10_000
	maxKeys   = 1024
	maxBitLen = 16
)

func TestRangeLocator_Correctness(t *testing.T) {
	for run := 0; run < testRuns; run++ {
		t.Run(fmt.Sprintf("run=%d", run), func(t *testing.T) {
			t.Parallel()
			seed := time.Now().UnixNano()
			keys := GenUniqueBitStrings(seed, maxKeys, maxBitLen)

			zt := zft.Build(keys)
			rl, err := NewRangeLocator(zt)
			if err != nil {
				t.Fatalf("NewRangeLocator failed (seed: %d): %v", seed, err)
			}

			it := zft.NewIterator(zt)
			for it.Next() {
				node := it.Node()
				if node == nil {
					continue
				}

				start, end, err := rl.Query(node.Extent)
				if err != nil {
					t.Fatalf("Query failed for existing node (seed: %d): %v", seed, err)
				}

				expectedStart, expectedEnd := FindRange(keys, node.Extent)

				if start != expectedStart || end != expectedEnd {
					t.Errorf("Mismatch for node %s (seed: %d). Got: [%d, %d), Exp: [%d, %d)",
						node.Extent.PrettyString(), seed, start, end, expectedStart, expectedEnd)
					t.FailNow()
				}
			}
		})
	}
}
