package hzft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

const (
	hzftTestRuns   = 1000
	hzftIterations = 1000
	hzftMaxKeys    = 64
	hzftMaxBitLen  = 16
)

func TestHZFastTrie_Correctness_AllPrefixes(t *testing.T) {
	t.Parallel()

	errorCount := 0
	totalChecks := 0
	runsWithErrors := 0

	for run := 0; run < hzftTestRuns; run++ {
		seed := time.Now().UnixNano() + int64(run)
		r := rand.New(rand.NewSource(seed))

		numKeys := r.Intn(hzftMaxKeys-1) + 2
		bitLen := r.Intn(hzftMaxBitLen-8) + 8
		keys := zft.GenerateRandomBitStrings(numKeys, bitLen, r)

		hzft := NewHZFastTrie[uint32](keys)
		if hzft == nil {
			continue
		}
		// Build reference trie separately for comparison
		referenceTrie, _ := zft.BuildFromIterator(bits.NewSliceBitStringIterator(keys))
		errorsInRun := 0

		for _, key := range keys {
			for prefixLen := 1; prefixLen <= int(key.Size()); prefixLen++ {
				prefix := key.Prefix(prefixLen)
				totalChecks++

				hzftResult := hzft.GetExistingPrefix(prefix)

				expectedNode := referenceTrie.GetExitNode(prefix)

				var expectedLength int64
				if expectedNode == referenceTrie.Root {
					expectedLength = 0
				} else {
					// Имя узла выхода n_alpha имеет длину |e_parent| + 1
					expectedLength = int64(expectedNode.NameLength)
				}

				if hzftResult != expectedLength {
					errorsInRun++
					errorCount++
					t.Errorf("Mismatch for prefix %s (key %s). HZFT length: %d, Ref length: %d (Seed: %d)",
						prefix.PrettyString(), key.PrettyString(), hzftResult, expectedLength, seed)
					fmt.Println(keys)
					fmt.Println(hzft)
					fmt.Println(hzft.GetExistingPrefix(prefix))
					break
				}
			}
		}

		if errorsInRun > 0 {
			runsWithErrors++
		}
	}

	t.Logf("Total checks: %d, Errors: %d", totalChecks, errorCount)
	if errorCount > 0 {
		t.Fatalf("HZFT failed %d checks out of %d. It must be deterministic.", errorCount, totalChecks)
	}
}
