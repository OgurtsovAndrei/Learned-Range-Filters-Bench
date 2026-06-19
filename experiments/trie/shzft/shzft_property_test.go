package shzft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const (
	shzftTestRuns   = 1000
	shzftMaxKeys    = 64
	shzftMaxBitLen  = 32
)

func TestSHZFastTrie_Correctness_AllPrefixes(t *testing.T) {
	t.Parallel()

	errorCount := 0
	totalChecks := 0

	for run := 0; run < shzftTestRuns; run++ {
		seed := time.Now().UnixNano() + int64(run)
		r := rand.New(rand.NewSource(seed))

		numKeys := r.Intn(shzftMaxKeys-1) + 2
		bitLen := r.Intn(shzftMaxBitLen-8) + 8
		keys := zft.GenerateRandomBitStrings(numKeys, bitLen, r)

		shzft := NewSuccinctHZFastTrie(keys)
		if shzft == nil {
			continue
		}
		
		// Build reference trie separately for comparison
		referenceTrie, _ := zft.BuildFromIterator(bits.NewSliceBitStringIterator(keys))

		for _, key := range keys {
			for prefixLen := 0; prefixLen <= int(key.Size()); prefixLen++ {
				var prefix bits.BitString
				if prefixLen == 0 {
					prefix = bits.NewFromText("")
				} else {
					prefix = key.Prefix(prefixLen)
				}
				totalChecks++

				shzftResult := shzft.GetExistingPrefix(prefix)

				expectedNode := referenceTrie.GetExitNode(prefix)

				var expectedLength int64
				if expectedNode == referenceTrie.Root {
					expectedLength = 0
				} else {
					// In HZFT, name length of child is parent.extentLen + 1
					expectedLength = int64(expectedNode.NameLength)
				}

				if shzftResult != expectedLength {
					errorCount++
					msg := fmt.Sprintf("Mismatch for prefix %s (key %s). SHZFT length: %d, Ref length: %d (Seed: %d)",
						prefix.PrettyString(), key.PrettyString(), shzftResult, expectedLength, seed)
					
					saveErrorReport(t, seed, keys, prefix, shzftResult, expectedLength)
					t.Errorf("%s", msg)
					
					// Verify reproducibility
					verifyReproducibility(t, seed, keys, prefix, expectedLength)
					
					return // Stop on first error to avoid flooding
				}
			}
		}
	}

	t.Logf("Total checks: %d, Errors: %d", totalChecks, errorCount)
}

func saveErrorReport(t *testing.T, seed int64, keys []bits.BitString, prefix bits.BitString, actual, expected int64) {
	reportDir := "error_reports"
	_ = os.MkdirAll(reportDir, 0755)
	
	filename := filepath.Join(reportDir, fmt.Sprintf("shzft_error_%d.txt", seed))
	f, err := os.Create(filename)
	if err != nil {
		t.Logf("Failed to create error report file: %v", err)
		return
	}
	defer f.Close()

	fmt.Fprintf(f, "Seed: %d\n", seed)
	fmt.Fprintf(f, "Prefix: %s\n", prefix.PrettyString())
	fmt.Fprintf(f, "Actual: %d\n", actual)
	fmt.Fprintf(f, "Expected: %d\n", expected)
	fmt.Fprintf(f, "Keys:\n")
	for _, k := range keys {
		fmt.Fprintf(f, "  %s\n", k.PrettyString())
	}
	
	t.Logf("Error report saved to %s", filename)
}

func verifyReproducibility(t *testing.T, seed int64, keys []bits.BitString, prefix bits.BitString, expected int64) {
	shzft := NewSuccinctHZFastTrie(keys)
	actual := shzft.GetExistingPrefix(prefix)
	if actual != expected {
		t.Logf("Reproduced error successfully with seed %d", seed)
	} else {
		t.Errorf("FAILED to reproduce error with seed %d. Actual: %d, Expected: %d", seed, actual, expected)
	}
}
