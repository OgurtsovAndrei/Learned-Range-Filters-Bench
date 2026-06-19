package rloc

import (
	"Thesis-bench-industry/experiments/trie/zft"
	"testing"
)

// TestReplaySpecificFailure tests a specific failing seed multiple times
func TestReplaySpecificFailure(t *testing.T) {
	seed := int64(1770219547974520041)
	mmphSeed := uint64(4010545232912815505)

	// Generate keys with the exact same seed
	keys := GenUniqueBitStringsDebug(seed, maxKeys, maxBitLen)
	t.Logf("Generated %d keys", len(keys))

	// Try building 10 times with same keys and same mmphSeed
	successCount := 0
	failCount := 0

	for i := 0; i < 10; i++ {
		zt := zft.Build(keys)
		_, err := NewRangeLocatorSeeded(zt, mmphSeed)

		if err != nil {
			t.Logf("Attempt %d: FAILED - %v", i+1, err)
			failCount++
		} else {
			t.Logf("Attempt %d: SUCCEEDED", i+1)
			successCount++
		}
	}

	t.Logf("Results: %d successes, %d failures out of 10 attempts", successCount, failCount)

	if successCount > 0 && failCount > 0 {
		t.Errorf("NON-DETERMINISTIC: Same input produced different results!")
	} else if successCount == 10 {
		t.Logf("DETERMINISTIC: All attempts succeeded")
	} else if failCount == 10 {
		t.Logf("DETERMINISTIC: All attempts failed (this is the expected behavior for this seed)")
	}
}
