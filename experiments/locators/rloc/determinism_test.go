package rloc

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"testing"
)

// TestDeterministicMMPHConstruction verifies that MMPH construction is deterministic
func TestDeterministicMMPHConstruction(t *testing.T) {
	// Create a simple set of keys
	keys := []bits.BitString{
		bits.NewFromUint64(10),
		bits.NewFromUint64(20),
		bits.NewFromUint64(30),
		bits.NewFromUint64(40),
		bits.NewFromUint64(50),
	}

	seed := uint64(12345)

	// Build twice with same seed
	zt1 := zft.Build(keys)
	rl1, err1 := NewRangeLocatorSeeded(zt1, seed)

	zt2 := zft.Build(keys)
	rl2, err2 := NewRangeLocatorSeeded(zt2, seed)

	// Both should succeed or both should fail
	if (err1 == nil) != (err2 == nil) {
		t.Fatalf("Non-deterministic: err1=%v, err2=%v", err1, err2)
	}

	// If both succeeded, verify they produce same results
	if err1 == nil && err2 == nil {
		for _, key := range keys {
			r1start, r1end, e1 := rl1.Query(key)
			r2start, r2end, e2 := rl2.Query(key)

			if e1 != nil || e2 != nil {
				t.Fatalf("Query error: e1=%v, e2=%v", e1, e2)
			}

			if r1start != r2start || r1end != r2end {
				t.Fatalf("Non-deterministic query results for key %v: (%d,%d) vs (%d,%d)",
					key, r1start, r1end, r2start, r2end)
			}
		}
		t.Logf("SUCCESS: Both constructions succeeded and produced identical results")
	} else {
		// Both failed - check if errors match
		if err1.Error() != err2.Error() {
			t.Fatalf("Non-deterministic errors: %v vs %v", err1, err2)
		}
		t.Logf("SUCCESS: Both constructions failed with same error: %v", err1)
	}
}
