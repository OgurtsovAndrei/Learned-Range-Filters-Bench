package lerloc

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/locators/rloc"
	"fmt"
	"sort"
	"testing"
	"time"
)

const (
	testRuns  = 1_000
	maxKeys   = 1 << 12
	maxBitLen = 16
)

func TestLocalExactRangeLocator_EmptyKeys(t *testing.T) {
	lerl, err := NewLocalExactRangeLocator([]bits.BitString{})
	if err != nil {
		t.Fatalf("NewLocalExactRangeLocator failed: %v", err)
	}
	start, end, err := lerl.WeakPrefixSearch(bits.NewFromText("test"))
	if err != nil {
		t.Errorf("Expected no error for empty keys, got: %v", err)
	}
	if start != 0 || end != 0 {
		t.Errorf("Expected [0, 0) for empty keys, got: [%d, %d)", start, end)
	}
}

func TestLocalExactRangeLocator_EmptyPrefix(t *testing.T) {
	keys := []bits.BitString{
		bits.NewFromText("abc"),
		bits.NewFromText("def"),
	}

	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Compare(keys[j]) < 0
	})

	lerl, err := NewLocalExactRangeLocator(keys)
	if err != nil {
		t.Fatalf("NewLocalExactRangeLocator failed: %v", err)
	}

	start, end, err := lerl.WeakPrefixSearch(bits.NewFromText(""))
	if err != nil {
		t.Errorf("Expected no error for empty prefix, got: %v", err)
	}
	if start != 0 || end != 2 {
		t.Errorf("Expected [0, 2) for empty prefix, got: [%d, %d)", start, end)
	}
}

func TestLocalExactRangeLocator_AllPrefixes(t *testing.T) {
	for run := 0; run < testRuns; run++ {
		t.Run(fmt.Sprintf("run=%d", run), func(t *testing.T) {
			t.Parallel()
			seed := time.Now().UnixNano()
			keys := rloc.GenUniqueBitStrings(seed, maxKeys, maxBitLen)

			lerl, err := NewLocalExactRangeLocator(keys)
			if err != nil {
				t.Fatalf("NewLocalExactRangeLocator failed (seed: %d): %v", seed, err)
			}

			for _, key := range keys {
				for prefixLen := uint32(0); prefixLen <= key.Size(); prefixLen++ {
					var prefix bits.BitString
					if prefixLen == 0 {
						prefix = bits.NewFromText("")
					} else {
						prefix = key.Prefix(int(prefixLen))
					}

					start, end, err := lerl.WeakPrefixSearch(prefix)
					if err != nil {
						t.Fatalf("WeakPrefixSearch failed for prefix %s of key %s (seed: %d): %v",
							prefix.PrettyString(), key.PrettyString(), seed, err)
					}

					expectedStart, expectedEnd := rloc.FindRange(keys, prefix)

					if start != expectedStart || end != expectedEnd {
						t.Errorf("Mismatch for prefix %s (seed: %d). Got: [%d, %d), Exp: [%d, %d)",
							prefix.PrettyString(), seed, start, end, expectedStart, expectedEnd)
						t.FailNow()
					}
				}
			}
		})
	}
}

func TestCompactLocalExactRangeLocator_AllPrefixes(t *testing.T) {
	// Run fewer iterations for the compact version as a basic correctness check
	for run := 0; run < 100; run++ {
		t.Run(fmt.Sprintf("run=%d", run), func(t *testing.T) {
			t.Parallel()
			seed := time.Now().UnixNano() + 1000
			keys := rloc.GenUniqueBitStrings(seed, 256, 32)

			lerl, err := NewCompactLocalExactRangeLocator(keys)
			if err != nil {
				t.Fatalf("NewCompactLocalExactRangeLocator failed (seed: %d): %v", seed, err)
			}

			for _, key := range keys {
				for prefixLen := uint32(0); prefixLen <= key.Size(); prefixLen++ {
					var prefix bits.BitString
					if prefixLen == 0 {
						prefix = bits.NewFromText("")
					} else {
						prefix = key.Prefix(int(prefixLen))
					}

					start, end, err := lerl.WeakPrefixSearch(prefix)
					if err != nil {
						t.Fatalf("WeakPrefixSearch failed for prefix %s of key %s (seed: %d): %v",
							prefix.PrettyString(), key.PrettyString(), seed, err)
					}

					expectedStart, expectedEnd := rloc.FindRange(keys, prefix)

					if start != expectedStart || end != expectedEnd {
						t.Errorf("Mismatch for prefix %s (seed: %d). Got: [%d, %d), Exp: [%d, %d)",
							prefix.PrettyString(), seed, start, end, expectedStart, expectedEnd)
						t.FailNow()
					}
				}
			}
		})
	}
}
