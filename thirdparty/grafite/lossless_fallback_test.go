package grafite

import (
	"math/rand"
	"testing"
)

// TestLosslessFallback_BuildsAndQueries verifies that when bpk > log2(u/n)+2 —
// the regime where upstream Grafite throws an exception — the wrapper now
// silently falls back to a lossless Elias-Fano on raw keys and answers
// queries with FPR = 0.
func TestLosslessFallback_BuildsAndQueries(t *testing.T) {
	// Tiny universe u = 100, n = 16 keys. Any bpk >= log2(100/16) + 2 ≈ 4.6
	// would fall into the lossless regime; bpk = 20 makes r = 16 * 2^18,
	// way above max(S) < 100.
	keys := []uint64{3, 7, 11, 19, 23, 29, 37, 41, 47, 53, 61, 67, 71, 79, 89, 97}

	f := New(keys, 20.0)
	if f == nil || f.ptr == nil {
		t.Fatalf("expected non-nil filter; got nil — lossless fallback did not engage")
	}

	// Membership queries on stored keys: must report non-empty (IsEmpty=false).
	for _, k := range keys {
		if f.IsEmpty(k, k) {
			t.Errorf("stored key %d reported as empty", k)
		}
	}

	// Range queries entirely inside a gap: must report empty (IsEmpty=true)
	// because in lossless mode FPR must be 0 by construction.
	gaps := [][2]uint64{
		{0, 2}, {4, 6}, {12, 18}, {24, 28}, {30, 36}, {42, 46},
		{48, 52}, {54, 60}, {62, 66}, {72, 78}, {80, 88}, {90, 96}, {98, 99},
	}
	for _, g := range gaps {
		if !f.IsEmpty(g[0], g[1]) {
			t.Errorf("gap [%d,%d] reported as non-empty (FPR > 0 in lossless mode)", g[0], g[1])
		}
	}

	// A range covering at least one key must report non-empty.
	covers := [][2]uint64{
		{0, 5}, {6, 12}, {25, 30}, {0, 99}, {95, 99},
	}
	for _, r := range covers {
		if f.IsEmpty(r[0], r[1]) {
			t.Errorf("range [%d,%d] covering a stored key reported as empty (FN!)", r[0], r[1])
		}
	}
}

// TestLosslessFallback_ZeroFPR_Stress sweeps random non-overlapping ranges in
// the gaps and confirms zero false positives across many queries.
func TestLosslessFallback_ZeroFPR_Stress(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	u := uint64(1 << 26) // 26-bit universe ≈ SOSD-Books scale
	n := 1024
	keysSet := make(map[uint64]struct{}, n)
	keys := make([]uint64, 0, n)
	for len(keys) < n {
		k := rng.Uint64() % u
		if _, ok := keysSet[k]; ok {
			continue
		}
		keysSet[k] = struct{}{}
		keys = append(keys, k)
	}

	// bpk = 25 gives r = n * 2^23 = 2^33, much larger than u = 2^26 → lossless mode.
	f := New(keys, 25.0)
	if f == nil || f.ptr == nil {
		t.Fatal("expected non-nil filter")
	}
	t.Logf("filter built, size = %d bits = %.2f BPK", f.SizeInBits(), float64(f.SizeInBits())/float64(n))

	// Generate random empty queries: pick a range, check it's actually empty,
	// then ensure the filter agrees.
	const numQueries = 20000
	emptyChecked := 0
	for q := 0; q < numQueries; q++ {
		lo := rng.Uint64() % u
		span := rng.Uint64()%64 + 1
		hi := lo + span
		if hi >= u {
			hi = u - 1
		}
		if lo > hi {
			continue
		}
		// Is this range actually empty in the key set?
		actuallyEmpty := true
		for _, k := range keys {
			if k >= lo && k <= hi {
				actuallyEmpty = false
				break
			}
		}
		if !actuallyEmpty {
			continue
		}
		emptyChecked++
		if !f.IsEmpty(lo, hi) {
			t.Errorf("FALSE POSITIVE in lossless mode: range [%d,%d] reported as non-empty but actually empty", lo, hi)
			break
		}
	}
	t.Logf("checked %d genuinely-empty queries; zero false positives", emptyChecked)
}
