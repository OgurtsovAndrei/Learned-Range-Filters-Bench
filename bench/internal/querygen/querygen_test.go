package querygen

import (
	"math/rand"
	"sort"
	"testing"
)

// containsKey reports whether [a, b] contains any element of sorted keys.
func containsKey(keys []uint64, a, b uint64) bool {
	idx := sort.Search(len(keys), func(i int) bool { return keys[i] >= a })
	return idx < len(keys) && keys[idx] <= b
}

// sparseSortedKeys returns n sorted uint64 keys spread evenly across [0, span)
// with a deterministic small jitter so adjacent keys leave non-trivial gaps.
func sparseSortedKeys(n int, span uint64, seed int64) []uint64 {
	rng := rand.New(rand.NewSource(seed))
	keys := make([]uint64, n)
	step := span / uint64(n)
	for i := 0; i < n; i++ {
		jitter := uint64(rng.Int63n(int64(step) / 2))
		keys[i] = uint64(i)*step + jitter
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

// TestGenerateMixedQueries_EmitsNonEmpty checks that the mixed generator
// produces a non-trivial fraction of queries that contain at least one
// stored key when near-key weight is non-zero. The smart-mix generator,
// in contrast, guarantees zero non-empty queries.
func TestGenerateMixedQueries_EmitsNonEmpty(t *testing.T) {
	keys := sparseSortedKeys(4096, 1<<32, 42)
	rng := rand.New(rand.NewSource(1))
	const L uint64 = 16
	const count = 4096

	w := SmartMixWeights{NearKey: 1.0, InGap: 0.0, Uniform: 0.0}
	qs := GenerateMixedQueriesWeighted(keys, count, L, w, rng)
	if len(qs) < count/2 {
		t.Fatalf("expected close to %d queries, got %d", count, len(qs))
	}

	nonEmpty := 0
	for _, q := range qs {
		if containsKey(keys, q[0], q[1]) {
			nonEmpty++
		}
	}
	// With offset range [-5L, 5L] and L=16, ≈10% of near-key queries
	// land on the reference key. Require at least 5% to keep the test
	// robust against RNG variance.
	frac := float64(nonEmpty) / float64(len(qs))
	if frac < 0.05 {
		t.Fatalf("near-key-only mix produced only %.2f%% non-empty queries; expected ≥5%%", frac*100)
	}
}

// TestGenerateMixedQueries_GapHeavyMostlyEmpty checks the gap-heavy weighting
// (the variant used in evaluation_tables): in-gap queries stay guaranteed
// empty, uniform queries are mostly empty on sparse keys, so the overall
// non-empty fraction should be small but typically positive at L=1024.
func TestGenerateMixedQueries_GapHeavyMostlyEmpty(t *testing.T) {
	keys := sparseSortedKeys(4096, 1<<32, 7)
	rng := rand.New(rand.NewSource(2))
	const L uint64 = 1024
	const count = 4096

	w := SmartMixWeights{NearKey: 0.0, InGap: 0.7, Uniform: 0.3}
	qs := GenerateMixedQueriesWeighted(keys, count, L, w, rng)
	if len(qs) == 0 {
		t.Fatalf("expected queries, got 0")
	}
	nonEmpty := 0
	for _, q := range qs {
		if containsKey(keys, q[0], q[1]) {
			nonEmpty++
		}
	}
	frac := float64(nonEmpty) / float64(len(qs))
	if frac > 0.6 {
		t.Fatalf("gap-heavy mix produced %.2f%% non-empty queries; expected <60%%", frac*100)
	}
}

// TestGenerateMixedQueries_NoTruncation makes the contrast with the smart
// generator explicit: re-running the smart generator with the same input
// should yield zero non-empty queries while the mixed one should not.
func TestGenerateMixedQueries_NoTruncation(t *testing.T) {
	keys := sparseSortedKeys(4096, 1<<32, 13)
	const L uint64 = 16
	const count = 4096

	w := SmartMixWeights{NearKey: 1.0, InGap: 0.0, Uniform: 0.0}

	smartRng := rand.New(rand.NewSource(99))
	smart := GenerateSmartQueriesWeighted(keys, count, L, w, smartRng)
	for _, q := range smart {
		if containsKey(keys, q[0], q[1]) {
			t.Fatalf("smart-mix produced non-empty query [%d, %d]", q[0], q[1])
		}
	}

	mixedRng := rand.New(rand.NewSource(99))
	mixed := GenerateMixedQueriesWeighted(keys, count, L, w, mixedRng)
	mixedNonEmpty := 0
	for _, q := range mixed {
		if containsKey(keys, q[0], q[1]) {
			mixedNonEmpty++
		}
	}
	if mixedNonEmpty == 0 {
		t.Fatalf("expected mixed mix to contain ≥1 non-empty query; got 0")
	}
}

// TestGenerateMixedQueries_Deterministic verifies that the generator is
// deterministic given the same seed and inputs.
func TestGenerateMixedQueries_Deterministic(t *testing.T) {
	keys := sparseSortedKeys(2048, 1<<30, 5)
	const L uint64 = 128
	const count = 2048

	w := SmartMixWeights{NearKey: 0.5, InGap: 0.3, Uniform: 0.2}

	a := GenerateMixedQueriesWeighted(keys, count, L, w, rand.New(rand.NewSource(2026)))
	b := GenerateMixedQueriesWeighted(keys, count, L, w, rand.New(rand.NewSource(2026)))
	if len(a) != len(b) {
		t.Fatalf("non-deterministic length: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic query at i=%d: %v vs %v", i, a[i], b[i])
		}
	}
}
