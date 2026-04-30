package bench_test

import (
	"math/rand"
	"testing"

	"Thesis/emptiness/approx/are_soda_hash"
)

// BenchmarkSodaIsEmptyVsFast compares the production SODA.IsEmpty (which
// goes through ere.IsEmpty → rsdic.Select1) against a "Fast" variant
// that goes through ere.IsEmptyFast → rsdic.Select1Fast (bracketed
// binary search over rankBlocks). End-to-end query latency at the B6
// headline cell: n=2^24, sosd_fb, L=65536, eps=0.01.
//
// Predicted: ~3× speedup (2 × 67 ns Select + 600 ns binsearch + 720 ns
// wrap = ~1450 ns, vs measured ~4500 ns for the original).
func BenchmarkSodaIsEmptyVsFast(b *testing.B) {
	const (
		n   = 1 << 24
		L   = uint64(65536)
		eps = 0.01
	)
	keys := loadKeysForSpec(b, "sosd_fb", n)
	soda, err := are_soda_hash.NewSodaARE(append([]uint64(nil), keys...), L, eps)
	if err != nil {
		b.Fatalf("NewSodaARE: %v", err)
	}
	ere := soda.ERE()

	rng := rand.New(rand.NewSource(42))
	const kIters = 200_000
	queries := generateSmartQueriesAudit(rng, keys, L, kIters)

	// SODA.IsEmpty on FB at L=65536 routes 100% of queries to the
	// blockA == blockB path, which calls ere.IsEmpty(hA, hB) once with
	// the hashed endpoints. Since SODA degenerates here, hA = a, hB = b.
	// So calling ere.IsEmpty(q.lo, q.hi) directly is equivalent to
	// SODA.IsEmpty(q.lo, q.hi) in this regime.

	b.Run("Original_IsEmpty", func(b *testing.B) {
		var fp int
		for i := 0; i < b.N; i++ {
			q := queries[i%kIters]
			if !ere.IsEmpty(q.lo, q.hi) {
				fp++
			}
		}
		b.ReportMetric(float64(fp)/float64(b.N), "fp_rate")
	})

	b.Run("IsEmptyFast", func(b *testing.B) {
		var fp int
		for i := 0; i < b.N; i++ {
			q := queries[i%kIters]
			if !ere.IsEmptyFast(q.lo, q.hi) {
				fp++
			}
		}
		b.ReportMetric(float64(fp)/float64(b.N), "fp_rate")
	})

	// Sanity: same answer. Verifies IsEmptyFast is functionally identical.
	b.Run("Equivalence", func(b *testing.B) {
		mismatches := 0
		for i, q := range queries {
			if i >= 1000 {
				break
			}
			if ere.IsEmpty(q.lo, q.hi) != ere.IsEmptyFast(q.lo, q.hi) {
				mismatches++
			}
		}
		if mismatches > 0 {
			b.Fatalf("IsEmpty / IsEmptyFast disagree on %d/1000 queries", mismatches)
		}
	})
}
