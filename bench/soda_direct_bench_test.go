package bench_test

import (
	"math/rand"
	"testing"

	"Thesis/emptiness/approx/are_soda_hash"
)

// BenchmarkSodaIsEmptyDirect measures SODA.IsEmpty without any closure /
// method-value / interface indirection. Direct method call on a typed
// concrete pointer in a tight loop.
//
// This isolates the question: is the headline ~5000 ns/query at L=65536
// real algorithmic cost, or benchmark-harness overhead?
func BenchmarkSodaIsEmptyDirect(b *testing.B) {
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

	rng := rand.New(rand.NewSource(42))
	const kIters = 200_000
	queries := generateSmartQueriesAudit(rng, keys, L, kIters)

	b.ReportMetric(float64(soda.SizeInBits())/float64(n), "bpk")
	b.ResetTimer()
	var fp int
	for i := 0; i < b.N; i++ {
		q := queries[i%kIters]
		if !soda.IsEmpty(q.lo, q.hi) {
			fp++
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(fp)/float64(b.N), "fp_rate")
}

// BenchmarkSodaIsEmptyClosure measures the same path through a method
// value (f.IsEmpty stored as `func(a,b uint64) bool`). This mirrors what
// b6_latency_test does. If this matches Direct within a few ns, the
// benchmark harness is not the source of the elevated latency.
func BenchmarkSodaIsEmptyClosure(b *testing.B) {
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

	rng := rand.New(rand.NewSource(42))
	const kIters = 200_000
	queries := generateSmartQueriesAudit(rng, keys, L, kIters)

	isEmpty := soda.IsEmpty // method value
	b.ReportMetric(float64(soda.SizeInBits())/float64(n), "bpk")
	b.ResetTimer()
	var fp int
	for i := 0; i < b.N; i++ {
		q := queries[i%kIters]
		if !isEmpty(q.lo, q.hi) {
			fp++
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(fp)/float64(b.N), "fp_rate")
}
