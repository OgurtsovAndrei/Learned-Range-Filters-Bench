package bench_test

import (
	"math/rand"
	"testing"

	"Thesis/emptiness/approx/are_bloom"
	"Thesis/emptiness/approx/are_greedy_scan"
	"Thesis/emptiness/approx/are_hybrid_scan"
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/emptiness/approx/are_trunc"
)

// BenchmarkAllAREDirect runs a direct microbench (no harness wrapping)
// of every ARE filter's IsEmpty over the same smart-mix-empty query
// stream on the B6 headline cell: n=2^24, sosd_fb, L=65536. Used to
// regression-check Select1Fast adoption — all filters that use ERE
// under the hood (SODA, Truncation, Scan, Greedy) should benefit;
// BloomARE doesn't use ERE so should be unchanged.
func BenchmarkAllAREDirect(b *testing.B) {
	const (
		n   = 1 << 24
		L   = uint64(65536)
		eps = 0.01
	)
	keys := loadKeysForSpec(b, "sosd_fb", n)
	keyBits := uint32(33)

	rng := rand.New(rand.NewSource(42))
	const kIters = 200_000
	queries := generateSmartQueriesAudit(rng, keys, L, kIters)

	type filter struct {
		name    string
		isEmpty func(a, b uint64) bool
	}
	filters := []filter{}

	{
		ks := append([]uint64(nil), keys...)
		f, err := are_soda_hash.NewSodaARE(ks, L, eps)
		if err != nil {
			b.Fatalf("SODA: %v", err)
		}
		filters = append(filters, filter{"SODA", f.IsEmpty})
	}
	{
		ks := append([]uint64(nil), keys...)
		f, err := are_trunc.NewTruncARE(ks, keyBits, are_trunc.Config{K: 18})
		if err != nil {
			b.Fatalf("Truncation: %v", err)
		}
		filters = append(filters, filter{"Truncation_K18", f.IsEmpty})
	}
	{
		ks := append([]uint64(nil), keys...)
		f, err := are_hybrid_scan.NewHybridScanARE(ks, keyBits, are_hybrid_scan.Config{K: 18})
		if err != nil {
			b.Fatalf("Scan-ARE: %v", err)
		}
		filters = append(filters, filter{"Scan-ARE_K18", f.IsEmpty})
	}
	{
		ks := append([]uint64(nil), keys...)
		f, err := are_greedy_scan.NewGreedyScanARE(ks, keyBits, are_greedy_scan.Config{K: 18})
		if err != nil {
			b.Fatalf("Greedy: %v", err)
		}
		filters = append(filters, filter{"Greedy_K18", f.IsEmpty})
	}
	{
		ks := append([]uint64(nil), keys...)
		f, err := are_bloom.NewBloomARE(ks, L, eps)
		if err != nil {
			b.Fatalf("BloomARE: %v", err)
		}
		filters = append(filters, filter{"BloomARE", f.IsEmpty})
	}

	for _, fl := range filters {
		fl := fl
		b.Run(fl.name, func(b *testing.B) {
			var fp int
			for i := 0; i < b.N; i++ {
				q := queries[i%kIters]
				if !fl.isEmpty(q.lo, q.hi) {
					fp++
				}
			}
			b.ReportMetric(float64(fp)/float64(b.N), "fp_rate")
		})
	}
}
