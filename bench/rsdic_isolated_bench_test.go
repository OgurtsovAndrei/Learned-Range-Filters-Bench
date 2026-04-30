package bench_test

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/succinct_bit_vector/rsdic"
)

// rsdicCachePath returns the on-disk dump path for an inner-SODA rsdic.
func rsdicCachePath(distName string, n int, L uint64) string {
	return filepath.Join("bench_results", "cache",
		fmt.Sprintf("rsdic_soda_%s_n%d_L%d.bin", distName, n, L))
}

// dumpSodaRSDic builds SODA on the supplied keys at (L, eps), extracts the
// inner ERE's rsdic, and persists it to disk. Idempotent.
func dumpSodaRSDic(tb testing.TB, distName string, keys []uint64, L uint64, eps float64) string {
	tb.Helper()
	path := rsdicCachePath(distName, len(keys), L)
	if _, err := os.Stat(path); err == nil {
		return path
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		tb.Fatalf("mkdir cache: %v", err)
	}
	keysCopy := append([]uint64(nil), keys...)
	soda, err := are_soda_hash.NewSodaARE(keysCopy, L, eps)
	if err != nil {
		tb.Fatalf("NewSodaARE(%s, n=%d, L=%d): %v", distName, len(keys), L, err)
	}
	rsd := soda.ERE().D
	blob, err := rsd.MarshalBinary()
	if err != nil {
		tb.Fatalf("MarshalBinary: %v", err)
	}
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		tb.Fatalf("write %s: %v", path, err)
	}
	tb.Logf("dumped rsdic %s (%.2f MB, %d ones / %d bits) -> %s",
		distName, float64(len(blob))/(1024*1024), rsd.OneNum(), rsd.Num(), path)
	return path
}

// loadRSDic reads an rsdic dump from disk.
func loadRSDic(tb testing.TB, path string) *rsdic.RSDic {
	tb.Helper()
	blob, err := os.ReadFile(path)
	if err != nil {
		tb.Fatalf("read %s: %v", path, err)
	}
	rsd := rsdic.New()
	if err := rsd.UnmarshalBinary(blob); err != nil {
		tb.Fatalf("UnmarshalBinary: %v", err)
	}
	return rsd
}

// rsdicIsolatedSpec defines one (distribution, L) cell for the bench.
type rsdicIsolatedSpec struct {
	distName string
	L        uint64
}

// loadKeysForSpec reads/generates the key set for the given distribution.
// For SOSD distributions it streams the canonical file; synthetic
// distributions reuse the same generator the headline B6 bench uses.
func loadKeysForSpec(tb testing.TB, distName string, n int) []uint64 {
	tb.Helper()
	switch distName {
	case "sosd_fb":
		ks, err := loadSOSDUint64(sosdPath("fb_200M_uint64"), n)
		if err != nil {
			tb.Skipf("sosd_fb unavailable: %v", err)
		}
		return ks
	case "uniform":
		rng := rand.New(rand.NewSource(0xBEEF))
		ks := make([]uint64, n)
		for i := range ks {
			ks[i] = rng.Uint64()
		}
		return ks
	default:
		tb.Skipf("loadKeysForSpec: unsupported %q", distName)
		return nil
	}
}

// TestDumpSodaRSDic builds SODA on each (dist, L) and persists the inner
// rsdic. Idempotent — reuse existing dumps in bench_results/cache/. Run
// once before BenchmarkSodaRSDicSelect1Isolated.
//
//	go test -run TestDumpSodaRSDic -timeout 30m ./bench/
func TestDumpSodaRSDic(t *testing.T) {
	const (
		n   = 1 << 24
		eps = 0.01
	)
	specs := []rsdicIsolatedSpec{
		{"sosd_fb", 1},
		{"sosd_fb", 1024},
		{"sosd_fb", 65536},
		{"uniform", 65536},
	}
	for _, s := range specs {
		s := s
		t.Run(fmt.Sprintf("%s/L=%d", s.distName, s.L), func(t *testing.T) {
			keys := loadKeysForSpec(t, s.distName, n)
			dumpSodaRSDic(t, s.distName, keys, s.L, eps)
		})
	}
}

// BenchmarkSodaRSDicSelect1Isolated measures Select1 on the inner rsdic of
// a SODA filter, in isolation (no surrounding ERE/SODA work). Loads dumps
// produced by TestDumpSodaRSDic. Compare against the synthetic 50%-density
// scaling numbers in Thesis/succinct_bit_vector/rsdic/scaling_test.go.
func BenchmarkSodaRSDicSelect1Isolated(b *testing.B) {
	const n = 1 << 24
	specs := []rsdicIsolatedSpec{
		{"sosd_fb", 1},
		{"sosd_fb", 1024},
		{"sosd_fb", 65536},
		{"uniform", 65536},
	}
	for _, s := range specs {
		s := s
		b.Run(fmt.Sprintf("%s/L=%d", s.distName, s.L), func(b *testing.B) {
			path := rsdicCachePath(s.distName, n, s.L)
			if _, err := os.Stat(path); err != nil {
				b.Skipf("missing dump %s — run TestDumpSodaRSDic first", path)
			}
			rsd := loadRSDic(b, path)
			ones := rsd.OneNum()
			if ones == 0 {
				b.Fatalf("rsdic has zero ones")
			}
			rng := rand.New(rand.NewSource(7))
			const kIters = 200_000
			ranks := make([]uint64, kIters)
			for i := range ranks {
				ranks[i] = uint64(rng.Int63n(int64(ones)))
			}
			b.ReportMetric(float64(rsd.AllocSize())/(1024*1024), "rsdic_MB")
			b.ReportMetric(float64(ones)/float64(rsd.Num()), "density")
			b.ResetTimer()
			var sink uint64
			for i := 0; i < b.N; i++ {
				sink ^= rsd.Select1(ranks[i%kIters])
			}
			b.StopTimer()
			if sink == 0xDEADBEEF {
				b.Log("sink trick")
			}
		})
	}
}

// BenchmarkSodaRSDicInterleavedSelect mirrors getBlockRange: Select1(r)
// followed by Select1(r+1). One b.N iteration = one such pair.
func BenchmarkSodaRSDicInterleavedSelect(b *testing.B) {
	const n = 1 << 24
	specs := []rsdicIsolatedSpec{
		{"sosd_fb", 1},
		{"sosd_fb", 1024},
		{"sosd_fb", 65536},
		{"uniform", 65536},
	}
	for _, s := range specs {
		s := s
		b.Run(fmt.Sprintf("%s/L=%d", s.distName, s.L), func(b *testing.B) {
			path := rsdicCachePath(s.distName, n, s.L)
			if _, err := os.Stat(path); err != nil {
				b.Skipf("missing dump %s — run TestDumpSodaRSDic first", path)
			}
			rsd := loadRSDic(b, path)
			ones := rsd.OneNum()
			if ones < 2 {
				b.Fatalf("rsdic has <2 ones")
			}
			rng := rand.New(rand.NewSource(7))
			const kIters = 200_000
			ranks := make([]uint64, kIters)
			for i := range ranks {
				ranks[i] = uint64(rng.Int63n(int64(ones - 1)))
			}
			b.ReportMetric(float64(rsd.AllocSize())/(1024*1024), "rsdic_MB")
			b.ResetTimer()
			var sink uint64
			for i := 0; i < b.N; i++ {
				r := ranks[i%kIters]
				sink ^= rsd.Select1(r)
				sink ^= rsd.Select1(r + 1)
			}
			b.StopTimer()
			if sink == 0xDEADBEEF {
				b.Log("sink trick")
			}
		})
	}
}
