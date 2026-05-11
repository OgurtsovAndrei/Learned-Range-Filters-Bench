package bench_test

import (
	"Thesis/emptiness/approx/are_soda_hash"
	exactbackend "Thesis/emptiness/exact"
	"Thesis/testutils"
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

// SodaARE PEF vs OneD comparison at n=2^24 across all 6 distributions.
//
// Motivation: SodaARE applies a 2-universal *linear* hash (a*x+b)>>(64-K),
// which preserves arithmetic-progression structure of clustered inputs as
// arithmetic progressions in hash space. PEF's DP partition can therefore
// still find long all-ones runs in the SodaARE's underlying ERE — meaning
// the bpk advantage of PEF on clustered/sosd-* data is expected to carry
// through SodaARE, even after hashing.
//
// This contradicts a naive "Soda hash makes everything uniform" mental
// model. The whole point of the SODA construction is that its hash is
// *not* random — it is multiplicative-additive specifically so locality
// can be exploited downstream.
const (
	sodaPEFN          = 1 << 24
	sodaPEFRangeLen   = uint64(128)
	sodaPEFEpsilon    = 0.01
	sodaPEFQueryCount = 1 << 15
	sodaPEFRounds     = 3
	sodaPEFSeed       = int64(424242)
)

type sodaPEFDataset struct {
	name string
	keys []uint64
}

type sodaPEFMetrics struct {
	buildMS float64
	queryNS float64
	bpk     float64
	K       uint32
}

type sodaPEFRow struct {
	dataset string
	n       int
	oneD    sodaPEFMetrics
	pef     sodaPEFMetrics
}

func loadSodaPEFDatasets(t *testing.T) []sodaPEFDataset {
	t.Helper()
	loaders := []struct {
		name string
		load func() ([]uint64, error)
	}{
		{
			name: "uniform",
			load: func() ([]uint64, error) {
				rng := rand.New(rand.NewSource(42))
				keys := make([]uint64, 0, sodaPEFN)
				seen := make(map[uint64]struct{}, sodaPEFN)
				for len(keys) < sodaPEFN {
					v := rng.Uint64()
					if _, ok := seen[v]; ok {
						continue
					}
					seen[v] = struct{}{}
					keys = append(keys, v)
				}
				sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
				return keys, nil
			},
		},
		{
			name: "clustered",
			load: func() ([]uint64, error) {
				rng := rand.New(rand.NewSource(42))
				keys, _ := testutils.GenerateClusterDistribution(sodaPEFN, 8, 0.10, rng)
				return keys, nil
			},
		},
		{
			name: "sosd_fb",
			load: func() ([]uint64, error) {
				return loadFacebookKeys( sodaPEFN)
			},
		},
		{
			name: "sosd_wiki",
			load: func() ([]uint64, error) {
				return loadWikiKeys( sodaPEFN)
			},
		},
		{
			name: "sosd_osm",
			load: func() ([]uint64, error) {
				return loadOSMKeys( sodaPEFN)
			},
		},
		{
			name: "sosd_books",
			load: func() ([]uint64, error) {
				return loadBooksKeys( sodaPEFN)
			},
		},
	}

	out := make([]sodaPEFDataset, 0, len(loaders))
	for _, ld := range loaders {
		t.Logf("loading %s ...", ld.name)
		keys, err := ld.load()
		if err != nil {
			if os.IsNotExist(err) {
				t.Logf("  skipping (file missing): %v", err)
				continue
			}
			t.Fatalf("load %s: %v", ld.name, err)
		}
		t.Logf("  %s: %d keys", ld.name, len(keys))
		out = append(out, sodaPEFDataset{name: ld.name, keys: keys})
	}
	return out
}

func sodaPEFK(n int, rangeLen uint64, eps float64) uint32 {
	rTarget := float64(n) * float64(rangeLen) / eps
	K := uint32(math.Ceil(math.Log2(rTarget)))
	if K > 64 {
		K = 64
	}
	return K
}

func measureSodaPEF(t *testing.T, ds sodaPEFDataset, variant exactbackend.Variant, K uint32, queries [][2]uint64) sodaPEFMetrics {
	t.Helper()
	start := time.Now()
	f, err := are_soda_hash.NewSodaAREFromKWithBackend(ds.keys, K, sodaPEFSeed, variant)
	if err != nil {
		t.Fatalf("build soda/%s/%s: %v", variant, ds.name, err)
	}
	buildMS := float64(time.Since(start).Microseconds()) / 1000.0

	queryNS := timeAREQueriesU64(queries, sodaPEFRounds, f.IsEmpty)
	bpk := float64(f.SizeInBits()) / float64(len(ds.keys))

	return sodaPEFMetrics{
		buildMS: buildMS,
		queryNS: queryNS,
		bpk:     bpk,
		K:       f.K,
	}
}

func TestSodaPEFvsOneDN24(t *testing.T) {
	if testing.Short() {
		t.Skip("heavy benchmark at n=2^24 across 6 distributions")
	}

	datasets := loadSodaPEFDatasets(t)
	if len(datasets) == 0 {
		t.Fatal("no datasets loaded")
	}

	rows := make([]sodaPEFRow, 0, len(datasets))
	for _, ds := range datasets {
		t.Logf("=== %s (n=%d) ===", ds.name, len(ds.keys))
		queries := generateARERawQueries(ds.keys, sodaPEFQueryCount, sodaPEFRangeLen, sodaPEFSeed)
		K := sodaPEFK(len(ds.keys), sodaPEFRangeLen, sodaPEFEpsilon)

		oneD := measureSodaPEF(t, ds, exactbackend.VariantOneD, K, queries)
		t.Logf("  OneD: build=%.1fms query=%.1fns bpk=%.3f", oneD.buildMS, oneD.queryNS, oneD.bpk)
		pef := measureSodaPEF(t, ds, exactbackend.VariantPEF, K, queries)
		t.Logf("  PEF : build=%.1fms query=%.1fns bpk=%.3f (Δbpk=%+.3f)", pef.buildMS, pef.queryNS, pef.bpk, pef.bpk-oneD.bpk)

		rows = append(rows, sodaPEFRow{
			dataset: ds.name,
			n:       len(ds.keys),
			oneD:    oneD,
			pef:     pef,
		})
	}

	writeSodaPEFReport(t, rows)
}

func writeSodaPEFReport(t *testing.T, rows []sodaPEFRow) {
	t.Helper()
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# SodaARE PEF vs OneD\n\n")
	fmt.Fprintf(&buf, "n=%d, rangeLen=%d, epsilon=%.4f, %d queries × %d rounds, mixed workload\n\n",
		sodaPEFN, sodaPEFRangeLen, sodaPEFEpsilon, sodaPEFQueryCount, sodaPEFRounds)
	fmt.Fprintf(&buf, "Hash linearity (a*x+b) preserves clusters → PEF should win bpk on clustered/sosd-* even *after* the SODA hash.\n\n")
	fmt.Fprintf(&buf, "| Dataset | n | K | OneD build ms | PEF build ms | OneD query ns | PEF query ns | Query PEF/OneD | OneD bpk | PEF bpk | Δ bpk |\n")
	fmt.Fprintf(&buf, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		querySpeedup := r.pef.queryNS / r.oneD.queryNS
		dBpk := r.pef.bpk - r.oneD.bpk
		fmt.Fprintf(&buf, "| %s | %d | %d | %.1f | %.1f | %.1f | %.1f | %.2fx | %.3f | %.3f | %+.3f |\n",
			r.dataset, r.n, r.oneD.K,
			r.oneD.buildMS, r.pef.buildMS,
			r.oneD.queryNS, r.pef.queryNS, querySpeedup,
			r.oneD.bpk, r.pef.bpk, dBpk,
		)
	}

	outDir := filepath.Join("..", "bench_results")
	if _, err := os.Stat(outDir); err != nil {
		outDir = "bench_results"
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", outDir, err)
	}
	outPath := filepath.Join(outDir, "are_soda_pef_n24_report.md")
	if err := os.WriteFile(outPath, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write report: %v", err)
	}
	t.Logf("Report written: %s", outPath)
	t.Logf("\n%s", buf.String())
}
