//go:build linux && heavy

package bench_test

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"Thesis-bench-industry/bench/internal/perf"
	"Thesis/testutils"
)

var ereCacheNValues = []int{1 << 20, 1 << 24, 1 << 28}

const (
	ereCacheQueryCount  = 100_000
	ereCacheWarmupCount = 10_000
)

var allPerfEvents = []perf.EventSpec{
	perf.L1DLoads,
	perf.L1DLoadMisses,
	perf.LLCLoads,
	perf.LLCLoadMisses,
	perf.Instructions,
}

type ereCacheRow struct {
	n           int
	dataset     string
	filter      string
	l1Loads     float64
	l1Misses    float64
	l1MissRate  float64
	llcLoads    float64
	llcMisses   float64
	llcMissRate float64
	instrs      float64
}

type ereCacheDataset struct {
	name    string
	keys    []uint64
	queries []ereQuery
}

// generateUniformKeysForCache generates n sorted unique uint64 keys via sort+dedup.
// Avoids map-based dedup to keep memory overhead low at large n.
func generateUniformKeysForCache(n int, seed int64) []uint64 {
	rng := rand.New(rand.NewSource(seed))
	raw := make([]uint64, n+1024)
	for i := range raw {
		raw[i] = rng.Uint64()
	}
	sort.Slice(raw, func(i, j int) bool { return raw[i] < raw[j] })
	out := raw[:1]
	for _, k := range raw[1:] {
		if k != out[len(out)-1] {
			out = append(out, k)
		}
	}
	if len(out) > n {
		out = out[:n]
	}
	return out
}

func loadEreCacheDatasets(t *testing.T, n int) []ereCacheDataset {
	t.Helper()

	var datasets []ereCacheDataset

	uniformKeys := generateUniformKeysForCache(n, 42)
	if len(uniformKeys) >= n {
		datasets = append(datasets, ereCacheDataset{
			name:    "uniform",
			keys:    uniformKeys,
			queries: generateEREMixedQueries(uniformKeys, ereCacheQueryCount, ereCompareRangeLen, 12345),
		})
	}

	rng := rand.New(rand.NewSource(42))
	clusteredKeys, _ := testutils.GenerateClusterDistribution(n, 8, 0.10, rng)
	if len(clusteredKeys) >= n {
		datasets = append(datasets, ereCacheDataset{
			name:    "clustered",
			keys:    clusteredKeys,
			queries: generateEREMixedQueries(clusteredKeys, ereCacheQueryCount, ereCompareRangeLen, 12345),
		})
	}

	type sosdLoader struct {
		name     string
		capacity int
		load     func(int) ([]uint64, error)
	}
	for _, l := range []sosdLoader{
		{"sosd_fb", 200_000_000, loadFacebookKeys},
		{"sosd_wiki", 200_000_000, loadWikiKeys},
		{"sosd_osm", 800_000_000, loadOSMKeys},
		{"sosd_books", 200_000_000, loadBooksKeys},
	} {
		if n > l.capacity {
			continue
		}
		keys, err := l.load(n)
		if err != nil || len(keys) == 0 {
			continue
		}
		datasets = append(datasets, ereCacheDataset{
			name:    l.name,
			keys:    keys,
			queries: generateEREMixedQueries(keys, ereCacheQueryCount, ereCompareRangeLen, 12345),
		})
	}
	return datasets
}

func measureERECacheEvents(t *testing.T, filter ereExactFilter, queries []ereQuery) ereCacheRow {
	t.Helper()

	for i := 0; i < ereCacheWarmupCount; i++ {
		q := queries[i%len(queries)]
		ereQuerySink = filter.IsEmpty(q.a, q.b)
	}

	g, err := perf.OpenGroup(allPerfEvents)
	if err != nil {
		t.Fatal("open perf group:", err)
	}
	defer g.Close()

	if err := g.Reset(); err != nil {
		t.Fatal("reset:", err)
	}
	if err := g.Enable(); err != nil {
		t.Fatal("enable:", err)
	}
	for i := 0; i < ereCacheQueryCount; i++ {
		q := queries[i%len(queries)]
		ereQuerySink = filter.IsEmpty(q.a, q.b)
	}
	if err := g.Disable(); err != nil {
		t.Fatal("disable:", err)
	}

	res, err := g.Read()
	if err != nil {
		t.Fatal("read:", err)
	}

	nq := float64(ereCacheQueryCount)
	l1Loads := float64(res.Values[0]) / nq
	l1Misses := float64(res.Values[1]) / nq
	llcLoads := float64(res.Values[2]) / nq
	llcMisses := float64(res.Values[3]) / nq
	instrs := float64(res.Values[4]) / nq

	l1Rate := 0.0
	if l1Loads > 0 {
		l1Rate = l1Misses / l1Loads * 100
	}
	llcRate := 0.0
	if llcLoads > 0 {
		llcRate = llcMisses / llcLoads * 100
	}

	return ereCacheRow{
		l1Loads: l1Loads, l1Misses: l1Misses, l1MissRate: l1Rate,
		llcLoads: llcLoads, llcMisses: llcMisses, llcMissRate: llcRate,
		instrs: instrs,
	}
}

func TestERECacheHitMiss(t *testing.T) {
	probe, err := perf.OpenGroup([]perf.EventSpec{perf.Instructions})
	if err != nil {
		t.Skip("perf_event_open unavailable:", err)
	}
	probe.Close()

	var rows []ereCacheRow

	for _, n := range ereCacheNValues {
		t.Logf("=== N=2^%d (%d) ===", ilog2(uint64(n)), n)
		datasets := loadEreCacheDatasets(t, n)
		if len(datasets) == 0 {
			t.Logf("N=2^%d: no datasets, skip", ilog2(uint64(n)))
			continue
		}

		for _, ds := range datasets {
			ereFilter, err := buildEREFilter(ds.keys)
			if err != nil {
				t.Fatalf("N=2^%d %s: build ere: %v", ilog2(uint64(n)), ds.name, err)
			}
			oneDFilter, err := buildEREOneDFilter(ds.keys)
			if err != nil {
				t.Fatalf("N=2^%d %s: build ere_one_d: %v", ilog2(uint64(n)), ds.name, err)
			}

			r1 := measureERECacheEvents(t, ereFilter, ds.queries)
			r1.n = n
			r1.dataset = ds.name
			r1.filter = "ere"

			r2 := measureERECacheEvents(t, oneDFilter, ds.queries)
			r2.n = n
			r2.dataset = ds.name
			r2.filter = "ere_one_d"

			rows = append(rows, r1, r2)
			t.Logf("  %s/ere       LLC-miss/q=%.2f  instrs/q=%.0f", ds.name, r1.llcMisses, r1.instrs)
			t.Logf("  %s/ere_one_d LLC-miss/q=%.2f  instrs/q=%.0f", ds.name, r2.llcMisses, r2.instrs)
		}
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# ERE vs ERE One-D — Cache Events per Query (N sweep)\n\n")
	fmt.Fprintf(&buf, "queries=%d (warmup=%d), rangeLen=%d\n\n",
		ereCacheQueryCount, ereCacheWarmupCount, ereCompareRangeLen)
	fmt.Fprintf(&buf, "Machine: Linux 6.17, x86-64. User-space events only (exclude_kernel).\n\n")
	fmt.Fprintf(&buf, "| N | Dataset | Filter | L1-loads/q | L1-misses/q | L1-miss%% | LLC-loads/q | LLC-misses/q | LLC-miss%% | Instrs/q |\n")
	fmt.Fprintf(&buf, "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&buf, "| 2^%d | %s | %s | %.1f | %.2f | %.1f%% | %.2f | %.2f | %.1f%% | %.1f |\n",
			ilog2(uint64(r.n)), r.dataset, r.filter,
			r.l1Loads, r.l1Misses, r.l1MissRate,
			r.llcLoads, r.llcMisses, r.llcMissRate,
			r.instrs)
	}
	fmt.Print("\n" + buf.String())

	reportPath := filepath.Join("..", "bench_results", "ere_cache_report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0755); err == nil {
		if err := os.WriteFile(reportPath, buf.Bytes(), 0644); err != nil {
			t.Logf("warning: could not write report: %v", err)
		}
	}
}

