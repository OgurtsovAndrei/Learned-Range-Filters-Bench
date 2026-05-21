//go:build linux && heavy

package bench_test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"Thesis/emptiness/approx/are_soda_hash"
	exactbackend "Thesis/emptiness/exact"
	"Thesis-bench-industry/bench/internal/perf"
)

const (
	sodaCacheQueryCount  = 100_000
	sodaCacheWarmupCount = 10_000
)

type sodaCacheRow struct {
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

func measureSodaCacheEvents(t *testing.T, queries [][2]uint64, fn func(a, b uint64) bool) sodaCacheRow {
	t.Helper()

	for i := 0; i < sodaCacheWarmupCount; i++ {
		q := queries[i%len(queries)]
		ereQuerySink = fn(q[0], q[1])
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
	for i := 0; i < sodaCacheQueryCount; i++ {
		q := queries[i%len(queries)]
		ereQuerySink = fn(q[0], q[1])
	}
	if err := g.Disable(); err != nil {
		t.Fatal("disable:", err)
	}

	res, err := g.Read()
	if err != nil {
		t.Fatal("read:", err)
	}

	nq := float64(sodaCacheQueryCount)
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

	return sodaCacheRow{
		l1Loads: l1Loads, l1Misses: l1Misses, l1MissRate: l1Rate,
		llcLoads: llcLoads, llcMisses: llcMisses, llcMissRate: llcRate,
		instrs: instrs,
	}
}

func TestSodaBackendCacheHitMiss(t *testing.T) {
	probe, err := perf.OpenGroup([]perf.EventSpec{perf.Instructions})
	if err != nil {
		t.Skip("perf_event_open unavailable:", err)
	}
	probe.Close()

	datasets := loadAREDatasets(t)

	var rows []sodaCacheRow

	for _, ds := range datasets {
		K := sodaK(len(ds.keysU64), areCompareRangeLen, areCompareEpsilon)

		classic, err := are_soda_hash.NewSodaAREFromKWithBackend(ds.keysU64, K, int64(areCompareRangeLen), exactbackend.VariantClassic)
		if err != nil {
			t.Fatalf("%s: build classic: %v", ds.name, err)
		}
		oneD, err := are_soda_hash.NewSodaAREFromKWithBackend(ds.keysU64, K, int64(areCompareRangeLen), exactbackend.VariantOneD)
		if err != nil {
			t.Fatalf("%s: build one_d: %v", ds.name, err)
		}

		r1 := measureSodaCacheEvents(t, ds.rawQueries, classic.IsEmpty)
		r1.dataset = ds.name
		r1.filter = "soda/classic"

		r2 := measureSodaCacheEvents(t, ds.rawQueries, oneD.IsEmpty)
		r2.dataset = ds.name
		r2.filter = "soda/one_d"

		rows = append(rows, r1, r2)
		t.Logf("  %s/classic LLC-miss/q=%.2f  instrs/q=%.0f", ds.name, r1.llcMisses, r1.instrs)
		t.Logf("  %s/one_d   LLC-miss/q=%.2f  instrs/q=%.0f", ds.name, r2.llcMisses, r2.instrs)
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# SODA ARE: Classic vs One-D ERE Backend — Cache Events per Query\n\n")
	fmt.Fprintf(&buf, "n=%d, rangeLen=%d, epsilon=%.4f\n", areCompareN, areCompareRangeLen, areCompareEpsilon)
	fmt.Fprintf(&buf, "queries=%d (warmup=%d)\n\n", sodaCacheQueryCount, sodaCacheWarmupCount)
	fmt.Fprintf(&buf, "Machine: Linux 6.17, x86-64. User-space events only (exclude_kernel).\n\n")
	fmt.Fprintf(&buf, "| Dataset | Filter | L1-loads/q | L1-misses/q | L1-miss%% | LLC-loads/q | LLC-misses/q | LLC-miss%% | Instrs/q |\n")
	fmt.Fprintf(&buf, "|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&buf, "| %s | %s | %.1f | %.2f | %.1f%% | %.2f | %.2f | %.1f%% | %.1f |\n",
			r.dataset, r.filter,
			r.l1Loads, r.l1Misses, r.l1MissRate,
			r.llcLoads, r.llcMisses, r.llcMissRate,
			r.instrs)
	}
	fmt.Print("\n" + buf.String())

	reportPath := filepath.Join("..", "bench_results", "soda_cache_report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0755); err == nil {
		_ = os.WriteFile(reportPath, buf.Bytes(), 0644)
	}
}
