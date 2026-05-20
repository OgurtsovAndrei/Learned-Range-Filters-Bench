//go:build linux

package bench_test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"Thesis-bench-industry/bench/internal/perf"
)

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

	n := float64(ereCacheQueryCount)
	l1Loads := float64(res.Values[0]) / n
	l1Misses := float64(res.Values[1]) / n
	llcLoads := float64(res.Values[2]) / n
	llcMisses := float64(res.Values[3]) / n
	instrs := float64(res.Values[4]) / n

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

	datasets := mustLoadEREDatasets(t)

	var rows []ereCacheRow

	for _, ds := range datasets {
		ereFilter, err := buildEREFilter(ds.keys)
		if err != nil {
			t.Fatalf("%s: build ere: %v", ds.name, err)
		}
		oneDFilter, err := buildEREOneDFilter(ds.keys)
		if err != nil {
			t.Fatalf("%s: build ere_one_d: %v", ds.name, err)
		}

		r1 := measureERECacheEvents(t, ereFilter, ds.queries)
		r1.dataset = ds.name
		r1.filter = "ere"

		r2 := measureERECacheEvents(t, oneDFilter, ds.queries)
		r2.dataset = ds.name
		r2.filter = "ere_one_d"

		rows = append(rows, r1, r2)
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# ERE vs ERE One-D — Cache Events per Query\n\n")
	fmt.Fprintf(&buf, "n=%d, queries=%d (warmup=%d)\n\n",
		ereCompareN, ereCacheQueryCount, ereCacheWarmupCount)
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

	reportPath := filepath.Join("..", "bench_results", "ere_cache_report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0755); err == nil {
		if err := os.WriteFile(reportPath, buf.Bytes(), 0644); err != nil {
			t.Logf("warning: could not write report to %s: %v", reportPath, err)
		}
	}
}
