package bench_test

import (
	"Thesis/emptiness/exact/ere_one_d"
	"Thesis/emptiness/exact/ere_pef"
	"Thesis/testutils"
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"
)

const (
	pefScalingRangeLen   = uint64(4096)
	pefScalingQueryCount = 1 << 14 // 16K — keep query bench short at huge n
	pefScalingKeyBits    = 60
)

type pefScalingSpec struct {
	name string
	// load returns up to n keys, masked to 60 bits, sorted+deduped.
	// Implementations should NOT cache (memory pressure at 2^28).
	load func(n int) ([]uint64, error)
}

type pefScalingRow struct {
	dataset     string
	n           int
	pefChunks   int
	pefBuildMS  float64
	oneDBuildMS float64
	pefThruMK   float64
	oneDThruMK  float64
	pefQueryNS  float64
	oneDQueryNS float64
	pefBPK      float64
	oneDBPK     float64
	bpkDelta    float64
	queryRatio  float64
	peakRSSGB   float64
}

func mask60AndDedupe(keys []uint64) []uint64 {
	mask := (uint64(1) << pefScalingKeyBits) - 1
	for i, k := range keys {
		keys[i] = k & mask
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	j := 0
	for i, k := range keys {
		if i == 0 || k != keys[i-1] {
			keys[j] = k
			j++
		}
	}
	return keys[:j]
}

func makePEFScalingSpecs() []pefScalingSpec {
	return []pefScalingSpec{
		{
			name: "uniform",
			load: func(n int) ([]uint64, error) {
				rng := rand.New(rand.NewSource(42))
				mask := (uint64(1) << pefScalingKeyBits) - 1
				keys := make([]uint64, n)
				for i := range keys {
					keys[i] = rng.Uint64() & mask
				}
				return mask60AndDedupe(keys), nil
			},
		},
		{
			name: "clustered",
			load: func(n int) ([]uint64, error) {
				rng := rand.New(rand.NewSource(42))
				keys, _ := testutils.GenerateClusterDistribution(n, 8, 0.10, rng)
				return mask60AndDedupe(keys), nil
			},
		},
		{
			name: "sosd_fb",
			load: func(n int) ([]uint64, error) {
				return loadAndMaskSOSD64(sosdPath("fb_200M_uint64"), n)
			},
		},
		{
			name: "sosd_wiki",
			load: func(n int) ([]uint64, error) {
				return loadAndMaskSOSD64(sosdPath("wiki_ts_200M_uint64"), n)
			},
		},
		{
			name: "sosd_osm",
			load: func(n int) ([]uint64, error) {
				return loadAndMaskSOSD64(sosdPath("osm_cellids_800M_uint64"), n)
			},
		},
		{
			name: "sosd_books",
			load: func(n int) ([]uint64, error) {
				return loadAndMaskSOSD64(sosdPath("books_800M_uint64"), n)
			},
		},
	}
}

func loadAndMaskSOSD64(path string, n int) ([]uint64, error) {
	keys, err := loadSOSDUint64(path, n)
	if err != nil {
		return nil, err
	}
	return mask60AndDedupe(keys), nil
}

func runOneScalingPoint(t *testing.T, spec pefScalingSpec, n int) (pefScalingRow, bool) {
	t.Helper()
	t.Logf("[%s n=%d] loading…", spec.name, n)
	keys, err := spec.load(n)
	if err != nil {
		if os.IsNotExist(err) {
			t.Logf("[%s n=%d] dataset missing, skipping", spec.name, n)
			return pefScalingRow{}, false
		}
		t.Fatalf("[%s n=%d] load: %v", spec.name, n, err)
	}
	if len(keys) == 0 {
		t.Logf("[%s n=%d] dataset empty after dedup, skipping", spec.name, n)
		return pefScalingRow{}, false
	}
	if len(keys) > n {
		keys = keys[:n]
	}
	if len(keys) < n {
		t.Logf("[%s] requested n=%d, using available %d (after mask60+dedup)",
			spec.name, n, len(keys))
	}
	queries := generateEREMixedQueries(keys, pefScalingQueryCount, pefScalingRangeLen, 12345)

	row := pefScalingRow{dataset: spec.name, n: len(keys)}

	// Build PEF (timed)
	t.Logf("[%s n=%d] building PEF…", spec.name, len(keys))
	start := time.Now()
	pef, err := ere_pef.NewPEF(keys, pefScalingKeyBits)
	if err != nil {
		t.Fatalf("PEF build: %v", err)
	}
	pefDur := time.Since(start)
	row.pefBuildMS = float64(pefDur.Microseconds()) / 1000.0
	if row.pefBuildMS > 0 {
		row.pefThruMK = float64(len(keys)) / 1000.0 / row.pefBuildMS
	}
	row.pefBPK = float64(pef.ByteSize()*8) / float64(len(keys))
	row.pefChunks = pef.NumChunks()

	// Build ere_one_d (timed)
	t.Logf("[%s n=%d] building ere_one_d…", spec.name, len(keys))
	start = time.Now()
	one, err := ere_one_d.NewExactRangeEmptiness(keys, pefScalingKeyBits)
	if err != nil {
		t.Fatalf("ere_one_d build: %v", err)
	}
	oneDur := time.Since(start)
	row.oneDBuildMS = float64(oneDur.Microseconds()) / 1000.0
	if row.oneDBuildMS > 0 {
		row.oneDThruMK = float64(len(keys)) / 1000.0 / row.oneDBuildMS
	}
	row.oneDBPK = float64(one.ByteSize()*8) / float64(len(keys))
	row.bpkDelta = row.pefBPK - row.oneDBPK

	// Parity (subset)
	parityCount := 1024
	if parityCount > len(queries) {
		parityCount = len(queries)
	}
	for i := 0; i < parityCount; i++ {
		q := queries[i]
		if pef.IsEmpty(q.a, q.b) != one.IsEmpty(q.a, q.b) {
			t.Fatalf("[%s n=%d] parity divergence at q#%d (%d,%d)",
				spec.name, len(keys), i, q.a, q.b)
		}
	}

	// Query latency
	t.Logf("[%s n=%d] measuring query latency…", spec.name, len(keys))
	row.pefQueryNS = float64(testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			pefQuerySink = pef.IsEmpty(q.a, q.b)
		}
	}).NsPerOp())
	row.oneDQueryNS = float64(testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			pefQuerySink = one.IsEmpty(q.a, q.b)
		}
	}).NsPerOp())
	if row.oneDQueryNS > 0 {
		row.queryRatio = row.pefQueryNS / row.oneDQueryNS
	}

	// Peak heap snapshot (rough)
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	row.peakRSSGB = float64(ms.Sys) / (1 << 30)

	// Free for next iteration
	keys = nil
	pef = nil
	one = nil
	queries = nil
	runtime.GC()
	return row, true
}

func TestEREPEFScalingReport(t *testing.T) {
	if testing.Short() {
		t.Skip("scaling report is slow")
	}

	specs := makePEFScalingSpecs()
	type plan struct {
		n     int
		specs []pefScalingSpec
	}
	plans := []plan{
		{n: 1 << 24, specs: specs}, // 16.8M — full set
		{n: 1 << 28, specs: []pefScalingSpec{ // 268M — only 800M-row SOSD
			// (synthetic generators OOM on a `seen` map at 268M;
			// SOSD-200M files don't carry enough deduped-after-mask keys.)
			specs[4], // sosd_osm (800M)
			specs[5], // sosd_books (800M)
		}},
	}

	var rows []pefScalingRow
	for _, p := range plans {
		t.Logf("=== n=%d (2^%d) ===", p.n, ilog2(uint64(p.n)))
		for _, s := range p.specs {
			row, ok := runOneScalingPoint(t, s, p.n)
			if ok {
				rows = append(rows, row)
			}
		}
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# ere_pef vs ere_one_d — scaling\n\n")
	fmt.Fprintf(&buf, "keyBits=%d (mask), rangeLen=%d, queries=%d (50%% hits / 50%% smart misses)\n\n",
		pefScalingKeyBits, pefScalingRangeLen, pefScalingQueryCount)
	fmt.Fprintf(&buf, "| Dataset | n | PEF chunks | PEF build s | One-D build s | PEF Mkeys/s | One-D Mkeys/s | PEF query ns | One-D query ns | PEF/One-D | PEF bpk | One-D bpk | Δ bpk |\n")
	fmt.Fprintf(&buf, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&buf, "| %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %.0f | %.0f | %.2fx | %.3f | %.3f | %+.3f |\n",
			r.dataset, r.n, r.pefChunks,
			r.pefBuildMS/1000.0, r.oneDBuildMS/1000.0,
			r.pefThruMK, r.oneDThruMK,
			r.pefQueryNS, r.oneDQueryNS, r.queryRatio,
			r.pefBPK, r.oneDBPK, r.bpkDelta)
	}
	fmt.Print("\n" + buf.String())

	reportPath := filepath.Join("..", "bench_results", "ere_pef_scaling_report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Logf("warning: mkdir: %v", err)
	}
	if err := os.WriteFile(reportPath, buf.Bytes(), 0o644); err != nil {
		t.Logf("warning: write report: %v", err)
	}
}

func ilog2(x uint64) int {
	r := 0
	for x > 1 {
		x >>= 1
		r++
	}
	return r
}
