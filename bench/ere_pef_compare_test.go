//go:build heavy

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
	"sort"
	"sync"
	"testing"
	"time"
)

const (
	pefCompareN          = 1 << 20
	pefCompareRangeLen   = uint64(4096)
	pefCompareQueryCount = 1 << 15
	pefCompareKeyBits    = 60
)

type pefLoadedDataset struct {
	name    string
	keys    []uint64
	queries []ereQuery
}

type pefBuildResult struct {
	filter       ereExactFilter
	buildDur     time.Duration
	practicalBPK float64
	chunkCount   int
}

type pefReportRow struct {
	dataset      string
	n            int
	pefBuildMS   float64
	oneDBuildMS  float64
	pefQueryNS   float64
	oneDQueryNS  float64
	pefBPK       float64
	oneDBPK      float64
	pefChunks    int
	pefThruMK    float64 // build throughput, Mkeys/s
	oneDThruMK   float64
	bpkDelta     float64
	queryRatio   float64 // pefQueryNS / oneDQueryNS — >1 means PEF slower
}

var (
	pefDatasetsOnce sync.Once
	pefDatasets     []pefLoadedDataset
	pefDatasetsErr  error
	pefQuerySink    bool
)

// loadAndMask60 wraps an underlying loader, masking keys to 60 bits and
// re-deduplicating + re-sorting (mask collisions are possible).
func loadAndMask60(load func() ([]uint64, error)) func() ([]uint64, error) {
	return func() ([]uint64, error) {
		raw, err := load()
		if err != nil {
			return nil, err
		}
		mask := (uint64(1) << pefCompareKeyBits) - 1
		for i, k := range raw {
			raw[i] = k & mask
		}
		sort.Slice(raw, func(i, j int) bool { return raw[i] < raw[j] })
		j := 0
		for i, k := range raw {
			if i == 0 || k != raw[i-1] {
				raw[j] = k
				j++
			}
		}
		return raw[:j], nil
	}
}

var pefDatasetLoaders = []ereDatasetLoader{
	{
		name: "uniform",
		load: loadAndMask60(func() ([]uint64, error) {
			rng := rand.New(rand.NewSource(42))
			keys := make([]uint64, 0, pefCompareN)
			seen := make(map[uint64]struct{}, pefCompareN)
			for len(keys) < pefCompareN {
				v := rng.Uint64()
				if _, ok := seen[v]; ok {
					continue
				}
				seen[v] = struct{}{}
				keys = append(keys, v)
			}
			return keys, nil
		}),
	},
	{
		name: "clustered",
		load: loadAndMask60(func() ([]uint64, error) {
			rng := rand.New(rand.NewSource(42))
			keys, _ := testutils.GenerateClusterDistribution(pefCompareN, 8, 0.10, rng)
			return keys, nil
		}),
	},
	{
		name: "sosd_fb",
		load: loadAndMask60(func() ([]uint64, error) {
			return loadFacebookKeys( pefCompareN)
		}),
	},
	{
		name: "sosd_wiki",
		load: loadAndMask60(func() ([]uint64, error) {
			return loadWikiKeys( pefCompareN)
		}),
	},
	{
		name: "sosd_osm",
		load: loadAndMask60(func() ([]uint64, error) {
			return loadOSMKeys( pefCompareN)
		}),
	},
	{
		name: "sosd_books",
		load: loadAndMask60(func() ([]uint64, error) {
			return loadBooksKeys( pefCompareN)
		}),
	},
}

func mustLoadPEFDatasets(tb testing.TB) []pefLoadedDataset {
	tb.Helper()
	pefDatasetsOnce.Do(func() {
		for _, loader := range pefDatasetLoaders {
			keys, err := loader.load()
			if err != nil {
				if os.IsNotExist(err) {
					continue
				}
				pefDatasetsErr = fmt.Errorf("%s: %w", loader.name, err)
				return
			}
			if len(keys) == 0 {
				continue
			}
			pefDatasets = append(pefDatasets, pefLoadedDataset{
				name:    loader.name,
				keys:    keys,
				queries: generateEREMixedQueries(keys, pefCompareQueryCount, pefCompareRangeLen, 12345),
			})
		}
	})
	if pefDatasetsErr != nil {
		tb.Fatalf("load PEF datasets: %v", pefDatasetsErr)
	}
	if len(pefDatasets) == 0 {
		tb.Fatal("no PEF datasets loaded")
	}
	return pefDatasets
}

func buildPEFFilter(keys []uint64) (ereExactFilter, int, error) {
	p, err := ere_pef.NewPEF(keys, pefCompareKeyBits)
	if err != nil {
		return nil, 0, err
	}
	return p, p.NumChunks(), nil
}

func buildOneDFilterPEF(keys []uint64) (ereExactFilter, error) {
	return ere_one_d.NewExactRangeEmptiness(keys, pefCompareKeyBits)
}

func measurePEFBuild(tb testing.TB, keys []uint64) pefBuildResult {
	tb.Helper()
	start := time.Now()
	f, chunks, err := buildPEFFilter(keys)
	if err != nil {
		tb.Fatalf("build PEF: %v", err)
	}
	return pefBuildResult{
		filter:       f,
		buildDur:     time.Since(start),
		practicalBPK: float64(f.ByteSize()*8) / float64(len(keys)),
		chunkCount:   chunks,
	}
}

func measureOneDBuildForPEF(tb testing.TB, keys []uint64) pefBuildResult {
	tb.Helper()
	start := time.Now()
	f, err := buildOneDFilterPEF(keys)
	if err != nil {
		tb.Fatalf("build ere_one_d: %v", err)
	}
	return pefBuildResult{
		filter:       f,
		buildDur:     time.Since(start),
		practicalBPK: float64(f.ByteSize()*8) / float64(len(keys)),
	}
}

func measurePEFQueryNS(tb testing.TB, filter ereExactFilter, queries []ereQuery) float64 {
	tb.Helper()
	bm := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			pefQuerySink = filter.IsEmpty(q.a, q.b)
		}
	})
	return float64(bm.NsPerOp())
}

func verifyPEFParity(tb testing.TB, a, b ereExactFilter, queries []ereQuery) {
	tb.Helper()
	for i, q := range queries {
		if a.IsEmpty(q.a, q.b) != b.IsEmpty(q.a, q.b) {
			tb.Fatalf("PEF↔ere_one_d parity mismatch at q#%d (%d,%d)", i, q.a, q.b)
		}
	}
}

func BenchmarkEREPEFCompareBuild(b *testing.B) {
	for _, ds := range mustLoadPEFDatasets(b) {
		ds := ds
		b.Run(ds.name+"/pef", func(b *testing.B) {
			b.ReportAllocs()
			var first ereExactFilter
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f, _, err := buildPEFFilter(ds.keys)
				if err != nil {
					b.Fatal(err)
				}
				if i == 0 {
					first = f
				}
			}
			if first != nil {
				b.ReportMetric(float64(first.ByteSize()*8)/float64(len(ds.keys)), "practical_bpk")
			}
		})
		b.Run(ds.name+"/ere_one_d", func(b *testing.B) {
			b.ReportAllocs()
			var first ereExactFilter
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f, err := buildOneDFilterPEF(ds.keys)
				if err != nil {
					b.Fatal(err)
				}
				if i == 0 {
					first = f
				}
			}
			if first != nil {
				b.ReportMetric(float64(first.ByteSize()*8)/float64(len(ds.keys)), "practical_bpk")
			}
		})
	}
}

func BenchmarkEREPEFCompareQuery(b *testing.B) {
	for _, ds := range mustLoadPEFDatasets(b) {
		ds := ds
		pef, _, err := buildPEFFilter(ds.keys)
		if err != nil {
			b.Fatal(err)
		}
		one, err := buildOneDFilterPEF(ds.keys)
		if err != nil {
			b.Fatal(err)
		}
		verifyPEFParity(b, pef, one, ds.queries)

		b.Run(ds.name+"/pef", func(b *testing.B) {
			b.ReportAllocs()
			b.ReportMetric(float64(pef.ByteSize()*8)/float64(len(ds.keys)), "practical_bpk")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := ds.queries[i%len(ds.queries)]
				pefQuerySink = pef.IsEmpty(q.a, q.b)
			}
		})
		b.Run(ds.name+"/ere_one_d", func(b *testing.B) {
			b.ReportAllocs()
			b.ReportMetric(float64(one.ByteSize()*8)/float64(len(ds.keys)), "practical_bpk")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := ds.queries[i%len(ds.queries)]
				pefQuerySink = one.IsEmpty(q.a, q.b)
			}
		})
	}
}

func TestEREPEFCompareReport(t *testing.T) {
	datasets := mustLoadPEFDatasets(t)

	rows := make([]pefReportRow, 0, len(datasets))
	for _, ds := range datasets {
		pefBuilt := measurePEFBuild(t, ds.keys)
		oneDBuilt := measureOneDBuildForPEF(t, ds.keys)
		verifyPEFParity(t, pefBuilt.filter, oneDBuilt.filter, ds.queries)

		pefNS := measurePEFQueryNS(t, pefBuilt.filter, ds.queries)
		oneDNS := measurePEFQueryNS(t, oneDBuilt.filter, ds.queries)

		pefMS := float64(pefBuilt.buildDur.Microseconds()) / 1000.0
		oneMS := float64(oneDBuilt.buildDur.Microseconds()) / 1000.0
		row := pefReportRow{
			dataset:      ds.name,
			n:            len(ds.keys),
			pefBuildMS:   pefMS,
			oneDBuildMS:  oneMS,
			pefQueryNS:   pefNS,
			oneDQueryNS:  oneDNS,
			pefBPK:       pefBuilt.practicalBPK,
			oneDBPK:      oneDBuilt.practicalBPK,
			pefChunks:    pefBuilt.chunkCount,
			bpkDelta:     pefBuilt.practicalBPK - oneDBuilt.practicalBPK,
		}
		if pefMS > 0 {
			row.pefThruMK = float64(len(ds.keys)) / 1000.0 / pefMS
		}
		if oneMS > 0 {
			row.oneDThruMK = float64(len(ds.keys)) / 1000.0 / oneMS
		}
		if oneDNS > 0 {
			row.queryRatio = pefNS / oneDNS
		}
		rows = append(rows, row)
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# ere_pef vs ere_one_d\n\n")
	fmt.Fprintf(&buf, "n=%d, keyBits=%d (mask), rangeLen=%d, queries=%d (50%% hits / 50%% smart misses)\n\n",
		pefCompareN, pefCompareKeyBits, pefCompareRangeLen, pefCompareQueryCount)
	fmt.Fprintf(&buf, "| Dataset | n | PEF chunks | PEF build ms | One-D build ms | PEF Mkeys/s | One-D Mkeys/s | PEF query ns | One-D query ns | PEF/One-D | PEF bpk | One-D bpk | Δ bpk |\n")
	fmt.Fprintf(&buf, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&buf, "| %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2fx | %.3f | %.3f | %+.3f |\n",
			r.dataset, r.n, r.pefChunks,
			r.pefBuildMS, r.oneDBuildMS,
			r.pefThruMK, r.oneDThruMK,
			r.pefQueryNS, r.oneDQueryNS, r.queryRatio,
			r.pefBPK, r.oneDBPK, r.bpkDelta)
	}
	fmt.Print("\n" + buf.String())

	reportPath := filepath.Join("..", "bench_results", "ere_pef_compare_report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Logf("warning: mkdir: %v", err)
	}
	if err := os.WriteFile(reportPath, buf.Bytes(), 0o644); err != nil {
		t.Logf("warning: write report: %v", err)
	}
}
