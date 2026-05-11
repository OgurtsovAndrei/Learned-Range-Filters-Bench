package bench_test

import (
	"Thesis/emptiness/exact/ere"
	"Thesis/emptiness/exact/ere_one_d"
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
	ereCompareN          = 1 << 20
	ereCompareRangeLen   = uint64(4096)
	ereCompareQueryCount = 1 << 15
)

type ereExactFilter interface {
	IsEmpty(a, b uint64) bool
	ByteSize() int
	SizeInBits() uint64
}

type ereQuery struct {
	a uint64
	b uint64
}

type ereLoadedDataset struct {
	name    string
	keys    []uint64
	queries []ereQuery
}

type ereBuildResult struct {
	filter       ereExactFilter
	buildDur     time.Duration
	practicalBPK float64
}

type ereReportRow struct {
	dataset      string
	n            int
	ereBuildMS   float64
	oneDBuildMS  float64
	ereQueryNS   float64
	oneDQueryNS  float64
	ereBPK       float64
	oneDBPK      float64
	speedup      float64
	bpkReduction float64
}

type ereDatasetLoader struct {
	name string
	load func() ([]uint64, error)
}

var (
	ereDatasetsOnce sync.Once
	ereDatasets     []ereLoadedDataset
	ereDatasetsErr  error
	ereQuerySink    bool
)

var ereDatasetLoaders = []ereDatasetLoader{
	{
		name: "uniform",
		load: func() ([]uint64, error) {
			rng := rand.New(rand.NewSource(42))
			keys := make([]uint64, 0, ereCompareN)
			seen := make(map[uint64]struct{}, ereCompareN)
			for len(keys) < ereCompareN {
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
			keys, _ := testutils.GenerateClusterDistribution(ereCompareN, 8, 0.10, rng)
			return keys, nil
		},
	},
	{
		name: "sosd_fb",
		load: func() ([]uint64, error) {
			return loadFacebookKeys(ereCompareN)
		},
	},
	{
		name: "sosd_wiki",
		load: func() ([]uint64, error) {
			return loadWikiKeys(ereCompareN)
		},
	},
	{
		name: "sosd_osm",
		load: func() ([]uint64, error) {
			return loadOSMKeys(ereCompareN)
		},
	},
	{
		name: "sosd_books",
		load: func() ([]uint64, error) {
			return loadBooksKeys(ereCompareN)
		},
	},
}

func mustLoadEREDatasets(tb testing.TB) []ereLoadedDataset {
	tb.Helper()

	ereDatasetsOnce.Do(func() {
		for _, loader := range ereDatasetLoaders {
			keys, err := loader.load()
			if err != nil {
				// Missing SOSD data is acceptable; keep going.
				if os.IsNotExist(err) {
					continue
				}
				ereDatasetsErr = fmt.Errorf("%s: %w", loader.name, err)
				return
			}
			if len(keys) == 0 {
				continue
			}
			ereDatasets = append(ereDatasets, ereLoadedDataset{
				name:    loader.name,
				keys:    keys,
				queries: generateEREMixedQueries(keys, ereCompareQueryCount, ereCompareRangeLen, 12345),
			})
		}
	})

	if ereDatasetsErr != nil {
		tb.Fatalf("load ERE datasets: %v", ereDatasetsErr)
	}
	if len(ereDatasets) == 0 {
		tb.Fatal("no ERE datasets loaded")
	}
	return ereDatasets
}

func generateEREMixedQueries(keys []uint64, count int, rangeLen uint64, seed int64) []ereQuery {
	rng := rand.New(rand.NewSource(seed))
	hitCount := count / 2
	missCount := count - hitCount

	hitQueries := make([][2]uint64, hitCount)
	for i := range hitQueries {
		key := keys[rng.Intn(len(keys))]
		var leftSlack uint64
		if rangeLen > 1 {
			maxSlack := rangeLen - 1
			if key < maxSlack {
				maxSlack = key
			}
			if maxSlack > 0 {
				leftSlack = uint64(rng.Int63n(int64(maxSlack + 1)))
			}
		}
		a := key - leftSlack
		b := a
		if rangeLen > 1 {
			if ^uint64(0)-a < rangeLen-1 {
				b = ^uint64(0)
			} else {
				b = a + rangeLen - 1
			}
		}
		hitQueries[i] = [2]uint64{a, b}
	}

	missQueries := generateERESafeEmptyQueries(keys, missCount, rangeLen, rand.New(rand.NewSource(seed^0x5eed5eed)))
	rawQueries := make([][2]uint64, 0, count)
	rawQueries = append(rawQueries, hitQueries...)
	rawQueries = append(rawQueries, missQueries...)
	rng.Shuffle(len(rawQueries), func(i, j int) {
		rawQueries[i], rawQueries[j] = rawQueries[j], rawQueries[i]
	})

	queries := make([]ereQuery, len(rawQueries))
	for i, q := range rawQueries {
		queries[i] = ereQuery{a: q[0], b: q[1]}
	}
	return queries
}

func generateERESafeEmptyQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	type gap struct {
		lo uint64
		hi uint64
	}

	if len(keys) == 0 || count == 0 {
		return nil
	}

	minK, maxK := keys[0], keys[len(keys)-1]
	gaps := make([]gap, 0, len(keys)-1)
	for i := 0; i+1 < len(keys); i++ {
		if keys[i+1] > keys[i]+1 {
			gaps = append(gaps, gap{lo: keys[i] + 1, hi: keys[i+1] - 1})
		}
	}

	tryAdd := func(out *[][2]uint64, a, b uint64) {
		if b < a {
			return
		}
		idx := sort.Search(len(keys), func(i int) bool { return keys[i] >= a })
		if idx < len(keys) && keys[idx] <= b {
			if keys[idx] == 0 || keys[idx]-1 < a {
				return
			}
			b = keys[idx] - 1
		}
		if b >= a {
			*out = append(*out, [2]uint64{a, b})
		}
	}

	nNear := count / 2
	nGap := count * 3 / 10
	nUnif := count - nNear - nGap

	queries := make([][2]uint64, 0, count)

	for i := 0; i < nNear*3 && len(queries) < nNear; i++ {
		key := keys[rng.Intn(len(keys))]
		window := rangeLen * 10
		var delta uint64
		if window > 0 {
			delta = randUint64n(rng, window)
		}
		var a uint64
		if delta < rangeLen*5 {
			if key >= rangeLen*5-delta {
				a = key - (rangeLen*5 - delta)
			}
		} else {
			if ^uint64(0)-key < delta-rangeLen*5 {
				a = ^uint64(0)
			} else {
				a = key + (delta - rangeLen*5)
			}
		}
		b := a
		if rangeLen > 1 {
			if ^uint64(0)-a < rangeLen-1 {
				b = ^uint64(0)
			} else {
				b = a + rangeLen - 1
			}
		}
		tryAdd(&queries, a, b)
	}

	for i := 0; i < nGap*3 && len(queries) < nNear+nGap && len(gaps) > 0; i++ {
		g := gaps[rng.Intn(len(gaps))]
		gapLen := g.hi - g.lo + 1
		if gapLen == 0 {
			continue
		}
		a := g.lo + randUint64n(rng, gapLen)
		b := a
		if rangeLen > 1 {
			if ^uint64(0)-a < rangeLen-1 {
				b = ^uint64(0)
			} else {
				b = a + rangeLen - 1
			}
			if b > g.hi {
				b = g.hi
			}
		}
		if b >= a {
			queries = append(queries, [2]uint64{a, b})
		}
	}

	span := maxK - minK
	for i := 0; i < nUnif*3 && len(queries) < count; i++ {
		a := minK
		if span > 0 {
			a += randUint64n(rng, span)
		}
		b := a
		if rangeLen > 1 {
			if ^uint64(0)-a < rangeLen-1 {
				b = ^uint64(0)
			} else {
				b = a + rangeLen - 1
			}
		}
		tryAdd(&queries, a, b)
	}

	for len(queries) < count {
		key := keys[rng.Intn(len(keys))]
		a := key + 1
		b := a
		if rangeLen > 1 && ^uint64(0)-a >= rangeLen-1 {
			b = a + rangeLen - 1
		}
		tryAdd(&queries, a, b)
		if len(queries) == 0 && a == ^uint64(0) {
			break
		}
	}

	if len(queries) > count {
		queries = queries[:count]
	}
	return queries
}

func randUint64n(rng *rand.Rand, n uint64) uint64 {
	if n == 0 {
		return 0
	}
	if n&(n-1) == 0 {
		return rng.Uint64() & (n - 1)
	}
	limit := ^uint64(0) - (^uint64(0) % n)
	for {
		v := rng.Uint64()
		if v < limit {
			return v % n
		}
	}
}

func buildEREFilter(keys []uint64) (ereExactFilter, error) {
	return ere.NewExactRangeEmptiness(keys, 64)
}

func buildEREOneDFilter(keys []uint64) (ereExactFilter, error) {
	return ere_one_d.NewExactRangeEmptiness(keys, 64)
}

func measureEREBuild(tb testing.TB, keys []uint64, build func([]uint64) (ereExactFilter, error)) ereBuildResult {
	tb.Helper()
	start := time.Now()
	filter, err := build(keys)
	if err != nil {
		tb.Fatalf("build failed: %v", err)
	}
	return ereBuildResult{
		filter:       filter,
		buildDur:     time.Since(start),
		practicalBPK: float64(filter.ByteSize()*8) / float64(len(keys)),
	}
}

func measureEREQueryNS(tb testing.TB, filter ereExactFilter, queries []ereQuery) float64 {
	tb.Helper()
	bm := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q := queries[i%len(queries)]
			ereQuerySink = filter.IsEmpty(q.a, q.b)
		}
	})
	return float64(bm.NsPerOp())
}

func verifyEREParity(tb testing.TB, a, b ereExactFilter, queries []ereQuery) {
	tb.Helper()
	for i, q := range queries {
		gotA := a.IsEmpty(q.a, q.b)
		gotB := b.IsEmpty(q.a, q.b)
		if gotA != gotB {
			tb.Fatalf("parity mismatch at query %d", i)
		}
	}
}

func BenchmarkERECompareBuild(b *testing.B) {
	for _, ds := range mustLoadEREDatasets(b) {
		ds := ds
		b.Run(ds.name+"/ere", func(b *testing.B) {
			b.ReportAllocs()
			var first ereExactFilter
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				filter, err := buildEREFilter(ds.keys)
				if err != nil {
					b.Fatalf("build ere: %v", err)
				}
				if i == 0 {
					first = filter
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
				filter, err := buildEREOneDFilter(ds.keys)
				if err != nil {
					b.Fatalf("build ere_one_d: %v", err)
				}
				if i == 0 {
					first = filter
				}
			}
			if first != nil {
				b.ReportMetric(float64(first.ByteSize()*8)/float64(len(ds.keys)), "practical_bpk")
			}
		})
	}
}

func BenchmarkERECompareQuery(b *testing.B) {
	for _, ds := range mustLoadEREDatasets(b) {
		ds := ds
		ereFilter, err := buildEREFilter(ds.keys)
		if err != nil {
			b.Fatalf("build ere: %v", err)
		}
		oneDFilter, err := buildEREOneDFilter(ds.keys)
		if err != nil {
			b.Fatalf("build ere_one_d: %v", err)
		}
		verifyEREParity(b, ereFilter, oneDFilter, ds.queries)

		b.Run(ds.name+"/ere", func(b *testing.B) {
			b.ReportAllocs()
			b.ReportMetric(float64(ereFilter.ByteSize()*8)/float64(len(ds.keys)), "practical_bpk")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := ds.queries[i%len(ds.queries)]
				ereQuerySink = ereFilter.IsEmpty(q.a, q.b)
			}
		})
		b.Run(ds.name+"/ere_one_d", func(b *testing.B) {
			b.ReportAllocs()
			b.ReportMetric(float64(oneDFilter.ByteSize()*8)/float64(len(ds.keys)), "practical_bpk")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := ds.queries[i%len(ds.queries)]
				ereQuerySink = oneDFilter.IsEmpty(q.a, q.b)
			}
		})
	}
}

func TestERECompareReport(t *testing.T) {
	datasets := mustLoadEREDatasets(t)

	rows := make([]ereReportRow, 0, len(datasets))
	for _, ds := range datasets {
		ereBuilt := measureEREBuild(t, ds.keys, buildEREFilter)
		oneDBuilt := measureEREBuild(t, ds.keys, buildEREOneDFilter)
		verifyEREParity(t, ereBuilt.filter, oneDBuilt.filter, ds.queries)

		ereNS := measureEREQueryNS(t, ereBuilt.filter, ds.queries)
		oneDNS := measureEREQueryNS(t, oneDBuilt.filter, ds.queries)

		row := ereReportRow{
			dataset:      ds.name,
			n:            len(ds.keys),
			ereBuildMS:   float64(ereBuilt.buildDur.Microseconds()) / 1000.0,
			oneDBuildMS:  float64(oneDBuilt.buildDur.Microseconds()) / 1000.0,
			ereQueryNS:   ereNS,
			oneDQueryNS:  oneDNS,
			ereBPK:       ereBuilt.practicalBPK,
			oneDBPK:      oneDBuilt.practicalBPK,
			bpkReduction: ereBuilt.practicalBPK - oneDBuilt.practicalBPK,
		}
		if oneDNS > 0 {
			row.speedup = ereNS / oneDNS
		}
		rows = append(rows, row)
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# ERE vs ERE One-D\n\n")
	fmt.Fprintf(&buf, "n=%d, rangeLen=%d, mixed query workload (%d queries, 50%% hits / 50%% smart misses)\n\n",
		ereCompareN, ereCompareRangeLen, ereCompareQueryCount)
	fmt.Fprintf(&buf, "| Dataset | n | ERE build ms | One-D build ms | ERE query ns | One-D query ns | Speedup | ERE bpk | One-D bpk | Delta bpk |\n")
	fmt.Fprintf(&buf, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, row := range rows {
		fmt.Fprintf(&buf, "| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2fx | %.3f | %.3f | %.3f |\n",
			row.dataset, row.n, row.ereBuildMS, row.oneDBuildMS, row.ereQueryNS, row.oneDQueryNS,
			row.speedup, row.ereBPK, row.oneDBPK, row.bpkReduction)
	}
	fmt.Print("\n" + buf.String())

	reportPath := filepath.Join("..", "bench_results", "ere_compare_report.md")
	if err := os.WriteFile(reportPath, buf.Bytes(), 0644); err != nil {
		t.Logf("warning: could not write report: %v", err)
	}
}
