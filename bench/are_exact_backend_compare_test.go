package bench_test

import (
	"Thesis/emptiness/are_greedy_scan"
	"Thesis/emptiness/are_soda_hash"
	exactbackend "Thesis/emptiness/exact"
	"bytes"
	"fmt"
	"math"
	mathbits "math/bits"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const (
	areCompareN          = 1 << 20
	areCompareRangeLen   = uint64(4096)
	areCompareEpsilon    = 0.01
	areCompareQueryCount = 1 << 15
	areQueryRounds       = 3
)

type areLoadedDataset struct {
	name       string
	keysU64    []uint64
	rawQueries [][2]uint64
}

type areVariantMetrics struct {
	buildMS float64
	queryNS float64
	bpk     float64
	extra   string
}

type areCompareRow struct {
	filter       string
	dataset      string
	n            int
	classic      areVariantMetrics
	oneD         areVariantMetrics
	querySpeedup float64
	bpkReduction float64
}

func loadAREDatasets(tb testing.TB) []areLoadedDataset {
	tb.Helper()

	out := make([]areLoadedDataset, 0, len(ereDatasetLoaders))
	for _, loader := range ereDatasetLoaders {
		keys, err := loader.load()
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			tb.Fatalf("load %s: %v", loader.name, err)
		}
		rawQueries := generateARERawQueries(keys, areCompareQueryCount, areCompareRangeLen, 424242)
		out = append(out, areLoadedDataset{
			name:       loader.name,
			keysU64:    keys,
			rawQueries: rawQueries,
		})
	}
	if len(out) == 0 {
		tb.Fatal("no ARE datasets loaded")
	}
	return out
}

func generateARERawQueries(keys []uint64, count int, rangeLen uint64, seed int64) [][2]uint64 {
	bitQueries := generateEREMixedQueries(keys, count, rangeLen, seed)
	raw := make([][2]uint64, len(bitQueries))
	for i, q := range bitQueries {
		raw[i] = [2]uint64{q.a, q.b}
	}
	return raw
}

func sodaK(n int, rangeLen uint64, epsilon float64) uint32 {
	rTarget := float64(n) * float64(rangeLen) / epsilon
	K := uint32(math.Ceil(math.Log2(rTarget)))
	if K > 64 {
		K = 64
	}
	return K
}

func greedyK(n int, rangeLen uint64, epsilon float64) uint32 {
	rTarget := float64(n) * float64(rangeLen+1) / epsilon
	K := uint32(math.Ceil(math.Log2(rTarget)))
	if K > 64 {
		K = 64
	}
	return K
}

func timeAREQueriesU64(queries [][2]uint64, rounds int, fn func(a, b uint64) bool) float64 {
	total := time.Duration(0)
	count := 0
	for r := 0; r < rounds; r++ {
		start := time.Now()
		for _, q := range queries {
			ereQuerySink = fn(q[0], q[1])
		}
		total += time.Since(start)
		count += len(queries)
	}
	return float64(total.Nanoseconds()) / float64(count)
}


func measureSODA(tb testing.TB, ds areLoadedDataset, variant exactbackend.Variant) areVariantMetrics {
	tb.Helper()

	if err := exactbackend.SetVariant(variant); err != nil {
		tb.Fatalf("set variant: %v", err)
	}
	start := time.Now()
	filter, err := are_soda_hash.NewSodaAREFromK(ds.keysU64, areCompareRangeLen, sodaK(len(ds.keysU64), areCompareRangeLen, areCompareEpsilon))
	if err != nil {
		tb.Fatalf("build soda/%s/%s: %v", variant.String(), ds.name, err)
	}
	buildDur := time.Since(start)

	return areVariantMetrics{
		buildMS: float64(buildDur.Microseconds()) / 1000.0,
		queryNS: timeAREQueriesU64(ds.rawQueries, areQueryRounds, filter.IsEmpty),
		bpk:     float64(filter.SizeInBits()) / float64(len(ds.keysU64)),
		extra:   fmt.Sprintf("K=%d", filter.K),
	}
}

func measureGreedyMerge(tb testing.TB, ds areLoadedDataset, variant exactbackend.Variant) areVariantMetrics {
	tb.Helper()

	if err := exactbackend.SetVariant(variant); err != nil {
		tb.Fatalf("set variant: %v", err)
	}
	K := greedyK(len(ds.keysU64), areCompareRangeLen, areCompareEpsilon)
	keyBits := uint32(max(1, mathbits.Len64(ds.keysU64[len(ds.keysU64)-1])))
	start := time.Now()
	filter, err := are_greedy_scan.NewGreedyScanAREFromK(ds.keysU64, keyBits, are_greedy_scan.ConfigFromK{RangeLen: float64(areCompareRangeLen), K: K})
	if err != nil {
		tb.Fatalf("build greedy/%s/%s: %v", variant.String(), ds.name, err)
	}
	buildDur := time.Since(start)
	nc, nf, _ := filter.Stats()

	return areVariantMetrics{
		buildMS: float64(buildDur.Microseconds()) / 1000.0,
		queryNS: timeAREQueriesU64(ds.rawQueries, areQueryRounds, filter.IsEmpty),
		bpk:     float64(filter.SizeInBits()) / float64(len(ds.keysU64)),
		extra:   fmt.Sprintf("K=%d clusters=%d fallback=%d", K, nc, nf),
	}
}

func TestAREExactBackendReport(t *testing.T) {
	defer func() {
		_ = exactbackend.SetVariant(exactbackend.VariantClassic)
	}()

	datasets := loadAREDatasets(t)
	rows := make([]areCompareRow, 0, len(datasets)*2)

	for _, ds := range datasets {
		sodaClassic := measureSODA(t, ds, exactbackend.VariantClassic)
		sodaOneD := measureSODA(t, ds, exactbackend.VariantOneD)
		rows = append(rows, areCompareRow{
			filter:       "SODA",
			dataset:      ds.name,
			n:            len(ds.keysU64),
			classic:      sodaClassic,
			oneD:         sodaOneD,
			querySpeedup: sodaClassic.queryNS / sodaOneD.queryNS,
			bpkReduction: sodaClassic.bpk - sodaOneD.bpk,
		})

		greedyClassic := measureGreedyMerge(t, ds, exactbackend.VariantClassic)
		greedyOneD := measureGreedyMerge(t, ds, exactbackend.VariantOneD)
		rows = append(rows, areCompareRow{
			filter:       "Greedy+Merge",
			dataset:      ds.name,
			n:            len(ds.keysU64),
			classic:      greedyClassic,
			oneD:         greedyOneD,
			querySpeedup: greedyClassic.queryNS / greedyOneD.queryNS,
			bpkReduction: greedyClassic.bpk - greedyOneD.bpk,
		})
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# ARE Exact Backend Comparison\n\n")
	fmt.Fprintf(&buf, "Filters: SODA, Greedy+Merge\n\n")
	fmt.Fprintf(&buf, "n=%d, rangeLen=%d, epsilon=%.4f, mixed query workload (%d queries, %d rounds)\n\n",
		areCompareN, areCompareRangeLen, areCompareEpsilon, areCompareQueryCount, areQueryRounds)
	fmt.Fprintf(&buf, "| Filter | Dataset | n | Classic build ms | One-D build ms | Classic query ns | One-D query ns | Speedup | Classic bpk | One-D bpk | Delta bpk |\n")
	fmt.Fprintf(&buf, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, row := range rows {
		fmt.Fprintf(&buf, "| %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %.2fx | %.3f | %.3f | %.3f |\n",
			row.filter, row.dataset, row.n,
			row.classic.buildMS, row.oneD.buildMS,
			row.classic.queryNS, row.oneD.queryNS,
			row.querySpeedup,
			row.classic.bpk, row.oneD.bpk, row.bpkReduction)
	}

	fmt.Fprintf(&buf, "\n## Notes\n\n")
	for _, row := range rows {
		fmt.Fprintf(&buf, "- %s / %s: classic `%s`, one_d `%s`\n",
			row.filter, row.dataset, row.classic.extra, row.oneD.extra)
	}

	fmt.Print("\n" + buf.String())

	reportPath := filepath.Join("..", "bench_results", "are_exact_backend_report.md")
	if err := os.WriteFile(reportPath, buf.Bytes(), 0644); err != nil {
		t.Logf("warning: could not write report: %v", err)
	}
}
