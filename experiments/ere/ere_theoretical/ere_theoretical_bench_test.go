package ere_theoretical

import (
	"Thesis/bits"
	"Thesis/emptiness/exact/ere"
	"Thesis-bench-industry/experiments/ere/ere_global"
	"Thesis/testutils"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

const (
	mask60   = (uint64(1) << 60) - 1
	bitLen60 = uint32(60)
	nRuns    = 3
	nQueries = 10000
)

var (
	benchNs = []int{1 << 16, 1 << 18, 1 << 20, 1 << 24}
	benchLs = []uint64{1, 16, 128, 1024}
)

var buildNames = []string{"ERE", "TheoreticalERE", "GlobalERE"}
var queryNames = []string{"ERE", "ERE (linear)", "TheoreticalERE", "GlobalERE"}

func generateBenchKeys(n int, rng *rand.Rand) ([]uint64, []bits.BitString) {
	seen := make(map[uint64]bool, n)
	raw := make([]uint64, 0, n)
	for len(raw) < n {
		v := rng.Uint64() & mask60
		if !seen[v] {
			seen[v] = true
			raw = append(raw, v)
		}
	}
	sort.Slice(raw, func(i, j int) bool { return raw[i] < raw[j] })

	bs := make([]bits.BitString, n)
	for i, v := range raw {
		bs[i] = bits.NewFromTrieUint64(v, bitLen60)
	}
	return raw, bs
}

func generateQueries(L uint64, rng *rand.Rand) ([]bits.BitString, []bits.BitString) {
	a := make([]bits.BitString, nQueries)
	b := make([]bits.BitString, nQueries)
	for i := 0; i < nQueries; i++ {
		lo := rng.Uint64() & mask60
		hi := lo + L - 1
		if hi > mask60 {
			hi = mask60
		}
		a[i] = bits.NewFromTrieUint64(lo, bitLen60)
		b[i] = bits.NewFromTrieUint64(hi, bitLen60)
	}
	return a, b
}

type isEmptyFunc func(a, b bits.BitString) bool

// Driven by its own N×L sweep, not by b.N — invoke with -benchtime=1x.
func BenchmarkEREComparison(b *testing.B) {
	universe := bits.NewBitString(bitLen60)

	type buildResult struct {
		name    string
		nKeys   int
		buildNs float64
		bpk     float64
	}
	type queryResult struct {
		name    string
		nKeys   int
		L       uint64
		queryNs float64
	}

	var buildResults []buildResult
	var queryResults []queryResult

	for _, n := range benchNs {
		rng := rand.New(rand.NewSource(42))
		keysU64, keysBS := generateBenchKeys(n, rng)

		fmt.Printf("\n--- N=%d ---\n", n)

		// ---- Build ERE ----
		{
			var totalNs int64
			var lastERE *ere.ExactRangeEmptiness
			for r := 0; r < nRuns; r++ {
				start := time.Now()
				e, err := ere.NewExactRangeEmptiness(keysU64, bitLen60)
				totalNs += time.Since(start).Nanoseconds()
				if err != nil {
					b.Fatalf("ERE build failed (N=%d): %v", n, err)
				}
				lastERE = e
			}
			nsPerKey := float64(totalNs) / float64(nRuns) / float64(n)
			bpk := float64(lastERE.ByteSize()) * 8 / float64(n)
			buildResults = append(buildResults, buildResult{"ERE", n, nsPerKey, bpk})
			fmt.Printf("  ERE build:         %8.1f ns/key, %.2f bits/key\n", nsPerKey, bpk)
		}

		// ---- Build TheoreticalERE ----
		{
			var totalNs int64
			var lastTERE *TheoreticalExactRangeEmptiness
			for r := 0; r < nRuns; r++ {
				start := time.Now()
				e, err := NewTheoreticalExactRangeEmptiness(keysBS, universe)
				totalNs += time.Since(start).Nanoseconds()
				if err != nil {
					b.Fatalf("TheoreticalERE build failed (N=%d): %v", n, err)
				}
				lastTERE = e
			}
			nsPerKey := float64(totalNs) / float64(nRuns) / float64(n)
			bpk := float64(lastTERE.ByteSize()) * 8 / float64(n)
			buildResults = append(buildResults, buildResult{"TheoreticalERE", n, nsPerKey, bpk})
			fmt.Printf("  TheoreticalERE:    %8.1f ns/key, %.2f bits/key\n", nsPerKey, bpk)
		}

		// ---- Build GlobalERE ----
		{
			var totalNs int64
			var lastGERE *ere_global.GlobalExactRangeEmptiness
			for r := 0; r < nRuns; r++ {
				start := time.Now()
				e, err := ere_global.NewGlobalExactRangeEmptiness(keysBS, universe)
				totalNs += time.Since(start).Nanoseconds()
				if err != nil {
					b.Fatalf("GlobalERE build failed (N=%d): %v", n, err)
				}
				lastGERE = e
			}
			nsPerKey := float64(totalNs) / float64(nRuns) / float64(n)
			bpk := float64(lastGERE.ByteSize()) * 8 / float64(n)
			buildResults = append(buildResults, buildResult{"GlobalERE", n, nsPerKey, bpk})
			fmt.Printf("  GlobalERE:         %8.1f ns/key, %.2f bits/key\n", nsPerKey, bpk)
		}

		// Pre-build once for queries
		ereStruct, _ := ere.NewExactRangeEmptiness(keysU64, bitLen60)
		tereStruct, _ := NewTheoreticalExactRangeEmptiness(keysBS, universe)
		gereStruct, _ := ere_global.NewGlobalExactRangeEmptiness(keysBS, universe)

		queryFuncs := []struct {
			name string
			fn   isEmptyFunc
		}{
			{"ERE", func(a, b bits.BitString) bool { return ereStruct.IsEmpty(a.TrieUint64(), b.TrieUint64()) }},
			{"ERE (linear)", func(a, b bits.BitString) bool { return ereStruct.LinearIsEmpty(a.TrieUint64(), b.TrieUint64()) }},
			{"TheoreticalERE", tereStruct.IsEmpty},
			{"GlobalERE", gereStruct.IsEmpty},
		}

		for _, L := range benchLs {
			qRng := rand.New(rand.NewSource(int64(n) ^ int64(L)))
			qA, qB := generateQueries(L, qRng)

			for _, qf := range queryFuncs {
				var totalNs int64
				for r := 0; r < nRuns; r++ {
					start := time.Now()
					for q := 0; q < nQueries; q++ {
						qf.fn(qA[q], qB[q])
					}
					totalNs += time.Since(start).Nanoseconds()
				}
				nsPerQ := float64(totalNs) / float64(nRuns) / float64(nQueries)
				queryResults = append(queryResults, queryResult{qf.name, n, L, nsPerQ})
			}
		}
	}

	// ---- Print build table ----
	colW := 15
	nameW := 18
	printHeader := func(title string) {
		fmt.Printf("\n=== %s ===\n", title)
		fmt.Printf("%-*s", nameW, "Filter")
		for _, n := range benchNs {
			fmt.Printf(" | %*s", colW-2, fmt.Sprintf("N=%d", n))
		}
		fmt.Println()
		fmt.Println(strings.Repeat("-", nameW+len(benchNs)*colW))
	}

	printHeader("Build Time (ns/key)")
	for _, name := range buildNames {
		fmt.Printf("%-*s", nameW, name)
		for _, n := range benchNs {
			for _, br := range buildResults {
				if br.name == name && br.nKeys == n {
					fmt.Printf(" | %*.1f", colW-2, br.buildNs)
				}
			}
		}
		fmt.Println()
	}

	printHeader("Memory (bits/key)")
	for _, name := range buildNames {
		fmt.Printf("%-*s", nameW, name)
		for _, n := range benchNs {
			for _, br := range buildResults {
				if br.name == name && br.nKeys == n {
					fmt.Printf(" | %*.2f", colW-2, br.bpk)
				}
			}
		}
		fmt.Println()
	}

	for _, L := range benchLs {
		printHeader(fmt.Sprintf("Query Time (ns/query), L=%d", L))
		for _, name := range queryNames {
			fmt.Printf("%-*s", nameW, name)
			for _, n := range benchNs {
				for _, qr := range queryResults {
					if qr.name == name && qr.nKeys == n && qr.L == L {
						fmt.Printf(" | %*.1f", colW-2, qr.queryNs)
					}
				}
			}
			fmt.Println()
		}
	}

	// ---- SVG plots ----
	plotDir := "../../bench_results/plots/ere_comparison"
	os.MkdirAll(plotDir, 0755)

	colors := map[string]string{"ERE": "#2a7fff", "ERE (linear)": "#f59e0b", "TheoreticalERE": "#ef4444", "GlobalERE": "#22c55e"}
	markers := map[string]string{"ERE": "square", "ERE (linear)": "triangle", "TheoreticalERE": "circle", "GlobalERE": "diamond"}

	makeSeriesFor := func(names []string) map[string]*testutils.SeriesData {
		m := make(map[string]*testutils.SeriesData)
		for _, name := range names {
			m[name] = &testutils.SeriesData{Name: name, Color: colors[name], Marker: markers[name]}
		}
		return m
	}
	toSlice := func(m map[string]*testutils.SeriesData, names []string) []testutils.SeriesData {
		out := make([]testutils.SeriesData, 0, len(names))
		for _, name := range names {
			if s := m[name]; len(s.Points) > 0 {
				out = append(out, *s)
			}
		}
		return out
	}

	// Build time plot
	{
		m := makeSeriesFor(buildNames)
		for _, br := range buildResults {
			m[br.name].Points = append(m[br.name].Points, testutils.Point{X: float64(br.nKeys), Y: br.buildNs})
		}
		testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title: "ERE Variants — Build Time", XLabel: "N (keys)", YLabel: "ns / key",
			XScale: testutils.Log10, YScale: testutils.Log10,
			XMax: 25,
		}, toSlice(m, buildNames), plotDir+"/build_time.svg")
	}

	// Memory plot
	{
		m := makeSeriesFor(buildNames)
		for _, br := range buildResults {
			m[br.name].Points = append(m[br.name].Points, testutils.Point{X: float64(br.nKeys), Y: br.bpk})
		}
		testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title: "ERE Variants — Memory", XLabel: "N (keys)", YLabel: "bits / key",
			XScale: testutils.Log10, YScale: testutils.Linear,
			XMax: 25,
		}, toSlice(m, buildNames), plotDir+"/memory.svg")
	}

	// Query time plots (one per L)
	for _, L := range benchLs {
		m := makeSeriesFor(queryNames)
		for _, qr := range queryResults {
			if qr.L != L {
				continue
			}
			m[qr.name].Points = append(m[qr.name].Points, testutils.Point{X: float64(qr.nKeys), Y: qr.queryNs})
		}
		testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title:  fmt.Sprintf("ERE Variants — Query Time (L=%d)", L),
			XLabel: "N (keys)", YLabel: "ns / query",
			XScale: testutils.Log10, YScale: testutils.Log10,
			XMax: 25,
		}, toSlice(m, queryNames), fmt.Sprintf("%s/query_L%d.svg", plotDir, L))
	}

	fmt.Printf("\nSVG plots written to %s/\n", plotDir)
}
