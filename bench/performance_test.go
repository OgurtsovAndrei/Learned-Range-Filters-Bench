package bench_test

import (
	"Thesis-bench-industry/snarf"
	"Thesis-bench-industry/surf"
	"Thesis/bits"
	"Thesis/emptiness/are_bloom"
	"Thesis/emptiness/are_hybrid"
	"Thesis/emptiness/are_hybrid_scan"
	"Thesis/emptiness/are_adaptive"
	"Thesis/emptiness/are_pgm"
	"Thesis/emptiness/are_soda_hash"
	"Thesis/emptiness/are_trunc"
	"Thesis/testutils"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

func TestBuildTimePerKey(t *testing.T) {
	sizes := []int{1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20}
	const (
		rangeLen  = uint64(100)
		nClusters = 5
		unifFrac  = 0.15
		eps       = 0.01
	)

	bpk := math.Log2(float64(rangeLen) / eps)

	type filterDef struct {
		name   string
		color  string
		marker string
		dashed bool
		build  func(bs []bits.BitString, u64 []uint64, masked []uint64) error
	}

	filters := []filterDef{
		{"Adaptive(t=0)", "#2a7fff", "square", false, func(bs []bits.BitString, _ []uint64, _ []uint64) error {
			_, err := are_adaptive.NewAdaptiveARE(bs, rangeLen, eps, 0)
			return err
		}},
		{"SODA", "#22a06b", "diamond", false, func(_ []bits.BitString, u64 []uint64, _ []uint64) error {
			_, err := are_soda_hash.NewApproximateRangeEmptinessSoda(u64, rangeLen, eps)
			return err
		}},
		{"Truncation", "#e6a800", "triangle", false, func(bs []bits.BitString, _ []uint64, _ []uint64) error {
			_, err := are_trunc.NewApproximateRangeEmptiness(bs, eps)
			return err
		}},
		{"Hybrid", "#9b59b6", "star", false, func(bs []bits.BitString, _ []uint64, _ []uint64) error {
			_, err := are_hybrid.NewHybridARE(bs, rangeLen, eps)
			return err
		}},
		{"Scan-ARE", "#06b6d4", "star", false, func(bs []bits.BitString, _ []uint64, _ []uint64) error {
			_, err := are_hybrid_scan.NewHybridScanARE(bs, rangeLen, eps)
			return err
		}},
		{"CDF-ARE", "#e05d10", "circle", false, func(_ []bits.BitString, u64 []uint64, _ []uint64) error {
			_, err := are_pgm.NewPGMApproximateRangeEmptiness(u64, rangeLen, eps, 64)
			return err
		}},
		{"Bloom V3", "#888888", "circle", true, func(_ []bits.BitString, u64 []uint64, _ []uint64) error {
			_, err := are_bloom.NewBloomARE(u64, rangeLen, eps)
			return err
		}},
		{"Grafite", "#1a6b3c", "diamond", false, func(_ []bits.BitString, _ []uint64, masked []uint64) error {
			if tryGrafite(masked, bpk) == nil {
				return fmt.Errorf("grafite: unsupported bpk=%.2f for this key set", bpk)
			}
			return nil
		}},
		{"SNARF", "#1a3a6b", "star", false, func(_ []bits.BitString, _ []uint64, masked []uint64) error {
			snarf.New(masked, bpk)
			return nil
		}},
		{"SuRF", "#111111", "square", false, func(_ []bits.BitString, _ []uint64, masked []uint64) error {
			surf.New(masked, surf.SuffixNone, 0, 0)
			return nil
		}},
		{"SuRFHash(8)", "#111111", "triangle", false, func(_ []bits.BitString, _ []uint64, masked []uint64) error {
			surf.New(masked, surf.SuffixHash, 8, 0)
			return nil
		}},
		{"SuRFReal(8)", "#111111", "diamond", false, func(_ []bits.BitString, _ []uint64, masked []uint64) error {
			surf.New(masked, surf.SuffixReal, 0, 8)
			return nil
		}},
	}

	var allSeries []testutils.SeriesData
	for _, f := range filters {
		allSeries = append(allSeries, testutils.SeriesData{
			Name: f.name, Color: f.color, Marker: f.marker, Dashed: f.dashed,
		})
	}

	fmt.Printf("\n=== Build Time per Key (ε=%.3f, L=%d) ===\n", eps, rangeLen)
	fmt.Printf("%-16s", "Filter")
	for _, n := range sizes {
		fmt.Printf(" | %10s", fmt.Sprintf("n=%d", n))
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 16+len(sizes)*13))

	for fi, fd := range filters {
		fmt.Printf("%-16s", fd.name)
		for _, n := range sizes {
			rng := rand.New(rand.NewSource(99))
			keysU64, _ := testutils.GenerateClusterDistribution(n, nClusters, unifFrac, rng)
			keysBS := make([]bits.BitString, len(keysU64))
			for i, v := range keysU64 {
				keysBS[i] = testutils.TrieBS(v)
			}
			maskedKeys := mask60Keys(keysU64)

			start := time.Now()
			err := fd.build(keysBS, keysU64, maskedKeys)
			dur := time.Since(start)

			if err != nil {
				fmt.Printf(" | %10s", "err")
				continue
			}

			nsPerKey := float64(dur.Nanoseconds()) / float64(n)
			allSeries[fi].Points = append(allSeries[fi].Points, testutils.Point{X: float64(n), Y: nsPerKey})
			fmt.Printf(" | %8.1f ns", nsPerKey)
		}
		fmt.Println()
	}

	os.MkdirAll("../bench_results/plots", 0755)
	err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Build Time per Key (ε=%.3f, L=%d)", eps, rangeLen),
		XLabel: "Number of Keys (n)",
		YLabel: "Build Time (ns/key)",
		XScale: testutils.Log10,
		YScale: testutils.Log10,
	}, allSeries, "../bench_results/plots/build_time_per_key.svg")
	if err != nil {
		t.Errorf("SVG generation failed: %v", err)
	} else {
		fmt.Println("\nSVG written to bench_results/plots/build_time_per_key.svg")
	}
}

func TestQueryTimeVsRangeLen(t *testing.T) {
	rangeLens := []uint64{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
	const (
		n          = 1 << 16
		queryCount = 200_000
		nClusters  = 5
		unifFrac   = 0.15
		eps        = 0.01
	)

	bpk := math.Log2(float64(rangeLens[len(rangeLens)/2]) / eps)

	rng := rand.New(rand.NewSource(99))
	keysU64, clusters := testutils.GenerateClusterDistribution(n, nClusters, unifFrac, rng)
	keysBS := make([]bits.BitString, len(keysU64))
	for i, v := range keysU64 {
		keysBS[i] = testutils.TrieBS(v)
	}
	maskedKeys := mask60Keys(keysU64)

	type filterDef struct {
		name   string
		color  string
		marker string
		dashed bool
		build  func(L uint64) (func(a, b uint64) bool, error)
	}

	filters := []filterDef{
		{"Adaptive(t=0)", "#2a7fff", "square", false, func(L uint64) (func(a, b uint64) bool, error) {
			f, err := are_adaptive.NewAdaptiveARE(keysBS, L, eps, 0)
			if err != nil {
				return nil, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }, nil
		}},
		{"SODA", "#22a06b", "diamond", false, func(L uint64) (func(a, b uint64) bool, error) {
			f, err := are_soda_hash.NewApproximateRangeEmptinessSoda(keysU64, L, eps)
			if err != nil {
				return nil, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, nil
		}},
		{"Truncation", "#e6a800", "triangle", false, func(_ uint64) (func(a, b uint64) bool, error) {
			f, err := are_trunc.NewApproximateRangeEmptiness(keysBS, eps)
			if err != nil {
				return nil, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }, nil
		}},
		{"Hybrid", "#9b59b6", "star", false, func(L uint64) (func(a, b uint64) bool, error) {
			f, err := are_hybrid.NewHybridARE(keysBS, L, eps)
			if err != nil {
				return nil, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }, nil
		}},
		{"Scan-ARE", "#06b6d4", "star", false, func(L uint64) (func(a, b uint64) bool, error) {
			f, err := are_hybrid_scan.NewHybridScanARE(keysBS, L, eps)
			if err != nil {
				return nil, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }, nil
		}},
		{"CDF-ARE", "#e05d10", "circle", false, func(L uint64) (func(a, b uint64) bool, error) {
			f, err := are_pgm.NewPGMApproximateRangeEmptiness(keysU64, L, eps, 64)
			if err != nil {
				return nil, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, nil
		}},
		{"Bloom V3", "#888888", "circle", true, func(L uint64) (func(a, b uint64) bool, error) {
			f, err := are_bloom.NewBloomARE(keysU64, L, eps)
			if err != nil {
				return nil, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, nil
		}},
		{"Grafite", "#1a6b3c", "diamond", false, func(_ uint64) (func(a, b uint64) bool, error) {
			f := tryGrafite(maskedKeys, bpk)
			if f == nil {
				return nil, fmt.Errorf("grafite: unsupported bpk=%.2f for this key set", bpk)
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, nil
		}},
		{"SNARF", "#1a3a6b", "star", false, func(_ uint64) (func(a, b uint64) bool, error) {
			f := snarf.New(maskedKeys, bpk)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, nil
		}},
		{"SuRF", "#111111", "square", false, func(_ uint64) (func(a, b uint64) bool, error) {
			f := surf.New(maskedKeys, surf.SuffixNone, 0, 0)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, nil
		}},
		{"SuRFHash(8)", "#111111", "triangle", false, func(_ uint64) (func(a, b uint64) bool, error) {
			f := surf.New(maskedKeys, surf.SuffixHash, 8, 0)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, nil
		}},
		{"SuRFReal(8)", "#111111", "diamond", false, func(_ uint64) (func(a, b uint64) bool, error) {
			f := surf.New(maskedKeys, surf.SuffixReal, 0, 8)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, nil
		}},
	}

	var allSeries []testutils.SeriesData
	for _, f := range filters {
		allSeries = append(allSeries, testutils.SeriesData{
			Name: f.name, Color: f.color, Marker: f.marker, Dashed: f.dashed,
		})
	}

	fmt.Printf("\n=== Query Time vs Range Length (n=%d, ε=%.3f) ===\n", n, eps)
	fmt.Printf("%-16s", "Filter")
	for _, L := range rangeLens {
		fmt.Printf(" | %10s", fmt.Sprintf("L=%d", L))
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 16+len(rangeLens)*13))

	for fi, fd := range filters {
		fmt.Printf("%-16s", fd.name)
		for _, L := range rangeLens {
			qrng := rand.New(rand.NewSource(12345))
			rawQueries := testutils.GenerateClusterQueries(queryCount, clusters, unifFrac, L, qrng)

			check, err := fd.build(L)
			if err != nil {
				fmt.Printf(" | %10s", "err")
				continue
			}

			// CGo filters use masked queries
			queries := rawQueries
			isCGo := fd.name == "Grafite" || fd.name == "SNARF" ||
				fd.name == "SuRF" || fd.name == "SuRFHash(8)" || fd.name == "SuRFReal(8)"
			if isCGo {
				queries = mask60Queries(rawQueries)
			}

			start := time.Now()
			for _, q := range queries {
				check(q[0], q[1])
			}
			dur := time.Since(start)
			nsPerQuery := float64(dur.Nanoseconds()) / float64(queryCount)

			allSeries[fi].Points = append(allSeries[fi].Points, testutils.Point{X: float64(L), Y: nsPerQuery})
			fmt.Printf(" | %8.1f ns", nsPerQuery)
		}
		fmt.Println()
	}

	os.MkdirAll("../bench_results/plots", 0755)
	err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Query Time vs Range Length (n=%d, ε=%.3f)", n, eps),
		XLabel: "Range Length (L)",
		YLabel: "Query Time (ns/op)",
		XScale: testutils.Log10,
		YScale: testutils.Linear,
	}, allSeries, "../bench_results/plots/query_time_vs_rangelen.svg")
	if err != nil {
		t.Errorf("SVG generation failed: %v", err)
	} else {
		fmt.Println("\nSVG written to bench_results/plots/query_time_vs_rangelen.svg")
	}
}

func TestScalability(t *testing.T) {
	sizes := []int{1 << 20}
	const (
		rangeLen   = uint64(128)
		queryCount = 1_000_000
		nClusters  = 5
		unifFrac   = 0.15
		eps        = 0.001
	)

	bpk := math.Log2(float64(rangeLen) / eps)

	type filterEntry struct {
		name  string
		build func(keysBS []bits.BitString, keysU64 []uint64, masked []uint64) (func(a, b uint64) bool, uint64, string, error)
	}

	filters := []filterEntry{
		{"Adaptive(t=0)", func(bs []bits.BitString, u64 []uint64, _ []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f, err := are_adaptive.NewAdaptiveARE(bs, rangeLen, eps, 0)
			if err != nil {
				return nil, 0, "", err
			}
			return func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }, f.SizeInBits(), "-", nil
		}},
		{"SODA", func(_ []bits.BitString, u64 []uint64, _ []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f, err := are_soda_hash.NewApproximateRangeEmptinessSoda(u64, rangeLen, eps)
			if err != nil {
				return nil, 0, "", err
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), "-", nil
		}},
		{"Truncation", func(bs []bits.BitString, _ []uint64, _ []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f, err := are_trunc.NewApproximateRangeEmptiness(bs, eps)
			if err != nil {
				return nil, 0, "", err
			}
			return func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }, f.SizeInBits(), "-", nil
		}},
		{"Hybrid", func(bs []bits.BitString, _ []uint64, _ []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f, err := are_hybrid.NewHybridARE(bs, rangeLen, eps)
			if err != nil {
				return nil, 0, "", err
			}
			nc, nf, nt := f.Stats()
			info := fmt.Sprintf("%dc/%d%%fb", nc, 100*nf/nt)
			return func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }, f.SizeInBits(), info, nil
		}},
		{"Scan-ARE", func(bs []bits.BitString, _ []uint64, _ []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f, err := are_hybrid_scan.NewHybridScanARE(bs, rangeLen, eps)
			if err != nil {
				return nil, 0, "", err
			}
			nc, nf, nt := f.Stats()
			info := fmt.Sprintf("%dc/%d%%fb", nc, 100*nf/nt)
			return func(a, b uint64) bool { return f.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) }, f.SizeInBits(), info, nil
		}},
		{"CDF-ARE", func(_ []bits.BitString, u64 []uint64, _ []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f, err := are_pgm.NewPGMApproximateRangeEmptiness(u64, rangeLen, eps, 64)
			if err != nil {
				return nil, 0, "", err
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.TotalSizeInBits(), "-", nil
		}},
		{"Bloom V3", func(_ []bits.BitString, u64 []uint64, _ []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f, err := are_bloom.NewBloomARE(u64, rangeLen, eps)
			if err != nil {
				return nil, 0, "", err
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), "-", nil
		}},
		{"Grafite", func(_ []bits.BitString, _ []uint64, masked []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f := tryGrafite(masked, bpk)
			if f == nil {
				return nil, 0, "", fmt.Errorf("grafite: unsupported bpk=%.2f", bpk)
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), "-", nil
		}},
		{"SNARF", func(_ []bits.BitString, _ []uint64, masked []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f := snarf.New(masked, bpk)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), "-", nil
		}},
		{"SuRF", func(_ []bits.BitString, _ []uint64, masked []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f := surf.New(masked, surf.SuffixNone, 0, 0)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), "-", nil
		}},
		{"SuRFHash(8)", func(_ []bits.BitString, _ []uint64, masked []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f := surf.New(masked, surf.SuffixHash, 8, 0)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), "-", nil
		}},
		{"SuRFReal(8)", func(_ []bits.BitString, _ []uint64, masked []uint64) (func(a, b uint64) bool, uint64, string, error) {
			f := surf.New(masked, surf.SuffixReal, 0, 8)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), "-", nil
		}},
	}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(99))
			keysU64, clusters := testutils.GenerateClusterDistribution(n, nClusters, unifFrac, rng)
			keysBS := make([]bits.BitString, len(keysU64))
			for i, v := range keysU64 {
				keysBS[i] = testutils.TrieBS(v)
			}
			maskedKeys := mask60Keys(keysU64)

			qrng := rand.New(rand.NewSource(12345))
			rawQueries := testutils.GenerateClusterQueries(queryCount, clusters, unifFrac, rangeLen, qrng)
			maskedQueries := mask60Queries(rawQueries)

			fmt.Printf("\n=== n=%d (ε=%.3f, rangeLen=%d, %d queries) ===\n", n, eps, rangeLen, queryCount)
			fmt.Printf("%-16s | %8s | %12s | %12s | %12s | %s\n",
				"Filter", "BPK", "FPR", "Build(ms)", "Query(ns/op)", "Info")
			fmt.Println(strings.Repeat("-", 90))

			for _, fe := range filters {
				isCGo := fe.name == "Grafite" || fe.name == "SNARF" ||
					fe.name == "SuRF" || fe.name == "SuRFHash(8)" || fe.name == "SuRFReal(8)"

				buildStart := time.Now()
				check, sizeBits, info, err := fe.build(keysBS, keysU64, maskedKeys)
				buildDur := time.Since(buildStart)
				if err != nil {
					fmt.Printf("%-16s | %8s | %12s | %12s | %12s | err: %v\n",
						fe.name, "N/A", "N/A", "N/A", "N/A", err)
					continue
				}

				queryKeys := keysU64
				queries := rawQueries
				if isCGo {
					queryKeys = maskedKeys
					queries = maskedQueries
				}

				bpkActual := float64(sizeBits) / float64(n)

				fp, totalEmpty := 0, 0
				for _, q := range queries {
					a, b := q[0], q[1]
					if b < a {
						continue
					}
					if !testutils.GroundTruth(queryKeys, a, b) {
						continue
					}
					totalEmpty++
					if !check(a, b) {
						fp++
					}
				}
				var fpr float64
				if totalEmpty > 0 {
					fpr = float64(fp) / float64(totalEmpty)
				}

				queryStart := time.Now()
				for _, q := range queries {
					check(q[0], q[1])
				}
				queryDur := time.Since(queryStart)
				queryNs := float64(queryDur.Nanoseconds()) / float64(queryCount)

				fmt.Printf("%-16s | %8.2f | %12.6f | %12.1f | %12.1f | %s\n",
					fe.name, bpkActual, fpr, float64(buildDur.Milliseconds()), queryNs, info)
			}
		})
	}
}

func TestTradeoff_Full(t *testing.T) {
	const (
		n          = 10000
		rangeLen   = uint64(100)
		queryCount = 100_000
	)

	epsilons := []float64{0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001}

	rng := rand.New(rand.NewSource(42))
	seen := make(map[uint64]bool)
	unifU64 := make([]uint64, 0, n)
	for len(unifU64) < n {
		v := rng.Uint64()
		if !seen[v] {
			seen[v] = true
			unifU64 = append(unifU64, v)
		}
	}
	sort.Slice(unifU64, func(i, j int) bool { return unifU64[i] < unifU64[j] })
	unifBS := make([]bits.BitString, n)
	for i, v := range unifU64 {
		unifBS[i] = testutils.TrieBS(v)
	}

	qrng := rand.New(rand.NewSource(12345))
	queries := make([][2]uint64, queryCount)
	for i := range queries {
		a := qrng.Uint64()
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}

	seqGap := uint64(1000)
	seqBase := uint64(1) << 40
	seqU64 := make([]uint64, n)
	seqBS := make([]bits.BitString, n)
	for i := 0; i < n; i++ {
		v := seqBase + uint64(i)*seqGap
		seqU64[i] = v
		seqBS[i] = testutils.TrieBS(v)
	}

	qrngS := rand.New(rand.NewSource(12345))
	seqQueries := make([][2]uint64, queryCount)
	for i := range seqQueries {
		a := qrngS.Uint64()
		seqQueries[i] = [2]uint64{a, a + rangeLen - 1}
	}

	allSeries := map[string]*testutils.SeriesData{
		"Theoretical":       {Name: "Theoretical", Color: "#ef4444", Dashed: true, Marker: "circle"},
		"Adaptive (Unif)":   {Name: "Adaptive (Unif)", Color: "#2a7fff", Marker: "square"},
		"Adaptive (Seq)":    {Name: "Adaptive (Seq)", Color: "#2a7fff", Dashed: true, Marker: "square"},
		"SODA (Unif)":       {Name: "SODA (Unif)", Color: "#22a06b", Marker: "diamond"},
		"SODA (Seq)":        {Name: "SODA (Seq)", Color: "#22a06b", Dashed: true, Marker: "diamond"},
		"Truncation (Unif)": {Name: "Truncation (Unif)", Color: "#e6a800", Marker: "triangle"},
		"Truncation (Seq)":  {Name: "Truncation (Seq)", Color: "#e6a800", Dashed: true, Marker: "triangle"},
		"Hybrid (Unif)":     {Name: "Hybrid (Unif)", Color: "#9b59b6", Marker: "star"},
		"Hybrid (Seq)":      {Name: "Hybrid (Seq)", Color: "#9b59b6", Dashed: true, Marker: "star"},
		"Scan-ARE (Unif)":   {Name: "Scan-ARE (Unif)", Color: "#06b6d4", Marker: "star"},
		"Scan-ARE (Seq)":    {Name: "Scan-ARE (Seq)", Color: "#06b6d4", Dashed: true, Marker: "star"},
		"CDF-ARE (Unif)":    {Name: "CDF-ARE (Unif)", Color: "#e05d10", Marker: "circle"},
		"CDF-ARE (Seq)":     {Name: "CDF-ARE (Seq)", Color: "#e05d10", Dashed: true, Marker: "circle"},
		"Bloom V3 (Unif)":   {Name: "Bloom V3 (Unif)", Color: "#888888", Marker: "circle"},
		"Bloom V3 (Seq)":    {Name: "Bloom V3 (Seq)", Color: "#888888", Dashed: true, Marker: "circle"},
	}

	os.MkdirAll("../bench_results/plots", 0755)
	csvF, _ := os.Create("../bench_results/plots/are_tradeoff_data.csv")
	defer csvF.Close()
	fmt.Fprintln(csvF, "Epsilon,Series,BPK,FPR")

	fmt.Printf("\n=== Full Comparison (Uniform + Sequential, %d keys) ===\n", n)
	fmt.Printf("%-6s | %-20s | %8s | %12s\n", "Eps", "Series", "BPK", "FPR")
	fmt.Println(strings.Repeat("-", 55))

	for _, eps := range epsilons {
		thBPK := math.Log2(float64(rangeLen) / eps)
		allSeries["Theoretical"].Points = append(allSeries["Theoretical"].Points, testutils.Point{X: thBPK, Y: eps})
		fmt.Fprintf(csvF, "%f,Theoretical,%f,%f\n", eps, thBPK, eps)
		fmt.Printf("%-6.3f | %-20s | %8.2f | %12.6f\n", eps, "Theoretical", thBPK, eps)

		fOptU, errOptU := are_adaptive.NewAdaptiveARE(unifBS, rangeLen, eps, 0)
		fOptS, errOptS := are_adaptive.NewAdaptiveARE(seqBS, rangeLen, eps, 0)
		fSodaU, errSodaU := are_soda_hash.NewApproximateRangeEmptinessSoda(unifU64, rangeLen, eps)
		fSodaS, errSodaS := are_soda_hash.NewApproximateRangeEmptinessSoda(seqU64, rangeLen, eps)
		fTruncU, errTruncU := are_trunc.NewApproximateRangeEmptiness(unifBS, eps)
		fTruncS, errTruncS := are_trunc.NewApproximateRangeEmptiness(seqBS, eps)
		fHybridU, errHybridU := are_hybrid.NewHybridARE(unifBS, rangeLen, eps)
		fHybridS, errHybridS := are_hybrid.NewHybridARE(seqBS, rangeLen, eps)
		fScanU, errScanU := are_hybrid_scan.NewHybridScanARE(unifBS, rangeLen, eps)
		fScanS, errScanS := are_hybrid_scan.NewHybridScanARE(seqBS, rangeLen, eps)
		fCdfU, errCdfU := are_pgm.NewPGMApproximateRangeEmptiness(unifU64, rangeLen, eps, 64)
		fCdfS, errCdfS := are_pgm.NewPGMApproximateRangeEmptiness(seqU64, rangeLen, eps, 64)
		fBloomU, errBloomU := are_bloom.NewBloomARE(unifU64, rangeLen, eps)
		fBloomS, errBloomS := are_bloom.NewBloomARE(seqU64, rangeLen, eps)

		type mm struct {
			name    string
			err     error
			bpk     float64
			keys    []uint64
			queries [][2]uint64
			check   func(a, b uint64) bool
		}

		var ms []mm
		add := func(name string, err error, sizeBits uint64, keys []uint64, qs [][2]uint64, fn func(a, b uint64) bool) {
			var bpk float64
			if err == nil {
				bpk = float64(sizeBits) / float64(n)
			}
			ms = append(ms, mm{name, err, bpk, keys, qs, fn})
		}

		add("Adaptive (Unif)", errOptU, perfSafeSize(fOptU), unifU64, queries, func(a, b uint64) bool { return fOptU.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) })
		add("Adaptive (Seq)", errOptS, perfSafeSize(fOptS), seqU64, seqQueries, func(a, b uint64) bool { return fOptS.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) })
		add("SODA (Unif)", errSodaU, perfSafeSizeSoda(fSodaU), unifU64, queries, func(a, b uint64) bool { return fSodaU.IsEmpty(a, b) })
		add("SODA (Seq)", errSodaS, perfSafeSizeSoda(fSodaS), seqU64, seqQueries, func(a, b uint64) bool { return fSodaS.IsEmpty(a, b) })
		add("Truncation (Unif)", errTruncU, perfSafeSizeTrunc(fTruncU), unifU64, queries, func(a, b uint64) bool { return fTruncU.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) })
		add("Truncation (Seq)", errTruncS, perfSafeSizeTrunc(fTruncS), seqU64, seqQueries, func(a, b uint64) bool { return fTruncS.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) })
		add("Hybrid (Unif)", errHybridU, perfSafeSizeHybrid(fHybridU), unifU64, queries, func(a, b uint64) bool { return fHybridU.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) })
		add("Hybrid (Seq)", errHybridS, perfSafeSizeHybrid(fHybridS), seqU64, seqQueries, func(a, b uint64) bool { return fHybridS.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) })
		add("Scan-ARE (Unif)", errScanU, perfSafeSizeScan(fScanU), unifU64, queries, func(a, b uint64) bool { return fScanU.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) })
		add("Scan-ARE (Seq)", errScanS, perfSafeSizeScan(fScanS), seqU64, seqQueries, func(a, b uint64) bool { return fScanS.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b)) })
		add("CDF-ARE (Unif)", errCdfU, perfSafeSizeCdf(fCdfU), unifU64, queries, func(a, b uint64) bool { return fCdfU.IsEmpty(a, b) })
		add("CDF-ARE (Seq)", errCdfS, perfSafeSizeCdf(fCdfS), seqU64, seqQueries, func(a, b uint64) bool { return fCdfS.IsEmpty(a, b) })
		add("Bloom V3 (Unif)", errBloomU, perfSafeSizeBloom(fBloomU), unifU64, queries, func(a, b uint64) bool { return fBloomU.IsEmpty(a, b) })
		add("Bloom V3 (Seq)", errBloomS, perfSafeSizeBloom(fBloomS), seqU64, seqQueries, func(a, b uint64) bool { return fBloomS.IsEmpty(a, b) })

		for _, me := range ms {
			if me.err != nil {
				fmt.Printf("%-6.3f | %-20s | %8s | %12s (err: %v)\n", eps, me.name, "N/A", "N/A", me.err)
				continue
			}
			fpr := testutils.MeasureFPR(me.keys, me.queries, me.check)
			allSeries[me.name].Points = append(allSeries[me.name].Points, testutils.Point{X: me.bpk, Y: fpr})
			fmt.Fprintf(csvF, "%f,%s,%f,%f\n", eps, me.name, me.bpk, fpr)
			fmt.Printf("%-6.3f | %-20s | %8.2f | %12.6f\n", eps, me.name, me.bpk, fpr)
		}
	}

	orderedSeries := []testutils.SeriesData{
		*allSeries["Theoretical"],
		*allSeries["Adaptive (Unif)"],
		*allSeries["Adaptive (Seq)"],
		*allSeries["SODA (Unif)"],
		*allSeries["SODA (Seq)"],
		*allSeries["Truncation (Unif)"],
		*allSeries["Truncation (Seq)"],
		*allSeries["Hybrid (Unif)"],
		*allSeries["Hybrid (Seq)"],
		*allSeries["Scan-ARE (Unif)"],
		*allSeries["Scan-ARE (Seq)"],
		*allSeries["CDF-ARE (Unif)"],
		*allSeries["CDF-ARE (Seq)"],
		*allSeries["Bloom V3 (Unif)"],
		*allSeries["Bloom V3 (Seq)"],
	}

	err := testutils.GenerateTradeoffSVG(
		"Range Emptiness: FPR vs Bits per Key",
		"Bits per Key (BPK)",
		"False Positive Rate (FPR)",
		orderedSeries,
		"../bench_results/plots/are_full_comparison.svg",
	)
	if err != nil {
		t.Errorf("SVG generation failed: %v", err)
	} else {
		fmt.Println("\nSVG written to bench_results/plots/are_full_comparison.svg")
	}
}

// Safe size helpers for TestTradeoff_Full (prefixed to avoid conflict with comparison_test.go)
func perfSafeSize(f *are_adaptive.AdaptiveApproximateRangeEmptiness) uint64 {
	if f == nil {
		return 0
	}
	return f.SizeInBits()
}

func perfSafeSizeSoda(f *are_soda_hash.ApproximateRangeEmptinessSoda) uint64 {
	if f == nil {
		return 0
	}
	return f.SizeInBits()
}

func perfSafeSizeTrunc(f *are_trunc.ApproximateRangeEmptiness) uint64 {
	if f == nil {
		return 0
	}
	return f.SizeInBits()
}

func perfSafeSizeHybrid(f *are_hybrid.HybridARE) uint64 {
	if f == nil {
		return 0
	}
	return f.SizeInBits()
}

func perfSafeSizeCdf(f *are_pgm.PGMApproximateRangeEmptiness) uint64 {
	if f == nil {
		return 0
	}
	return f.TotalSizeInBits()
}

func perfSafeSizeBloom(f *are_bloom.BloomARE) uint64 {
	if f == nil {
		return 0
	}
	return f.SizeInBits()
}

func perfSafeSizeScan(f *are_hybrid_scan.HybridScanARE) uint64 {
	if f == nil {
		return 0
	}
	return f.SizeInBits()
}
