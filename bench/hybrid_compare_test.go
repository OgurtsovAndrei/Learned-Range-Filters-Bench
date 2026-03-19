package bench_test

import (
	"Thesis/bits"
	"Thesis/emptiness/are_greedy_scan"
	"Thesis/emptiness/are_hybrid"
	"Thesis/emptiness/are_hybrid_scan"
	"Thesis/testutils"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"sync"
	"testing"
	"time"
)

// ---- helpers ----

func toTrieBS(keys []uint64) []bits.BitString {
	bs := make([]bits.BitString, len(keys))
	for i, v := range keys {
		bs[i] = testutils.TrieBS(v)
	}
	return bs
}

// hybridFPRPoint builds all three filters at a given K and returns (bpk, fpr) for each.
// FPR is averaged over the provided seeds using avgFPRParallel.
type hybridPoint struct {
	hybrid, scan, greedy testutils.Point
}

func hybridMeasureK(
	keysBS []bits.BitString,
	keys []uint64,
	queryFunc func(uint64, int64) [][2]uint64,
	rangeLen uint64,
	K uint32,
	seeds []int64,
) hybridPoint {
	type task struct {
		name    string
		bpk     float64
		isEmpty func(a, b uint64) bool
	}

	var tasks []task

	if fh, err := are_hybrid.NewHybridAREFromK(keysBS, rangeLen, K); err == nil {
		bpk := float64(fh.SizeInBits()) / float64(len(keys))
		tasks = append(tasks, task{"Hybrid", bpk, func(a, b uint64) bool {
			return fh.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
		}})
	}

	if fs, err := are_hybrid_scan.NewHybridScanAREFromK(keysBS, rangeLen, K); err == nil {
		bpk := float64(fs.SizeInBits()) / float64(len(keys))
		tasks = append(tasks, task{"Scan-ARE", bpk, func(a, b uint64) bool {
			return fs.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
		}})
	}

	if fg, err := are_greedy_scan.NewGreedyScanAREFromK(keysBS, rangeLen, K); err == nil {
		bpk := float64(fg.SizeInBits()) / float64(len(keys))
		tasks = append(tasks, task{"Greedy-ARE", bpk, func(a, b uint64) bool {
			return fg.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
		}})
	}

	results := make([]float64, len(tasks))
	var wg sync.WaitGroup
	for i, tk := range tasks {
		i, tk := i, tk
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i] = avgFPRParallel(keys, queryFunc, rangeLen, seeds, tk.isEmpty)
		}()
	}
	wg.Wait()

	var hp hybridPoint
	for i, tk := range tasks {
		switch tk.name {
		case "Hybrid":
			hp.hybrid = testutils.Point{X: tk.bpk, Y: results[i]}
		case "Scan-ARE":
			hp.scan = testutils.Point{X: tk.bpk, Y: results[i]}
		case "Greedy-ARE":
			hp.greedy = testutils.Point{X: tk.bpk, Y: results[i]}
		}
	}
	return hp
}

func runHybridCompareFPR(t *testing.T, distName string, keys []uint64, queryFunc func(uint64, int64) [][2]uint64) {
	t.Helper()

	const rangeLen = uint64(128)
	const queryCount = 1 << 18
	kGrid := []uint32{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48}
	seeds := []int64{12345, 54321, 99999}

	keysBS := toTrieBS(keys)

	hybridSeries := &testutils.SeriesData{Name: "Hybrid", Color: "#ff6b6b", Marker: "circle"}
	scanSeries := &testutils.SeriesData{Name: "Scan-ARE", Color: "#06b6d4", Marker: "square"}
	greedySeries := &testutils.SeriesData{Name: "Greedy-ARE", Color: "#22c55e", Marker: "diamond"}

	fmt.Printf("\n=== Hybrid Compare FPR — %s (n=%d, L=%d) ===\n", distName, len(keys), rangeLen)
	fmt.Printf("%-12s | %10s | %10s | %10s | %10s | %10s | %10s\n",
		"K", "Hyb-BPK", "Hyb-FPR", "Scan-BPK", "Scan-FPR", "Grdy-BPK", "Grdy-FPR")
	fmt.Println("--------------------------------------------------------------------------------------------")

	for _, K := range kGrid {
		hp := hybridMeasureK(keysBS, keys, queryFunc, rangeLen, K, seeds)
		hybridSeries.Points = append(hybridSeries.Points, hp.hybrid)
		scanSeries.Points = append(scanSeries.Points, hp.scan)
		greedySeries.Points = append(greedySeries.Points, hp.greedy)
		fmt.Printf("%-12d | %10.2f | %10.6f | %10.2f | %10.6f | %10.2f | %10.6f\n",
			K, hp.hybrid.X, hp.hybrid.Y, hp.scan.X, hp.scan.Y, hp.greedy.X, hp.greedy.Y)
	}

	outDir := fmt.Sprintf("../bench_results/plots/hybrid_compare/%s", distName)
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Errorf("mkdir: %v", err)
		return
	}

	svgPath := fmt.Sprintf("%s/L%d.svg", outDir, rangeLen)
	series := []testutils.SeriesData{*hybridSeries, *scanSeries, *greedySeries}
	err := testutils.GenerateTradeoffSVG(
		fmt.Sprintf("Hybrid ARE Variants — %s (n=%d, L=%d)", distName, len(keys), rangeLen),
		"Bits per Key (BPK)",
		"False Positive Rate (FPR)",
		series,
		svgPath,
	)
	if err != nil {
		t.Errorf("SVG generation failed: %v", err)
	} else {
		fmt.Printf("SVG written to %s\n", svgPath)
	}
}

func runHybridCompareBuildTime(t *testing.T, distName string, nValues []int, genKeys func(n int) []uint64) {
	t.Helper()

	const (
		rangeLen = uint64(128)
		epsilon  = 0.01
	)

	hybridSeries := &testutils.SeriesData{Name: "Hybrid", Color: "#ff6b6b", Marker: "circle"}
	scanSeries := &testutils.SeriesData{Name: "Scan-ARE", Color: "#06b6d4", Marker: "square"}
	greedySeries := &testutils.SeriesData{Name: "Greedy-ARE", Color: "#22c55e", Marker: "diamond"}

	fmt.Printf("\n=== Hybrid Compare Build Time — %s ===\n", distName)
	fmt.Printf("%-10s | %12s | %12s | %12s\n", "N", "Hybrid", "Scan-ARE", "Greedy-ARE")
	fmt.Println("------------------------------------------------------------")

	for _, n := range nValues {
		keys := genKeys(n)
		keysBS := toTrieBS(keys)

		start := time.Now()
		_, hybErr := are_hybrid.NewHybridARE(keysBS, rangeLen, epsilon)
		hybTime := time.Since(start)

		start = time.Now()
		_, scanErr := are_hybrid_scan.NewHybridScanARE(keysBS, rangeLen, epsilon)
		scanTime := time.Since(start)

		start = time.Now()
		_, greedyErr := are_greedy_scan.NewGreedyScanARE(keysBS, rangeLen, epsilon)
		greedyTime := time.Since(start)

		hybNs := float64(hybTime.Nanoseconds()) / float64(n)
		scanNs := float64(scanTime.Nanoseconds()) / float64(n)
		greedyNs := float64(greedyTime.Nanoseconds()) / float64(n)

		if hybErr == nil {
			hybridSeries.Points = append(hybridSeries.Points, testutils.Point{X: float64(n), Y: hybNs})
		}
		if scanErr == nil {
			scanSeries.Points = append(scanSeries.Points, testutils.Point{X: float64(n), Y: scanNs})
		}
		if greedyErr == nil {
			greedySeries.Points = append(greedySeries.Points, testutils.Point{X: float64(n), Y: greedyNs})
		}

		fmt.Printf("%-10d | %10.1f ns | %10.1f ns | %10.1f ns\n", n, hybNs, scanNs, greedyNs)
	}

	outDir := "../bench_results/plots/hybrid_compare/build_time"
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Errorf("mkdir: %v", err)
		return
	}

	svgPath := fmt.Sprintf("%s/%s.svg", outDir, distName)
	series := []testutils.SeriesData{*hybridSeries, *scanSeries, *greedySeries}
	err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Build Time per Key — Hybrid ARE Variants (%s)", distName),
		XLabel: "Number of Keys (n)",
		YLabel: "Build Time (ns/key)",
		XScale: testutils.Log10,
		YScale: testutils.Log10,
	}, series, svgPath)
	if err != nil {
		t.Errorf("SVG generation failed: %v", err)
	} else {
		fmt.Printf("SVG written to %s\n", svgPath)
	}
}

// ---- FPR benchmarks for synthetic distributions ----

func TestHybridCompare_FPR_Clustered(t *testing.T) {
	const (
		n         = 1 << 18
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "clustered", n, func() []uint64 {
		rng := rand.New(rand.NewSource(99))
		raw, _ := testutils.GenerateClusterDistribution(n, 5, 0.15, rng)
		return mask60Keys(raw)
	})

	// We need cluster info for query generation — re-generate with same seed.
	rng := rand.New(rand.NewSource(99))
	_, clusters := testutils.GenerateClusterDistribution(n, 5, 0.15, rng)

	runHybridCompareFPR(t, "clustered", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return mask60Queries(testutils.GenerateClusterQueries(queryCount, clusters, 0.15, rangeLen, qrng))
	})
}

func TestHybridCompare_FPR_Uniform(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "uniform", n, func() []uint64 {
		rng := rand.New(rand.NewSource(42))
		return generateUniformKeys(n, rng)
	})
	runHybridCompareFPR(t, "uniform", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateUniformQueries(queryCount, rangeLen, qrng)
	})
}

func TestHybridCompare_FPR_Spread(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "spread", n, func() []uint64 {
		return generateSpreadKeys(n)
	})
	runHybridCompareFPR(t, "spread", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateUniformQueries(queryCount, rangeLen, qrng)
	})
}

func TestHybridCompare_FPR_Zipfian(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
		nPrefixes  = 100
	)
	keysPath := fmt.Sprintf("../bench/synthetic_data/zipfian_%d.bin", n)
	prefixesPath := fmt.Sprintf("../bench/synthetic_data/zipfian_%d_prefixes.bin", n)

	os.MkdirAll("../bench/synthetic_data", 0755)

	var keys, prefixes []uint64
	cachedKeys, keyErr := loadSyntheticKeys(keysPath)
	cachedPrefixes, prefixErr := loadSyntheticKeys(prefixesPath)

	if keyErr == nil && prefixErr == nil {
		keys = cachedKeys
		prefixes = cachedPrefixes
	} else {
		rng := rand.New(rand.NewSource(77))
		keys, prefixes = generateZipfianKeys(n, nPrefixes, rng)
		saveSyntheticKeys(keysPath, keys)
		saveSyntheticKeys(prefixesPath, prefixes)
	}

	runHybridCompareFPR(t, "zipfian", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateZipfianQueries(queryCount, prefixes, rangeLen, qrng)
	})
}

func TestHybridCompare_FPR_Temporal(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "temporal", n, func() []uint64 {
		rng := rand.New(rand.NewSource(55))
		return generateTemporalKeys(n, rng)
	})
	runHybridCompareFPR(t, "temporal", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateTemporalQueries(queryCount, keys, rangeLen, qrng)
	})
}

// ---- FPR benchmarks for SOSD distributions ----

func TestHybridCompare_FPR_SOSD_Facebook(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	path := sosdPath("fb_200M_uint64")
	keys, err := loadSOSDUint64(path, n)
	if err != nil {
		t.Skipf("SOSD fb_200M_uint64 not available: %v", err)
	}
	keys = mask60Keys(keys)

	runHybridCompareFPR(t, "sosd_fb", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestHybridCompare_FPR_SOSD_Wiki(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	path := sosdPath("wiki_ts_200M_uint64")
	keys, err := loadSOSDUint64(path, n)
	if err != nil {
		t.Skipf("SOSD wiki_ts_200M_uint64 not available: %v", err)
	}
	keys = mask60Keys(keys)

	runHybridCompareFPR(t, "sosd_wiki", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestHybridCompare_FPR_SOSD_OSM(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	path := sosdPath("osm_cellids_800M_uint64")
	keys, err := loadSOSDUint64(path, n)
	if err != nil {
		t.Skipf("SOSD osm_cellids_800M_uint64 not available: %v", err)
	}
	keys = mask60Keys(keys)

	runHybridCompareFPR(t, "sosd_osm", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestHybridCompare_FPR_SOSD_Books(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	path := sosdPath("books_200M_uint32")
	keys, err := loadSOSDUint32(path, n)
	if err != nil {
		t.Skipf("SOSD books_200M_uint32 not available: %v", err)
	}
	keys = mask60Keys(keys)

	runHybridCompareFPR(t, "sosd_books", keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

// ---- Build time benchmarks for synthetic distributions ----

var hybridBuildNValues = []int{1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20}

func TestHybridCompare_BuildTime_Clustered(t *testing.T) {
	runHybridCompareBuildTime(t, "clustered", hybridBuildNValues, func(n int) []uint64 {
		rng := rand.New(rand.NewSource(99))
		raw, _ := testutils.GenerateClusterDistribution(n, 5, 0.15, rng)
		keys := mask60Keys(raw)
		sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
		return keys
	})
}

func TestHybridCompare_BuildTime_Uniform(t *testing.T) {
	runHybridCompareBuildTime(t, "uniform", hybridBuildNValues, func(n int) []uint64 {
		rng := rand.New(rand.NewSource(42))
		return generateUniformKeys(n, rng)
	})
}

func TestHybridCompare_BuildTime_Spread(t *testing.T) {
	runHybridCompareBuildTime(t, "spread", hybridBuildNValues, func(n int) []uint64 {
		return generateSpreadKeys(n)
	})
}

func TestHybridCompare_BuildTime_Zipfian(t *testing.T) {
	runHybridCompareBuildTime(t, "zipfian", hybridBuildNValues, func(n int) []uint64 {
		rng := rand.New(rand.NewSource(77))
		keys, _ := generateZipfianKeys(n, 100, rng)
		return keys
	})
}

func TestHybridCompare_BuildTime_Temporal(t *testing.T) {
	runHybridCompareBuildTime(t, "temporal", hybridBuildNValues, func(n int) []uint64 {
		rng := rand.New(rand.NewSource(55))
		return generateTemporalKeys(n, rng)
	})
}

// ---- Build time benchmarks for SOSD distributions ----

func sosdBuildKeys(path string, n int, uint32Keys bool) ([]uint64, error) {
	if uint32Keys {
		return loadSOSDUint32(path, n)
	}
	return loadSOSDUint64(path, n)
}

func TestHybridCompare_BuildTime_SOSD_Facebook(t *testing.T) {
	path := sosdPath("fb_200M_uint64")
	runHybridCompareBuildTime(t, "sosd_fb", hybridBuildNValues, func(n int) []uint64 {
		keys, err := loadSOSDUint64(path, n)
		if err != nil {
			t.Skipf("SOSD fb_200M_uint64 not available: %v", err)
			return nil
		}
		return mask60Keys(keys)
	})
}

func TestHybridCompare_BuildTime_SOSD_Wiki(t *testing.T) {
	path := sosdPath("wiki_ts_200M_uint64")
	runHybridCompareBuildTime(t, "sosd_wiki", hybridBuildNValues, func(n int) []uint64 {
		keys, err := loadSOSDUint64(path, n)
		if err != nil {
			t.Skipf("SOSD wiki_ts_200M_uint64 not available: %v", err)
			return nil
		}
		return mask60Keys(keys)
	})
}

func TestHybridCompare_BuildTime_SOSD_OSM(t *testing.T) {
	path := sosdPath("osm_cellids_800M_uint64")
	runHybridCompareBuildTime(t, "sosd_osm", hybridBuildNValues, func(n int) []uint64 {
		keys, err := loadSOSDUint64(path, n)
		if err != nil {
			t.Skipf("SOSD osm_cellids_800M_uint64 not available: %v", err)
			return nil
		}
		return mask60Keys(keys)
	})
}

func TestHybridCompare_BuildTime_SOSD_Books(t *testing.T) {
	path := sosdPath("books_200M_uint32")
	runHybridCompareBuildTime(t, "sosd_books", hybridBuildNValues, func(n int) []uint64 {
		keys, err := loadSOSDUint32(path, n)
		if err != nil {
			t.Skipf("SOSD books_200M_uint32 not available: %v", err)
			return nil
		}
		return mask60Keys(keys)
	})
}
