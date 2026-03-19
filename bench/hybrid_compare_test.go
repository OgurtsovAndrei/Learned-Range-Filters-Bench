package bench_test

import (
	"Thesis/bits"
	"Thesis/emptiness/are_dp_scan"
	"Thesis/emptiness/are_greedy_scan"
	"Thesis/emptiness/are_hybrid"
	"Thesis/emptiness/are_hybrid_scan"
	"Thesis/testutils"
	"encoding/json"
	"fmt"
	"math"
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

// dpMaxNFPR caps the N at which DP-Optimal is included in FPR benchmarks.
// segmentDP is O(n²); at n=262144 with 24 K values this would take hours.
const dpMaxNFPR = 1 << 14 // 16384

// dpMaxNBuild caps the N at which DP-Optimal is included in build-time benchmarks.
const dpMaxNBuild = 1 << 16 // 65536

type hybridPoint struct {
	hybrid, scan, greedyRaw, greedyMerge, dp testutils.Point
	dpValid                                   bool
}

func hybridMeasureK(
	keysBS []bits.BitString,
	keys []uint64,
	queryFunc func(uint64, int64) [][2]uint64,
	rangeLen uint64,
	K uint32,
	seeds []int64,
	includDP bool,
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

	if fg, err := are_greedy_scan.NewGreedyScanAREFromKRaw(keysBS, rangeLen, K); err == nil {
		bpk := float64(fg.SizeInBits()) / float64(len(keys))
		tasks = append(tasks, task{"Greedy-raw", bpk, func(a, b uint64) bool {
			return fg.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
		}})
	}

	if fg, err := are_greedy_scan.NewGreedyScanAREFromK(keysBS, rangeLen, K); err == nil {
		bpk := float64(fg.SizeInBits()) / float64(len(keys))
		tasks = append(tasks, task{"Greedy+Merge", bpk, func(a, b uint64) bool {
			return fg.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
		}})
	}

	if includDP {
		if fd, err := are_dp_scan.NewDPScanAREFromK(keysBS, rangeLen, K); err == nil {
			bpk := float64(fd.SizeInBits()) / float64(len(keys))
			tasks = append(tasks, task{"DP-Optimal", bpk, func(a, b uint64) bool {
				return fd.IsEmpty(testutils.TrieBS(a), testutils.TrieBS(b))
			}})
		}
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
		p := testutils.Point{X: tk.bpk, Y: results[i]}
		switch tk.name {
		case "Hybrid":
			hp.hybrid = p
		case "Scan-ARE":
			hp.scan = p
		case "Greedy-raw":
			hp.greedyRaw = p
		case "Greedy+Merge":
			hp.greedyMerge = p
		case "DP-Optimal":
			hp.dp = p
			hp.dpValid = true
		}
	}
	return hp
}

// filterBPK returns a copy of series with points where X > maxBPK removed.
func filterBPK(s testutils.SeriesData, maxBPK float64) testutils.SeriesData {
	out := s
	out.Points = nil
	for _, p := range s.Points {
		if p.X <= maxBPK {
			out.Points = append(out.Points, p)
		}
	}
	return out
}

// hybridSaveJSON writes the FPR series data for a single (distName, rangeLen, n) combination.
func hybridSaveJSON(t *testing.T, distName string, n int, rangeLen uint64, allSeries []*testutils.SeriesData) {
	t.Helper()
	dir := fmt.Sprintf("../bench_results/data/hybrid_compare/N%d/%s", n, distName)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Errorf("mkdir JSON dir: %v", err)
		return
	}
	path := fmt.Sprintf("%s/L%d.json", dir, rangeLen)
	seriesMap := make(map[string]*testutils.SeriesData, len(allSeries))
	for _, s := range allSeries {
		seriesMap[s.Name] = s
	}
	if err := saveSeriesData(path, seriesMap); err != nil {
		t.Errorf("save JSON: %v", err)
	}
}

// hybridSaveBuildTimeJSON writes the build-time series data for a single distribution.
func hybridSaveBuildTimeJSON(t *testing.T, distName string, allSeries []*testutils.SeriesData) {
	t.Helper()
	dir := "../bench_results/data/hybrid_compare/build_time"
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Errorf("mkdir JSON dir: %v", err)
		return
	}
	path := fmt.Sprintf("%s/%s.json", dir, distName)
	seriesMap := make(map[string]*testutils.SeriesData, len(allSeries))
	for _, s := range allSeries {
		seriesMap[s.Name] = s
	}
	if err := saveSeriesData(path, seriesMap); err != nil {
		t.Errorf("save JSON: %v", err)
	}
}

// hybridJSONToSavedSeries converts a slice of SeriesData pointers to []savedSeries for JSON.
func hybridJSONToSavedSeries(allSeries []*testutils.SeriesData) []savedSeries {
	var out []savedSeries
	for _, s := range allSeries {
		if len(s.Points) > 0 {
			pts := make([]testutils.Point, len(s.Points))
			copy(pts, s.Points)
			out = append(out, savedSeries{Name: s.Name, Points: pts})
		}
	}
	return out
}

// saveHybridJSON writes []savedSeries as JSON to path, creating parent dirs.
func saveHybridJSON(path string, data []savedSeries) error {
	b, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

func runHybridCompareFPR(t *testing.T, distName string, n int, keys []uint64, queryFunc func(uint64, int64) [][2]uint64) {
	t.Helper()

	rangeLens := []uint64{16, 128, 1024}
	kGrid := []uint32{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48, 52, 56}
	seeds := []int64{12345, 54321, 99999}

	const maxBPKForSVG = 35.0

	keysBS := toTrieBS(keys)
	includDP := len(keys) <= dpMaxNFPR

	outDir := fmt.Sprintf("../bench_results/plots/hybrid_compare/N%d/%s", n, distName)
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Errorf("mkdir: %v", err)
		return
	}

	for _, rangeLen := range rangeLens {
		rangeLen := rangeLen
		t.Run(fmt.Sprintf("L=%d", rangeLen), func(t *testing.T) {
			hybridSeries := &testutils.SeriesData{Name: "Hybrid", Color: "#ff6b6b", Marker: "circle"}
			scanSeries := &testutils.SeriesData{Name: "Scan-ARE", Color: "#06b6d4", Marker: "square"}
			greedyRawSeries := &testutils.SeriesData{Name: "Greedy-raw", Color: "#a3e635", Marker: "triangle"}
			greedyMergeSeries := &testutils.SeriesData{Name: "Greedy+Merge", Color: "#22c55e", Marker: "diamond"}
			dpSeries := &testutils.SeriesData{Name: "DP-Optimal", Color: "#8b5cf6", Marker: "star"}

			fmt.Printf("\n=== Hybrid Compare FPR — %s (n=%d, L=%d) ===\n", distName, len(keys), rangeLen)
			if !includDP {
				fmt.Printf("    (DP-Optimal skipped: n=%d > dpMaxNFPR=%d)\n", len(keys), dpMaxNFPR)
			}
			fmt.Printf("%-12s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s\n",
				"K",
				"Hyb-BPK", "Hyb-FPR",
				"Scan-BPK", "Scan-FPR",
				"GrRaw-BPK", "GrRaw-FPR",
				"GrMrg-BPK", "GrMrg-FPR",
				"DP-BPK", "DP-FPR")
			fmt.Println("---------------------------------------------------------------------------------------------------------------------------")

			for _, K := range kGrid {
				hp := hybridMeasureK(keysBS, keys, queryFunc, rangeLen, K, seeds, includDP)
				hybridSeries.Points = append(hybridSeries.Points, hp.hybrid)
				scanSeries.Points = append(scanSeries.Points, hp.scan)
				greedyRawSeries.Points = append(greedyRawSeries.Points, hp.greedyRaw)
				greedyMergeSeries.Points = append(greedyMergeSeries.Points, hp.greedyMerge)
				if hp.dpValid {
					dpSeries.Points = append(dpSeries.Points, hp.dp)
				}

				dpBPK, dpFPR := 0.0, 0.0
				if hp.dpValid {
					dpBPK, dpFPR = hp.dp.X, hp.dp.Y
				}
				fmt.Printf("%-12d | %10.2f | %10.6f | %10.2f | %10.6f | %10.2f | %10.6f | %10.2f | %10.6f | %10.2f | %10.6f\n",
					K,
					hp.hybrid.X, hp.hybrid.Y,
					hp.scan.X, hp.scan.Y,
					hp.greedyRaw.X, hp.greedyRaw.Y,
					hp.greedyMerge.X, hp.greedyMerge.Y,
					dpBPK, dpFPR)
			}

			// Save full data (including BPK > 35 points) to JSON.
			allSeries := []*testutils.SeriesData{hybridSeries, scanSeries, greedyRawSeries, greedyMergeSeries}
			if len(dpSeries.Points) > 0 {
				allSeries = append(allSeries, dpSeries)
			}
			jsonDir := fmt.Sprintf("../bench_results/data/hybrid_compare/N%d/%s", n, distName)
			if err := os.MkdirAll(jsonDir, 0755); err != nil {
				t.Errorf("mkdir JSON dir: %v", err)
			} else {
				jsonPath := fmt.Sprintf("%s/L%d.json", jsonDir, rangeLen)
				seriesMap := make(map[string]*testutils.SeriesData, len(allSeries))
				for _, s := range allSeries {
					seriesMap[s.Name] = s
				}
				if err := saveSeriesData(jsonPath, seriesMap); err != nil {
					t.Errorf("save JSON: %v", err)
				}
			}

			// Filter BPK > 35 for SVG to avoid scale distortion from Greedy-raw at low K.
			svgSeries := make([]testutils.SeriesData, 0, len(allSeries))
			for _, s := range allSeries {
				filtered := filterBPK(*s, maxBPKForSVG)
				if len(filtered.Points) > 0 {
					svgSeries = append(svgSeries, filtered)
				}
			}

			svgPath := fmt.Sprintf("%s/L%d.svg", outDir, rangeLen)
			err := testutils.GenerateTradeoffSVG(
				fmt.Sprintf("Hybrid ARE Variants — %s (n=%d, L=%d)", distName, len(keys), rangeLen),
				"Bits per Key (BPK)",
				"False Positive Rate (FPR)",
				svgSeries,
				svgPath,
			)
			if err != nil {
				t.Errorf("SVG generation failed: %v", err)
			} else {
				fmt.Printf("SVG written to %s\n", svgPath)
			}
		})
	}
}

// greedyKFromEpsilon replicates the K derivation used inside NewGreedyScanARE.
func greedyKFromEpsilon(n int, rangeLen uint64, epsilon float64) uint32 {
	rTarget := float64(n) * float64(rangeLen+1) / epsilon
	K := uint32(math.Ceil(math.Log2(rTarget)))
	if K > 64 {
		K = 64
	}
	return K
}

func runHybridCompareBuildTime(t *testing.T, distName string, nValues []int, genKeys func(n int) []uint64) {
	t.Helper()

	const (
		rangeLen = uint64(128)
		epsilon  = 0.01
	)

	hybridSeries := &testutils.SeriesData{Name: "Hybrid", Color: "#ff6b6b", Marker: "circle"}
	scanSeries := &testutils.SeriesData{Name: "Scan-ARE", Color: "#06b6d4", Marker: "square"}
	greedyRawSeries := &testutils.SeriesData{Name: "Greedy-raw", Color: "#a3e635", Marker: "triangle"}
	greedyMergeSeries := &testutils.SeriesData{Name: "Greedy+Merge", Color: "#22c55e", Marker: "diamond"}
	dpSeries := &testutils.SeriesData{Name: "DP-Optimal", Color: "#8b5cf6", Marker: "star"}

	fmt.Printf("\n=== Hybrid Compare Build Time — %s ===\n", distName)
	fmt.Printf("%-10s | %12s | %12s | %14s | %14s | %12s\n",
		"N", "Hybrid", "Scan-ARE", "Greedy-raw", "Greedy+Merge", "DP-Optimal")
	fmt.Println("--------------------------------------------------------------------------------------------")

	for _, n := range nValues {
		keys := genKeys(n)
		keysBS := toTrieBS(keys)
		K := greedyKFromEpsilon(n, rangeLen, epsilon)

		start := time.Now()
		_, hybErr := are_hybrid.NewHybridARE(keysBS, rangeLen, epsilon)
		hybTime := time.Since(start)

		start = time.Now()
		_, scanErr := are_hybrid_scan.NewHybridScanARE(keysBS, rangeLen, epsilon)
		scanTime := time.Since(start)

		start = time.Now()
		_, greedyRawErr := are_greedy_scan.NewGreedyScanAREFromKRaw(keysBS, rangeLen, K)
		greedyRawTime := time.Since(start)

		start = time.Now()
		_, greedyMergeErr := are_greedy_scan.NewGreedyScanAREFromK(keysBS, rangeLen, K)
		greedyMergeTime := time.Since(start)

		hybNs := float64(hybTime.Nanoseconds()) / float64(n)
		scanNs := float64(scanTime.Nanoseconds()) / float64(n)
		greedyRawNs := float64(greedyRawTime.Nanoseconds()) / float64(n)
		greedyMergeNs := float64(greedyMergeTime.Nanoseconds()) / float64(n)

		if hybErr == nil {
			hybridSeries.Points = append(hybridSeries.Points, testutils.Point{X: float64(n), Y: hybNs})
		}
		if scanErr == nil {
			scanSeries.Points = append(scanSeries.Points, testutils.Point{X: float64(n), Y: scanNs})
		}
		if greedyRawErr == nil {
			greedyRawSeries.Points = append(greedyRawSeries.Points, testutils.Point{X: float64(n), Y: greedyRawNs})
		}
		if greedyMergeErr == nil {
			greedyMergeSeries.Points = append(greedyMergeSeries.Points, testutils.Point{X: float64(n), Y: greedyMergeNs})
		}

		dpNsStr := "skipped"
		if n <= dpMaxNBuild {
			start = time.Now()
			_, dpErr := are_dp_scan.NewDPScanARE(keysBS, rangeLen, epsilon)
			dpTime := time.Since(start)
			dpNs := float64(dpTime.Nanoseconds()) / float64(n)
			if dpErr == nil {
				dpSeries.Points = append(dpSeries.Points, testutils.Point{X: float64(n), Y: dpNs})
				dpNsStr = fmt.Sprintf("%.1f ns", dpNs)
			}
		}

		fmt.Printf("%-10d | %10.1f ns | %10.1f ns | %12.1f ns | %12.1f ns | %s\n",
			n, hybNs, scanNs, greedyRawNs, greedyMergeNs, dpNsStr)
	}

	outDir := "../bench_results/plots/hybrid_compare/build_time"
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Errorf("mkdir: %v", err)
		return
	}

	allSeries := []*testutils.SeriesData{hybridSeries, scanSeries, greedyRawSeries, greedyMergeSeries}
	if len(dpSeries.Points) > 0 {
		allSeries = append(allSeries, dpSeries)
	}

	// Save JSON.
	jsonDir := "../bench_results/data/hybrid_compare/build_time"
	if err := os.MkdirAll(jsonDir, 0755); err != nil {
		t.Errorf("mkdir JSON dir: %v", err)
	} else {
		jsonPath := fmt.Sprintf("%s/%s.json", jsonDir, distName)
		seriesMap := make(map[string]*testutils.SeriesData, len(allSeries))
		for _, s := range allSeries {
			seriesMap[s.Name] = s
		}
		if err := saveSeriesData(jsonPath, seriesMap); err != nil {
			t.Errorf("save JSON: %v", err)
		}
	}

	svgSeries := make([]testutils.SeriesData, 0, len(allSeries))
	for _, s := range allSeries {
		svgSeries = append(svgSeries, *s)
	}

	svgPath := fmt.Sprintf("%s/%s.svg", outDir, distName)
	err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title:  fmt.Sprintf("Build Time per Key — Hybrid ARE Variants (%s)", distName),
		XLabel: "Number of Keys (n)",
		YLabel: "Build Time (ns/key)",
		XScale: testutils.Log10,
		YScale: testutils.Log10,
	}, svgSeries, svgPath)
	if err != nil {
		t.Errorf("SVG generation failed: %v", err)
	} else {
		fmt.Printf("SVG written to %s\n", svgPath)
	}
}

// ---- FPR benchmarks for synthetic distributions (N=1<<18) ----

func TestHybridCompare_FPR_Clustered(t *testing.T) {
	const (
		n          = 1 << 18
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "clustered", n, func() []uint64 {
		rng := rand.New(rand.NewSource(99))
		raw, _ := testutils.GenerateClusterDistribution(n, 5, 0.15, rng)
		return mask60Keys(raw)
	})

	rng := rand.New(rand.NewSource(99))
	_, clusters := testutils.GenerateClusterDistribution(n, 5, 0.15, rng)

	runHybridCompareFPR(t, "clustered", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
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
	runHybridCompareFPR(t, "uniform", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
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
	runHybridCompareFPR(t, "spread", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
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

	runHybridCompareFPR(t, "zipfian", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
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
	runHybridCompareFPR(t, "temporal", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateTemporalQueries(queryCount, keys, rangeLen, qrng)
	})
}

// ---- FPR benchmarks for SOSD distributions (N=1<<18) ----

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

	runHybridCompareFPR(t, "sosd_fb", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
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

	runHybridCompareFPR(t, "sosd_wiki", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
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

	runHybridCompareFPR(t, "sosd_osm", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
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

	runHybridCompareFPR(t, "sosd_books", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

// ---- FPR benchmarks for synthetic distributions (N=1<<20) ----

func TestHybridCompare_FPR_1M_Clustered(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "clustered", n, func() []uint64 {
		rng := rand.New(rand.NewSource(99))
		raw, _ := testutils.GenerateClusterDistribution(n, 5, 0.15, rng)
		return mask60Keys(raw)
	})

	rng := rand.New(rand.NewSource(99))
	_, clusters := testutils.GenerateClusterDistribution(n, 5, 0.15, rng)

	runHybridCompareFPR(t, "clustered", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return mask60Queries(testutils.GenerateClusterQueries(queryCount, clusters, 0.15, rangeLen, qrng))
	})
}

func TestHybridCompare_FPR_1M_Uniform(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "uniform", n, func() []uint64 {
		rng := rand.New(rand.NewSource(42))
		return generateUniformKeys(n, rng)
	})
	runHybridCompareFPR(t, "uniform", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateUniformQueries(queryCount, rangeLen, qrng)
	})
}

func TestHybridCompare_FPR_1M_Spread(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "spread", n, func() []uint64 {
		return generateSpreadKeys(n)
	})
	runHybridCompareFPR(t, "spread", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateUniformQueries(queryCount, rangeLen, qrng)
	})
}

func TestHybridCompare_FPR_1M_Zipfian(t *testing.T) {
	const (
		n          = 1 << 20
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

	runHybridCompareFPR(t, "zipfian", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateZipfianQueries(queryCount, prefixes, rangeLen, qrng)
	})
}

func TestHybridCompare_FPR_1M_Temporal(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
	)
	keys := cacheOrGenerate("../bench/synthetic_data", "temporal", n, func() []uint64 {
		rng := rand.New(rand.NewSource(55))
		return generateTemporalKeys(n, rng)
	})
	runHybridCompareFPR(t, "temporal", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		qrng := rand.New(rand.NewSource(seed))
		return generateTemporalQueries(queryCount, keys, rangeLen, qrng)
	})
}

// ---- FPR benchmarks for SOSD distributions (N=1<<20) ----

func TestHybridCompare_FPR_1M_SOSD_Facebook(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
	)
	path := sosdPath("fb_200M_uint64")
	keys, err := loadSOSDUint64(path, n)
	if err != nil {
		t.Skipf("SOSD fb_200M_uint64 not available: %v", err)
	}
	keys = mask60Keys(keys)

	runHybridCompareFPR(t, "sosd_fb", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestHybridCompare_FPR_1M_SOSD_Wiki(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
	)
	path := sosdPath("wiki_ts_200M_uint64")
	keys, err := loadSOSDUint64(path, n)
	if err != nil {
		t.Skipf("SOSD wiki_ts_200M_uint64 not available: %v", err)
	}
	keys = mask60Keys(keys)

	runHybridCompareFPR(t, "sosd_wiki", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestHybridCompare_FPR_1M_SOSD_OSM(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
	)
	path := sosdPath("osm_cellids_800M_uint64")
	keys, err := loadSOSDUint64(path, n)
	if err != nil {
		t.Skipf("SOSD osm_cellids_800M_uint64 not available: %v", err)
	}
	keys = mask60Keys(keys)

	runHybridCompareFPR(t, "sosd_osm", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
		return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
	})
}

func TestHybridCompare_FPR_1M_SOSD_Books(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 18
	)
	path := sosdPath("books_200M_uint32")
	keys, err := loadSOSDUint32(path, n)
	if err != nil {
		t.Skipf("SOSD books_200M_uint32 not available: %v", err)
	}
	keys = mask60Keys(keys)

	runHybridCompareFPR(t, "sosd_books", n, keys, func(rangeLen uint64, seed int64) [][2]uint64 {
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
