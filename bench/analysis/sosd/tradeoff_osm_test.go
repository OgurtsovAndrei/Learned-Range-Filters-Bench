package sosd_test

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"testing"

	"Thesis-bench-industry/bench/datasets"
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/emptiness/approx/are_trunc"
	"Thesis/emptiness/exact"
	"Thesis/testutils"
)

func osmDataPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "..", "sosd_data", name)
}

func TestTradeoff_OSM_TruncVsSoda(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 15
		nRuns      = 3
		rangeLen   = uint64(128)
	)

	reader := datasets.SOSDReader{
		Path:    osmDataPath("osm_cellids_800M_uint64"),
		Label:   "OSM",
		KeyType: datasets.SOSDUint64,
		MaxKeys: n,
	}
	keys, err := reader.Keys()
	if err != nil {
		t.Skipf("OSM dataset not available: %v", err)
	}

	seeds := []int64{12345, 54321, 99999}
	querySets := make([][][2]uint64, nRuns)
	for r := 0; r < nRuns; r++ {
		rng := rand.New(rand.NewSource(seeds[r]))
		querySets[r] = epsFixSmartQueries(keys, queryCount, rangeLen, rng)
		if len(querySets[r]) == 0 {
			t.Fatal("no queries generated")
		}
	}

	kGridTrunc := []uint32{
		20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
	}
	kGridSoda := []uint32{
		20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
	}

	theoretical := &testutils.SeriesData{Name: "Theoretical", Color: "#ef4444", Dashed: true, Marker: "none"}
	truncSeries := &testutils.SeriesData{Name: "Truncation", Color: "#9b59b6", Marker: "triangle"}
	sodaSeries := &testutils.SeriesData{Name: "SODA", Color: "#4dd88a", Marker: "diamond"}

	// Extend theoretical curve down to small K so it covers the full BPK range.
	for K := uint32(7); K <= 52; K++ {
		thEps := float64(rangeLen) / math.Exp2(float64(K))
		if thEps >= 1e-6 && thEps <= 1 {
			theoretical.Points = append(theoretical.Points,
				testutils.Point{X: float64(K), Y: thEps})
		}
	}

	type task struct {
		series    string
		K         uint32
		bpk       float64
		isEmptyFn func(a, b uint64) bool
	}
	var tasks []task

	for _, K := range kGridTrunc {
		if f, err := are_trunc.NewTruncAREFromKWithBackend(keys, 64, K, exact.VariantOneD); err == nil {
			bpk := float64(f.SizeInBits()) / float64(n)
			f := f
			tasks = append(tasks, task{"Truncation", K, bpk,
				func(a, b uint64) bool { return f.IsEmpty(a, b) }})
		}
	}
	for _, K := range kGridSoda {
		if f, err := are_soda_hash.NewSodaAREFromKWithBackend(keys, K, int64(rangeLen), exact.VariantOneD); err == nil {
			bpk := float64(f.SizeInBits()) / float64(n)
			f := f
			tasks = append(tasks, task{"SODA", K, bpk,
				func(a, b uint64) bool { return f.IsEmpty(a, b) }})
		}
	}

	results := make([]testutils.Point, len(tasks))
	seriesNames := make([]string, len(tasks))
	var wg sync.WaitGroup
	for i, tk := range tasks {
		i, tk := i, tk
		wg.Add(1)
		go func() {
			defer wg.Done()
			sum := 0.0
			for _, qs := range querySets {
				sum += testutils.MeasureFPR(keys, qs, tk.isEmptyFn)
			}
			results[i] = testutils.Point{X: tk.bpk, Y: sum / float64(nRuns)}
			seriesNames[i] = tk.series
		}()
	}
	wg.Wait()

	for i, pt := range results {
		switch seriesNames[i] {
		case "Truncation":
			truncSeries.Points = append(truncSeries.Points, pt)
		case "SODA":
			sodaSeries.Points = append(sodaSeries.Points, pt)
		}
	}

	// Sort points by BPK for clean lines.
	sortPoints := func(s *testutils.SeriesData) {
		pts := s.Points
		sort.Slice(pts, func(i, j int) bool { return pts[i].X < pts[j].X })
	}
	sortPoints(truncSeries)
	sortPoints(sodaSeries)

	svgPath := "tradeoff_osm_L128.svg"
	err = testutils.GenerateTradeoffSVG(
		fmt.Sprintf("FPR vs BPK — OSM (n=%d, L=%d)", n, rangeLen),
		"Bits per Key (BPK)",
		"False Positive Rate (FPR)",
		[]testutils.SeriesData{*theoretical, *truncSeries, *sodaSeries},
		svgPath,
		1.0/float64(queryCount*nRuns),
	)
	if err != nil {
		t.Errorf("SVG generation failed: %v", err)
	} else {
		fmt.Printf("SVG written to %s\n", svgPath)
	}
}


// inGapQueries generates empty range queries placed uniformly inside random gaps.
func inGapQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	n := len(keys)
	type gap struct{ lo, hi uint64 }
	var gaps []gap
	for i := 0; i < n-1; i++ {
		if keys[i+1]-keys[i] > rangeLen {
			gaps = append(gaps, gap{keys[i] + 1, keys[i+1] - rangeLen})
		}
	}
	if len(gaps) == 0 {
		return nil
	}
	queries := make([][2]uint64, 0, count)
	for len(queries) < count {
		g := gaps[rng.Intn(len(gaps))]
		span := g.hi - g.lo + 1
		a := g.lo + epsFixRandUint64Below(rng, span)
		b := a + rangeLen - 1
		if b <= g.hi+rangeLen-1 {
			queries = append(queries, [2]uint64{a, b})
		}
	}
	return queries
}

func TestTradeoff_OSM_TruncVsSoda_InGap(t *testing.T) {
	const (
		n          = 1 << 20
		queryCount = 1 << 15
		nRuns      = 3
		rangeLen   = uint64(128)
	)

	reader := datasets.SOSDReader{
		Path:    osmDataPath("osm_cellids_800M_uint64"),
		Label:   "OSM",
		KeyType: datasets.SOSDUint64,
		MaxKeys: n,
	}
	keys, err := reader.Keys()
	if err != nil {
		t.Skipf("OSM dataset not available: %v", err)
	}

	seeds := []int64{12345, 54321, 99999}
	querySets := make([][][2]uint64, nRuns)
	for r := 0; r < nRuns; r++ {
		rng := rand.New(rand.NewSource(seeds[r]))
		querySets[r] = inGapQueries(keys, queryCount, rangeLen, rng)
		if len(querySets[r]) == 0 {
			t.Fatal("no queries generated")
		}
	}

	kGridTrunc := []uint32{20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50}
	kGridSoda  := []uint32{20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52}

	theoretical := &testutils.SeriesData{Name: "Theoretical", Color: "#ef4444", Dashed: true, Marker: "none"}
	truncSeries := &testutils.SeriesData{Name: "Truncation", Color: "#9b59b6", Marker: "triangle"}
	sodaSeries  := &testutils.SeriesData{Name: "SODA", Color: "#4dd88a", Marker: "diamond"}

	for K := uint32(7); K <= 52; K++ {
		thEps := float64(rangeLen) / math.Exp2(float64(K))
		if thEps >= 1e-6 && thEps <= 1 {
			theoretical.Points = append(theoretical.Points, testutils.Point{X: float64(K), Y: thEps})
		}
	}

	type task struct {
		series    string
		bpk       float64
		isEmptyFn func(a, b uint64) bool
	}
	var tasks []task

	for _, K := range kGridTrunc {
		if f, ferr := are_trunc.NewTruncAREFromKWithBackend(keys, 64, K, exact.VariantOneD); ferr == nil {
			bpk := float64(f.SizeInBits()) / float64(n)
			f := f
			tasks = append(tasks, task{"Truncation", bpk, func(a, b uint64) bool { return f.IsEmpty(a, b) }})
		}
	}
	for _, K := range kGridSoda {
		if f, ferr := are_soda_hash.NewSodaAREFromKWithBackend(keys, K, int64(rangeLen), exact.VariantOneD); ferr == nil {
			bpk := float64(f.SizeInBits()) / float64(n)
			f := f
			tasks = append(tasks, task{"SODA", bpk, func(a, b uint64) bool { return f.IsEmpty(a, b) }})
		}
	}

	results     := make([]testutils.Point, len(tasks))
	seriesNames := make([]string, len(tasks))
	var wg sync.WaitGroup
	for i, tk := range tasks {
		i, tk := i, tk
		wg.Add(1)
		go func() {
			defer wg.Done()
			sum := 0.0
			for _, qs := range querySets {
				sum += testutils.MeasureFPR(keys, qs, tk.isEmptyFn)
			}
			results[i]     = testutils.Point{X: tk.bpk, Y: sum / float64(nRuns)}
			seriesNames[i] = tk.series
		}()
	}
	wg.Wait()

	for i, pt := range results {
		switch seriesNames[i] {
		case "Truncation":
			truncSeries.Points = append(truncSeries.Points, pt)
		case "SODA":
			sodaSeries.Points = append(sodaSeries.Points, pt)
		}
	}

	sortPts := func(s *testutils.SeriesData) {
		sort.Slice(s.Points, func(i, j int) bool { return s.Points[i].X < s.Points[j].X })
	}
	sortPts(truncSeries)
	sortPts(sodaSeries)

	svgPath := "tradeoff_osm_ingap_L128.svg"
	if err = testutils.GenerateTradeoffSVG(
		fmt.Sprintf("FPR vs BPK — OSM in-gap (n=%d, L=%d)", n, rangeLen),
		"Bits per Key (BPK)", "False Positive Rate (FPR)",
		[]testutils.SeriesData{*theoretical, *truncSeries, *sodaSeries},
		svgPath, 1.0/float64(queryCount*nRuns),
	); err != nil {
		t.Errorf("SVG generation failed: %v", err)
	} else {
		fmt.Printf("SVG written to %s\n", svgPath)
	}
}
