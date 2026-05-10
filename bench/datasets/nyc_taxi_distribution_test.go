package datasets_test

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"Thesis-bench-industry/bench/datasets"
	"Thesis/testutils"
)

// TestNYCTaxi2009Jan_Distribution prints gap-percentile statistics, cluster
// counts at several gap thresholds, and writes a 1000-bin SVG histogram for
// the pickup-timestamp keys of yellow_tripdata_2009-01.parquet.
func TestNYCTaxi2009Jan_Distribution(t *testing.T) {
	path := nycTaxiPath("yellow_tripdata_2009-01.parquet")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("file not available: %v", err)
	}
	r := &datasets.NYCTaxiPickupReader{
		Files: []string{path},
		Label: "nyc_yellow_2009_01",
	}
	keys, err := r.Keys()
	if err != nil {
		t.Fatalf("Keys: %v", err)
	}
	n := len(keys)
	if n < 2 {
		t.Fatalf("not enough keys: %d", n)
	}

	span := keys[n-1] - keys[0]
	meanGap := float64(span) / float64(n-1)
	fmt.Printf("\n=== nyc_yellow_2009_01 ===\n")
	fmt.Printf("n = %d unique pickup timestamps (s-precision)\n", n)
	fmt.Printf("range = [%s, %s]\n",
		time.Unix(0, int64(keys[0])).UTC().Format(time.RFC3339),
		time.Unix(0, int64(keys[n-1])).UTC().Format(time.RFC3339))
	fmt.Printf("span  = %d ns  (%.3f days)\n",
		span, float64(span)/86_400e9)
	fmt.Printf("mean gap = %.0f ns  (%.3f s)\n", meanGap, meanGap/1e9)

	gaps := make([]uint64, n-1)
	for i := 1; i < n; i++ {
		gaps[i-1] = keys[i] - keys[i-1]
	}
	sortedGaps := make([]uint64, len(gaps))
	copy(sortedGaps, gaps)
	sort.Slice(sortedGaps, func(i, j int) bool { return sortedGaps[i] < sortedGaps[j] })
	pct := func(p float64) uint64 {
		idx := int(p * float64(len(sortedGaps)-1))
		return sortedGaps[idx]
	}

	fmt.Printf("\ngap percentiles (ns / x mean):\n")
	for _, p := range []float64{0.5, 0.9, 0.99, 0.999, 0.9999} {
		g := pct(p)
		fmt.Printf("  p%-6s = %12d  (%.3fs, %.2fx mean)\n",
			fmtPct(p), g, float64(g)/1e9, float64(g)/meanGap)
	}
	fmt.Printf("  max     = %12d  (%.3fs, %.2fx mean)\n",
		sortedGaps[len(sortedGaps)-1],
		float64(sortedGaps[len(sortedGaps)-1])/1e9,
		float64(sortedGaps[len(sortedGaps)-1])/meanGap)

	fmt.Printf("\ncluster boundaries by threshold:\n")
	for _, k := range []float64{10, 100, 1000, 10000} {
		thr := uint64(k * meanGap)
		cnt := 0
		for _, g := range gaps {
			if g >= thr {
				cnt++
			}
		}
		fmt.Printf("  gaps ≥ %5.0fx mean (≥ %.1fs): %d  (=> %d clusters)\n",
			k, float64(thr)/1e9, cnt, cnt+1)
	}

	histDir := filepath.Join(filepath.Dir(path), "..", "bench_results",
		"plots", "distributions")
	os.MkdirAll(histDir, 0755)
	histPath := filepath.Join(histDir, "hist_nyc_yellow_2009_01.svg")
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title: fmt.Sprintf("Key Density — nyc_yellow_2009_01 (n=%d, 1000 bins)", n),
		XLabel: "Normalized Pickup-Timestamp Position (Jan 2009)",
		YLabel: "Relative Density",
		YScale: testutils.Log10,
		XMax:   25,
	}, []testutils.SeriesData{{
		Name:   "nyc_yellow_2009_01",
		Color:  "#f4b400",
		Marker: "none",
		Points: histogramUint64(keys, 1000),
	}}, histPath)
	if err != nil {
		t.Fatalf("histogram SVG failed: %v", err)
	}
	fmt.Printf("\nHistogram written to %s\n", histPath)
}

func fmtPct(p float64) string {
	switch p {
	case 0.5:
		return "50"
	case 0.9:
		return "90"
	case 0.99:
		return "99"
	case 0.999:
		return "99.9"
	case 0.9999:
		return "99.99"
	}
	return fmt.Sprintf("%.4g", p*100)
}

// histogramUint64 mirrors bench.histogram() but accepts already-sorted keys.
func histogramUint64(keys []uint64, nBins int) []testutils.Point {
	n := len(keys)
	minK, maxK := float64(keys[0]), float64(keys[n-1])
	span := maxK - minK
	if span == 0 {
		span = 1
	}
	counts := make([]int, nBins)
	for _, k := range keys {
		bin := int((float64(k) - minK) / span * float64(nBins))
		if bin >= nBins {
			bin = nBins - 1
		}
		counts[bin]++
	}
	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}
	pts := make([]testutils.Point, nBins)
	for i, c := range counts {
		pts[i] = testutils.Point{
			X: float64(i) / float64(nBins),
			Y: float64(c) / float64(maxCount),
		}
	}
	return pts
}
