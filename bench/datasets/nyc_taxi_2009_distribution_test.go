//go:build heavy

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

// TestNYCTaxi2009Year_Distribution aggregates every yellow_tripdata_2009-*.parquet
// present on disk and analyses its pickup-timestamp gap structure. The intent
// is to surface daily/weekly periodicity that a single month cannot show.
func TestNYCTaxi2009Year_Distribution(t *testing.T) {
	matches, _ := filepath.Glob(nycTaxiPath("yellow_tripdata_2009-*.parquet"))
	if len(matches) == 0 {
		t.Skip("no 2009 yellow_tripdata files (run download.sh)")
	}
	sort.Strings(matches)
	t.Logf("aggregating %d monthly files", len(matches))

	r := &datasets.NYCTaxiPickupReader{
		Files: matches,
		Label: "nyc_yellow_2009",
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
	fmt.Printf("\n=== nyc_yellow_2009 (aggregate of %d files) ===\n", len(matches))
	fmt.Printf("n     = %d unique pickup timestamps (s-precision)\n", n)
	fmt.Printf("range = [%s, %s]\n",
		time.Unix(0, int64(keys[0])).UTC().Format(time.RFC3339),
		time.Unix(0, int64(keys[n-1])).UTC().Format(time.RFC3339))
	fmt.Printf("span  = %d ns  (%.2f days)\n",
		span, float64(span)/86_400e9)
	fmt.Printf("mean gap = %.0f ns  (%.3fs)\n", meanGap, meanGap/1e9)

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
	fmt.Printf("\ngap percentiles:\n")
	for _, p := range []float64{0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999} {
		g := pct(p)
		fmt.Printf("  p%-8s = %12d ns  (%.3fs, %.2fx mean)\n",
			fmtPct(p), g, float64(g)/1e9, float64(g)/meanGap)
	}
	fmt.Printf("  max       = %12d ns  (%.3fs, %.2fx mean)\n",
		sortedGaps[len(sortedGaps)-1],
		float64(sortedGaps[len(sortedGaps)-1])/1e9,
		float64(sortedGaps[len(sortedGaps)-1])/meanGap)

	fmt.Printf("\ncluster boundaries (gap thresholds in absolute units):\n")
	for _, label := range []struct {
		name string
		ns   uint64
	}{
		{"≥ 5s", 5e9},
		{"≥ 10s", 10e9},
		{"≥ 30s", 30e9},
		{"≥ 60s", 60e9},
		{"≥ 5min", 300e9},
		{"≥ 30min", 1800e9},
		{"≥ 1h", 3600e9},
	} {
		cnt := 0
		for _, g := range gaps {
			if g >= label.ns {
				cnt++
			}
		}
		fmt.Printf("  gaps %-9s : %6d  (=> %d clusters)\n",
			label.name, cnt, cnt+1)
	}

	// Find the top-10 largest gaps and print them with surrounding timestamps,
	// so we can see whether they correspond to nighttime quiet, holidays, etc.
	type idxGap struct {
		idx int
		g   uint64
	}
	all := make([]idxGap, len(gaps))
	for i, g := range gaps {
		all[i] = idxGap{i, g}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].g > all[j].g })
	fmt.Printf("\ntop-10 largest gaps:\n")
	for i := 0; i < 10 && i < len(all); i++ {
		ig := all[i]
		fmt.Printf("  #%2d  gap=%9.2fs  before=%s  after=%s\n",
			i+1, float64(ig.g)/1e9,
			time.Unix(0, int64(keys[ig.idx])).UTC().Format(time.RFC3339),
			time.Unix(0, int64(keys[ig.idx+1])).UTC().Format(time.RFC3339))
	}

	histDir := filepath.Join(filepath.Dir(matches[0]), "..", "bench_results",
		"plots", "distributions")
	os.MkdirAll(histDir, 0755)
	histPath := filepath.Join(histDir, "hist_nyc_yellow_2009_year.svg")
	err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
		Title: fmt.Sprintf("Key Density — nyc_yellow_2009 aggregate (n=%d, %d files, 1000 bins)",
			n, len(matches)),
		XLabel: "Normalized Pickup-Timestamp Position (2009)",
		YLabel: "Relative Density",
		YScale: testutils.Log10,
		XMax:   25,
	}, []testutils.SeriesData{{
		Name:   "nyc_yellow_2009",
		Color:  "#f4b400",
		Marker: "none",
		Points: histogramUint64(keys, 1000),
	}}, histPath)
	if err != nil {
		t.Fatalf("histogram SVG failed: %v", err)
	}
	fmt.Printf("\nHistogram written to %s\n", histPath)
}
