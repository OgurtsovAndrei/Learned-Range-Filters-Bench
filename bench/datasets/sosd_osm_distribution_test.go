//go:build heavy

package datasets_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"Thesis-bench-industry/bench/datasets"
	"Thesis/testutils"
)

// TestSOSD_OSM_Distribution rebuilds 1000-bin histograms for sosd_osm in
// log + linear forms so it can be directly compared to the legacy obsolete
// hist_sosd_osm.svg (which was log-only). On log-y, isolated keys at
// y ≈ 10^-6 falsely look like "structure"; the linear copy reveals whether
// they are dense regions or single-key outliers.
func TestSOSD_OSM_Distribution(t *testing.T) {
	r := &datasets.SOSDReader{
		Path:    sosdPath("osm_cellids_800M_uint64"),
		Label:   "sosd_osm",
		KeyType: datasets.SOSDUint64,
	}
	keys, err := r.Keys()
	if err != nil {
		t.Skipf("osm_cellids_800M_uint64 not available: %v", err)
	}
	n := len(keys)

	const nBins = 1000
	minK, maxK := keys[0], keys[n-1]
	span := maxK - minK
	if span == 0 {
		span = 1
	}
	binWidth := span / nBins
	if binWidth == 0 {
		binWidth = 1
	}
	counts := make([]uint64, nBins)
	for _, k := range keys {
		b := int((k - minK) / binWidth)
		if b >= nBins {
			b = nBins - 1
		}
		counts[b]++
	}
	var maxCount uint64
	emptyBins := 0
	singletons := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
		if c == 0 {
			emptyBins++
		}
		if c == 1 {
			singletons++
		}
	}
	fmt.Printf("\n=== sosd_osm (uint64) ===\n")
	fmt.Printf("n=%d  span=%d (~2^%.1f)\n", n, span, log2u64(span))
	fmt.Printf("max-bin=%d  mean-bin=%.0f\n", maxCount, float64(n)/float64(nBins))
	fmt.Printf("empty bins: %d / %d  (%.1f%%)\n",
		emptyBins, nBins, 100*float64(emptyBins)/float64(nBins))
	fmt.Printf("singleton bins (count=1): %d / %d  (%.1f%%)\n",
		singletons, nBins, 100*float64(singletons)/float64(nBins))

	histDir := filepath.Join(filepath.Dir(r.Path), "..", "bench_results",
		"plots", "distributions")
	histDirLinear := filepath.Join(filepath.Dir(r.Path), "..", "bench_results",
		"plots", "distributions_linear")
	histDirSmooth := filepath.Join(filepath.Dir(r.Path), "..", "bench_results",
		"plots", "distributions_smoothed")
	os.MkdirAll(histDir, 0755)
	os.MkdirAll(histDirLinear, 0755)
	os.MkdirAll(histDirSmooth, 0755)

	ptsLog := make([]testutils.Point, nBins)
	ptsLin := make([]testutils.Point, nBins)
	for i, c := range counts {
		x := float64(i) / float64(nBins)
		yLin := float64(c) / float64(maxCount)
		yLog := yLin
		if yLog == 0 {
			yLog = 1e-9
		}
		ptsLog[i] = testutils.Point{X: x, Y: yLog}
		ptsLin[i] = testutils.Point{X: x, Y: yLin}
	}
	title := fmt.Sprintf("Key Density — sosd_osm (n=%d, %d bins)", n, nBins)

	for _, target := range []struct {
		dir   string
		path  string
		scale testutils.AxisScale
		pts   []testutils.Point
	}{
		{histDir, "hist_sosd_osm.svg", testutils.Log10, ptsLog},
		{histDirLinear, "hist_sosd_osm.svg", testutils.Linear, ptsLin},
	} {
		out := filepath.Join(target.dir, target.path)
		err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title: title, XLabel: "Normalized Key Position",
			YLabel: "Density (bin / max)",
			YScale: target.scale, XMax: 25,
		}, []testutils.SeriesData{{
			Name: r.Label, Color: "#27ae60", Marker: "none", Points: target.pts,
		}}, out)
		if err != nil {
			t.Fatalf("svg %s: %v", out, err)
		}
		fmt.Printf("svg → %s\n", out)
	}

	for _, w := range []int{5, 7, 11} {
		smoothed := movingAverage(ptsLin, w)
		path := filepath.Join(histDirSmooth,
			fmt.Sprintf("hist_sosd_osm_w%d.svg", w))
		err := testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title:  fmt.Sprintf("%s — smoothed (window=%d bins)", title, w),
			XLabel: "Normalized Key Position",
			YLabel: fmt.Sprintf("Mean of %d-bin window", w),
			YScale: testutils.Linear, XMax: 25,
		}, []testutils.SeriesData{{
			Name: r.Label, Color: "#27ae60", Marker: "none", Points: smoothed,
		}}, path)
		if err != nil {
			t.Fatalf("smoothed w=%d: %v", w, err)
		}
		fmt.Printf("svg → %s\n", path)
	}
}
