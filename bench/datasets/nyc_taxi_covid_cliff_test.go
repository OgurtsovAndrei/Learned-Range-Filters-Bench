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

// TestNYCTaxi_COVIDCliff aggregates Jan..May 2020 yellow trip data and
// renders linear + log + smoothed histograms over the 5-month window. The
// expected feature: a sharp drop ~Mar 22 2020 when NYC entered lockdown.
func TestNYCTaxi_COVIDCliff(t *testing.T) {
	matches, _ := filepath.Glob(nycTaxiPath("yellow_tripdata_2020-0[1-5].parquet"))
	if len(matches) == 0 {
		t.Skip("no 2020 Q1/Q2 files yet")
	}
	sort.Strings(matches)
	if len(matches) < 3 {
		t.Skipf("only %d/5 Q1+Q2 files present, skipping", len(matches))
	}
	t.Logf("aggregating %d files: %v", len(matches),
		filenames(matches))

	r := &datasets.NYCTaxiPickupReader{
		Files: matches,
		Label: "nyc_yellow_2020_q1_q2",
	}
	lo := uint64(time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).UnixNano())
	hi := uint64(time.Date(2020, 6, 1, 0, 0, 0, 0, time.UTC).UnixNano())

	const nBins = 1000
	t0 := time.Now()
	counts, err := r.BinHistogram(lo, hi, nBins)
	if err != nil {
		t.Fatalf("BinHistogram: %v", err)
	}
	elapsed := time.Since(t0)

	var total, maxCount uint64
	for _, c := range counts {
		total += c
		if c > maxCount {
			maxCount = c
		}
	}
	mean := float64(total) / float64(nBins)
	fmt.Printf("\n=== nyc_yellow_2020_q1_q2 (Jan..May) ===\n")
	fmt.Printf("files=%d  trips=%d  bins=%d  binWidth=%v  read=%s\n",
		len(matches), total, nBins,
		time.Duration((hi-lo)/uint64(nBins)), elapsed.Round(time.Millisecond))
	fmt.Printf("mean trips/bin=%.0f  max=%d  mean/max=%.3f\n",
		mean, maxCount, mean/float64(maxCount))

	// Find the bin whose normalized count first drops below 0.4 of the
	// previous bin's count by a sustained margin — the visual cliff.
	for i := 5; i < nBins; i++ {
		prev := float64(counts[i-1])
		cur := float64(counts[i])
		if prev > mean && cur < 0.5*prev && cur < 0.4*float64(maxCount) {
			t0 := time.Unix(0, int64(lo+(hi-lo)*uint64(i-1)/uint64(nBins))).UTC()
			t1 := time.Unix(0, int64(lo+(hi-lo)*uint64(i)/uint64(nBins))).UTC()
			fmt.Printf("first cliff candidate: bin %d → %d  (%.0f → %.0f)\n",
				i-1, i, prev, cur)
			fmt.Printf("  time window: [%s, %s)\n",
				t0.Format(time.RFC3339), t1.Format(time.RFC3339))
			break
		}
	}

	histDir := filepath.Join(filepath.Dir(matches[0]), "..", "bench_results",
		"plots", "distributions")
	histDirLinear := filepath.Join(filepath.Dir(matches[0]), "..", "bench_results",
		"plots", "distributions_linear")
	histDirSmooth := filepath.Join(filepath.Dir(matches[0]), "..", "bench_results",
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
	title := fmt.Sprintf("Trip Count — nyc_yellow_2020_q1_q2 (n=%d, %d bins, raw)",
		total, nBins)
	xLabel := "Day of period (Jan–May 2020, COVID-19 lockdown ~Mar 22)"

	for _, target := range []struct {
		dir   string
		path  string
		scale testutils.AxisScale
		title string
		pts   []testutils.Point
	}{
		{histDir, "hist_nyc_yellow_2020_q1_q2_raw.svg", testutils.Log10, title, ptsLog},
		{histDirLinear, "hist_nyc_yellow_2020_q1_q2_raw.svg", testutils.Linear, title, ptsLin},
	} {
		out := filepath.Join(target.dir, target.path)
		err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title: target.title, XLabel: xLabel,
			YLabel: "Trips per bin / max bin",
			YScale: target.scale, XMax: 25,
		}, []testutils.SeriesData{{
			Name: r.Label, Color: "#db4437", Marker: "none", Points: target.pts,
		}}, out)
		if err != nil {
			t.Fatalf("svg %s: %v", target.path, err)
		}
		fmt.Printf("svg → %s\n", out)
	}

	for _, w := range []int{5, 7, 11} {
		smoothed := movingAverage(ptsLin, w)
		path := filepath.Join(histDirSmooth,
			fmt.Sprintf("hist_nyc_yellow_2020_q1_q2_raw_w%d.svg", w))
		err = testutils.GeneratePerformanceSVG(testutils.PlotConfig{
			Title:  fmt.Sprintf("%s — smoothed (window=%d bins)", title, w),
			XLabel: xLabel,
			YLabel: fmt.Sprintf("Mean of %d-bin window (normalized)", w),
			YScale: testutils.Linear, XMax: 25,
		}, []testutils.SeriesData{{
			Name: r.Label, Color: "#db4437", Marker: "none", Points: smoothed,
		}}, path)
		if err != nil {
			t.Fatalf("smoothed w=%d: %v", w, err)
		}
		fmt.Printf("svg → %s\n", path)
	}
}

func filenames(paths []string) []string {
	out := make([]string, len(paths))
	for i, p := range paths {
		out[i] = filepath.Base(p)
	}
	return out
}
